import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time

from PIL import Image, ImageDraw

from perception import prompts as P
from perception.gpt import GPT
from service.holding import HoldingSettings, RULE_VERSION, H_RULE_VERSION
from robot.home import dataset_states, load_home
from bon.fk import YamFK

ROOT = Path(__file__).resolve().parent
VERSION = 'sisyphus_v1'
ABLATIONS = ('full', 'noverify', 'nomemory', 'noabort')
CROP_HALF, CROP_ZOOM = 48, 4
OPEN = .8
POSITIONS = dict(left='on the left', right='on the right', top='on the top', bottom='on the bottom')


class HomeCheckFailed(Exception):
    pass


def capture():
    from perception import camera
    t = camera.capture(camera.TOP_SERIAL, depth=True)
    w = camera.capture(camera.WRIST_SERIAL, depth=False)
    return t['rgb'], w['rgb'], t['depth_m'], t['intrinsics']


def project(depth, intr, x, y):
    from perception import camera
    return camera.point_3d(depth, intr, x, y)


def run_once(command, env):
    return subprocess.call(command, env=env)


def xy(p):
    try:
        return float(p['x']), float(p['y'])
    except (TypeError, KeyError, ValueError):
        return None


def finite_pixel(p):
    return (isinstance(p, (list, tuple)) and len(p) == 2 and not any(isinstance(v, bool) for v in p)
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p))


def ticks(img, step=64):
    img = img.convert('RGB').copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    for x in range(0, w, step):
        d.line([(x, 0), (x, 8)], fill=(255, 255, 0), width=2)
        d.text((x + 2, 10), str(x), fill=(255, 255, 0))
    for y in range(0, h, step):
        d.line([(0, y), (8, y)], fill=(255, 255, 0), width=2)
        d.text((10, y + 2), str(y), fill=(255, 255, 0))
    return img


def region_item(region):
    return 'bowl' if region == 'bowl' else region + ' region'


def instruction_for(color, shape, position, region):
    obj = color + ' ' + shape + ('' if position is None else ' ' + position)
    if region == 'bowl':
        return 'pick up the %s and place it into the bowl' % obj
    return 'pick up the %s and place it at the center of the %s region' % (obj, region)


def safe_name(text):
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_') or 'object'


def crop_box(size, point):
    if not isinstance(point, (list, tuple)) or len(point) != 2 or any(isinstance(v, bool) for v in point):
        raise ValueError('missing object pixel')
    try:
        x, y = (float(v) for v in point)
    except (TypeError, ValueError) as exc:
        raise ValueError('object pixel must contain finite numbers') from exc
    w, h = size
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < w and 0 <= y < h):
        raise ValueError('object pixel outside the image or nonfinite')
    left, top = round(x) - CROP_HALF, round(y) - CROP_HALF
    return (left, top, left + 2 * CROP_HALF, top + 2 * CROP_HALF)


def zoom_crop(image, point):
    image = image.convert('RGB')
    box = crop_box(image.size, point)
    source = (max(0, box[0]), max(0, box[1]), min(image.width, box[2]), min(image.height, box[3]))
    crop = Image.new('RGB', (2 * CROP_HALF, 2 * CROP_HALF), (112, 112, 112))
    crop.paste(image.crop(source), (source[0] - box[0], source[1] - box[1]))
    crop = crop.resize((crop.width * CROP_ZOOM, crop.height * CROP_ZOOM), Image.Resampling.LANCZOS)
    return crop, dict(attr_crop_box=list(box), attr_crop_source_size=list(image.size),
                      attr_crop_size=list(crop.size), attr_object_px=[float(v) for v in point])


def check_attributes(value):
    if not isinstance(value, dict) or set(value) != {'color', 'shape'}:
        return 'Return exactly the color and shape fields'
    if not isinstance(value['color'], str) or value['color'] not in P.COLORS:
        return 'color must be one word from COLORS'
    if not isinstance(value['shape'], str) or value['shape'] not in P.SHAPES:
        return 'shape must be one word from SHAPES'
    return None


def translate(ask, image, point, crop_path):
    began = time.perf_counter()
    try:
        crop, meta = zoom_crop(image, point)
    except ValueError as exc:
        return dict(attr_status='attr_invalid', attr_color=None, attr_shape=None, attr_raw=None,
                    attr_replies=[], attr_calls=0, attr_error=str(exc), attr_seconds=time.perf_counter() - began)
    with open(crop_path, 'xb') as stream:
        crop.save(stream, format='PNG')
    replies = []
    for index in range(2):
        t0 = time.perf_counter()
        value, raw, error = ask('attr', P.ATTR, [crop], check=check_attributes, tries=1)
        error = error or check_attributes(value)
        replies.append(dict(call=index + 1, raw=raw, error=error, seconds=time.perf_counter() - t0))
        if error is None:
            return dict(meta, attr_status='valid', attr_color=value['color'], attr_shape=value['shape'],
                        attr_raw=raw, attr_replies=replies, attr_calls=len(replies), attr_error=None,
                        attr_seconds=time.perf_counter() - began, attr_crop_path=crop_path)
    return dict(meta, attr_status='attr_invalid', attr_color=None, attr_shape=None, attr_raw=replies[-1]['raw'],
                attr_replies=replies, attr_calls=len(replies), attr_error=replies[-1]['error'],
                attr_seconds=time.perf_counter() - began, attr_crop_path=crop_path)


def refine_point(ask, image, coarse, name, max_shift, tag='refine'):
    info = dict(coarse_px=list(coarse) if coarse else None, refined_px=None, refine_shift_px=None,
                refine_max_shift_px=float(max_shift), refine_status='no_coarse')
    if not coarse:
        return None, None, info
    half, zoom = CROP_HALF, CROP_ZOOM
    x0 = int(max(0, min(image.size[0] - 2 * half, coarse[0] - half)))
    y0 = int(max(0, min(image.size[1] - 2 * half, coarse[1] - half)))
    crop = image.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)).resize((2 * half * zoom, 2 * half * zoom), Image.LANCZOS)
    reply, _, _ = ask(tag, P.REFINE.format(w=crop.size[0], h=crop.size[1], step=64, name=name), [ticks(crop)])
    p = xy(reply or {})
    if not p or not all(math.isfinite(v) for v in p):
        info['refine_status'] = 'refine_failed_keep_coarse'
        return None, tuple(coarse), info
    refined = (round(x0 + p[0] / zoom, 2), round(y0 + p[1] / zoom, 2))
    shift = math.hypot(refined[0] - coarse[0], refined[1] - coarse[1])
    info.update(refined_px=list(refined), refine_shift_px=round(shift, 3))
    if shift > max_shift:
        info['refine_status'] = 'shift_exceeds_delta_keep_coarse'
        return refined, tuple(coarse), info
    info['refine_status'] = 'refined'
    return refined, refined, info


def position_phrase(target, duplicates, min_sep):
    if not duplicates:
        return None, 'unique_color_shape'
    if not finite_pixel(target):
        return None, 'ambiguous_no_target_pixel'
    tx, ty = float(target[0]), float(target[1])
    xs = [float(p[0]) for p in duplicates]
    ys = [float(p[1]) for p in duplicates]
    margins = dict(left=min(xs) - tx, right=tx - max(xs), top=min(ys) - ty, bottom=ty - max(ys))
    side, margin = max(margins.items(), key=lambda kv: kv[1])
    if margin < min_sep:
        return None, 'ambiguous_not_separable'
    return POSITIONS[side], 'position_' + side


class SceneMemory:
    def __init__(self, radius_m, radius_px):
        if not (math.isfinite(radius_m) and radius_m > 0 and math.isfinite(radius_px) and radius_px > 0):
            raise ValueError('memory radii must be positive and finite')
        self.radius_m, self.radius_px = float(radius_m), float(radius_px)
        self.entries = {}

    def __contains__(self, obj):
        return obj in self.entries

    def placed(self):
        return list(self.entries)

    def add(self, obj, **entry):
        self.entries[obj] = dict(entry)

    def remove(self, obj):
        return self.entries.pop(obj, None)

    def distance(self, entry, base, pixel):
        eb = entry.get('base_xy')
        if eb is not None and base is not None:
            return math.hypot(eb[0] - base[0], eb[1] - base[1]), 'base_xy_m', self.radius_m
        ep = entry.get('pixel')
        if ep is not None and finite_pixel(pixel):
            return math.hypot(ep[0] - pixel[0], ep[1] - pixel[1]), 'pixel', self.radius_px
        return None, 'unavailable', None

    def near(self, base, pixel, exclude=()):
        hits = []
        for obj, entry in self.entries.items():
            if obj in exclude:
                continue
            d, metric, radius = self.distance(entry, base, pixel)
            if d is not None and d <= radius:
                hits.append((obj, round(d, 4), metric))
        return hits

    def occupancy(self):
        return {e['region']: (o, e['step']) for o, e in self.entries.items()}

    def report(self):
        return {o: {k: v for k, v in e.items() if k != 'crop'} for o, e in self.entries.items()}


def reselect_check(k):
    def check(value):
        if not isinstance(value, dict) or set(value) != {'index'}:
            return 'Return exactly the index field'
        index = value['index']
        if index is None or (type(index) is int and 1 <= index <= k):
            return None
        return 'index must be an integer from 1 to %d, or null' % k
    return check


def memory_check(value):
    if not isinstance(value, dict) or set(value) != {'still_there'} or not isinstance(value['still_there'], bool):
        return 'Return exactly the still_there field as true or false'
    return None


def scene_points(objects, regions, obj, obj_px, loc, depth, intr, calib):
    import numpy as np
    pixels = {o: (obj_px if o == obj else xy((loc or {}).get(o))) for o in objects}
    pixels.update({region_item(r): xy((loc or {}).get(region_item(r))) for r in regions})
    camera_points, base, missing = {}, {}, {}
    T = None
    if depth is not None and intr is not None:
        T = np.asarray(json.loads(Path(calib).read_text())['T_base_cam'], float)
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
            raise ValueError('Invalid scene-point calibration')
    for name, pixel in pixels.items():
        if pixel is None or not all(math.isfinite(float(v)) for v in pixel):
            missing[name] = 'missing_pixel'
            continue
        if depth is None or intr is None:
            missing[name] = 'no_depth_or_intrinsics'
            continue
        if not (0 <= pixel[0] < depth.shape[1] and 0 <= pixel[1] < depth.shape[0]):
            missing[name] = 'pixel_out_of_bounds'
            continue
        point = project(depth, intr, *pixel)
        try:
            point = np.asarray(point, float)
        except (TypeError, ValueError):
            point = np.asarray(float('nan'))
        if point.shape != (3,) or not np.isfinite(point).all():
            missing[name] = 'invalid_depth'
            continue
        camera_points[name] = point.tolist()
        base[name] = (T[:3, :3] @ point + T[:3, 3]).tolist()
    return dict(
        scene_points={o: base[o][:2] for o in objects if o in base},
        scene_points_missing={o: missing.get(o, 'missing_point') for o in objects if o not in base},
        scene_points_frame='base_xy_m', scene_object_pixels={o: pixels[o] for o in objects},
        scene_object_cam_xyz={o: camera_points[o] for o in objects if o in camera_points},
        region_points={r: base[region_item(r)][:2] for r in regions if region_item(r) in base},
        region_points_missing={r: missing[region_item(r)] for r in regions if region_item(r) in missing},
        object_cam_xyz=camera_points.get(obj), all_camera_points=camera_points,
        calibration_sha256=hashlib.sha256(Path(calib).read_bytes()).hexdigest() if T is not None else None)


def read_execution(data_root, dataset, require_recovery, abort_action):
    row = json.loads((Path(data_root) / dataset / 'execution.json').read_text())
    if row.get('h_rule_version') != H_RULE_VERSION:
        raise ValueError('Missing/mismatched h rule version in execution report')
    if (row.get('abort_enabled') is not True or row.get('abort_rule_version') != RULE_VERSION
            or not isinstance(row.get('wrong_grasp_abort'), bool) or row.get('dataset') != dataset):
        raise ValueError('Missing/mismatched abort execution report')
    HoldingSettings.parse(row.get('holding_parameters'))
    enabled = row.get('abort_action_enabled', True)
    if type(enabled) is not bool or enabled != abort_action:
        raise ValueError('abort_action_enabled differs from requested mode')
    if not enabled and type(row.get('would_abort')) is not bool:
        raise ValueError('Shadow mode requires an explicit would_abort verdict')
    would = row.get('would_abort', row['wrong_grasp_abort'])
    if type(would) is not bool or row['wrong_grasp_abort'] != (would and enabled):
        raise ValueError('Inconsistent active/shadow wrong-grasp report')
    if not enabled and row.get('abort_recovery') is not None:
        raise ValueError('Shadow mode must never execute abort recovery')
    if require_recovery and row['wrong_grasp_abort'] and (row.get('abort_recovery') or {}).get('ok') is not True:
        raise ValueError('Wrong-grasp recovery did not complete successfully')
    keys = ('abort_enabled', 'abort_rule_version', 'wrong_grasp_abort', 'wrong_grasp_check', 'holding_parameters',
            'holding_seen', 'first_holding', 'holding_conditions', 'stop_reason', 'empty_grasp', 'empty_grasp_count',
            'empty_grasp_events', 'd_o', 'd_o_m', 'd_grasp', 'd_grasp_m', 'nearest_other_object',
            'nearest_other_distance_m', 'other_distances_m', 'grasp_check', 'abort_recovery', 'h_rule_version',
            'released_after_holding', 'release_count', 'first_release', 'conditions', 'relocate_on_empty',
            'continuation')
    return dict({k: row.get(k) for k in keys}, abort_action_enabled=enabled, would_abort=would)


def shuffled_regions(scene, run_name, out_root, seed):
    match = re.search(r'(?:^|[-_])(?:r|run|trial|episode)[-_]?(\d+)(?=$|[-_])', run_name, re.I)
    if match is None:
        match = re.search(r'_(\d{1,3})$', run_name)
    if match:
        index, source = int(match[1]), 'run_name'
    else:
        index, source = 1, 'count_started_runs_of_same_scene'
        for log_file in Path(out_root).expanduser().glob('*/log.jsonl'):
            try:
                with log_file.open() as stream:
                    first = next((json.loads(line) for line in stream if line.strip()), {})
                old = first.get('scene') or {}
                if first.get('event') == 'start' and isinstance(old, dict) and old.get('name') == scene.get('name'):
                    index += 1
            except (OSError, ValueError, TypeError):
                continue
    before = list(scene.get('regions', ['red', 'blue', 'white']))
    after = before.copy()
    random.Random(seed + index).shuffle(after)
    return after, dict(event='region_order', run_number=index, run_number_source=source, region_seed=seed,
                       region_random_seed=seed + index, original_regions=before, presented_regions=after)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--ablation', choices=ABLATIONS, default='full')
    for key, value in HoldingSettings().report().items():
        ap.add_argument('--holding-' + key.replace('_', '-'), type=int if key in ('n_min', 'm') else float, default=value)
    ap.add_argument('--bon', type=int, default=5)
    ap.add_argument('--noise-scale', type=float, default=0.5)
    ap.add_argument('--seconds', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-attempts', type=int, default=9)
    ap.add_argument('--cap-per-object', type=int, default=3)
    ap.add_argument('--region-seed', type=int, default=0)
    ap.add_argument('--gpt-model', default='gpt-6.1-sol')
    ap.add_argument('--effort', default='medium')
    ap.add_argument('--calib', default=os.path.expanduser('~/calib_cam_base/calib.json'))
    ap.add_argument('--once', default=str(ROOT / 'once.sh'))
    ap.add_argument('--out-root', default=str(ROOT / 'runs'))
    ap.add_argument('--data-root', default=str(ROOT / 'data'))
    ap.add_argument('--home-tol', type=float, default=0.06)
    ap.add_argument('--home-wait', type=float, default=3.0)
    source = ap.add_mutually_exclusive_group()
    source.add_argument('--home-reference')
    source.add_argument('--home-dataset')
    ap.add_argument('--position-min-sep-px', type=float, default=10.)
    ap.add_argument('--refine-max-shift-px', type=float, default=24.)
    ap.add_argument('--radius-m', type=float, default=.025)
    ap.add_argument('--max-relocations', type=int, default=2)
    ap.add_argument('--memory-radius-px', type=float, default=20.)
    args = ap.parse_args(argv)
    for name in ('position_min_sep_px', 'refine_max_shift_px', 'radius_m', 'memory_radius_px'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            ap.error('--' + name.replace('_', '-') + ' must be positive and finite')
    if args.cap_per_object < 1 or args.max_attempts < 1 or args.max_relocations < 0:
        ap.error('--cap-per-object and --max-attempts must be at least 1, --max-relocations at least 0')
    if not math.isfinite(args.home_tol) or not 0 < args.home_tol <= .06:
        ap.error('--home-tol must be in (0,0.06] metres')
    if args.bon > 1 and not os.path.exists(args.calib):
        ap.error('--bon %d needs the calibration file %s' % (args.bon, args.calib))
    return args


def run(args):
    started = time.perf_counter()
    holding = HoldingSettings.parse({k: getattr(args, 'holding_' + k) for k in HoldingSettings().report()})
    scene = json.loads(Path(args.scene).read_text())
    objects = list(scene['objects'])
    if not objects or any(not isinstance(o, str) or not o.strip() for o in objects) or len(set(objects)) != len(objects):
        raise SystemExit('scene objects must be unique nonempty text identifiers')
    regions, region_order = shuffled_regions(scene, args.run_name, args.out_root, args.region_seed)
    out = Path(args.out_root).expanduser() / args.run_name
    if out.exists():
        raise SystemExit('%s already exists, use another --run-name' % out)
    out.mkdir(parents=True)
    log_path, gpt_path = out / 'log.jsonl', out / 'gpt.jsonl'
    memory_on = args.ablation != 'nomemory'
    abort_action = args.ablation != 'noabort'
    context = {'attempt': None}

    def log(row, path=log_path):
        row = dict(row, version=VERSION, ablation=args.ablation)
        with open(path, 'a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    gpt = GPT(args.gpt_model, args.effort, log=lambda r: log(dict(r, scene_attempt=context['attempt']), gpt_path))
    ask = gpt.ask

    def look(tag):
        top, wrist, depth, intr = capture()
        if depth is not None:
            import numpy as np
            np.save(out / (tag + '_top_depth.npy'), depth.astype('float16'))
        top.save(out / (tag + '_top.png'))
        wrist.save(out / (tag + '_wrist.png'))
        return top, wrist, depth, intr

    log(dict(event='start', scene=scene, args=vars(args), time=time.strftime('%F %T'), wall_start_unix_s=time.time(),
             holding_parameters=holding.report(), abort_rule_version=RULE_VERSION, h_rule_version=H_RULE_VERSION,
             abort_action_enabled=abort_action, memory_enabled=memory_on))
    log(region_order)
    print('run folder:', out)

    home = {}
    if args.home_reference or args.home_dataset:
        reference = load_home(home_reference=args.home_reference, home_dataset=args.home_dataset)
        home['q'] = reference.state
        log(dict(event='home_reference', state=list(reference.state), source=reference.source))

    def home_check(name):
        try:
            first, last = dataset_states(Path(args.data_root) / name)
        except Exception as exc:
            return dict(ok=False, reason='no dataset: ' + type(exc).__name__)
        if 'q' not in home:
            home['q'] = first
            log(dict(event='home_reference', dataset=name, state=first))
        try:
            import numpy as np
            home.setdefault('fk', YamFK())
            dist = float(np.linalg.norm(home['fk'].tip(np.array(last[:6])) - home['fk'].tip(np.array(home['q'][:6]))))
            if not math.isfinite(dist):
                raise ValueError('nonfinite FK home distance')
        except Exception as exc:
            return dict(ok=False, reason='FK home check failed: ' + type(exc).__name__)
        return dict(ok=dist <= args.home_tol, tip_dist_m=round(dist, 4), last_state=last,
                    reason='gripper %.1f cm from home (limit %.1f cm)' % (dist * 100, args.home_tol * 100))

    state = dict(attempts=0, interventions=0, stop_reason='method_reported')
    history, failed_by_object = [], {}
    memory = SceneMemory(args.radius_m, args.memory_radius_px)
    removals, attr_cache = [], {}
    t_scene = time.time()

    def summary(**extra):
        row = dict(event='summary', run_name=args.run_name, scene=scene.get('name'), model=args.model,
                   gpt_model=args.gpt_model, attempts_total=state['attempts'], interventions=state['interventions'],
                   stop_reason=state['stop_reason'], failed_attempts_by_object=failed_by_object,
                   cycle_time_s=round(time.time() - t_scene, 1), total_time_s=round(time.perf_counter() - started, 6),
                   gpt_calls=gpt.calls, gpt_seconds=round(gpt.seconds, 1), region_order=region_order,
                   placed_set=memory.placed() if memory_on else None,
                   scene_memory=memory.report() if memory_on else None, memory_removals=removals)
        row.update(extra)
        if memory_on and not all(o in memory for o in objects):
            row.update(handover=True, interventions=max(1, row['interventions']))
        elif memory_on:
            row['handover'] = False
        log(row)
        return row

    def stop_home(k, reason):
        state['stop_reason'] = 'home_check_failed'
        state['attempts'] += 1
        state['interventions'] += 1
        log(dict(event='home_check_failed', attempt=k, reason=reason))
        summary(status='stopped', home_check=dict(ok=False, reason=reason))
        raise HomeCheckFailed(reason)

    def execute(k, instruction, target, sub, relocate):
        name = 'rollout_sis_%s_a%02d' % (args.run_name, k) + ('r%d' % sub if sub else '')
        if sub and 'q' not in home:
            return stop_home(k, 'no home state for a continuation after an empty grasp')
        env = dict(os.environ, PI05_NOISE_SCALE=str(args.noise_scale), SIS_BON=str(args.bon), SIS_NO_PROMPT='1',
                   SIS_A2_ABORT_ACTION_ENABLED='1' if abort_action else '0',
                   SIS_RELOCATE_ON_EMPTY='1' if relocate else '0', SIS_CONTINUATION='1' if sub else '0',
                   SIS_CALIB_JSON=os.path.abspath(args.calib))
        for key, value in holding.report().items():
            env['SIS_A2_HOLDING_' + key.upper()] = str(value)
        if 'q' in home:
            env['SIS_A2_HOME_STATE'] = json.dumps(list(home['q']))
        else:
            env.pop('SIS_A2_HOME_STATE', None)
        target_path = out / ('attempt_%02d%s_target.json' % (k, 'r%d' % sub if sub else ''))
        target_path.write_text(json.dumps(target, indent=1))
        env['SIS_TARGET_JSON'] = str(target_path)
        if args.bon > 1:
            env['SIS_BON_LOG'] = str(out / ('attempt_%02d%s_bon.jsonl' % (k, 'r%d' % sub if sub else '')))
        print('\n>>> attempt %d%s: %s' % (k, ' (continuation %d)' % sub if sub else '', instruction))
        t0 = time.time()
        rc = run_once(['bash', args.once, args.model, name, instruction, str(args.seed), str(args.seconds)], env)
        t1 = time.time()
        report = dict(abort_enabled=True, abort_rule_version=RULE_VERSION, wrong_grasp_abort=False,
                      abort_action_enabled=abort_action, would_abort=False, holding_parameters=holding.report(),
                      holding_seen=False, stop_reason=None, empty_grasp=False, d_grasp=None, d_grasp_m=None,
                      h_rule_version=H_RULE_VERSION)
        try:
            report = read_execution(args.data_root, name, rc == 0, abort_action)
            if report['holding_parameters'] != holding.report():
                raise ValueError('Holding parameters differ from requested values')
            if report.get('relocate_on_empty') is not relocate or report.get('continuation') is not bool(sub):
                raise ValueError('Relocation/continuation flags differ from requested values')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if rc == 0:
                rc = 65
                log(dict(event='abort_protocol_failed', attempt=k, error=type(exc).__name__ + ': ' + str(exc)))
        if report['wrong_grasp_abort']:
            log(dict(report, event='abort', attempt=k, instruction=instruction, dataset=name, failure='wrong_object'))
        elif report.get('would_abort'):
            log(dict(report, event='would_abort', attempt=k, instruction=instruction, dataset=name))
        if report.get('empty_grasp'):
            log(dict(event='empty_grasp', attempt=k, dataset=name, empty_grasp_count=report.get('empty_grasp_count'),
                     events=report.get('empty_grasp_events')))
        if rc == 0 and report.get('stop_reason') == 'empty_grasp':
            if 'q' not in home:
                try:
                    home['q'] = dataset_states(Path(args.data_root) / name)[0]
                    log(dict(event='home_reference', dataset=name, state=home['q']))
                except Exception as exc:
                    return stop_home(k, 'cannot read the home state: ' + type(exc).__name__)
            print('    empty grasp: locating the %s again' % target['object'])
            return dict(report, dataset=name, seed=args.seed, return_code=rc, t_start=round(t0, 2),
                        t_end=round(t1, 2), duration_s=round(t1 - t0, 1), home_check=None)
        hc = home_check(name)
        print('    home check: %s (%s)' % ('AT HOME' if hc['ok'] else 'NOT AT HOME', hc['reason']))
        result = dict(report, dataset=name, seed=args.seed, return_code=rc, t_start=round(t0, 2), t_end=round(t1, 2),
                      duration_s=round(t1 - t0, 1), home_check=hc)
        if not hc['ok']:
            state['stop_reason'] = 'home_check_failed'
            state['attempts'] += 1
            state['interventions'] += 1
            log(dict(result, event='home_check_failed', attempt=k, instruction=instruction))
            summary(status='stopped', home_check=hc)
            raise HomeCheckFailed(hc['reason'])
        if rc != 0:
            print('    execution failed (exit code %s); the scene stops.' % rc)
            return dict(result, failed=True)
        time.sleep(args.home_wait)
        return result

    def base_xy(depth, intr, pixel):
        if depth is None or intr is None or not finite_pixel(pixel):
            return None
        data = scene_points([], ['_'], None, None, {'_ region': {'x': pixel[0], 'y': pixel[1]}}, depth, intr, args.calib)
        point = data['region_points'].get('_')
        return None if point is None else [float(point[0]), float(point[1])]

    def not_executed(k, obj, region, plan, obj_px, coarse, region_px, target, top, wrist, failure, t_start, fields):
        state['attempts'] += 1
        now = time.time()
        top.save(out / ('attempt_%02d_after_top.png' % k))
        wrist.save(out / ('attempt_%02d_after_wrist.png' % k))
        with (out / ('attempt_%02d_target.json' % k)).open('x') as stream:
            json.dump(target or dict(object=obj, region=region, object_px=obj_px, region_px=region_px), stream, indent=1)
        log(dict(event='attempt', attempt=k, object=obj, region=region, instruction=None, plan=plan, object_px=obj_px,
                 coarse_px=coarse, region_px=region_px, target=target, dataset=None, return_code=failure,
                 executed=False, t_start=t_start, t_end=now, duration_s=0., verification=None, placed=False,
                 failure=failure, **fields))
        history.append(dict(step=k, object=obj, region=region, result='not placed (%s)' % failure, executed=False))
        return count_failure(k, obj)

    def count_failure(k, obj):
        failed_by_object[obj] = failed_by_object.get(obj, 0) + 1
        if failed_by_object[obj] >= args.cap_per_object:
            state['stop_reason'] = 'cap_reached'
            state['interventions'] += 1
            log(dict(event='cap_reached', attempt=k, object=obj, failed_attempts=failed_by_object[obj],
                     cap_per_object=args.cap_per_object))
            return True
        return False

    def reselect(k, obj, top, points, obj_px, hits):
        candidates = []
        for other in objects:
            if other == obj:
                continue
            pixel = points['scene_object_pixels'].get(other)
            if not finite_pixel(pixel) or memory.near(points['scene_points'].get(other), pixel):
                continue
            try:
                crop_box(top.size, pixel)
            except ValueError:
                continue
            candidates.append((other, tuple(pixel)))
        row = dict(event='reselect', attempt=k, object=obj, object_px=list(obj_px) if obj_px else None,
                   near_placed=hits, candidates=[dict(name=n, pixel=list(p)) for n, p in candidates])
        if not candidates:
            log(dict(row, result='no_candidate_away_from_P'))
            return None
        crops = []
        for index, (other, pixel) in enumerate(candidates, 1):
            crop, _ = zoom_crop(top, pixel)
            crop.save(out / ('attempt_%02d_reselect_%d.png' % (k, index)))
            crops.append(crop)
        answer, raw, err = ask('reselect', P.RESELECT.format(obj=obj, k=len(crops)), crops, check=reselect_check(len(crops)))
        index = None if answer is None else answer.get('index')
        if index is None:
            log(dict(row, result='none_matches' if answer is not None else 'gpt_failed', raw=raw, error=err))
            return None
        name, pixel = candidates[index - 1]
        log(dict(row, result='reselected', index=index, chosen_point_of=name, chosen_px=list(pixel), raw=raw))
        return pixel, name

    def recheck(k, image, exclude):
        removed = []
        for other in [o for o in memory.placed() if o not in exclude]:
            entry = memory.entries[other]
            row = dict(event='memory_check', attempt=k, object=other, region=entry['region'],
                       pixel=entry.get('pixel'), stored_crop=entry.get('crop_path'))
            if entry.get('crop') is None or not finite_pixel(entry.get('pixel')):
                log(dict(row, result='no_stored_crop_kept'))
                continue
            try:
                current, _ = zoom_crop(image, entry['pixel'])
            except ValueError:
                log(dict(row, result='crop_failed_kept'))
                continue
            current_path = out / ('attempt_%02d_memcheck_%s.png' % (k, safe_name(other)))
            current.save(current_path)
            answer, raw, err = ask('memory_check', P.MEMORYCHECK.format(obj=other, region=region_item(entry['region'])),
                                   [entry['crop'], current], check=memory_check)
            still = None if answer is None else answer['still_there']
            if still is False:
                memory.remove(other)
                removed.append(other)
                removals.append(dict(attempt=k, object=other))
            log(dict(row, current_crop=str(current_path), still_there=still, raw=raw, error=err,
                     result='removed_from_P' if still is False else 'kept'))
        return removed

    def remember(k, obj, region, image, depth, intr, region_px, region_point, located=None):
        if located is not None:
            pixel, point = located
            source, info, lerr = 'verified', {}, None
        else:
            loc, _, lerr = ask('relocate', P.LOCATE.format(w=image.size[0], h=image.size[1], items=json.dumps([obj])), [ticks(image)])
            _, pixel, info = refine_point(ask, image, xy((loc or {}).get(obj)), obj, args.refine_max_shift_px,
                                          tag='relocate_refine')
            source, point = 'relocated', None
        if not finite_pixel(pixel):
            pixel, source = (tuple(region_px), 'region_center_fallback') if finite_pixel(region_px) else (None, 'missing')
        if point is None:
            point = base_xy(depth, intr, pixel) if pixel is not None else None
        if point is None and source == 'region_center_fallback' and region_point is not None:
            point = list(region_point)
        crop, crop_path = None, None
        if pixel is not None:
            try:
                crop, _ = zoom_crop(image, pixel)
                crop_path = str(out / ('memory_%s_a%02d.png' % (safe_name(obj), k)))
                crop.save(crop_path)
            except ValueError:
                crop = None
        memory.add(obj, region=region, step=k, pixel=list(pixel) if pixel is not None else None,
                   base_xy=point, crop=crop, crop_path=crop_path, point_source=source)
        log(dict(event='memory_add', attempt=k, object=obj, region=region, point_source=source,
                 pixel=list(pixel) if pixel is not None else None, base_xy=point, crop_path=crop_path,
                 locate_error=lerr, **info))

    def plan_check(p, pending):
        if p.get('all_done') is True:
            if memory_on and pending:
                return 'all_done must be false while Objects to reset is nonempty'
            return None
        if p.get('next') not in pending:
            return 'next must be exactly one of %s' % json.dumps(pending)
        if p.get('region') not in regions:
            return 'region must be one of: %s' % ', '.join(regions)
        occ = memory.occupancy() if memory_on else {}
        free = [r for r in regions if r not in occ or occ[r][0] == p.get('next')]
        held = occ.get(p['region'])
        if memory_on and held and held[0] != p.get('next') and free:
            return ('the %s region already holds the %s, placed at step %d; each region takes one object. '
                    'Regions still free: %s' % (p['region'], held[0], held[1], ', '.join(free)))
        return None

    if os.environ.get('SIS_NO_PROMPT') != '1':
        input('Enter = start the scene on the robot: ')
    def ground(k, obj, region, top, depth, intr, tag):
        items = list(objects) + [region_item(r) for r in regions]
        loc, _, _ = ask('locate', P.LOCATE.format(w=top.size[0], h=top.size[1], items=json.dumps(items)), [ticks(top)])
        coarse = xy((loc or {}).get(obj))
        region_px = xy((loc or {}).get(region_item(region)))
        _, obj_px, refine_info = refine_point(ask, top, coarse, obj, args.refine_max_shift_px)
        log(dict(event='refine', attempt=k, stage=tag, object=obj, **refine_info))
        points = scene_points(objects, regions, obj, obj_px, loc, depth, intr, args.calib)
        g = dict(coarse=coarse, region_px=region_px, obj_px=obj_px, refine=refine_info, points=points,
                 chosen=None, hits=None)
        if memory_on and memory.placed():
            hits = memory.near(points['scene_points'].get(obj), obj_px)
            if hits:
                g['hits'] = hits
                g['chosen'] = reselect(k, obj, top, points, obj_px, hits)
                if g['chosen'] is not None:
                    g['obj_px'] = g['chosen'][0]
                    loc = dict(loc or {}, **{g['chosen'][1]: None})
                    g['points'] = scene_points(objects, regions, obj, g['obj_px'], loc, depth, intr, args.calib)
        camera_points = g['points'].pop('all_camera_points')
        g['target'] = dict(object=obj, region=region, object_px=g['obj_px'], region_px=region_px,
                           region_cam_xyz=camera_points.get(region_item(region)), intrinsics=intr,
                           frame='top camera (convert with the calibrated extrinsics)', **g['points'])
        return g

    def verify(k, obj, region, ex, image, depth, intr, region_point):
        held = ex.get('holding_seen') is True
        g = (ex.get('conditions') or {}).get('gripper_measured')
        opened = isinstance(g, (int, float)) and not isinstance(g, bool) and math.isfinite(g) and g > OPEN
        info = dict(held=held, gripper_end=g, opened=opened, radius_m=args.radius_m, region_xy=region_point,
                    object_xy=None, distance_m=None, object_px=None)
        if not held:
            return False, 'not_grasped', info, None
        if not opened:
            return False, 'not_released', info, None
        loc, _, _ = ask('relocate', P.LOCATE.format(w=image.size[0], h=image.size[1], items=json.dumps([obj])), [ticks(image)])
        _, pixel, refine_info = refine_point(ask, image, xy((loc or {}).get(obj)), obj, args.refine_max_shift_px,
                                             tag='relocate_refine')
        point = base_xy(depth, intr, pixel) if finite_pixel(pixel) else None
        info.update(object_px=list(pixel) if finite_pixel(pixel) else None, object_xy=point, refine=refine_info)
        if point is None:
            return False, 'not_located', info, None
        if region_point is None:
            return False, 'no_region_point', info, None
        info['distance_m'] = round(math.hypot(point[0] - region_point[0], point[1] - region_point[1]), 4)
        if info['distance_m'] > args.radius_m:
            return False, 'off_target', info, None
        return True, None, info, (tuple(pixel), point)

    k = 0
    while k < args.max_attempts:
        k += 1
        context['attempt'] = k
        top, wrist, depth, intr = look('attempt_%02d_before' % k)
        pending = [o for o in objects if o not in memory] if memory_on else objects
        prompt = P.PLAN.format(objects=json.dumps(pending), regions=', '.join(regions),
                               history=json.dumps(history, indent=1) if memory_on and history else '[]')
        log(dict(event='gpt_plan_prompt', attempt=k, prompt=prompt, presented_regions=regions))
        plan, _, err = ask('plan', prompt, [top, wrist], check=lambda p: plan_check(p, pending))
        if plan is None:
            log(dict(event='gpt_error', attempt=k, error=err))
            state['stop_reason'] = 'gpt_error'
            break
        if plan.get('all_done') is True:
            log(dict(event='gpt_all_done', attempt=k, plan=plan))
            state['stop_reason'] = 'gpt_all_done'
            break
        obj, region = plan['next'], plan['region']
        g = ground(k, obj, region, top, depth, intr, 'plan')
        coarse, region_px, obj_px, target = g['coarse'], g['region_px'], g['obj_px'], g['target']
        if g['hits'] and g['chosen'] is None:
            if not_executed(k, obj, region, plan, obj_px, coarse, region_px, target, top, wrist,
                            'target_on_placed', time.time(), dict(near_placed=g['hits'], refine=g['refine'])):
                break
            continue
        marked = top.copy()
        d = ImageDraw.Draw(marked)
        for pt, col, lab in ((coarse, (255, 0, 255), 'coarse'), (obj_px, (0, 255, 0), obj), (region_px, (255, 255, 0), region)):
            if pt:
                d.ellipse([pt[0] - 7, pt[1] - 7, pt[0] + 7, pt[1] + 7], outline=col, width=3)
                d.text((pt[0] + 9, pt[1] - 6), lab, fill=col)
        marked.save(out / ('attempt_%02d_plan.png' % k))
        attrs = translate(ask, top, obj_px, str(out / ('attempt_%02d_attr_crop.png' % k)))
        attrs.update(object_id=obj)
        log(dict(event='attribute_translation', attempt=k, object=obj, region=region, **attrs))
        if attrs['attr_status'] != 'valid':
            if not_executed(k, obj, region, plan, obj_px, coarse, region_px, target, top, wrist, 'attr_invalid',
                            time.time() - attrs['attr_seconds'], attrs):
                break
            continue
        attr_cache[obj] = (attrs['attr_color'], attrs['attr_shape'])
        duplicates, unknown = [], []
        for other in objects:
            if other == obj:
                continue
            pixel = g['points']['scene_object_pixels'].get(other)
            if not finite_pixel(pixel):
                unknown.append(dict(object=other, reason='no_pixel'))
                continue
            if other not in attr_cache:
                other_attrs = translate(ask, top, pixel, str(out / ('attempt_%02d_attr_crop_%s.png' % (k, safe_name(other)))))
                log(dict(event='attribute_translation_other', attempt=k, object=other, **other_attrs))
                if other_attrs['attr_status'] != 'valid':
                    unknown.append(dict(object=other, reason='attr_invalid'))
                    continue
                attr_cache[other] = (other_attrs['attr_color'], other_attrs['attr_shape'])
            if attr_cache[other] == attr_cache[obj]:
                duplicates.append(dict(object=other, pixel=list(pixel)))
        phrase, status = position_phrase(obj_px, [p['pixel'] for p in duplicates], args.position_min_sep_px)
        attrs.update(position=phrase, position_status=status, position_duplicates=duplicates, position_unknown=unknown)
        instruction = instruction_for(attrs['attr_color'], attrs['attr_shape'], phrase, region)
        ex = execute(k, instruction, target, 0, args.max_relocations > 0)
        relocations = []
        while not ex.get('failed') and ex.get('stop_reason') == 'empty_grasp':
            sub = len(relocations) + 1
            top_r, _, depth_r, intr_r = look('attempt_%02d_relocate%d' % (k, sub))
            g_r = ground(k, obj, region, top_r, depth_r, intr_r, 'relocate%d' % sub)
            ok = finite_pixel(g_r['obj_px']) and not (g_r['hits'] and g_r['chosen'] is None)
            if ok:
                target = g_r['target']
            relocations.append(dict(sub=sub, located=ok, object_px=g_r['obj_px'], dataset=ex.get('dataset')))
            log(dict(event='relocate_after_empty_grasp', attempt=k, sub=sub, object=obj, located=ok,
                     object_px=g_r['obj_px'], kept_previous_target=not ok))
            ex = execute(k, instruction, target, sub, sub < args.max_relocations)
        if ex.get('failed'):
            log(dict(ex, event='run_once_failed', attempt=k, object=obj, region=region, instruction=instruction))
            state['stop_reason'] = 'run_once_failed'
            break
        state['attempts'] += 1
        top2, _, depth2, intr2 = look('attempt_%02d_after' % k)
        located, check = None, None
        if ex.get('wrong_grasp_abort'):
            placed, failure, source = False, 'wrong_object', 'abort'
        elif args.ablation == 'noverify':
            placed = ex.get('holding_seen') is True and ex.get('stop_reason') == 'early_done'
            failure, source = None if placed else 'not_held_released', 'held_released'
        else:
            placed, failure, check, located = verify(k, obj, region, ex, top2, depth2, intr2,
                                                     g['points']['region_points'].get(region))
            source = 'eq9'
        print('    placed=%s failure=%s' % (placed, failure))
        removed = recheck(k, top2, exclude=(obj,)) if memory_on else []
        if memory_on and placed:
            remember(k, obj, region, top2, depth2, intr2, region_px, g['points']['region_points'].get(region), located)
        log(dict(ex, event='attempt', attempt=k, object=obj, region=region, instruction=instruction, plan=plan,
                 object_px=obj_px, coarse_px=coarse, region_px=region_px, target=target, verification=check,
                 verification_source=source, placed=placed, failure=failure, refine=g['refine'],
                 reselected=g['chosen'] is not None, relocations=relocations, memory_removed=removed,
                 placed_set=memory.placed() if memory_on else None, **attrs))
        history.append(dict(step=k, object=obj, region=region,
                            result='placed' if placed else 'not placed (%s)' % failure, memory_removed=removed))
        if memory_on and all(o in memory for o in objects):
            state['stop_reason'] = 'all_placed'
            log(dict(event='all_placed', attempt=k, objects=list(objects)))
            break
        if not placed and count_failure(k, obj):
            break
    else:
        state['stop_reason'] = 'max_attempts'
        state['interventions'] += 1
        log(dict(event='max_attempts', max_attempts=args.max_attempts))
    context['attempt'] = None
    look('final')
    return summary()


def main(argv=None):
    args = parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=1))
        return 0
    except HomeCheckFailed as exc:
        print('HOME CHECK FAILED: %s. The scene stopped; return the arm home and start a new run.' % exc)
        return 4


if __name__ == '__main__':
    raise SystemExit(main())
