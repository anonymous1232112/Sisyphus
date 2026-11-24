from pathlib import Path
import argparse
from collections import Counter
from importlib import metadata
import json
import struct


def inspect_safetensors(path: Path) -> dict:
    size_before = path.stat().st_size
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Safetensors header is incomplete; copying may still be in progress.')
        header_size = struct.unpack('<Q', prefix)[0]
        if not (2 <= header_size <= 64 * 1024 * 1024):
            raise ValueError('Invalid safetensors header length.')
        raw = stream.read(header_size)
        if len(raw) != header_size:
            raise ValueError('Safetensors JSON header is incomplete.')
    header = json.loads(raw)
    entries = {key:value for key,value in header.items() if key != '__metadata__'}
    ends = []
    for value in entries.values():
        start, end = value['data_offsets']
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end):
            raise ValueError('Invalid tensor offsets.')
        ends.append(end)
    expected_size = 8 + header_size + max(ends, default=0)
    size_after = path.stat().st_size
    if size_before != size_after or size_after != expected_size:
        raise ValueError('Safetensors file size does not match its header; finish copying before execution.')
    return {'file': path.name, 'bytes': size_after, 'tensor_count': len(entries),
            'dtypes': dict(Counter(value['dtype'] for value in entries.values())),
            'lm_head_keys': [key for key in entries if key.endswith('lm_head.weight')],
            'header_and_length_complete': True,
            'note': 'Header and length check only; not a full-file checksum or model-load test.'}


def inspect_model(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError('The requested model directory does not exist yet.')
    config = json.loads((root/'config.json').read_text())
    problems = []
    if config.get('type') != 'pi05':
        problems.append('Expected type=pi05.')
    expected_inputs = {'observation.state': [7], 'observation.images.top': [3,480,640],
                       'observation.images.wrist': [3,480,640]}
    actual_inputs = {key: value.get('shape') for key,value in config.get('input_features',{}).items()}
    if actual_inputs != expected_inputs:
        problems.append('Input features differ from 7-D state and the existing two RGB cameras.')
    if config.get('output_features',{}).get('action',{}).get('shape') != [7]:
        problems.append('Action must have seven dimensions.')
    expected_names = [f'joint_{index}.pos' for index in range(6)] + ['gripper.pos']
    if config.get('action_feature_names') != expected_names:
        problems.append('Action joint names/order require user review.')
    if config.get('dtype') != 'bfloat16':
        problems.append('Policy dtype differs from the known bfloat16 setup; review before execution.')
    if config.get('n_action_steps') != 10:
        problems.append('n_action_steps differs from the reference execution horizon 10.')

    processors = {}
    tokenizer_cache = []
    for name in ['policy_preprocessor.json','policy_postprocessor.json']:
        data = json.loads((root/name).read_text())
        state_files = []
        for step in data.get('steps',[]):
            state_file = step.get('state_file')
            if state_file:
                path = (root/state_file).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError('Processor state references a file outside the model directory.')
                state_files.append(inspect_safetensors(path))
            if step.get('registry_name') == 'tokenizer_processor':
                tokenizer = step.get('config',{}).get('tokenizer_name')
                if tokenizer:
                    cache = Path.home()/'.cache/huggingface/hub'/('models--'+tokenizer.replace('/','--'))
                    tokenizer_cache.append({'name': tokenizer, 'default_cache_directory_exists': cache.is_dir()})
        processors[name] = {'state_files': state_files}

    index_path = root/'model.safetensors.index.json'
    if (root/'model.safetensors').is_file():
        weights = [root/'model.safetensors']
    elif index_path.is_file():
        index = json.loads(index_path.read_text())
        weights = [root/name for name in sorted(set(index['weight_map'].values()))]
    else:
        raise ValueError('No model.safetensors or shard index found; finish copying first.')
    weight_report = [inspect_safetensors(path) for path in weights]
    versions = {}
    for package in ['lerobot','torch','transformers','lerobot_robot_yam']:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = 'distribution metadata not found'
    return {'status': 'needs_user' if problems else 'metadata_checks_passed', 'model_directory': str(root.resolve()),
            'policy': {key: config.get(key) for key in ['type','dtype','chunk_size','n_action_steps','num_inference_steps','use_relative_actions','action_feature_names']},
            'input_features': actual_inputs, 'action_shape': config.get('output_features',{}).get('action',{}).get('shape'),
            'weights': weight_report, 'processors': processors, 'tokenizer_cache': tokenizer_cache,
            'versions': versions, 'issues_requiring_review': problems,
            'execution_not_performed': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = inspect_model(args.model.resolve())
    except Exception as exc:
        print(json.dumps({'status':'needs_user','error_type':type(exc).__name__,'error':str(exc), 'execution_not_performed':True},ensure_ascii=False,indent=2))
        raise SystemExit(1)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(0 if report['status'] == 'metadata_checks_passed' else 1)
