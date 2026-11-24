import json
from pathlib import Path

import numpy as np

from .holding import HoldingTracker,RULE_VERSION


def scene_from_target(target_path):
    if not target_path:return None,{},dict(_target='no_target_json')
    data=json.loads(Path(target_path).read_text())
    raw=data.get('scene_points') or {}
    if not isinstance(raw,dict):raise ValueError('scene_points must be a mapping of object names to base xy')
    if raw and data.get('scene_points_frame','base_xy_m')!='base_xy_m':
        raise ValueError('scene_points must already be base_xy_m; never transform twice')
    points={};missing=dict(data.get('scene_points_missing') or {})
    for name,value in raw.items():
        try:point=np.asarray(value,float)
        except (TypeError,ValueError):missing[name]='invalid_base_xy';continue
        if point.shape!=(2,) or not np.isfinite(point).all():missing[name]='invalid_base_xy';continue
        points[name]=point.copy()
    return data.get('object'),points,missing


class WrongGraspDetector:
    def __init__(self,target_name,scene_points=None,*,settings=None,fk=None,missing=None,abort_action_enabled=True):
        if type(abort_action_enabled) is not bool:
            raise ValueError('abort_action_enabled must be a boolean')
        self.abort_action_enabled=abort_action_enabled
        self.target_name=target_name;self.points={}
        self.missing=dict(missing or {})
        for name,value in (scene_points or {}).items():
            point=np.asarray(value,float)
            if point.shape!=(2,) or not np.isfinite(point).all():raise ValueError('Expected finite base xy')
            self.points[name]=point.copy()
        self.tracker=HoldingTracker(settings)
        self.fk=fk;self.last=None
        self.available=target_name in self.points and any(n!=target_name for n in self.points)
        self.unavailable_reason=('target_point_missing' if target_name not in self.points else
            'no_other_object_points' if not self.available else None)

    def update(self,state,elapsed_s,tip_xyz=None):
        state=np.asarray(state,float)
        if state.shape!=(7,) or not np.isfinite(state).all():raise ValueError('Seven finite measured values required')
        event=self.tracker.observe(state[6],elapsed_s)
        if event is None:return None
        if tip_xyz is None:
            if self.fk is None:
                from bon.core import YamFK
                self.fk=YamFK()
            tip_xyz=self.fk.tip(state[:6])
        tip=np.asarray(tip_xyz,float)
        if tip.shape!=(3,) or not np.isfinite(tip).all():raise ValueError('Invalid current FK tip')
        d_o=None if self.target_name not in self.points else float(np.linalg.norm(tip[:2]-self.points[self.target_name]))
        distances={n:float(np.linalg.norm(tip[:2]-p)) for n,p in self.points.items() if n!=self.target_name}
        nearest=min(distances,key=distances.get) if distances else None
        d_j=distances[nearest] if nearest is not None else None
        wrong=self.available and d_j<d_o
        status='unavailable' if not self.available else 'wrong_object' if wrong else 'target_nearest'
        self.last=dict(event,abort_rule_version=RULE_VERSION,would_abort=bool(wrong),
            abort_action_enabled=self.abort_action_enabled,
            wrong_grasp_abort=bool(wrong) and self.abort_action_enabled,
            wrong_grasp_check=status,unavailable_reason=self.unavailable_reason,
            d_o=d_o,d_o_m=d_o,d_grasp=d_o,d_grasp_m=d_o,
            nearest_other_object=nearest,nearest_other_distance_m=d_j,other_distances_m=distances,
            target_object=self.target_name,scene_points={n:p.tolist() for n,p in self.points.items()},
            scene_points_missing=self.missing,tip_xyz_now=tip.tolist(),measured_state=state.tolist())
        return dict(self.last)

    def holding(self):
        return bool(self.tracker.holding_now)

    def released_after_holding(self):
        return bool(self.tracker.released_after_holding)

    def note_action(self,action,state,elapsed_s):
        if isinstance(action,dict):g=action['gripper.pos']
        else:
            action=np.asarray(action,float)
            if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Seven finite submitted action values required')
            g=action[6]
        return self.tracker.command(g,float(state[6]),elapsed_s)

    def report(self):
        r=dict(abort_enabled=True,abort_rule_version=RULE_VERSION,
            abort_action_enabled=self.abort_action_enabled,
            would_abort=bool(self.last and self.last['would_abort']),
            wrong_grasp_abort=bool(self.last and self.last['wrong_grasp_abort']),
            wrong_grasp_check=self.last['wrong_grasp_check'] if self.last else 'pending_holding' if self.available else 'unavailable',
            unavailable_reason=self.unavailable_reason,grasp_check=self.last,**self.tracker.report())
        if self.last is not None:r['first_holding']=dict(self.last)
        for key in ('d_o','d_o_m','d_grasp','d_grasp_m','nearest_other_object','nearest_other_distance_m','other_distances_m'):
            r[key]=self.last.get(key) if self.last else None
        return r
