from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np

from bon.fk import YamFK


def endpoints(fk, candidates, state, h, s=1.0):
    p = fk.tip(np.asarray(state, float)[:6])
    return p + s * (fk.tip(np.asarray(candidates, float)[:, h, :6]) - p)


def load_targets(target_json: str, calib_json: str) -> Dict[str, Optional[np.ndarray]]:
    t = json.load(open(target_json))
    c = json.load(open(calib_json))
    T = np.asarray(c["T_base_cam"], float)
    out = {}
    for k in ("object", "region"):
        p = t.get("%s_cam_xyz" % k)
        out[k] = None if p is None else (T[:3, :3] @ np.asarray(p, float) + T[:3, 3])
    out["table_z_base"] = float(c.get("table_z_base", 0.0))
    return out


class BonSelector:
    def __init__(self, targets: Dict, holding, fk: YamFK = None, horizon_index: int = 9, near_off_m: float = 0.06,
                 lift_m: float = 0.03):
        if not callable(holding):
            raise ValueError("holding must be a callable returning h (Eq.8)")
        self.t, self.fk, self.holding = targets, fk or YamFK(), holding
        self.h, self.near, self.lift = horizon_index, near_off_m, lift_m
        self.stage, self.latched, self.close_z = None, False, None
        self.log = []

    def _stage(self, state: np.ndarray, tip_now: np.ndarray) -> int:
        carrying = bool(self.holding())
        if carrying and self.close_z is None:
            self.close_z = float(tip_now[2])
        if not carrying:
            self.close_z = None
        return 2 if carrying else 1

    def choose(self, candidates: np.ndarray, state: np.ndarray) -> Dict:
        cand = np.asarray(candidates, float)
        state = np.asarray(state, float)
        tip_now = self.fk.tip(state[:6])
        stage = self._stage(state, tip_now)
        if stage != self.stage:
            self.stage, self.latched = stage, False
        target = self.t.get("object" if stage == 1 else "region")
        rec = {"stage": stage, "index": 0, "reason": None, "dist_now": None, "scores": None}
        if target is None:
            rec["reason"] = "no_target"
        else:
            d_now = float(np.linalg.norm(tip_now[:2] - target[:2]))
            rec["dist_now"] = round(d_now, 4)
            height = 0.0 if self.close_z is None else float(tip_now[2] - self.close_z)
            if stage == 2 and height <= self.lift:
                rec["reason"] = "stage2_before_lift"
            elif self.latched:
                rec["reason"] = "near_target_latched"
            elif d_now < self.near:
                self.latched = True
                rec["reason"] = "near_target"
            else:
                h = min(self.h, cand.shape[1] - 1)
                ends = endpoints(self.fk, cand, state, h)[:, :2]
                dist = np.linalg.norm(ends - target[:2], axis=1)
                rec["index"] = int(np.argmin(dist))
                rec["scores"] = [round(float(x), 4) for x in dist]
                rec["reason"] = "searched"
        self.log.append(rec)
        return rec
