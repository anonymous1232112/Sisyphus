from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(HERE, "yam.urdf")
GRASP_OFFSET = (0.0, 0.0, 0.14465)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axis_angle(axis, q):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    q = np.asarray(q, float)[:, None, None]
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


class YamFK:
    def __init__(self, urdf: str = DEFAULT_URDF, tip_link: str = "gripper", grasp_offset=GRASP_OFFSET):
        root = ET.parse(urdf).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        by_child = {j.find("child").get("link"): j for j in joints.values()}
        chain: List[ET.Element] = []
        link = tip_link
        while link in by_child:
            j = by_child[link]
            chain.append(j)
            link = j.find("parent").get("link")
        chain.reverse()
        self.joints = []
        for j in chain:
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
            ax = j.find("axis")
            axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]
            self.joints.append((j.get("name"), j.get("type"), T, axis))
        self.revolute = [n for n, t, _, _ in self.joints if t in ("revolute", "continuous")]
        self.grasp_offset = np.append(np.asarray(grasp_offset, float), 1.0)

    def tip_pose(self, q: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(np.asarray(q, float))
        if q.shape[1] < len(self.revolute):
            raise ValueError("need %d joint angles, got %d" % (len(self.revolute), q.shape[1]))
        N = q.shape[0]
        T = np.tile(np.eye(4), (N, 1, 1))
        k = 0
        for name, typ, T0, axis in self.joints:
            T = T @ T0
            if typ in ("revolute", "continuous"):
                R = np.tile(np.eye(4), (N, 1, 1))
                R[:, :3, :3] = _axis_angle(axis, q[:, k])
                T = T @ R
                k += 1
        return T

    def tip(self, q: np.ndarray) -> np.ndarray:
        single = np.asarray(q).ndim == 1
        p = (self.tip_pose(q) @ self.grasp_offset)[:, :3]
        return p[0] if single else p
