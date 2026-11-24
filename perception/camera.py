import numpy as np

TOP_SERIAL = "260322272085"
WRIST_SERIAL = "260422275259"


def capture(serial, depth=True, warmup=15, w=640, h=480, fps=30):
    import pyrealsense2 as rs
    from PIL import Image
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
    if depth:
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    prof = pipe.start(cfg)
    try:
        align = rs.align(rs.stream.color) if depth else None
        frames = None
        for _ in range(warmup):
            frames = pipe.wait_for_frames(5000)
        if align is not None:
            frames = align.process(frames)
        color = frames.get_color_frame()
        rgb = np.asanyarray(color.get_data()).copy()
        out = {"rgb": Image.fromarray(rgb), "serial": serial}
        intr = color.profile.as_video_stream_profile().intrinsics
        out["intrinsics"] = {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy, "w": intr.width, "h": intr.height}
        if depth:
            scale = prof.get_device().first_depth_sensor().get_depth_scale()
            out["depth_m"] = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
            out["depth_scale"] = scale
        return out
    finally:
        pipe.stop()


def point_3d(depth_m, intr, x, y, half=3):
    if depth_m is None or x is None or y is None:
        return None
    h, w = depth_m.shape
    xi, yi = int(round(x)), int(round(y))
    win = depth_m[max(0, yi - half): min(h, yi + half + 1), max(0, xi - half): min(w, xi + half + 1)]
    vals = win[(win > 0.05) & (win < 3.0)]
    if vals.size == 0:
        return None
    z = float(np.median(vals))
    return [round((x - intr["ppx"]) * z / intr["fx"], 4), round((y - intr["ppy"]) * z / intr["fy"], 4), round(z, 4)]
