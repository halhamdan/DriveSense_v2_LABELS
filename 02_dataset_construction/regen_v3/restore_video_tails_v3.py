"""
Restores full-session video coverage in release v3 (second pass, 2026-09-14).

build_v3_video_realign.py realigned the videos by dropping the first L =
round(lead_in*30) frames of the released (v1) files, which left each video
ending L frames (~11 s) before its telemetry. This script rebuilds every v3
video so that frame k <-> elapsed_s k/30 AND the video covers the whole
telemetry window:

  FRONT  = released_v1_front[L:]  +  tail from the full-length blurred front
           (Front_Blur_V2_Fix/{tag}_Front_blurred_v2_full.mp4, the same blur
           run the release came from; frame f of it = raw frame f)
  SIDE   = [blurred(raw side frames missing at the start), only D1_S1]
           + released_v1_side[L:]
           + blurred(raw side tail frames)
  where the tail = raw frames [fs_old + N, fs_old + N + L - 1], N = released
  front frame count, fs_old = raw frame index of released frame 0 (estimated
  from UTC time-of-day, refined by frame matching against the full-length front).

Side frames that exist in no blurred source (74 tails of ~11 s, plus 40 s of
D1_S1 whose released side video had been trimmed twice) are blurred here with
the pipeline's own detector and rules (03_video_deidentification/
blur_side_video.py: MTCNN keep_all, p >= 0.7, carry-forward of the nearest
detection with extra padding, Gaussian blur k=51), followed by the session's
researcher mask (the black left strip found in the released side video) and a
generous STATIC safety blur over the driver-head region as a floor, since a
short clip gives the carry-forward little context. Contact sheets of every
restored segment are written for visual review.

Run with the environment that has torch + facenet_pytorch + cv2 (the
authors' Modelling_GPU venv). Resumable; writes build_v3_tails_report.csv.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DOCS = Path(r"C:\Users\halha\OneDrive - Durham University\Documents")
V1 = DOCS / "Published_Dataset_Final"; V3 = DOCS / "Published_Dataset_Final_v3_VIDEO_ALIGNED"
FULLFRONT = DOCS / "Front_Blur_V2_Fix"; RAWSIDE = DOCS / "Published_Dataset"; NATIVE = DOCS / "VBOX_Data" / "Raw"
REPO = Path(__file__).resolve().parents[2]
LEAD = pd.read_csv(REPO / "05_technical_validation" / "validation_output" / "video_lead_in_per_session.csv").set_index("tag")["lead_in_s"]
REPORT_V3 = pd.read_csv(Path(__file__).resolve().parent / "build_v3_report.csv").set_index("tag")
HERE = Path(__file__).resolve().parent; TMP = HERE / "tmp_tails"; TMP.mkdir(exist_ok=True)
SHEETS = REPO / "05_technical_validation" / "validation_output" / "video_tail_review"; SHEETS.mkdir(parents=True, exist_ok=True)
LOG = HERE / "restore_tails.log"
FPS = 30; W, H = 960, 540
NVENC = ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "26", "-preset", "p5", "-pix_fmt", "yuv420p", "-an"]
FACE_PROB = 0.7; BLUR_K = 51
# static safety floor for the side view: the driver's head region in this fixed framing
# (normalised x0, x1, y0, y1); generous on purpose
HEAD_REGION = (0.08, 0.42, 0.05, 0.55)
_mtcnn = None


def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as f: f.write(m + "\n")


def nframes(p: Path) -> int:
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
    return int(o.split(",")[0])


def grey_frames(video: Path, first: int, last: int, w=96, h=54) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='between(n\\,{first}\\,{last})',scale={w}:{h},format=gray", "-vsync", "0", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h * w).astype(np.int16)


def tod(u): return (u // 10000) * 3600 + ((u % 10000) // 100) * 60 + (u % 100)


def fs_old_for(tag: str, d: str, s: str, rel_front: Path, full_front: Path) -> tuple[int, float]:
    r = pd.read_csv(V1 / "Raw_Dataset" / f"D{d}" / f"Session_{s}" / f"{tag}_VBOX_raw.csv", usecols=["utc_tod_s"], nrows=2)
    n = pd.read_csv(NATIVE / f"{tag}.csv", encoding="utf-8-sig", usecols=["UTC time"], nrows=2)
    off = float(r.utc_tod_s.iloc[0] - tod(float(n["UTC time"].iloc[0])))
    if off < -1000: off += 86400
    est = int(round(off * FPS)); win = 60  # sessions with a UTC-vs-elapsed divergence need a wide window
    a = grey_frames(rel_front, 100, 100)[0]; cand = grey_frames(full_front, max(est + 100 - win, 0), est + 100 + win)
    diffs = [float(np.abs(c - a).mean()) for c in cand]; k = int(np.argmin(diffs))
    fs = max(est - win, 0) + k
    # confirm with a second frame far into the session (guards against a chance match)
    a2 = grey_frames(rel_front, 20000, 20000)[0]; c2 = grey_frames(full_front, fs + 20000 - 3, fs + 20000 + 3)
    d2 = [float(np.abs(c - a2).mean()) for c in c2]; k2 = int(np.argmin(d2))
    if k2 != 3: log(f"    fs_old check: second-frame match offset {k2 - 3:+d} frames (diff {d2[k2]:.2f}) -- constant offset not exact")
    return fs, diffs[k]


def mask_width(rel_side: Path, n: int) -> int:
    """Width (px) of the black left strip (researcher mask) in the released side video, from its last frames."""
    g = grey_frames(rel_side, max(n - 30, 0), n - 1, w=W, h=H).reshape(-1, H, W).mean(axis=0).mean(axis=0)  # per-column mean
    x = 0
    while x < W and g[x] < 10: x += 1
    return x if x >= 20 else 0


def init_mtcnn():
    global _mtcnn
    if _mtcnn is None:
        import torch
        from facenet_pytorch import MTCNN
        _mtcnn = MTCNN(image_size=160, margin=20, min_face_size=30, thresholds=[0.6, 0.7, 0.7], factor=0.709, keep_all=True,
                       device="cuda" if torch.cuda.is_available() else "cpu")


def read_frames(video: Path, first: int, last: int) -> list[np.ndarray]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='between(n\\,{first}\\,{last})'", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True).stdout
    arr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    return [arr[i].copy() for i in range(arr.shape[0])]


def blur_clip(frames: list[np.ndarray], mask_px: int) -> tuple[list[np.ndarray], dict]:
    """Pipeline rules on a short clip + static head floor + researcher mask."""
    init_mtcnn()
    boxes = [None] * len(frames)
    for i in range(0, len(frames), 8):
        batch = frames[i:i + 8]
        try:
            bb, pp = _mtcnn.detect([f[:, :, ::-1] for f in batch])
        except Exception as e:  # noqa: BLE001
            log(f"    detect error @{i}: {e}"); continue
        for j, (b, p) in enumerate(zip(bb, pp)):
            if b is None: continue
            kept = [x for x, q in zip(b, p) if q is not None and q >= FACE_PROB]
            if kept: boxes[i + j] = kept
    n_det = sum(b is not None for b in boxes)
    # carry-forward / backward fill
    filled = list(boxes); last = None
    for i in range(len(filled)):
        if filled[i] is not None: last = filled[i]
        elif last is not None: filled[i] = last
    first = next((b for b in boxes if b is not None), None)
    for i in range(len(filled)):
        if boxes[i] is not None: break
        if filled[i] is None and first is not None: filled[i] = first
    out = []
    hx0, hx1, hy0, hy1 = int(HEAD_REGION[0] * W), int(HEAD_REGION[1] * W), int(HEAD_REGION[2] * H), int(HEAD_REGION[3] * H)
    for f, b, orig in zip(frames, filled, boxes):
        o = f.copy()
        # static floor (always)
        o[hy0:hy1, hx0:hx1] = cv2.GaussianBlur(o[hy0:hy1, hx0:hx1], (BLUR_K, BLUR_K), 0)
        if b is not None:
            px, py = (0.22, 0.35) if orig is None else (0.10, 0.20)
            for box in b:
                x1, y1, x2, y2 = (int(v) for v in box); x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
                fw, fh = x2 - x1, y2 - y1
                if fw <= 10 or fh <= 10: continue
                bx, by = max(0, x1 - int(fw * px)), max(0, y1 - int(fh * py)); bw, bh = min(fw + 2 * int(fw * px), W - bx), min(fh + 2 * int(fh * py), H - by)
                o[by:by + bh, bx:bx + bw] = cv2.GaussianBlur(o[by:by + bh, bx:bx + bw], (BLUR_K, BLUR_K), 0)
        if mask_px > 0: o[:, :mask_px] = 0
        out.append(o)
    return out, {"n_frames": len(frames), "n_detected": n_det, "max_gap": _max_gap(boxes)}


def _max_gap(boxes):
    g = m = 0
    for b in boxes:
        g = 0 if b is not None else g + 1; m = max(m, g)
    return m


def write_clip(frames: list[np.ndarray], out: Path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{W}x{H}", "-framerate", str(FPS), "-i", "pipe:0"] + NVENC + [str(out)],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()


def concat(parts: list[tuple[Path, int | None, int | None]], out: Path):
    """parts: (video, first, last) with last None = to end. Frame-accurate select + concat, NVENC."""
    cmd = ["ffmpeg", "-v", "error", "-y"]; fc = []; labels = []
    for i, (v, a, b) in enumerate(parts):
        cmd += ["-i", str(v)]
        sel = f"gte(n\\,{a})" if b is None else f"between(n\\,{a}\\,{b})"
        fc.append(f"[{i}:v]select='{sel}',setpts=N/{FPS}/TB[p{i}]"); labels.append(f"[p{i}]")
    fc.append("".join(labels) + f"concat=n={len(parts)}:v=1:a=0[o]")
    cmd += ["-filter_complex", ";".join(fc), "-map", "[o]", "-r", str(FPS), "-vsync", "cfr"] + NVENC + [str(out)]
    with open(LOG, "a", encoding="utf-8") as err:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=err)
    if r.returncode != 0: raise RuntimeError(f"ffmpeg concat failed for {out.name}")


def sheet(video: Path, idxs: list[int], out: Path):
    fr = [grey_frames(video, i, i, w=320, h=180)[0].reshape(180, 320).astype(np.uint8) for i in idxs]
    cv2.imwrite(str(out), np.hstack(fr))


def process(tag: str, rows: list):
    d, s = tag[1:].split("_S"); raw1 = V1 / "Raw_Dataset" / f"D{d}" / f"Session_{s}"; raw3 = V3 / "Raw_Dataset" / f"D{d}" / f"Session_{s}"
    rel_front = raw1 / f"{tag}_Front_blurred.mp4"; full_front = FULLFRONT / f"D{d}" / f"Session_{s}" / f"{tag}_Front_blurred_v2_full.mp4"
    rel_side = raw1 / f"{tag}_Side_blurred.mp4"; raw_side = RAWSIDE / f"D{d}" / f"Session_{s}" / f"{tag}_Side.mp4"
    L = int(round(float(LEAD[tag]) * FPS)); N = int(REPORT_V3.loc[tag, "Front_frames_in"]); trunc = REPORT_V3.loc[tag, "truncated_at_elapsed_s"]
    fs_old, mdiff = fs_old_for(tag, d, s, rel_front, full_front); nfull = nframes(full_front)
    info = {"tag": tag, "L": L, "N_front": N, "fs_old": fs_old, "fs_match_diff": round(mdiff, 2), "full_front_frames": nfull}
    if not pd.isna(trunc):
        info["note"] = "truncated session: no tail"; rows.append(info); log(f"  {tag}: truncated, videos unchanged"); return
    t0, t1 = fs_old + N, min(fs_old + N + L - 1, nfull - 1); info["tail_raw_range"] = f"{t0}-{t1}"; info["tail_frames"] = t1 - t0 + 1
    expected = (N - L) + (t1 - t0 + 1)
    # ---- front
    outF = raw3 / f"{tag}_Front_blurred.mp4"
    if FORCE or nframes(outF) != expected:
        tt = time.time(); concat([(rel_front, L, None), (full_front, t0, t1)], outF); info["front_frames_out"] = nframes(outF)
        log(f"  {tag} front: {info['front_frames_out']} (expected {expected}) in {time.time() - tt:.0f} s")
    else:
        info["front_frames_out"] = expected
    sheet(outF, [expected - (t1 - t0 + 1) - 1, expected - (t1 - t0 + 1), expected - 1], SHEETS / f"{tag}_front_tail.png")
    # ---- side
    if rel_side.exists() and raw_side.exists():
        n_side = nframes(rel_side); mpx = mask_width(rel_side, n_side); info["side_mask_px"] = mpx; info["side_frames_in"] = n_side
        parts = []; clips = []
        if n_side != N:  # D1_S1: double-trimmed side (frame k = raw 2*fs_old + k) -> blur the missing head
            head0, head1 = fs_old + L, 2 * fs_old - 1
            frames = read_frames(raw_side, head0, head1); bl, st = blur_clip(frames, mpx); c = TMP / f"{tag}_side_head.mp4"; write_clip(bl, c)
            parts.append((c, 0, None)); clips.append(c); info["side_head_range"] = f"{head0}-{head1}"; info["side_head_detect"] = f"{st['n_detected']}/{st['n_frames']} maxgap {st['max_gap']}"
            parts.append((rel_side, 0, None)); log(f"  {tag} side: double-trimmed; head {head0}-{head1} blurred ({st})")
        else:
            parts.append((rel_side, L, None))
        frames = read_frames(raw_side, t0, t1); bl, st = blur_clip(frames, mpx); c = TMP / f"{tag}_side_tail.mp4"; write_clip(bl, c); parts.append((c, 0, None)); clips.append(c)
        info["side_tail_detect"] = f"{st['n_detected']}/{st['n_frames']} maxgap {st['max_gap']}"
        outS = raw3 / f"{tag}_Side_blurred.mp4"; tt = time.time(); concat(parts, outS); info["side_frames_out"] = nframes(outS)
        log(f"  {tag} side: {info['side_frames_out']} (expected {expected}) tail det {info['side_tail_detect']} in {time.time() - tt:.0f} s")
        sheet(outS, [expected - (t1 - t0 + 1) - 1, expected - (t1 - t0 + 1), expected - (t1 - t0 + 1) // 2, expected - 1], SHEETS / f"{tag}_side_tail.png")
        if n_side != N: sheet(outS, [0, (2 * fs_old - fs_old - L) // 2, 2 * fs_old - fs_old - L - 1, 2 * fs_old - fs_old - L], SHEETS / f"{tag}_side_head.png")
        for c in clips: c.unlink(missing_ok=True)
    rows.append(info)


FORCE = False


def main():
    global FORCE
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", nargs="*"); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    FORCE = a.force
    tags = a.sessions or sorted(REPORT_V3.index)
    prev = HERE / "build_v3_tails_report.csv"
    rows = [] if not (a.sessions and prev.exists()) else pd.read_csv(prev).query("tag not in @tags").to_dict("records")
    log(f"=== restore tails: {len(tags)} sessions (force={FORCE})")
    for i, t in enumerate(tags, 1):
        log(f"[{i}/{len(tags)}] {t}")
        try:
            process(t, rows)
        except Exception as e:  # noqa: BLE001
            log(f"  !! {t}: {e}"); rows.append({"tag": t, "error": str(e)})
        pd.DataFrame(rows).to_csv(HERE / "build_v3_tails_report.csv", index=False)
    log("=== done")


if __name__ == "__main__":
    main()
