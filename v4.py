"""IR dizilerinden etiketli objeleri silip arka planı dolduran v4 (tek dosya).

Fikir: objenin arkasındaki arka plan başka karelerde görünür. O karelerdeki gerçek pikseller kamera hareketine
göre hizalanıp taşınır; gerekirse kenarlar gradyan alanında (Poisson) birleştirilir. Hiç açılmayan piksel için
isteğe bağlı öğrenilmiş dolgu (--lama) vardır. Her silme işlemine objenin kendi yerindeki kanıtla bir karar
verilir: iyi, orta, kotu.

Ana kullanım:
    python v4.py --project "C:\\proje" --out "C:\\cikti" --preview-every 20
    python v4.py --frames tif --labels etiketler --out cikti --lama ir_lama.pt

Araçlar (aynı dosyada):
    python v4.py sentez --out KLASOR --scene gok --camera sabit --frames 600 --seed 1
    python v4.py degerlendir --bench SENTETIK --run CIKTI
    python v4.py kalibre --pairs BENCH1:RUN1 BENCH2:RUN2 --out kalibrasyon.json
    python v4.py sayfa --run CIKTI --frames KARELER --count 150 --out SAYFA
    python v4.py uyum --sheet SAYFA

Bağımlılıklar: numpy, opencv-python, scipy; yalnız --lama için torch.
"""
import argparse
import csv
import json
import math
import re
import shutil
import struct
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import factorized

# ---------------------------------------------------------------- sabitler

BOX_PAD_FRACTION, BOX_PAD_MIN, BOX_PAD_MAX = 0.25, 4, 10
MAX_REGION_FRACTION = 0.20
MAX_SOURCES_PER_SIDE = 6
MIN_GAIN_FRACTION, MIN_GAIN_PIXELS = 0.04, 16
RING_INNER, RING_WIDTH_MIN, RING_WIDTH_MAX, RING_MIN_PIXELS = 3, 4, 12, 30
WINSOR = 20
COPY_MAX_OFFSET, COPY_MAX_A1 = 0.5, 1.25
REFINE_SKIP_A1, REFINE_MIN_GAIN = 1.2, 0.10
IDENTITY_SNAP = 0.25
STATIC_PAIR_P95, STATIC_LONG_GAP, STATIC_LONG_P95 = 0.3, 50, 1.0
FEATURE_CLEARANCE = 12
CORE_Z, GROW_Z, CORE_MIN_AREA, HOLE_DILATE, HOLE_FULL_FRACTION = 4.0, 2.0, 4, 2, 0.85
UNREVEALED_MAX = 0.10
KAPPA, T_IYI, T_KOTU = 0.8, 1.4, 2.5
LEFT_Z, LEFT_ORTA, LEFT_KOTU = 4.0, 0.75, 1.5
SEAM_MIN_PAIRS, SEAM_ORTA, SEAM_KOTU = 6, 2.0, 3.0
TEX_ORTA, TEX_KOTU = 1.6, 2.5
FLICKER_ORTA = 2.5
VISIBLE_FRACTION = 0.005
NOISE_PAIRS, NOISE_BLOCK, NOISE_QUANTILE = 30, 32, 0.20
CACHE_BYTES = 300 * 2 ** 20

TIFF_SUFFIXES = (".tif", ".tiff")
FRAMES_SUBDIR = Path("2_Etiketlenecek_Veri") / "tif"
COCO_SUBDIR = Path("3_Etiketlenmis_Veri_Ciktisi")
CATEGORIES = ("iyi", "orta", "kotu")
CONFIDENCE = {"iyi": 200, "orta": 128, "kotu": 0}
COLORS = {"iyi": (0, 200, 0), "orta": (0, 200, 255), "kotu": (0, 0, 255)}
SQRT2 = math.sqrt(2.0)
REPORT_COLUMNS = ["frame", "sequence", "object", "label_x0", "label_y0", "label_x1", "label_y1", "region",
                  "hole_x0", "hole_y0", "hole_x1", "hole_y1", "hole_pixels", "rect_pixels", "mode", "n_sources",
                  "gap_near", "gap_far", "one_sided", "unrevealed", "a1", "a2", "a2_coverage", "offset", "leftover",
                  "seam", "texture", "flicker", "e_hat", "category", "reason", "refined", "chain_reliable",
                  "sigma_t", "sigma_eff"]
ALIGN_COLUMNS = ["sequence", "frame", "dx", "dy", "rot_deg", "scale", "corner_px", "n_points", "n_inliers", "reliable"]


# ---------------------------------------------------------------- okuma / yazma

def frame_number(path):
    match = re.search(r"(\d+)(?!.*\d)", Path(path).stem)
    if match is None:
        raise ValueError(f"Dosya adında kare numarası yok: {path}")
    return int(match.group(1))


def list_frames(folder):
    paths = [p for p in Path(folder).iterdir() if p.suffix.lower() in TIFF_SUFFIXES]
    if not paths:
        raise FileNotFoundError(f"TIFF bulunamadı: {folder}")
    return sorted(paths, key=lambda p: (frame_number(p), p.name))


def tiff_bits(data):
    """Klasik TIFF başlığındaki BitsPerSample (etiket 258); BigTIFF ya da bozuk başlıkta None."""
    if len(data) < 8:
        return None
    order = "<" if bytes(data[:2]) == b"II" else ">"
    if struct.unpack_from(order + "H", data, 2)[0] != 42:
        return None
    offset = struct.unpack_from(order + "I", data, 4)[0]
    count = struct.unpack_from(order + "H", data, offset)[0]
    for entry in range(count):
        position = offset + 2 + 12 * entry
        tag, kind, n = struct.unpack_from(order + "HHI", data, position)
        if tag == 258:
            value = position + 8
            if n * 2 > 4:
                value = struct.unpack_from(order + "I", data, value)[0]
            return struct.unpack_from(order + "H", data, value)[0]
    return 1


def read_image(path):
    """cv2.imread Windows'ta ASCII dışı yolda çalışmaz; paketli 12/14 bit TIFF reddedilir."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if Path(path).suffix.lower() in TIFF_SUFFIXES:
        bits = tiff_bits(data)
        if bits is not None and bits % 8:
            raise IOError(f"{bits} bit paketli TIFF desteklenmiyor, önce 16 bit kapsayıcıya çevrilmeli: {path}")
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Okunamadı: {path}")
    return image


def write_image(path, image):
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise IOError(f"Yazılamadı: {path}")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.dtype != image.dtype or not np.array_equal(decoded, image):
        raise IOError(f"Yazılan veri girdiyle aynı değil: {path}")
    encoded.tofile(str(path))


def plane(image):
    """Geometri ve kararlar için gri düzlem (float32): tek kanal ya da ilk 3 kanalın ortalaması."""
    if image.ndim == 2:
        return image.astype(np.float32)
    return image[..., :3].astype(np.float32).mean(axis=2)


def fill_channels(image):
    """Dolgunun uygulanacağı kanallar: tek kanal [None], çok kanalda ilk 3 (alfa değişmez)."""
    return [None] if image.ndim == 2 else list(range(min(image.shape[2], 3)))


def channel_plane(image, channel):
    return image.astype(np.float32) if channel is None else image[..., channel].astype(np.float32)


# ---------------------------------------------------------------- etiketler

def clip_box(x, y, w, h, width, height):
    x0, y0 = max(0.0, float(x)), max(0.0, float(y))
    x1, y1 = min(float(width), float(x) + float(w)), min(float(height), float(y) + float(h))
    return (x0, y0, x1, y1) if x1 - x0 >= 1 and y1 - y0 >= 1 else None


def load_coco(images_path, annotations_path, frames, width, height, missing):
    images = json.loads(Path(images_path).read_text(encoding="utf-8"))["images"]
    annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))["annotations"]
    position = {f.stem: i for i, f in enumerate(frames)}
    boxes = [None] * len(frames)
    by_id = {}
    for image in images:
        index = position.get(Path(image["file_name"]).stem)
        if index is not None:
            by_id[image["id"]] = index
            boxes[index] = []
    for annotation in annotations:
        index = by_id.get(annotation["image_id"])
        if index is not None:
            box = clip_box(*annotation["bbox"], width, height)
            if box:
                boxes[index].append(box)
    return apply_missing(boxes, missing, default="unknown")


def load_yolo(folder, frames, width, height, missing):
    boxes = []
    for frame in frames:
        path = Path(folder) / f"{frame.stem}.txt"
        if not path.exists():
            boxes.append(None)
            continue
        items = []
        for line in path.read_text().splitlines():
            if line.strip():
                cx, cy, w, h = map(float, line.split()[1:5])
                box = clip_box((cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height, width, height)
                if box:
                    items.append(box)
        boxes.append(items)
    return apply_missing(boxes, missing, default="unknown")


def load_csv(path, frames, width, height, missing):
    position = {}
    for i, frame in enumerate(frames):
        position.setdefault(frame_number(frame), i)
    boxes = [None] * len(frames)
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            index = position.get(int(row["frame"]))
            if index is None:
                continue
            if boxes[index] is None:
                boxes[index] = []
            box = clip_box(row["x"], row["y"], row["w"], row["h"], width, height)
            if box:
                boxes[index].append(box)
    # CSV boş kareyi ifade edemez: geçmeyen kare boş sayılır.
    return apply_missing(boxes, missing, default="empty")


def apply_missing(boxes, missing, default):
    mode = default if missing == "auto" else missing
    return [b if b is not None else ([] if mode == "empty" else None) for b in boxes]


def sequence_key(path):
    stem = Path(path).stem
    return stem[:-6] if stem[-6:].isdigit() else stem.rstrip("0123456789")


def split_sequences(frames, boxes):
    """Önek başına dizi; bilinmeyen kareler (kutu listesi None) çıkarılır."""
    groups = OrderedDict()
    unknown = Counter()
    for frame, frame_boxes in zip(frames, boxes):
        key = sequence_key(frame)
        groups.setdefault(key, ([], []))
        if frame_boxes is None:
            unknown[key] += 1
            continue
        groups[key][0].append(frame)
        groups[key][1].append(frame_boxes)
    return [(key, f, b, unknown[key]) for key, (f, b) in groups.items() if f]


# ---------------------------------------------------------------- geometri

def pad_rect(box, pad, width, height):
    """Etiket + pay, tam sayı, yarı açık [x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    return (max(0, int(math.floor(x0 - pad))), max(0, int(math.floor(y0 - pad))),
            min(width, int(math.ceil(x1 + pad))), min(height, int(math.ceil(y1 + pad))))


def box_pad(box, fixed):
    if fixed is not None:
        return int(fixed)
    w, h = box[2] - box[0], box[3] - box[1]
    return int(np.clip(round(BOX_PAD_FRACTION * max(w, h)), BOX_PAD_MIN, BOX_PAD_MAX))


def transform_rect(M, rect, width, height):
    x0, y0, x1, y1 = rect
    corners = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], np.float64).T
    moved = M @ corners
    xs, ys = moved[0] / moved[2], moved[1] / moved[2]
    return (max(0, int(math.floor(xs.min()))), max(0, int(math.floor(ys.min()))),
            min(width, int(math.ceil(xs.max()))), min(height, int(math.ceil(ys.max()))))


def near_identity(M, rect):
    x0, y0, x1, y1 = rect
    corners = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]], np.float64).T
    moved = M @ corners
    return float(np.abs(moved[:2] / moved[2] - corners[:2]).max()) < IDENTITY_SNAP


def map_coords(M, xs, ys, identity):
    """Hedef piksel koordinatlarını kaynak koordinatına çevirir, en yakın piksele yuvarlar."""
    if identity:
        return xs.astype(np.int64), ys.astype(np.int64)
    w = M[2, 0] * xs + M[2, 1] * ys + M[2, 2]
    X = (M[0, 0] * xs + M[0, 1] * ys + M[0, 2]) / w
    Y = (M[1, 0] * xs + M[1, 1] * ys + M[1, 2]) / w
    return np.floor(X + 0.5).astype(np.int64), np.floor(Y + 0.5).astype(np.int64)


def inside_and_free(sx, sy, rects, width, height):
    ok = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
    for x0, y0, x1, y1 in rects:
        ok &= ~((sx >= x0) & (sx < x1) & (sy >= y0) & (sy < y1))
    return ok


def translation(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], np.float64)


def to_3x3(A):
    return np.vstack([A.astype(np.float64), [0, 0, 1]])


def rms_clip(values, limit):
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.clip(values, -limit, limit) ** 2)))


def mad(values):
    return float(np.median(np.abs(values - np.median(values)))) if values.size else 0.0


# ---------------------------------------------------------------- kare önbelleği

class FrameStore:
    def __init__(self, paths, budget=CACHE_BYTES):
        self.paths, self.budget = paths, budget
        self.cache, self.gray_cache, self.used = OrderedDict(), OrderedDict(), 0

    def raw(self, t):
        if t in self.cache:
            self.cache.move_to_end(t)
            return self.cache[t]
        image = read_image(self.paths[t])
        self.cache[t] = image
        self.used += image.nbytes
        while self.used > self.budget and len(self.cache) > 2:
            _, old = self.cache.popitem(last=False)
            self.used -= old.nbytes
            self.gray_cache.pop(_, None)
        return image

    def gray(self, t):
        if t not in self.gray_cache:
            self.gray_cache[t] = plane(self.raw(t))
            while len(self.gray_cache) > max(4, len(self.cache)):
                self.gray_cache.popitem(last=False)
        return self.gray_cache[t]


# ---------------------------------------------------------------- global hizalama

def to8(image, low, high):
    scaled = np.clip((image - low) / max(high - low, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(scaled, (0, 0), 1.5)


def clearance_mask(shape, rects, clearance, border):
    mask = np.full(shape, 255, np.uint8)
    for x0, y0, x1, y1 in rects:
        mask[max(0, y0 - clearance):y1 + clearance, max(0, x0 - clearance):x1 + clearance] = 0
    if border:
        mask[:border], mask[-border:], mask[:, :border], mask[:, -border:] = 0, 0, 0, 0
    return mask


def track(prev8, cur8, points, window, levels, max_error):
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
    forward, st1, _ = cv2.calcOpticalFlowPyrLK(prev8, cur8, points, None, winSize=window, maxLevel=levels,
                                               criteria=criteria)
    backward, st2, _ = cv2.calcOpticalFlowPyrLK(cur8, prev8, forward, None, winSize=window, maxLevel=levels,
                                                criteria=criteria)
    ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((backward - points).reshape(-1, 2), axis=1) <= max_error)
    return points.reshape(-1, 2)[ok], forward.reshape(-1, 2)[ok]


def pair_transform(prev, cur, rects_prev, rects_cur):
    """Kare cur koordinatını prev koordinatına götüren 3x3 benzerlik dönüşümü ve bilgi."""
    low, high = np.percentile(prev, (1, 99))
    prev8, cur8 = to8(prev, low, high), to8(cur, low, high)
    mask = clearance_mask(prev.shape, rects_prev, FEATURE_CLEARANCE, 8)
    points = cv2.goodFeaturesToTrack(prev8, maxCorners=800, qualityLevel=0.01, minDistance=8, blockSize=7, mask=mask)
    info = {"n_points": 0, "n_inliers": 0, "reliable": False}
    if points is None or len(points) < 30:
        return None, info
    p_prev, p_cur = track(prev8, cur8, points.astype(np.float32), (21, 21), 4, 1.0)
    if len(p_cur):
        h, w = cur.shape
        keep = inside_and_free(np.floor(p_cur[:, 0]).astype(np.int64), np.floor(p_cur[:, 1]).astype(np.int64),
                               [(x0 - FEATURE_CLEARANCE, y0 - FEATURE_CLEARANCE, x1 + FEATURE_CLEARANCE,
                                 y1 + FEATURE_CLEARANCE) for x0, y0, x1, y1 in rects_cur], w, h)
        p_prev, p_cur = p_prev[keep], p_cur[keep]
    info["n_points"] = int(len(p_cur))
    if len(p_cur) < 30:
        return None, info
    cv2.setRNGSeed(0)
    A, inliers = cv2.estimateAffinePartial2D(p_cur, p_prev, method=cv2.RANSAC, ransacReprojThreshold=1.0,
                                             maxIters=2000, confidence=0.995, refineIters=10)
    if A is None:
        return None, info
    info["n_inliers"] = int(inliers.sum())
    info["reliable"] = info["n_inliers"] >= max(20, 0.4 * len(p_cur))
    return to_3x3(A), info


def describe(P, width, height):
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]], np.float64).T
    moved = P @ corners
    corner = float(np.linalg.norm(moved[:2] / moved[2] - corners[:2], axis=0).max())
    scale = float(math.hypot(P[0, 0], P[1, 0]))
    rot = float(math.degrees(math.atan2(P[1, 0], P[0, 0])))
    return float(P[0, 2]), float(P[1, 2]), rot, scale, corner


# ---------------------------------------------------------------- dizi hazırlığı

class SequenceInfo:
    """İşçilere gönderilen dizi bilgisi (küçük, pickle edilebilir)."""

    def __init__(self, name, frames, boxes, unknown, args):
        self.name, self.frames, self.boxes, self.unknown = name, [str(f) for f in frames], boxes, unknown
        first = read_image(frames[0])
        self.height, self.width = first.shape[:2]
        self.dtype = first.dtype
        self.ndim = first.ndim
        self.n = len(frames)
        self.rects = [[pad_rect(b, box_pad(b, args.box_pad), self.width, self.height) for b in fb] for fb in boxes]
        self.C = np.repeat(np.eye(3)[None], self.n, axis=0)
        self.Cinv = self.C.copy()
        self.pair_reliable = np.ones(self.n, bool)
        self.align_rows = []
        self.camera = "sabit"
        self.motion_p95 = 0.0
        self.occ = []
        self.sigma_t = 1.0
        self.range = 1.0
        self.sigma_eff = 1.0

    def M(self, a, b):
        """Kare a koordinatından kare b koordinatına."""
        return self.Cinv[b] @ self.C[a]

    def reliable_between(self, a, b):
        if self.camera == "sabit":
            return True
        lo, hi = min(a, b), max(a, b)
        return bool(self.pair_reliable[lo + 1:hi + 1].all())


def prepare_sequence(info, args, log):
    store = FrameStore(info.frames)
    n, width, height = info.n, info.width, info.height
    P = [np.eye(3)] * n
    reliable = np.ones(n, bool)
    corners = []
    pair_infos = [None] * n
    if args.camera != "static" and n > 1:
        last_good = None
        previous = store.gray(0)
        for t in range(1, n):
            current = store.gray(t)
            matrix, pinfo = pair_transform(previous, current, info.rects[t - 1], info.rects[t])
            if matrix is None or not pinfo["reliable"]:
                reliable[t] = False
                matrix = last_good if last_good is not None else np.eye(3)
            else:
                last_good = matrix
            P[t] = matrix
            pair_infos[t] = pinfo
            corners.append(describe(matrix, width, height)[4])
            previous = current
    camera = "sabit"
    if args.camera == "moving":
        camera = "hareketli"
    elif args.camera == "auto" and n > 1:
        if reliable[1:].mean() < 0.5:
            camera = "gozlenemiyor"
        else:
            long_moves = []
            starts = [int(s) for s in np.linspace(0, n - 1 - STATIC_LONG_GAP, 20)] if n > STATIC_LONG_GAP + 1 else []
            for s in starts:
                matrix, pinfo = pair_transform(store.gray(s), store.gray(s + STATIC_LONG_GAP), info.rects[s],
                                               info.rects[s + STATIC_LONG_GAP])
                if matrix is not None and pinfo["reliable"]:
                    long_moves.append(describe(matrix, width, height)[4])
            pair_p95 = float(np.percentile(corners, 95)) if corners else 0.0
            long_p95 = float(np.percentile(long_moves, 95)) if long_moves else 0.0
            camera = "sabit" if pair_p95 < STATIC_PAIR_P95 and long_p95 < STATIC_LONG_P95 else "hareketli"
    if camera == "gozlenemiyor":
        print(f"UYARI: {info.name}: kare çiftlerinin yarısından azı hizalanabildi; kamera sabit varsayıldı.")
    info.camera = camera
    info.motion_p95 = float(np.percentile(corners, 95)) if corners else 0.0
    for t in range(n):
        if t == 0:
            info.align_rows.append([info.name, Path(info.frames[0]).name, 0, 0, 0, 1, 0, 0, 0, 1])
            continue
        dx, dy, rot, scale, corner = describe(P[t], width, height)
        pinfo = pair_infos[t] or {"n_points": 0, "n_inliers": 0, "reliable": camera == "sabit"}
        info.align_rows.append([info.name, Path(info.frames[t]).name, round(dx, 3), round(dy, 3), round(rot, 4),
                                round(scale, 5), round(corner, 3), pinfo["n_points"], pinfo["n_inliers"],
                                int(bool(reliable[t]))])
    if camera == "hareketli":
        C = [np.eye(3)]
        for t in range(1, n):
            C.append(C[-1] @ P[t])
        info.C = np.array(C)
        info.Cinv = np.array([np.linalg.inv(c) for c in C])
        info.pair_reliable = reliable
    # Doluluk: karenin kutuları + ısıl iz için önceki margin karenin kutuları bu kareye taşınmış.
    occ = []
    for t in range(n):
        rects = list(info.rects[t])
        for k in range(1, args.margin + 1):
            if t - k >= 0:
                M = info.M(t - k, t)
                for rect in info.rects[t - k]:
                    rects.append(transform_rect(M, rect, width, height) if camera == "hareketli" else rect)
        occ.append(rects)
    info.occ = occ
    estimate_noise(info, store)
    log[info.name] = {"ad": info.name, "kare": n, "bilinmeyen_kare": info.unknown, "kamera": camera,
                      "hareket_p95_px": round(info.motion_p95, 3), "sigma_t": round(info.sigma_t, 4),
                      "sigma_eff": round(info.sigma_eff, 4), "aralik": round(info.range, 3)}


def estimate_noise(info, store):
    n, width, height = info.n, info.width, info.height
    values = []
    picks = sorted({int(round(v)) for v in np.linspace(1, n - 1, min(NOISE_PAIRS, n - 1))}) if n > 1 else []
    ys, xs = np.mgrid[:height, :width]
    for t in picks:
        current, previous = store.gray(t), store.gray(t - 1)
        M = info.M(t, t - 1)
        identity = info.camera != "hareketli" or near_identity(M, (0, 0, width, height))
        sx, sy = map_coords(M, xs.ravel(), ys.ravel(), identity)
        ok = inside_and_free(sx, sy, info.occ[t - 1], width, height)
        ok &= inside_and_free(xs.ravel(), ys.ravel(), info.occ[t], width, height)
        sample = np.zeros(width * height, np.float32)
        sample[ok] = previous[sy[ok], sx[ok]]
        d = (current.ravel() - sample).reshape(height, width)
        ok = ok.reshape(height, width)
        blocks = []
        for by in range(0, height - NOISE_BLOCK + 1, NOISE_BLOCK):
            for bx in range(0, width - NOISE_BLOCK + 1, NOISE_BLOCK):
                m = ok[by:by + NOISE_BLOCK, bx:bx + NOISE_BLOCK]
                if m.mean() >= 0.5:
                    v = d[by:by + NOISE_BLOCK, bx:bx + NOISE_BLOCK][m]
                    blocks.append(1.4826 * mad(v) / SQRT2)
        if blocks:
            values.append(float(np.quantile(blocks, NOISE_QUANTILE)))
    if values:
        info.sigma_t = max(float(np.median(values)), 1e-3)
    else:
        first = store.gray(0)
        info.sigma_t = max(1.4826 * float(np.median(np.abs(np.diff(first, axis=1)))) / SQRT2, 1e-3)
    sample_frames = sorted({int(round(v)) for v in np.linspace(0, n - 1, min(20, n))})
    info.range = float(np.median([np.subtract(*np.percentile(store.gray(t), (99, 1))) for t in sample_frames]))
    info.sigma_eff = max(info.sigma_t, info.visible_fraction * info.range)


# ---------------------------------------------------------------- öğrenilmiş dolgu

class LamaRunner:
    def __init__(self, path, device):
        import torch
        self.torch = torch
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            torch.set_num_threads(1)
        self.model = torch.jit.load(str(path), map_location=self.device)
        self.model.eval()

    def predict(self, window, hole):
        outside = window[~hole] if (~hole).any() else window.ravel()
        low, high = np.percentile(outside, (0.5, 99.5))
        high = max(float(high), float(low) + 1e-3)
        image = np.clip((window - low) / (high - low), 0, 1).astype(np.float32)
        mask = hole.astype(np.float32)
        h, w = image.shape
        scale = 1.0
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            mask = (cv2.resize(mask, size, interpolation=cv2.INTER_AREA) > 0).astype(np.float32)
        sh, sw = image.shape
        ph, pw = (-sh) % 8, (-sw) % 8
        image = np.pad(image, ((0, ph), (0, pw)), mode="reflect")
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
        with self.torch.no_grad():
            x = self.torch.from_numpy(image)[None, None].to(self.device).expand(-1, 3, -1, -1).contiguous()
            m = self.torch.from_numpy(mask)[None, None].to(self.device)
            out = self.model(x, m).mean(dim=1)[0].clamp(0, 1).cpu().numpy()
        out = out[:sh, :sw]
        if scale != 1.0:
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        return out.astype(np.float32) * (high - low) + low


# ---------------------------------------------------------------- Poisson

class Poisson:
    """Delik pikselleri için Poisson denklemi; matris bir kez çözümlenir, tüm kanallarda kullanılır.
    Delik dışı komşu sınır koşuludur (Dirichlet), görüntü dışı komşu denklemde yoktur (Neumann)."""

    def __init__(self, hole, image_edges):
        h, w = hole.shape
        self.ys, self.xs = np.nonzero(hole)
        n = len(self.ys)
        self.n = n
        index = np.full(hole.shape, -1, np.int64)
        index[self.ys, self.xs] = np.arange(n)
        top, bottom, left, right = image_edges
        diagonal = np.zeros(n)
        rows, cols = [], []
        self.links = []
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            qy, qx = self.ys + dy, self.xs + dx
            inside = (qy >= 0) & (qy < h) & (qx >= 0) & (qx < w)
            # Kırpıntı kenarı görüntü kenarı değilse delik oraya değmez (kırpıntıda pay var).
            diagonal += inside
            p = np.nonzero(inside)[0]
            qy, qx = qy[inside], qx[inside]
            q = index[qy, qx]
            self.links.append((p, qy, qx, q))
            linked = q >= 0
            rows.append(p[linked])
            cols.append(q[linked])
        data = np.concatenate([diagonal] + [-np.ones(len(r)) for r in rows])
        matrix = sparse.csc_matrix((data, (np.concatenate([np.arange(n)] + rows), np.concatenate([np.arange(n)] + cols))),
                                   shape=(n, n))
        self.solve = factorized(matrix)

    def run(self, target, guidance):
        """guidance(p_index, qy, qx, q_index) -> (değer, var_mı) dizileri."""
        b = np.zeros(self.n)
        for p, qy, qx, q in self.links:
            g, ok = guidance(p, qy, qx, q)
            value = np.where(ok, g, 0.0)
            outer = q < 0
            value[outer] += target[qy[outer], qx[outer]]
            b[p] += value
        return self.solve(b)


# ---------------------------------------------------------------- karar

def e_hat_of(a1, a2, kappa):
    if a1 is None and a2 is None:
        return None
    if a1 is None:
        return kappa * a2
    if a2 is None:
        return a1
    return max(a1, kappa * a2)


def decide(s, t_iyi, t_kotu, kappa):
    """Bölüm 6.11: ilk tutan kural."""
    mode = s["mode"]
    textured = mode in ("kopya", "poisson")
    e_hat = e_hat_of(s.get("a1"), s.get("a2"), kappa)

    def gt(name, limit):
        value = s.get(name)
        return value is not None and value > limit

    if s.get("too_big"):
        return "kotu", "cok_buyuk", e_hat
    if mode == "uzamsal":
        return "kotu", "kaynak_yok", e_hat
    if gt("leftover", LEFT_KOTU):
        return "kotu", "kalinti", e_hat
    if gt("seam", SEAM_KOTU):
        return "kotu", "dikis", e_hat
    if textured and gt("texture", TEX_KOTU):
        return "kotu", "doku", e_hat
    if e_hat is not None and e_hat > t_kotu:
        return "kotu", "hata", e_hat
    if mode == "lama":
        return "orta", "ogrenilmis", e_hat
    if s.get("one_sided"):
        return "orta", "tek_yon", e_hat
    if s.get("a1") is None:
        return "orta", "halka_yok", e_hat
    if gt("leftover", LEFT_ORTA):
        return "orta", "kalinti", e_hat
    if gt("seam", SEAM_ORTA):
        return "orta", "dikis", e_hat
    if textured and gt("texture", TEX_ORTA):
        return "orta", "doku", e_hat
    if gt("flicker", FLICKER_ORTA):
        return "orta", "titreme", e_hat
    if e_hat is not None and e_hat > t_iyi:
        return "orta", "hata", e_hat
    return "iyi", "tamam", e_hat


# ---------------------------------------------------------------- bölge işleme

class Source:
    __slots__ = ("t", "side", "distance", "M", "identity", "gray", "raw", "valid", "offset", "artik", "refined",
                 "ring_ok")


class RegionSolver:
    def __init__(self, info, store, params, lama):
        self.info, self.store, self.p, self.lama = info, store, params, lama

    # ---- kaynak örnekleme

    def sample(self, t_src, M, identity, crop):
        info = self.info
        cx0, cy0, cx1, cy1 = crop
        ys, xs = np.mgrid[cy0:cy1, cx0:cx1]
        sx, sy = map_coords(M, xs.ravel(), ys.ravel(), identity)
        valid = inside_and_free(sx, sy, info.occ[t_src], info.width, info.height)
        inside = (sx >= 0) & (sx < info.width) & (sy >= 0) & (sy < info.height)
        raw = self.store.raw(t_src)
        shape = (cy1 - cy0, cx1 - cx0)
        gray_full = self.store.gray(t_src)
        gray = np.zeros(sx.shape, np.float32)
        gray[inside] = gray_full[sy[inside], sx[inside]]
        channels = []
        for channel in fill_channels(raw):
            values = np.zeros(sx.shape, np.float32)
            plane_full = raw if channel is None else raw[..., channel]
            values[inside] = plane_full[sy[inside], sx[inside]]
            channels.append(values.reshape(shape))
        return gray.reshape(shape), channels, valid.reshape(shape)

    def ring_stats(self, target, gray, valid, ring):
        """Halka artığı: medyan farkı o, kırpılmış karekök-ortalama artik (Bölüm 6.7)."""
        use = ring & valid
        if use.sum() < RING_MIN_PIXELS:
            return None, None
        d = target[use] - gray[use]
        o = float(np.median(d))
        return o, rms_clip(d - o, WINSOR * SQRT2 * self.info.sigma_eff)

    # ---- yerel hizalama

    def local_refine(self, t, src, rect_global, others):
        info = self.info
        x0, y0, x1, y1 = rect_global
        pad = int(np.clip(1.5 * max(x1 - x0, y1 - y0), 40, 120))
        win = (max(0, x0 - pad), max(0, y0 - pad), min(info.width, x1 + pad), min(info.height, y1 + pad))
        target = self.store.gray(t)[win[1]:win[3], win[0]:win[2]]
        gray, _, valid = self.sample(src.t, src.M, src.identity, win)
        low, high = np.percentile(target, (1, 99))
        t8, s8 = to8(target, low, high), to8(np.where(valid, gray, target), low, high)
        blocked = ~valid
        for rx0, ry0, rx1, ry1 in info.occ[t]:
            blocked[max(0, ry0 - win[1]):max(0, ry1 - win[1]), max(0, rx0 - win[0]):max(0, rx1 - win[0])] = True
        size = 2 * FEATURE_CLEARANCE + 1
        blocked = cv2.dilate(blocked.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
        mask = np.where(blocked, 0, 255).astype(np.uint8)
        points = cv2.goodFeaturesToTrack(t8, maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7, mask=mask)
        if points is None or len(points) < 12:
            return None
        p_t, p_s = track(t8, s8, points.astype(np.float32), (21, 21), 3, 0.7)
        if len(p_t) < 12:
            return None
        cv2.setRNGSeed(0)
        A, inliers = cv2.estimateAffinePartial2D(p_t, p_s, method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if A is None or inliers.sum() < max(10, 0.5 * len(p_t)):
            return None
        scale = math.hypot(A[0, 0], A[1, 0])
        rot = abs(math.degrees(math.atan2(A[1, 0], A[0, 0])))
        if math.hypot(A[0, 2], A[1, 2]) > 20 or not 0.95 <= scale <= 1.05 or rot > 3:
            return None
        return src.M @ translation(win[0], win[1]) @ to_3x3(A) @ translation(-win[0], -win[1])

    # ---- ana akış

    def solve(self, t, region_mask, region_rect, objects, other_rects):
        """region_mask: bölge dikdörtgeninde R; region_rect global (x0, y0, x1, y1)."""
        info, p = self.info, self.p
        x0, y0, x1, y1 = region_rect
        rh, rw = y1 - y0, x1 - x0
        ring_width = int(np.clip(round(0.5 * min(rh, rw)), RING_WIDTH_MIN, RING_WIDTH_MAX))
        margin = RING_INNER + ring_width + 3
        crop = (max(0, x0 - margin), max(0, y0 - margin), min(info.width, x1 + margin), min(info.height, y1 + margin))
        cx0, cy0, cx1, cy1 = crop
        R = np.zeros((cy1 - cy0, cx1 - cx0), bool)
        R[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = region_mask
        edges = (cy0 == 0, cy1 == info.height, cx0 == 0, cx1 == info.width)
        target_raw = self.store.raw(t)[cy0:cy1, cx0:cx1]
        target = self.store.gray(t)[cy0:cy1, cx0:cx1]
        others = np.zeros(R.shape, bool)
        for ox0, oy0, ox1, oy1 in other_rects:
            others[max(0, oy0 - cy0):max(0, oy1 - cy0), max(0, ox0 - cx0):max(0, ox1 - cx0)] = True
        distance = cv2.distanceTransform((~R).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        ring = (distance >= RING_INNER) & (distance <= RING_INNER + ring_width) & ~others & ~R
        ctx = {"t": t, "crop": crop, "R": R, "target": target, "target_raw": target_raw, "others": others,
               "ring": ring, "ring_width": ring_width, "edges": edges, "objects": objects}
        result = {"crop": crop, "R": R, "rect_pixels": int(R.sum())}

        if R.sum() > MAX_REGION_FRACTION * info.width * info.height:
            result.update({"H": R.copy(), "signals": {"too_big": True, "mode": "yok"}, "values": None, "sources": []})
            return result

        sources = self.plan(t, R, crop)
        for src in sources:
            self.finish_source(t, src, crop, ring, region_rect, others)
        ctx["sources"] = sources
        H, core_sign, dominant = self.thin_hole(ctx)
        outcome = self.fill_and_assess(ctx, H, core_sign, dominant)
        if (outcome["signals"].get("leftover") or 0) > LEFT_ORTA and not np.array_equal(H, R):
            grown = (cv2.dilate(H.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0) & R
            retry = self.fill_and_assess(ctx, grown, core_sign, dominant)
            if (retry["signals"].get("leftover") or 0) < outcome["signals"]["leftover"]:
                outcome = retry
        result.update(outcome)
        result.update({"sources": sources, "ctx": ctx, "core_sign": core_sign, "dominant": dominant})
        return result

    def plan(self, t, R, crop):
        """Bölüm 6.6: geçmiş ve gelecekten, piksel açıldıkça en yakın kaynaklar."""
        info, p = self.info, self.p
        cx0, cy0 = crop[0], crop[1]
        ys, xs = np.nonzero(R)
        gx, gy = xs + cx0, ys + cy0
        n = len(gx)
        min_gain = max(MIN_GAIN_PIXELS, MIN_GAIN_FRACTION * n)
        remaining = {"gecmis": np.ones(n, bool), "gelecek": np.ones(n, bool)}
        chosen = {"gecmis": [], "gelecek": []}
        done = {"gecmis": False, "gelecek": False}
        rect = (int(gx.min()), int(gy.min()), int(gx.max()) + 1, int(gy.max()) + 1)
        for d in range(1, p["search"] + 1):
            for side, ts in (("gecmis", t - d), ("gelecek", t + d)):
                if done[side] or ts < 0 or ts >= info.n:
                    continue
                M = info.M(t, ts)
                identity = info.camera != "hareketli" or near_identity(M, rect)
                sx, sy = map_coords(M, gx, gy, identity)
                gain = inside_and_free(sx, sy, info.occ[ts], info.width, info.height) & remaining[side]
                count = int(gain.sum())
                if count == 0:
                    continue
                left = int(remaining[side].sum())
                if count == left or (count >= min_gain and len(chosen[side]) < MAX_SOURCES_PER_SIDE - 1):
                    src = Source()
                    src.t, src.side, src.distance, src.M, src.identity = ts, side, d, M, identity
                    src.refined = False
                    chosen[side].append(src)
                    remaining[side] &= ~gain
                    if not remaining[side].any():
                        done[side] = True
            past_over = done["gecmis"] or t - d <= 0
            future_over = done["gelecek"] or t + d >= info.n - 1
            if past_over and future_over:
                break
        sources = chosen["gecmis"] + chosen["gelecek"]
        sources.sort(key=lambda s: (s.distance, s.side != "gecmis"))
        return sources

    def finish_source(self, t, src, crop, ring, rect_global, others):
        target = self.store.gray(t)[crop[1]:crop[3], crop[0]:crop[2]]
        gray, raw, valid = self.sample(src.t, src.M, src.identity, crop)
        o, artik = self.ring_stats(target, gray, valid, ring)
        sigma_t = self.info.sigma_t
        if artik is None or artik / (SQRT2 * sigma_t) > REFINE_SKIP_A1:
            candidate = self.local_refine(t, src, rect_global, others)
            if candidate is not None:
                identity = near_identity(candidate, crop)
                g2, r2, v2 = self.sample(src.t, candidate, identity, crop)
                o2, artik2 = self.ring_stats(target, g2, v2, ring)
                if artik2 is not None and (artik is None or artik2 <= (1 - REFINE_MIN_GAIN) * artik):
                    src.M, src.identity, src.refined = candidate, identity, True
                    gray, raw, valid, o, artik = g2, r2, v2, o2, artik2
        src.gray, src.raw, src.valid = gray, raw, valid
        src.ring_ok = artik is not None
        src.offset = o if o is not None else 0.0
        src.artik = artik

    def assign(self, sources, mask):
        """Her piksel için yakın kaynak (en yakın zaman, eşitlikte geçmiş) ve karşı yöndeki uzak kaynak."""
        near = np.full(mask.shape, -1, np.int64)
        first_side = {"gecmis": np.full(mask.shape, -1, np.int64), "gelecek": np.full(mask.shape, -1, np.int64)}
        for k, src in enumerate(sources):
            take = src.valid & mask & (near < 0)
            near[take] = k
            side_map = first_side[src.side]
            side_take = src.valid & mask & (side_map < 0)
            side_map[side_take] = k
        far = np.full(mask.shape, -1, np.int64)
        for k, src in enumerate(sources):
            here = near == k
            opposite = first_side["gelecek" if src.side == "gecmis" else "gecmis"]
            far[here] = opposite[here]
        return near, far

    def thin_hole(self, ctx):
        """Bölüm 6.8: kutudan obje maskesine."""
        R, target, sources = ctx["R"], ctx["target"], ctx["sources"]
        if not sources:
            return R.copy(), None, None
        near, _ = self.assign(sources, R)
        counts = np.bincount(near[near >= 0], minlength=len(sources))
        if counts.sum() == 0:
            return R.copy(), None, None
        dominant = int(np.argmax(counts))
        background = np.zeros(R.shape, np.float32)
        known = near >= 0
        for k, src in enumerate(sources):
            take = near == k
            background[take] = src.gray[take] + src.offset
        D = np.where(known, target - background, 0.0)
        spread = max(SQRT2 * self.info.sigma_t, sources[dominant].artik or 0.0)
        core = known & (np.abs(D) > CORE_Z * spread)
        labels, count = ndimage.label(core, structure=np.ones((3, 3)))
        if count:
            sizes = ndimage.sum(core, labels, index=np.arange(1, count + 1))
            core = np.isin(labels, 1 + np.nonzero(sizes >= CORE_MIN_AREA)[0])
        if not core.any():
            return R.copy(), None, dominant
        grow = known & (np.abs(D) > GROW_Z * spread)
        glabels, _ = ndimage.label(grow | core, structure=np.ones((3, 3)))
        keep = np.unique(glabels[core])
        H = np.isin(glabels, keep[keep > 0]) | (R & ~known)
        H = cv2.morphologyEx(H.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        H = ndimage.binary_fill_holes(H)
        H = (cv2.dilate(H.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=HOLE_DILATE) > 0) & R
        if H.sum() > HOLE_FULL_FRACTION * R.sum():
            H = R.copy()
        sign = float(np.sign(D[core].sum())) or None
        return H, sign, dominant

    def fill_and_assess(self, ctx, H, core_sign, dominant):
        info, p = self.info, self.p
        sources, target, target_raw = ctx["sources"], ctx["target"], ctx["target_raw"]
        near, far = self.assign(sources, H) if sources else (np.full(H.shape, -1), np.full(H.shape, -1))
        n_hole = int(H.sum())
        unrevealed = float((near[H] < 0).mean()) if n_hole else 0.0
        used = sorted(set(near[H][near[H] >= 0].tolist()))
        if not used or unrevealed > UNREVEALED_MAX:
            mode = "lama" if self.lama is not None else "uzamsal"
        elif all(sources[k].ring_ok and abs(sources[k].offset) <= COPY_MAX_OFFSET * info.sigma_t
                 and sources[k].artik / (SQRT2 * info.sigma_t) <= COPY_MAX_A1 for k in used) and unrevealed == 0:
            mode = "kopya"
        else:
            mode = "poisson"
        lama_values = None
        if mode == "lama":
            lama_values = self.run_lama(ctx, H)
        channels = fill_channels(target_raw)
        final_raw = target_raw.copy()
        ys, xs = np.nonzero(H)
        low, high = float(target.min()), float(target.max())
        if target_raw.ndim == 3:
            low, high = float(target_raw[..., :3].min()), float(target_raw[..., :3].max())
        for k in used:
            values = np.concatenate([c[H & (near == k)] for c in sources[k].raw])
            if values.size:
                low, high = min(low, float(values.min())), max(high, float(values.max()))
        if mode == "kopya":
            for ci, channel in enumerate(channels):
                plane_values = np.zeros(n_hole, np.float32)
                for k in used:
                    take = near[ys, xs] == k
                    plane_values[take] = sources[k].raw[ci][ys[take], xs[take]]
                self.store_values(final_raw, channel, ys, xs, plane_values, low, high)
        else:
            system = Poisson(H, ctx["edges"])
            near_flat = near[system.ys, system.xs]
            for ci, channel in enumerate(channels):
                plane_target = channel_plane(target_raw, channel)
                guide = self.guidance(sources, near, near_flat, system, ci, mode, lama_values)
                solved = system.run(plane_target, guide)
                self.store_values(final_raw, channel, system.ys, system.xs, solved, low, high)
        final = plane(final_raw)
        signals = self.assess(ctx, final, H, near, far, used, mode, core_sign, dominant, unrevealed)
        return {"H": H, "values": final_raw, "signals": signals, "near": near, "far": far, "used": used,
                "unrevealed": unrevealed}

    def store_values(self, final_raw, channel, ys, xs, values, low, high):
        dtype = final_raw.dtype
        if np.issubdtype(dtype, np.integer):
            limits = np.iinfo(dtype)
            values = np.clip(np.rint(values), max(limits.min, math.floor(low)), min(limits.max, math.ceil(high)))
        if channel is None:
            final_raw[ys, xs] = values.astype(dtype)
        else:
            final_raw[ys, xs, channel] = values.astype(dtype)

    def guidance(self, sources, near, near_flat, system, ci, mode, lama_values):
        def fn(p_idx, qy, qx, q_idx):
            py, px = system.ys[p_idx], system.xs[p_idx]
            g = np.zeros(len(p_idx))
            count = np.zeros(len(p_idx))
            own = near_flat[p_idx]
            for k, src in enumerate(sources):
                sel = own == k
                if sel.any():
                    ok = src.valid[py[sel], px[sel]] & src.valid[qy[sel], qx[sel]]
                    values = src.raw[ci][py[sel], px[sel]] - src.raw[ci][qy[sel], qx[sel]]
                    g[np.nonzero(sel)[0][ok]] += values[ok]
                    count[np.nonzero(sel)[0][ok]] += 1
            inner = q_idx >= 0
            other = np.full(len(p_idx), -1)
            other[inner] = near_flat[q_idx[inner]]
            for k, src in enumerate(sources):
                sel = inner & (other == k)
                if sel.any():
                    ok = src.valid[py[sel], px[sel]] & src.valid[qy[sel], qx[sel]]
                    values = src.raw[ci][py[sel], px[sel]] - src.raw[ci][qy[sel], qx[sel]]
                    g[np.nonzero(sel)[0][ok]] += values[ok]
                    count[np.nonzero(sel)[0][ok]] += 1
            have = count > 0
            g[have] /= count[have]
            if mode == "lama" and lama_values is not None:
                miss = ~have
                g[miss] = lama_values[py[miss], px[miss]] - lama_values[qy[miss], qx[miss]]
                have = np.ones(len(p_idx), bool)
            return g, have
        return fn

    def run_lama(self, ctx, H):
        info = self.info
        cx0, cy0, cx1, cy1 = ctx["crop"]
        ys, xs = np.nonzero(H)
        hx0, hy0, hx1, hy1 = xs.min() + cx0, ys.min() + cy0, xs.max() + 1 + cx0, ys.max() + 1 + cy0
        grow = max(32, max(hx1 - hx0, hy1 - hy0))
        win = (max(0, hx0 - grow), max(0, hy0 - grow), min(info.width, hx1 + grow), min(info.height, hy1 + grow))
        full = self.store.gray(ctx["t"])
        window = full[win[1]:win[3], win[0]:win[2]]
        hole = np.zeros(window.shape, bool)
        hole[cy0 - win[1] + ys, cx0 - win[0] + xs] = True
        out = self.lama.predict(window, hole)
        result = full[cy0:cy1, cx0:cx1].copy()
        ox0, oy0 = max(win[0], cx0), max(win[1], cy0)
        ox1, oy1 = min(win[2], cx1), min(win[3], cy1)
        result[oy0 - cy0:oy1 - cy0, ox0 - cx0:ox1 - cx0] = out[oy0 - win[1]:oy1 - win[1], ox0 - win[0]:ox1 - win[0]]
        return result

    # ---- kalite sinyalleri (Bölüm 6.10)

    def assess(self, ctx, final, H, near, far, used, mode, core_sign, dominant, unrevealed):
        info = self.info
        sigma_t, sigma_eff = info.sigma_t, info.sigma_eff
        sources, target, ring, others, R = ctx["sources"], ctx["target"], ctx["ring"], ctx["others"], ctx["R"]
        limit = WINSOR * SQRT2 * sigma_eff
        n_hole = int(H.sum())
        s = {"mode": mode, "unrevealed": unrevealed, "n_sources": 0, "gap_near": None, "gap_far": None}
        used_all = set(used) | set(far[H][far[H] >= 0].tolist())
        s["n_sources"] = len(used_all)
        if used:
            s["gap_near"] = float(np.mean([sources[k].distance for k in near[H][near[H] >= 0]]))
        if (far[H] >= 0).any():
            s["gap_far"] = float(np.mean([sources[k].distance for k in far[H][far[H] >= 0]]))
        s["refined"] = int(any(sources[k].refined for k in used_all)) if used_all else 0
        s["chain_reliable"] = int(all(info.reliable_between(ctx["t"], sources[k].t) for k in used_all))
        # a1: deliğin hemen yanındaki halka artığı
        contributors = {k for k in used if (near[H] == k).sum() >= 0.10 * n_hole}
        if dominant is not None and dominant < len(sources):
            contributors.add(dominant)
        a1_values = [sources[k].artik / (SQRT2 * sigma_eff) for k in contributors if sources[k].ring_ok]
        s["a1"] = max(a1_values) if a1_values else None
        s["offset"] = max(abs(sources[k].offset) for k in used) / sigma_t if used else None
        # a2: son dolgu ile zamanın öbür yanındaki bağımsız görüntü
        have_far = H & (far >= 0)
        s["a2_coverage"] = float(have_far.sum() / n_hole) if n_hole else 0.0
        if s["a2_coverage"] >= 0.5 and mode != "uzamsal":
            d = np.zeros(int(have_far.sum()), np.float32)
            fy, fx = np.nonzero(have_far)
            ks = far[fy, fx]
            for k in np.unique(ks):
                sel = ks == k
                d[sel] = final[fy[sel], fx[sel]] - (sources[k].gray[fy[sel], fx[sel]] + sources[k].offset)
            s["a2"] = rms_clip(d, limit) / (SQRT2 * sigma_eff)
            s["one_sided"] = 0
        else:
            s["a2"] = None
            s["one_sided"] = 1
        s["leftover"] = self.leftover(ctx, H, dominant, core_sign)
        s["seam"] = self.seam(final, target, H, others)
        s["texture"] = None
        if n_hole >= 30 and (ring).sum() >= RING_MIN_PIXELS:
            hp = final - cv2.GaussianBlur(final, (0, 0), 1.5)
            inner, outer = mad(hp[H]), mad(hp[ring])
            if inner > 0 and outer > 0:
                r = inner / outer
                s["texture"] = max(r, 1 / r)
            else:
                s["texture"] = float("inf") if (inner > 0) != (outer > 0) else 1.0
        return s

    def leftover(self, ctx, H, dominant, sign):
        if dominant is None:
            return None
        src = ctx["sources"][dominant]
        info = self.info
        distance = cv2.distanceTransform((~H).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        rounded = np.rint(distance).astype(np.int64)
        usable = src.valid & ~ctx["others"] & ~H
        diff = ctx["target"] - src.gray - src.offset
        values, counts = {}, {}
        for d in range(1, 4 + ctx["ring_width"]):
            ring = usable & (rounded == d)
            counts[d] = int(ring.sum())
            values[d] = float(np.median(diff[ring])) if counts[d] else None
        far_values = np.array([values[d] for d in range(4, 4 + ctx["ring_width"]) if values[d] is not None])
        if far_values.size == 0:
            return None
        center, spread = float(np.median(far_values)), 1.4826 * mad(far_values)
        best = 0.0
        for d in (1, 2, 3):
            if values[d] is None or counts[d] < 6:
                continue
            excess = values[d] - center
            excess = excess * sign if sign else abs(excess)
            se = 1.2533 * SQRT2 * info.sigma_t / math.sqrt(counts[d])
            if excess > max(LEFT_Z * se, 3 * spread):
                best = max(best, excess)
        return best / info.sigma_eff

    def seam(self, final, target, H, others):
        h, w = H.shape
        ratios = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            py, px = np.nonzero(H)
            qy, qx = py + dy, px + dx
            ry, rx = qy + dy, qx + dx
            ok = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w)
            py, px, qy, qx, ry, rx = py[ok], px[ok], qy[ok], qx[ok], ry[ok], rx[ok]
            ok = ~H[qy, qx] & ~H[ry, rx] & ~others[qy, qx] & ~others[ry, rx]
            if ok.sum() < SEAM_MIN_PAIRS:
                continue
            inner = np.abs(final[py[ok], px[ok]] - target[qy[ok], qx[ok]]).mean()
            outer = np.abs(target[qy[ok], qx[ok]] - target[ry[ok], rx[ok]]).mean()
            ratios.append(float(inner / max(outer, 1.128 * self.info.sigma_t)))
        return max(ratios) if ratios else None


# ---------------------------------------------------------------- kare işleme

def regions_of(info, t):
    rects = info.rects[t]
    union = np.zeros((info.height, info.width), np.uint8)
    for x0, y0, x1, y1 in rects:
        union[y0:y1, x0:x1] = 1
    count, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=8)
    regions = []
    for c in range(1, count):
        x0, y0, w, h = (int(v) for v in stats[c][:4])
        mask = labels[y0:y0 + h, x0:x0 + w] == c
        members = [k for k, (rx0, ry0, rx1, ry1) in enumerate(rects) if (labels[ry0:ry1, rx0:rx1] == c).any()]
        regions.append((c, (x0, y0, x0 + w, y0 + h), mask, members))
    return regions


def flicker_of(info, store, t, result_gray, H_global, ring_global, prev):
    if prev is None:
        return None
    prev_out, prev_hole = prev
    ys, xs = np.nonzero(H_global)
    M = info.M(t, t - 1)
    identity = info.camera != "hareketli"
    sx, sy = map_coords(M, xs, ys, identity)
    inside = (sx >= 0) & (sx < info.width) & (sy >= 0) & (sy < info.height)
    overlap = np.zeros(len(xs), bool)
    overlap[inside] = prev_hole[sy[inside], sx[inside]]
    if overlap.sum() < 30:
        return None
    limit = WINSOR * SQRT2 * info.sigma_eff
    inner = rms_clip(result_gray[ys[overlap], xs[overlap]] - prev_out[sy[overlap], sx[overlap]], limit)
    ry, rx = np.nonzero(ring_global)
    tx, ty = map_coords(M, rx, ry, identity)
    ok = (tx >= 0) & (tx < info.width) & (ty >= 0) & (ty < info.height)
    ok[ok] &= ~prev_hole[ty[ok], tx[ok]]
    if ok.sum() < 30:
        return None
    current = store.gray(t)
    previous = store.gray(t - 1)
    outer = rms_clip(current[ry[ok], rx[ok]] - previous[ty[ok], tx[ok]], limit)
    return inner / outer if outer and outer > 0 else None


def process_frame(info, store, solver, t, params, prev):
    raw = store.raw(t)
    labels = info.boxes[t]
    result = raw.copy()
    hole_all = np.zeros((info.height, info.width), bool)
    confidence = np.full((info.height, info.width), 255, np.uint8)
    rows, drawn = [], []
    if not labels:
        return result, rows, "iyi", confidence, drawn, hole_all
    for region_id, rect, mask, members in regions_of(info, t):
        other_rects = [r for k, r in enumerate(info.rects[t]) if k not in members]
        solved = solver.solve(t, mask, rect, members, other_rects)
        cx0, cy0, cx1, cy1 = solved["crop"]
        H = solved["H"]
        signals = solved["signals"]
        H_global = np.zeros((info.height, info.width), bool)
        H_global[cy0:cy1, cx0:cx1] = H
        if solved["values"] is not None and not signals.get("too_big"):
            region_view = result[cy0:cy1, cx0:cx1]
            region_view[H] = solved["values"][H]
            hole_all |= H_global
            ring_global = np.zeros_like(H_global)
            ctx_ring = cv2.distanceTransform((~solved["R"]).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            ring_local = (ctx_ring >= RING_INNER) & (ctx_ring <= RING_INNER + RING_WIDTH_MAX) & ~H
            ring_global[cy0:cy1, cx0:cx1] = ring_local
            signals["flicker"] = flicker_of(info, store, t, plane(result), H_global, ring_global, prev) \
                if params["every"] == 1 else None
        else:
            signals.setdefault("flicker", None)
        category, reason, e_hat = decide(signals, params["t_iyi"], params["t_kotu"], params["kappa"])
        # KARAR: çok büyük bölgede pikseller değişmez ama güven haritasında bölge 0 işaretlenir (obje duruyor).
        confidence[H_global] = np.minimum(confidence[H_global], CONFIDENCE[category])
        drawn.append((H_global, rect, signals.get("mode", ""), reason, category))
        ys, xs = np.nonzero(H_global)
        hole_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else ("", "", "", "")
        for k in members:
            box = labels[k]
            rows.append({
                "frame": Path(info.frames[t]).name, "sequence": info.name, "object": k,
                "label_x0": round(box[0], 2), "label_y0": round(box[1], 2), "label_x1": round(box[2], 2),
                "label_y1": round(box[3], 2), "region": region_id, "hole_x0": hole_box[0], "hole_y0": hole_box[1],
                "hole_x1": hole_box[2], "hole_y1": hole_box[3], "hole_pixels": int(H_global.sum()),
                "rect_pixels": solved["rect_pixels"], "mode": signals.get("mode"),
                "n_sources": signals.get("n_sources"), "gap_near": signals.get("gap_near"),
                "gap_far": signals.get("gap_far"), "one_sided": signals.get("one_sided"),
                "unrevealed": signals.get("unrevealed"), "a1": signals.get("a1"), "a2": signals.get("a2"),
                "a2_coverage": signals.get("a2_coverage"), "offset": signals.get("offset"),
                "leftover": signals.get("leftover"), "seam": signals.get("seam"), "texture": signals.get("texture"),
                "flicker": signals.get("flicker"), "e_hat": e_hat, "category": category, "reason": reason,
                "refined": signals.get("refined"), "chain_reliable": signals.get("chain_reliable"),
                "sigma_t": info.sigma_t, "sigma_eff": info.sigma_eff})
    frame_category = max((r["category"] for r in rows), key=CATEGORIES.index) if rows else "iyi"
    return result, rows, frame_category, confidence, drawn, hole_all


def write_preview(path, original, result, drawn):
    gray = plane(original)
    low, high = np.percentile(gray, (0.5, 99.8))

    def eight(image):
        scaled = np.clip((plane(image) - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

    left, right = eight(original), eight(result)
    for H_global, rect, mode, reason, category in drawn:
        contours, _ = cv2.findContours(H_global.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(left, contours, -1, COLORS[category], 1)
        cv2.putText(left, f"{mode} {reason}", (rect[0], max(10, rect[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    COLORS[category], 1)
    write_image(path, cv2.hconcat([left, right]))


_WORKER = {}


def worker_state(info, params):
    key = (info.name, id(info))
    if _WORKER.get("key") != key:
        lama = _WORKER.get("lama")
        if params["lama"] and lama is None:
            lama = LamaRunner(params["lama"], params["lama_device"])
        _WORKER.clear()
        store = FrameStore(info.frames)
        _WORKER.update({"key": key, "store": store, "lama": lama,
                        "solver": RegionSolver(info, store, params, lama)})
    return _WORKER["store"], _WORKER["solver"]


def process_chunk(info, indices, params):
    info.visible_fraction = params["visible_fraction"]
    store, solver = worker_state(info, params)
    out = Path(params["out"])
    records = []
    prev = None
    if params["every"] == 1 and indices and indices[0] > 0 and info.boxes[indices[0] - 1]:
        t0 = indices[0] - 1
        result, _, _, _, _, hole = process_frame(info, store, solver, t0, params, None)
        prev = (plane(result), hole)
    for t in indices:
        path = Path(info.frames[t])
        result, rows, category, confidence, drawn, hole = process_frame(info, store, solver, t, params, prev)
        prev = (plane(result), hole) if rows else None
        if rows:
            write_image(out / category / f"bg_{path.name}", result)
            write_image(out / "guven" / f"{path.stem}.png", confidence)
            if params["preview_every"] and t % params["preview_every"] == 0:
                write_preview(out / "onizleme" / f"{path.stem}.png", store.raw(t), result, drawn)
        else:
            shutil.copyfile(path, out / "iyi" / f"bg_{path.name}")
        if params["originals"]:
            shutil.copyfile(path, out / "silinmemis" / path.name)
        records.append((t, category, rows))
    return info.name, records


# ---------------------------------------------------------------- ana program

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf" if value > 0 else ""
        return f"{value:.4f}"
    return value


def load_labels(args, frames, width, height):
    if args.labels:
        if Path(args.labels).is_file():
            return load_csv(args.labels, frames, width, height, args.missing_labels)
        return load_yolo(args.labels, frames, width, height, args.missing_labels)
    return load_coco(args.images_json, args.annotations_json, frames, width, height, args.missing_labels)


def run(args):
    started = time.time()
    if args.project:
        root = Path(args.project)
        args.frames = args.frames or root / FRAMES_SUBDIR
        args.images_json = args.images_json or root / COCO_SUBDIR / "coco-images.json"
        args.annotations_json = args.annotations_json or root / COCO_SUBDIR / "coco-annotations.json"
    if not args.frames or not (args.labels or (args.images_json and args.annotations_json)):
        raise SystemExit("--project ya da --frames ile birlikte bir etiket kaynağı verilmeli")
    kappa, t_iyi, t_kotu = KAPPA, T_IYI, T_KOTU
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        kappa, t_iyi, t_kotu = calibration["kappa"], calibration["t_iyi"], calibration["t_kotu"]
    if args.thresholds:
        t_iyi, t_kotu = (float(v) for v in args.thresholds.split(","))
    frames = list_frames(args.frames)
    first = read_image(frames[0])
    height, width = first.shape[:2]
    boxes = load_labels(args, frames, width, height)
    if all(b is None for b in boxes):
        raise ValueError("Hiçbir kare etiket kaynağıyla eşleşmedi")
    out = Path(args.out)
    for sub in CATEGORIES + ("onizleme", "guven") + (() if args.no_originals else ("silinmemis",)):
        (out / sub).mkdir(parents=True, exist_ok=True)
    params = {"search": args.search, "every": args.every, "t_iyi": t_iyi, "t_kotu": t_kotu, "kappa": kappa,
              "lama": args.lama, "lama_device": args.lama_device, "out": str(out), "preview_every": args.preview_every,
              "originals": not args.no_originals, "visible_fraction": args.visible_fraction}
    log = {}
    all_records, align_rows = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = []
        for name, seq_frames, seq_boxes, unknown in split_sequences(frames, boxes):
            if unknown:
                print(f"UYARI: {name}: etiket kaydı olmayan {unknown} kare işlenmedi ve kaynak olarak kullanılmadı.")
            info = SequenceInfo(name, seq_frames, seq_boxes, unknown, args)
            info.visible_fraction = args.visible_fraction
            prepare_sequence(info, args, log)
            align_rows.extend(info.align_rows)
            print(f"{name}: {info.n} kare, kamera {info.camera}, sigma_t {info.sigma_t:.3f}, "
                  f"sigma_eff {info.sigma_eff:.3f}", flush=True)
            indices = list(range(0, info.n, args.every))
            parts = max(1, min(len(indices), args.workers * 4))
            size = math.ceil(len(indices) / parts)
            for start in range(0, len(indices), size):
                futures.append(pool.submit(process_chunk, info, indices[start:start + size], params))
        for done, future in enumerate(futures, 1):
            name, records = future.result()
            all_records.extend((name, t, category, rows) for t, category, rows in records)
            print(f"{done}/{len(futures)} parça", flush=True)
    all_records.sort(key=lambda r: (r[0], r[1]))
    with open(out / "rapor.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(REPORT_COLUMNS)
        for _, _, _, rows in all_records:
            for row in sorted(rows, key=lambda r: r["object"]):
                writer.writerow([fmt(row[c]) for c in REPORT_COLUMNS])
    with open(out / "hizalama.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(ALIGN_COLUMNS)
        writer.writerows(align_rows)
    objects = [row for _, _, _, rows in all_records for row in rows]
    elapsed = time.time() - started
    summary = {
        "surum": "v4",
        "argumanlar": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "esikler": {"kappa": kappa, "t_iyi": t_iyi, "t_kotu": t_kotu},
        "sabitler": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float))},
        "diziler": list(log.values()),
        "kareler": dict(Counter(category for _, _, category, _ in all_records)),
        "objeler": dict(Counter(r["category"] for r in objects)),
        "nedenler": dict(Counter(r["reason"] for r in objects)),
        "modlar": dict(Counter(r["mode"] for r in objects)),
        "sure_sn": round(elapsed, 1),
        "saniye_obje": round(elapsed * args.workers / max(1, len(objects)), 3),
    }
    (out / "ozet.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Kare: {len(all_records)}  " + "  ".join(f"{c}: {summary['kareler'].get(c, 0)}" for c in CATEGORIES)
          + f"  obje: {summary['objeler']}  modlar: {summary['modlar']}  süre: {summary['sure_sn']} sn")
    return summary


def main_parser():
    parser = argparse.ArgumentParser(description="IR dizilerinden etiketli objeleri silip arka planı doldurur (v4)")
    parser.add_argument("--project", help="2_Etiketlenecek_Veri ve 3_Etiketlenmis_Veri_Ciktisi klasörlerinin kökü")
    parser.add_argument("--frames", help="kare klasörü, varsayılan <project>/2_Etiketlenecek_Veri/tif")
    parser.add_argument("--images-json", help="varsayılan <project>/3_Etiketlenmis_Veri_Ciktisi/coco-images.json")
    parser.add_argument("--annotations-json", help="varsayılan <project>/3_Etiketlenmis_Veri_Ciktisi/coco-annotations.json")
    parser.add_argument("--labels", help="COCO yerine YOLO klasörü ya da frame,x,y,w,h CSV dosyası")
    parser.add_argument("--missing-labels", choices=("auto", "unknown", "empty"), default="auto",
                        help="etiket kaydı olmayan kare: auto (kaynağa göre), unknown (atla), empty (objesiz say)")
    parser.add_argument("--out", required=True, help="çıktı kökü")
    parser.add_argument("--camera", choices=("auto", "static", "moving"), default="auto", help="kamera sınıfı")
    parser.add_argument("--search", type=int, default=1000, help="kaynağın hedefe en çok uzaklığı (kare)")
    parser.add_argument("--margin", type=int, default=3, help="obje çıktıktan sonra beklenecek kare (ısıl iz)")
    parser.add_argument("--box-pad", default="auto", help="kutu payı (piksel) ya da auto")
    parser.add_argument("--thresholds", default=None, help="T_IYI,T_KOTU (e_hat biriminde)")
    parser.add_argument("--calibration", help="kalibrasyon JSON'u (kappa, t_iyi, t_kotu)")
    parser.add_argument("--visible-fraction", type=float, default=VISIBLE_FRACTION, help="görünürlük payı")
    parser.add_argument("--lama", help="TorchScript inpainting modeli (ör. ir_lama.pt)")
    parser.add_argument("--lama-device", default="cuda", help="cuda yoksa kendiliğinden cpu")
    parser.add_argument("--every", type=int, default=1, help="yalnız her N. kareyi işle")
    parser.add_argument("--workers", type=int, default=4, help="süreç sayısı")
    parser.add_argument("--preview-every", type=int, default=0, help="her N karede bir önizleme, 0 kapalı")
    parser.add_argument("--no-originals", action="store_true", help="silinmemis/ yazma")
    return parser


def parse_main(argv):
    args = main_parser().parse_args(argv)
    args.box_pad = None if args.box_pad == "auto" else int(args.box_pad)
    return args


# ================================================================ araçlar

def fbm(shape, seed, base, octaves, gain):
    rng = np.random.default_rng(seed)
    field = np.zeros(shape, np.float32)
    amplitude, total = 1.0, 0.0
    for o in range(octaves):
        grid = rng.standard_normal((base[0] * 2 ** o, base[1] * 2 ** o)).astype(np.float32)
        field += amplitude * cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
        total += amplitude
        amplitude *= gain
    return field / total


def ellipse(shape, cx, cy, rx, ry):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def object_core(kind, w, h):
    core = np.zeros((h + 8, w + 8), np.float32)
    cx, cy = (w + 8) / 2, (h + 8) / 2
    if kind == "kus":
        core[ellipse(core.shape, cx, cy, w / 2, h / 2)] = 1
    elif kind == "insan":
        core[ellipse(core.shape, cx, 4 + 0.62 * h, w * 0.42, h * 0.36)] = 1
        core[ellipse(core.shape, cx, 4 + 0.14 * h, w * 0.26, h * 0.12)] = 1
    else:
        core[ellipse(core.shape, cx, cy, w * 0.28, h * 0.32)] = 1
        for fx in (0.2, 0.8):
            for fy in (0.25, 0.75):
                core[ellipse(core.shape, 4 + fx * w, 4 + fy * h, w * 0.16, h * 0.2)] = 0.8
    return core


def bounce(lo, hi, position):
    span = hi - lo
    x = position % (2 * span)
    return lo + (x if x < span else 2 * span - x)


def default_objects(scene, width, height):
    sx, sy = width / 640.0, height / 512.0
    sky = [
        {"name": "hizli_kus", "kind": "kus", "size": (8, 5), "amp": 800,
         "path": lambda t: (bounce(60 * sx, 540 * sx, 3.2 * t), (110 + 20 * math.sin(t / 25)) * sy)},
        {"name": "orta_drone", "kind": "drone", "size": (18, 10), "amp": 2500,
         "path": lambda t: (bounce(80 * sx, 560 * sx, 1.0 * t), 175 * sy)},
        {"name": "yavas_drone", "kind": "drone", "size": (22, 12), "amp": 2500,
         "path": lambda t: (150 * sx + 0.23 * t, 70 * sy)},
        {"name": "cok_yavas_drone", "kind": "drone", "size": (24, 13), "amp": 2500,
         "path": lambda t: (480 * sx - 0.10 * t, 240 * sy)},
        {"name": "sabit_drone", "kind": "drone", "size": (20, 12), "amp": 2500,
         "path": lambda t: (330 * sx, 120 * sy)},
    ]
    person = {"name": "yavas_insan", "kind": "insan", "size": (12, 30), "amp": 1500,
              "path": lambda t: (100 * sx + 0.3 * t, (420 if scene == "karma" else 380) * sy)}
    return sky if scene == "gok" else sky + [person]


CAMERAS = {
    "sabit": lambda t: (0.0, 0.0, 0.0, 1.0),
    "salinim": lambda t: (12 * math.sin(2 * math.pi * t / 60), 3.6 * math.sin(2 * math.pi * t / 60), 0.0, 1.0),
    "pan": lambda t: (0.27 * t, 0.0, 0.0, 1.0),
    "donme": lambda t: (0.0, 0.0, 1.0 * math.sin(2 * math.pi * t / 80), 1.0),
    "zoom": lambda t: (0.0, 0.0, 0.0, 1 + 0.02 * math.sin(2 * math.pi * t / 100)),
    "hizli": lambda t: (38 * math.sin(2 * math.pi * t / 40), 11.4 * math.sin(2 * math.pi * t / 40), 0.0, 1.0),
}


def synth(out, scene="gok", camera="sabit", frames=600, seed=1, width=640, height=512, objects=None,
          camera_fn=None, cloud_evolve=520.0, sigma=5.5):
    """Gerçek arka planı bilinen sentetik dizi (Bölüm 8.1)."""
    out = Path(out)
    tif, coco, gt_dir = out / FRAMES_SUBDIR, out / COCO_SUBDIR, out / "gt"
    for d in (tif, coco, gt_dir):
        d.mkdir(parents=True, exist_ok=True)
    border = 140
    CH, CW = height + 2 * border, width + 2 * border
    rng = np.random.default_rng(seed * 7919 + 13)
    A, B = fbm((CH, CW), seed + 1, (4, 6), 6, 0.55), fbm((CH, CW), seed + 2, (4, 6), 6, 0.55)
    D1, D2 = fbm((CH, CW), seed + 3, (12, 18), 4, 0.55), fbm((CH, CW), seed + 4, (12, 18), 4, 0.55)
    sky_base = (5300 + 700 * (np.arange(CH, dtype=np.float32) / CH))[:, None] * np.ones((1, CW), np.float32)
    ground = fbm((CH, CW), seed + 5, (8, 10), 7, 0.6)
    lo, hi = np.percentile(ground, (1, 99))
    ground = 6600 + (ground - lo) / (hi - lo) * 2600
    for _ in range(14):
        w, h = rng.integers(20, 121, 2)
        x, y = rng.integers(0, CW - w), rng.integers(0, CH - h)
        ground[y:y + h, x:x + w] += rng.choice([-1, 1]) * rng.uniform(150, 500)
    for _ in range(8):
        p1 = tuple(int(v) for v in (rng.integers(0, CW), rng.integers(0, CH)))
        p2 = tuple(int(v) for v in (rng.integers(0, CW), rng.integers(0, CH)))
        layer = np.zeros((CH, CW), np.float32)
        cv2.line(layer, p1, p2, float(rng.choice([-1, 1]) * 400), int(rng.integers(2, 5)))
        ground += layer
    ground = ground.astype(np.float32)
    sky_rows = np.arange(CH)[:, None] < (border + int(330 * height / 512)) if scene == "karma" else None
    objects = objects if objects is not None else default_objects(scene, width, height)
    camera_fn = camera_fn or CAMERAS[camera]
    noise_rng = np.random.default_rng(seed * 104729 + 7)
    fixed = (noise_rng.normal(0, 2.0, (1, width)) + noise_rng.normal(0, 1.0, (height, width))).astype(np.float32)
    label_rng = np.random.default_rng(seed * 31 + 5)
    cores = {o["name"]: object_core(o["kind"], *o["size"]) for o in objects}
    layers = {n: None for n in cores}
    for o in objects:
        core = cores[o["name"]]
        layers[o["name"]] = o["amp"] * np.maximum(core, 0.35 * cv2.GaussianBlur(core, (0, 0), 2.5))
    images, annotations, truth = [], [], []
    fc = np.array([width / 2.0, height / 2.0])
    cc = fc + border
    for t in range(frames):
        if scene == "zemin":
            canvas = ground
        else:
            phase = t / cloud_evolve
            field = math.cos(phase) * A + math.sin(phase) * B
            detail = math.cos(1.7 * phase) * D1 + math.sin(1.7 * phase) * D2
            shift = np.float32([[1, 0, -0.16 * t], [0, 1, -0.035 * t]])
            field = cv2.warpAffine(field, shift, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            detail = cv2.warpAffine(detail, shift, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            cover = np.clip((field - 0.02) / 0.42, 0, 1)
            cover = cover * cover * (3 - 2 * cover)
            sky = sky_base + cover * (650 + 260 * detail)
            canvas = np.where(sky_rows, sky, ground) if scene == "karma" else sky
        dx, dy, theta, scale = camera_fn(t)
        rad = math.radians(theta)
        rot = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]]) * scale
        A2 = np.zeros((2, 3))
        A2[:, :2] = rot
        A2[:, 2] = cc + np.array([dx, dy]) - rot @ fc
        background = cv2.warpAffine(canvas.astype(np.float32), A2, (width, height),
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        observed = background.copy()
        inv_rot = np.linalg.inv(rot)
        frame_truth = []
        for o in objects:
            qx, qy = o["path"](t)
            q = np.array([qx + border, qy + border])
            px, py = fc + inv_rot @ (q - cc - np.array([dx, dy]))
            layer = layers[o["name"]]
            core = cores[o["name"]]
            lh, lw = layer.shape
            x0, y0 = int(round(px - lw / 2)), int(round(py - lh / 2))
            ix0, iy0, ix1, iy1 = max(0, x0), max(0, y0), min(width, x0 + lw), min(height, y0 + lh)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            observed[iy0:iy1, ix0:ix1] += layer[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
            cys, cxs = np.nonzero(core > 0.5)
            bx0, by0 = x0 + cxs.min(), y0 + cys.min()
            bx1, by1 = x0 + cxs.max() + 1, y0 + cys.max() + 1
            bw, bh = bx1 - bx0, by1 - by0
            loose = label_rng.uniform(0, 0.15, 4) * np.array([bw, bh, bw, bh])
            jitter = label_rng.integers(-1, 2, 4)
            lx0, ly0 = bx0 - loose[0] + jitter[0], by0 - loose[1] + jitter[1]
            lx1, ly1 = bx1 + loose[2] + jitter[2], by1 + loose[3] + jitter[3]
            lx0, ly0, lx1, ly1 = max(0, lx0), max(0, ly0), min(width, lx1), min(height, ly1)
            if lx1 - lx0 < 1 or ly1 - ly0 < 1:
                continue
            annotations.append({"id": len(annotations), "image_id": t, "category_id": 1,
                                "bbox": [float(lx0), float(ly0), float(lx1 - lx0), float(ly1 - ly0)]})
            frame_truth.append({"name": o["name"], "core_box": [int(bx0), int(by0), int(bx1), int(by1)],
                                "label_box": [float(lx0), float(ly0), float(lx1), float(ly1)]})
        noise = noise_rng.normal(0, sigma, (height, width)).astype(np.float32)
        name = f"frame_{t:06d}.tiff"
        write_image(tif / name, np.clip(np.rint(observed + noise + fixed), 0, 65535).astype(np.uint16))
        write_image(gt_dir / name, np.clip(np.rint(background + noise + fixed), 0, 65535).astype(np.uint16))
        images.append({"id": t, "file_name": f"gorsel/frame_{t:06d}.png", "width": width, "height": height})
        truth.append(frame_truth)
    (coco / "coco-images.json").write_text(json.dumps({"images": images}), encoding="utf-8")
    (coco / "coco-annotations.json").write_text(json.dumps({"annotations": annotations}), encoding="utf-8")
    (out / "truth.json").write_text(json.dumps({"sigma": sigma, "scene": scene, "camera": camera, "frames": truth}),
                                    encoding="utf-8")
    return out


def read_report(run_dir):
    rows = list(csv.DictReader(open(Path(run_dir) / "rapor.csv", encoding="utf-8")))
    for r in rows:
        for key in ("a1", "a2", "a2_coverage", "offset", "leftover", "seam", "texture", "flicker", "e_hat",
                    "unrevealed"):
            r[key] = float(r[key]) if r[key] not in ("", None) else None
        r["one_sided"] = int(r["one_sided"]) if r["one_sided"] not in ("", None) else 0
        r["too_big"] = r["reason"] == "cok_buyuk"
    return rows


def find_output(run_dir, name):
    for sub in CATEGORIES:
        p = Path(run_dir) / sub / f"bg_{name}"
        if p.exists():
            return p
    return None


def true_errors(bench, run_dir):
    """Her rapor satırına gerçek hata e_true ekler (Bölüm 8.2)."""
    bench, run_dir = Path(bench), Path(run_dir)
    truth = json.loads((bench / "truth.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "ozet.json").read_text(encoding="utf-8"))
    vf = summary["argumanlar"].get("visible_fraction", 0.0)
    sigma = truth["sigma"]
    gt_files = sorted((bench / "gt").glob("*.tiff"))
    picks = gt_files[::max(1, len(gt_files) // 20)][:20]
    spread = float(np.median([np.subtract(*np.percentile(read_image(p).astype(np.float32), (99, 1))) for p in picks]))
    sigma_eff = max(sigma, vf * spread)
    rows = read_report(run_dir)
    cache = {}
    for r in rows:
        name = r["frame"]
        if name not in cache:
            cache.clear()
            t = int(frame_number(name))
            cache[name] = (read_image(find_output(run_dir, name)).astype(np.float32),
                           read_image(bench / FRAMES_SUBDIR / name).astype(np.float32),
                           read_image(bench / "gt" / name).astype(np.float32), truth["frames"][t])
        result, observed, gt, objects = cache[name]
        k = int(r["object"])
        r["name"] = objects[k]["name"] if k < len(objects) else "?"
        x0, y0, x1, y1 = objects[k]["label_box"] if k < len(objects) else (0, 0, 0, 0)
        h, w = gt.shape
        sl = (slice(max(0, int(y0) - 12), min(h, int(math.ceil(y1)) + 12)),
              slice(max(0, int(x0) - 12), min(w, int(math.ceil(x1)) + 12)))
        E = (result[sl] != observed[sl]) | (np.abs(observed[sl] - gt[sl]) > 0.5)
        # KARAR: pencereye giren başka objelerin pikselleri (çekirdek kutusu + 8 piksel hale) bu objenin hatasına
        # sayılmaz; yoksa yandan geçen sabit obje bu objeyi kötü gösterir.
        for j, other in enumerate(objects):
            if j != k:
                ox0, oy0, ox1, oy1 = other["core_box"]
                E[max(0, oy0 - 8 - sl[0].start):max(0, oy1 + 8 - sl[0].start),
                  max(0, ox0 - 8 - sl[1].start):max(0, ox1 + 8 - sl[1].start)] = False
        r["e_true"] = float(np.abs(result[sl] - gt[sl])[E].mean()) / (1.128 * sigma_eff) if E.any() else 0.0
    return rows


def rates(rows, categories=None):
    cats = categories or [r["category"] for r in rows]
    truth = ["iyi" if r["e_true"] <= 1.6 else "kotu" if r["e_true"] > 3.0 else "sinir" for r in rows]
    table = {c: {g: 0 for g in ("iyi", "sinir", "kotu")} for c in CATEGORIES}
    for c, g in zip(cats, truth):
        table[c][g] += 1

    def ratio(num, den):
        return (num, den, num / den if den else None)

    said_good = sum(table["iyi"].values())
    really_bad = sum(table[c]["kotu"] for c in CATEGORIES)
    really_good = sum(table[c]["iyi"] for c in CATEGORIES)
    return table, {
        "P(gercek kotu | karar iyi)": ratio(table["iyi"]["kotu"], said_good),
        "P(gercek iyi degil | karar iyi)": ratio(table["iyi"]["kotu"] + table["iyi"]["sinir"], said_good),
        "P(karar kotu | gercek kotu)": ratio(table["kotu"]["kotu"], really_bad),
        "P(karar iyi | gercek kotu)": ratio(table["iyi"]["kotu"], really_bad),
        "P(karar iyi | gercek iyi)": ratio(table["iyi"]["iyi"], really_good),
    }


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    ra, rb = np.argsort(np.argsort(a[ok])), np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def tool_evaluate(argv):
    parser = argparse.ArgumentParser(prog="v4.py degerlendir")
    parser.add_argument("--bench", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    rows = true_errors(args.bench, args.run)
    result = evaluation_summary(rows)
    print_evaluation(result)
    (Path(args.run) / "degerlendirme.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    return result


def evaluation_summary(rows):
    by_name = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r)
    objects = {}
    for name, group in sorted(by_name.items()):
        e = np.array([g["e_true"] for g in group])
        objects[name] = {"n": len(group), "e_true_medyan": float(np.median(e)), "e_true_p90": float(np.percentile(e, 90)),
                         "iyi_orani": float(np.mean([g["category"] == "iyi" for g in group])),
                         "kategoriler": dict(Counter(g["category"] for g in group)),
                         "nedenler": dict(Counter(g["reason"] for g in group))}
    table, conditional = rates(rows)
    e = np.array([r["e_true"] for r in rows]) if rows else np.zeros(1)
    return {"n": len(rows), "e_true_medyan": float(np.median(e)), "e_true_p90": float(np.percentile(e, 90)),
            "objeler": objects, "karisiklik": table,
            "kosullu_oranlar": {k: {"pay": v[0], "payda": v[1], "oran": v[2]} for k, v in conditional.items()},
            "spearman_ehat_etrue": spearman([r["e_hat"] if r["e_hat"] is not None else np.nan for r in rows],
                                            [r["e_true"] for r in rows]),
            "nedenler": dict(Counter(r["reason"] for r in rows)), "modlar": dict(Counter(r["mode"] for r in rows))}


def print_evaluation(result):
    print(f"obje sayısı {result['n']}, e_true medyan {result['e_true_medyan']:.2f}, %90 {result['e_true_p90']:.2f}")
    print(f"{'obje':<18}{'n':>5}{'medyan':>9}{'%90':>8}{'iyi%':>7}  kategoriler")
    for name, o in result["objeler"].items():
        print(f"{name:<18}{o['n']:>5}{o['e_true_medyan']:>9.2f}{o['e_true_p90']:>8.2f}{100 * o['iyi_orani']:>7.0f}  {o['kategoriler']}")
    print("karar \\ gerçek       iyi(<=1.6)   sınırda   kötü(>3)")
    for c, row in result["karisiklik"].items():
        print(f"{c:<20}{row['iyi']:>10}{row['sinir']:>10}{row['kotu']:>10}")
    for k, v in result["kosullu_oranlar"].items():
        oran = "ölçülemedi" if v["oran"] is None else f"{v['oran']:.3f}"
        print(f"  {k:<34} {oran:>10}  ({v['pay']}/{v['payda']})")
    sp = result["spearman_ehat_etrue"]
    print(f"  Spearman(e_hat, e_true): {'yok' if sp is None else f'{sp:.3f}'}")
    print(f"  nedenler: {result['nedenler']}  modlar: {result['modlar']}")


def tool_calibrate(argv):
    parser = argparse.ArgumentParser(prog="v4.py kalibre")
    parser.add_argument("--pairs", nargs="+", required=True, help="BENCH:RUN çiftleri")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = []
    for pair in args.pairs:
        cut = next(i for i, ch in enumerate(pair) if ch == ":" and pair[i + 1:i + 2] not in ("/", "\\"))
        rows.extend(true_errors(pair[:cut], pair[cut + 1:]))
    best, relax = None, 0.75
    while best is None and relax >= 0.4:
        for kappa in np.round(np.arange(0.5, 1.01, 0.1), 2):
            for t_iyi in np.round(np.arange(1.1, 2.001, 0.05), 3):
                for t_kotu in np.round(np.arange(t_iyi + 0.3, 4.001, 0.1), 3):
                    cats = [decide(r, t_iyi, t_kotu, kappa)[0] for r in rows]
                    _, cond = rates(rows, cats)
                    q1, q3, good = (cond["P(gercek kotu | karar iyi)"][2], cond["P(karar kotu | gercek kotu)"][2],
                                    cond["P(karar iyi | gercek iyi)"][2])
                    if q1 is None or q1 > 0.03 or (q3 is not None and q3 < relax):
                        continue
                    if best is None or (good or 0) > best[0]:
                        best = ((good or 0), float(kappa), float(t_iyi), float(t_kotu), cond)
        if best is None:
            relax = round(relax - 0.05, 2)
    if best is None:
        raise SystemExit("Kısıtları sağlayan eşik bulunamadı")
    result = {"kappa": best[1], "t_iyi": best[2], "t_kotu": best[3],
              "olusturma": {"ciftler": args.pairs, "tarih": time.strftime("%Y-%m-%d %H:%M"),
                            "kisitlar": {"P(gercek kotu | karar iyi)": "<= 0.03", "P(karar kotu | gercek kotu)": f">= {relax}"},
                            "ulasilan": {k: v[2] for k, v in best[4].items()}}}
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def tool_sheet(argv):
    parser = argparse.ArgumentParser(prog="v4.py sayfa")
    parser.add_argument("--run", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = read_report(args.run)
    cells = {}
    for r in rows:
        if r["hole_x0"] != "":
            cells.setdefault((r["category"], r["mode"]), []).append(r)
    per_cell = max(1, args.count // max(1, len(cells)))
    chosen = []
    for key in sorted(cells):
        group = cells[key]
        step = max(1, len(group) // per_cell)
        chosen.extend(group[::step][:per_cell])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "anahtar.csv", "w", newline="", encoding="utf-8") as key_file, \
            open(out / "puanlar.csv", "w", newline="", encoding="utf-8") as score_file:
        key_writer, score_writer = csv.writer(key_file), csv.writer(score_file)
        key_writer.writerow(["id", "frame", "object", "category", "e_hat", "mode"])
        score_writer.writerow(["id", "insan"])
        for i, r in enumerate(chosen):
            original = plane(read_image(Path(args.frames) / r["frame"]))
            result = plane(read_image(find_output(args.run, r["frame"])))
            x0, y0, x1, y1 = (int(r[k]) for k in ("hole_x0", "hole_y0", "hole_x1", "hole_y1"))
            h, w = original.shape
            sl = (slice(max(0, y0 - 30), min(h, y1 + 30)), slice(max(0, x0 - 30), min(w, x1 + 30)))
            lo, hi = np.percentile(original[sl], (1, 99.5))
            tiles = []
            for image in (original, result):
                v = np.clip((image[sl] - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
                tiles.append(cv2.resize(v, (v.shape[1] * 4, v.shape[0] * 4), interpolation=cv2.INTER_NEAREST))
            write_image(out / f"{i:03d}.png", cv2.hconcat(tiles))
            key_writer.writerow([i, r["frame"], r["object"], r["category"], fmt(r["e_hat"]), r["mode"]])
            score_writer.writerow([i, ""])
    print(f"{len(chosen)} örnek yazıldı: {out}")


def tool_agreement(argv):
    parser = argparse.ArgumentParser(prog="v4.py uyum")
    parser.add_argument("--sheet", required=True)
    args = parser.parse_args(argv)
    folder = Path(args.sheet)
    key = {r["id"]: r for r in csv.DictReader(open(folder / "anahtar.csv", encoding="utf-8"))}
    scores = [r for r in csv.DictReader(open(folder / "puanlar.csv", encoding="utf-8")) if r["insan"].strip()]
    table = {c: {h: 0 for h in CATEGORIES} for c in CATEGORIES}
    for s in scores:
        table[key[s["id"]]["category"]][s["insan"].strip()] += 1
    n = sum(sum(v.values()) for v in table.values())
    observed = sum(table[c][c] for c in CATEGORIES) / n if n else 0
    expected = sum(sum(table[c].values()) * sum(table[x][c] for x in CATEGORIES) for c in CATEGORIES) / n ** 2 if n else 0
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else None
    said_good = sum(table["iyi"].values())
    human_bad = sum(table[c]["kotu"] for c in CATEGORIES)
    result = {"tablo (script \\ insan)": table, "n": n, "cohen_kappa": kappa,
              "P(insan kotu | karar iyi)": table["iyi"]["kotu"] / said_good if said_good else None,
              "P(karar kotu | insan kotu)": table["kotu"]["kotu"] / human_bad if human_bad else None}
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def tool_synth(argv):
    parser = argparse.ArgumentParser(prog="v4.py sentez")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", choices=("gok", "zemin", "karma"), default="gok")
    parser.add_argument("--camera", choices=tuple(CAMERAS), default="sabit")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bulut-evrim", type=float, default=520.0)
    args = parser.parse_args(argv)
    synth(args.out, args.scene, args.camera, args.frames, args.seed, cloud_evolve=args.bulut_evrim)
    print(f"yazıldı: {args.out}")


TOOLS = {"sentez": tool_synth, "degerlendir": tool_evaluate, "kalibre": tool_calibrate, "sayfa": tool_sheet,
         "uyum": tool_agreement}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if argv and argv[0] in TOOLS:
        return TOOLS[argv[0]](argv[1:])
    return run(parse_main(argv))


if __name__ == "__main__":
    main()
