from __future__ import annotations

import argparse
import math
from PIL import Image
import numpy as np


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    s = hex_color.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError("Hex color must be 6 chars, e.g. #01e678")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def make_union_find(n: int):
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    return parent, rank, find, union


def find_key_boxes_downsample(
    rgba: np.ndarray,
    key_rgb: np.ndarray,
    step: int,
    core_tol: int,
    min_area: int,
    take_top: int,
) -> list[tuple[int, int, int, int, int]]:
    """Cari bounding box area hijau utama (4 kotak) pakai downsample + connected components."""
    h, w = rgba.shape[0], rgba.shape[1]
    gw = math.ceil(w / step)
    gh = math.ceil(h / step)

    mask = np.zeros(gw * gh, dtype=np.uint8)

    for gy in range(gh):
        y = min(h - 1, gy * step)
        row = rgba[y]
        for gx in range(gw):
            x = min(w - 1, gx * step)
            rgb = row[x, :3].astype(np.float32)
            d = float(np.linalg.norm(rgb - key_rgb))
            if d <= core_tol:
                mask[gy * gw + gx] = 1

    parent, rank, find, union = make_union_find(gw * gh)

    for gy in range(gh):
        base = gy * gw
        for gx in range(gw):
            i = base + gx
            if mask[i] == 0:
                continue
            if gx > 0 and mask[i - 1]:
                union(i, i - 1)
            if gy > 0 and mask[i - gw]:
                union(i, i - gw)

    boxes: dict[int, list[int]] = {}
    for gy in range(gh):
        base = gy * gw
        for gx in range(gw):
            i = base + gx
            if mask[i] == 0:
                continue
            r = find(i)
            if r not in boxes:
                boxes[r] = [gx, gy, gx, gy]  
            else:
                b = boxes[r]
                b[0] = min(b[0], gx)
                b[1] = min(b[1], gy)
                b[2] = max(b[2], gx)
                b[3] = max(b[3], gy)

    rects = []
    for b in boxes.values():
        minx, miny, maxx, maxy = b
        x0 = minx * step
        y0 = miny * step
        x1 = min(w, (maxx + 1) * step)
        y1 = min(h, (maxy + 1) * step)
        area = (x1 - x0) * (y1 - y0)
        if area >= min_area:
            rects.append((x0, y0, x1, y1, area))

    rects.sort(key=lambda t: t[4], reverse=True)
    return rects[:take_top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="newspaper.png")
    ap.add_argument("--out", dest="out_path", default="frame_transparent.png")

    ap.add_argument("--key", default="#01e678")

    # Deteksi kotak hijau 
    ap.add_argument("--step", type=int, default=3, help="Downsample step for box detection")
    ap.add_argument("--core-tol", type=int, default=6, help="Strict tol to find the solid green core")
    ap.add_argument("--min-area", type=int, default=1500, help="Min box area (approx) to keep")
    ap.add_argument("--boxes", type=int, default=4, help="How many biggest green boxes to process")
    ap.add_argument("--margin", type=int, default=18, help="Expand each box by N pixels to catch fringes")

    # Keying dalam box 
    ap.add_argument("--tol", type=int, default=90, help="RGB distance tolerance (bigger = more removed)")
    ap.add_argument("--soft", type=int, default=60, help="Feather range. 0 = hard cut")

    # Tambahan deteksi fringe
    ap.add_argument("--excess-low", type=int, default=10, help="Start removing when green excess >= this")
    ap.add_argument("--excess-soft", type=int, default=40, help="Full remove after excess-low + excess-soft")

    ap.add_argument("--no-despill", action="store_true", help="Disable green despill on semi-transparent edges")

    args = ap.parse_args()

    key_r, key_g, key_b = hex_to_rgb(args.key)
    key_rgb = np.array([key_r, key_g, key_b], dtype=np.float32)

    im = Image.open(args.in_path).convert("RGBA")
    rgba = np.array(im).astype(np.float32)  # H,W,4
    h, w = rgba.shape[0], rgba.shape[1]

    rects = find_key_boxes_downsample(
        rgba=rgba,
        key_rgb=key_rgb,
        step=max(1, args.step),
        core_tol=max(0, args.core_tol),
        min_area=max(0, args.min_area),
        take_top=max(1, args.boxes),
    )

    if not rects:
        raise RuntimeError("No green boxes detected. Coba kecilkan --core-tol atau cek inputnya.")

    out = rgba.copy()

    tol = max(0, args.tol)
    soft = max(0, args.soft)
    ex_low = args.excess_low
    ex_soft = max(0, args.excess_soft)
    despill = not args.no_despill

    for (x0, y0, x1, y1, _area) in rects:
        x0 = max(0, x0 - args.margin)
        y0 = max(0, y0 - args.margin)
        x1 = min(w, x1 + args.margin)
        y1 = min(h, y1 + args.margin)

        region = out[y0:y1, x0:x1]
        rgb = region[..., :3]
        a = region[..., 3]

        dist = np.sqrt(np.sum((rgb - key_rgb) ** 2, axis=-1))  

        # strength_d: 1 = remove, 0 = keep
        if soft == 0:
            strength_d = (dist <= tol).astype(np.float32)
        else:
            keep = np.clip((dist - tol) / float(soft), 0.0, 1.0)  
            strength_d = 1.0 - keep

        max_rb = np.maximum(rgb[..., 0], rgb[..., 2])
        excess = rgb[..., 1] - max_rb  

        if ex_soft == 0:
            strength_e = (excess >= ex_low).astype(np.float32)
        else:
            strength_e = np.clip((excess - ex_low) / float(ex_soft), 0.0, 1.0)

        strength = np.maximum(strength_d, strength_e)

        # Apply alpha
        region[..., 3] = a * (1.0 - strength)

        # Despill 
        if despill:
            region[..., 1] = rgb[..., 1] - strength * (rgb[..., 1] - max_rb)

        out[y0:y1, x0:x1] = region

    alpha = out[..., 3]
    zero = alpha <= 0.5
    out[zero, 0:3] = 0.0

    out_img = Image.fromarray(out.astype(np.uint8), mode="RGBA")
    out_img.save(args.out_path)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
