"""Maski kół SAM 2.1 z boxów detektora (Florence-2-large) i z pojedynczego kliknięcia.

    gpu-run.sh uv run python scripts/sam_masks.py facebook/sam2.1-hiera-large
Wymaga out/detect/florence2-large.json. Wynik: out/masks/<zdjęcie>_<i>.png (maska box),
out/masks/<zdjęcie>_<i>_point.png (maska z kliknięcia w środek boxa), out/masks/sam.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from wheeler_spike.memwatch import PeakMemory

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "docs" / "spike" / "samples"
OUT = Path(__file__).resolve().parents[1] / "out"
DEVICE = "mps"


def sync() -> None:
    torch.mps.synchronize()


def main() -> None:
    repo = sys.argv[1] if len(sys.argv) > 1 else "facebook/sam2.1-hiera-large"
    detections = json.loads((OUT / "detect" / "florence2-large.json").read_text())["runs"]["grounding"]
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    report: dict = {"repo": repo, "images": {}}
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        processor = Sam2Processor.from_pretrained(repo)
        model = Sam2Model.from_pretrained(repo).to(DEVICE).eval()
        sync()
        report["load_s"] = round(time.perf_counter() - t0, 2)
        for car, run in detections.items():
            image = Image.open(SAMPLES / car).convert("RGB")
            boxes = [d["box"] for d in run["detections"]]
            entry: dict = {"boxes": boxes}
            with torch.no_grad():
                for attempt in range(2):  # 0 = rozgrzewka
                    t0 = time.perf_counter()
                    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
                    embeddings = model.get_image_embeddings(inputs["pixel_values"])
                    sync()
                    entry["embed_s"] = round(time.perf_counter() - t0, 3)

                    t0 = time.perf_counter()
                    prompt = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(DEVICE)
                    out = model(
                        image_embeddings=embeddings, input_boxes=prompt["input_boxes"], multimask_output=False
                    )
                    masks = processor.post_process_masks(out.pred_masks.cpu(), prompt["original_sizes"].cpu())[0]
                    entry["boxes_decode_s"] = round(time.perf_counter() - t0, 3)

                    t0 = time.perf_counter()
                    centers = [[[[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]] for b in boxes]]
                    prompt = processor(
                        images=image, input_points=centers, input_labels=[[[1] for _ in boxes]], return_tensors="pt"
                    ).to(DEVICE)
                    out = model(
                        image_embeddings=embeddings,
                        input_points=prompt["input_points"],
                        input_labels=prompt["input_labels"],
                        multimask_output=True,
                    )
                    # Z 3 kandydatów bierzemy maskę o najwyższym przewidywanym IoU.
                    best = out.iou_scores.argmax(dim=-1)
                    point_masks = processor.post_process_masks(out.pred_masks.cpu(), prompt["original_sizes"].cpu())[0]
                    entry["point_decode_s"] = round(time.perf_counter() - t0, 3)
                    # Klik -> maska felgi (best IoU) -> box powiększony 1,5x wokół jej środka -> drugi prompt
                    # box + ten sam punkt. Felga zajmuje ~2/3 średnicy koła, więc 1,5x obejmuje oponę.
                    t0 = time.perf_counter()
                    grown_boxes = []
                    for i in range(len(boxes)):
                        m = point_masks[i, int(best[0, i])].numpy()
                        ys, xs = np.nonzero(m)
                        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                        half = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.75
                        grown_boxes.append([float(cx - half), float(cy - half), float(cx + half), float(cy + half)])
                    prompt = processor(
                        images=image, input_boxes=[grown_boxes], input_points=centers,
                        input_labels=[[[1] for _ in boxes]], return_tensors="pt",
                    ).to(DEVICE)
                    out = model(
                        image_embeddings=embeddings,
                        input_boxes=prompt["input_boxes"],
                        input_points=prompt["input_points"],
                        input_labels=prompt["input_labels"],
                        multimask_output=False,
                    )
                    click_masks = processor.post_process_masks(out.pred_masks.cpu(), prompt["original_sizes"].cpu())[0]
                    entry["click_box_decode_s"] = round(time.perf_counter() - t0, 3)
            stem = Path(car).stem
            for i in range(len(boxes)):
                cm = click_masks[i, 0].numpy().astype(np.uint8) * 255
                Image.fromarray(cm).save(OUT / "masks" / f"{stem}_{i}_clickbox.png")
                m = masks[i, 0].numpy().astype(np.uint8) * 255
                Image.fromarray(m).save(OUT / "masks" / f"{stem}_{i}.png")
                pm = point_masks[i, int(best[0, i])].numpy().astype(np.uint8) * 255
                Image.fromarray(pm).save(OUT / "masks" / f"{stem}_{i}_point.png")
                # Wariant: największa z 3 masek (zwykle całe koło zamiast samej felgi / piasty).
                areas = point_masks[i].flatten(1).sum(dim=1)
                pm = point_masks[i, int(areas.argmax())].numpy().astype(np.uint8) * 255
                Image.fromarray(pm).save(OUT / "masks" / f"{stem}_{i}_pointmax.png")
            report["images"][car] = entry
            print(car, entry, flush=True)
    report["memory"] = mem.report()
    (OUT / "masks" / f"sam-{repo.split('/')[-1]}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "images"}))


if __name__ == "__main__":
    main()
