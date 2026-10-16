"""Porównanie detektorów kół: Florence-2 (base/large) vs Grounding DINO (tiny/base).

Uruchamiać przez gpu-run.sh, jeden detektor na proces (czysty pomiar pamięci):
    gpu-run.sh uv run python scripts/detect_wheels.py florence2-large
Wynik: out/detect/<detektor>.json + nakładki out/detect/<detektor>__<zdjęcie>.jpg
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Florence2ForConditionalGeneration

from wheeler_spike.memwatch import PeakMemory

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "docs" / "spike" / "samples"
OUT = Path(__file__).resolve().parents[1] / "out" / "detect"
CARS = ["car_side_golf.jpg", "car_34front_sorento.jpg", "car_occluded_vios.jpg"]
DEVICE = "mps"

DETECTORS = {
    "florence2-base": "florence-community/Florence-2-base",
    "florence2-large": "florence-community/Florence-2-large",
    "gdino-tiny": "IDEA-Research/grounding-dino-tiny",
    "gdino-base": "IDEA-Research/grounding-dino-base",
}


def load(name: str):
    repo = DETECTORS[name]
    processor = AutoProcessor.from_pretrained(repo)
    if name.startswith("florence2"):
        model = Florence2ForConditionalGeneration.from_pretrained(repo, dtype=torch.float16)
    else:
        model = AutoModelForZeroShotObjectDetection.from_pretrained(repo)
    return processor, model.to(DEVICE).eval()


@torch.no_grad()
def detect_florence(processor, model, image: Image.Image, mode: str) -> list[dict]:
    if mode == "od":
        task, text = "<OD>", "<OD>"
    else:
        task, text = "<CAPTION_TO_PHRASE_GROUNDING>", "<CAPTION_TO_PHRASE_GROUNDING>wheel"
    inputs = processor(text=text, images=image, return_tensors="pt").to(DEVICE, torch.float16)
    ids = model.generate(**inputs, max_new_tokens=1024, num_beams=3, do_sample=False)
    raw = processor.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(raw, task=task, image_size=image.size)[task]
    dets = []
    for box, label in zip(parsed["bboxes"], parsed["labels"]):
        if mode == "od" and not any(k in label.lower() for k in ("wheel", "tire", "tyre")):
            continue
        dets.append({"box": [round(float(v), 1) for v in box], "label": label, "score": None})
    return dets


@torch.no_grad()
def detect_gdino(processor, model, image: Image.Image) -> list[dict]:
    inputs = processor(images=image, text="a car wheel.", return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.35, text_threshold=0.25, target_sizes=[image.size[::-1]]
    )[0]
    return [
        {"box": [round(float(v), 1) for v in box], "label": label, "score": round(float(score), 3)}
        for box, score, label in zip(result["boxes"].cpu(), result["scores"].cpu(), result["text_labels"])
    ]


def draw(image: Image.Image, dets: list[dict], path: Path) -> None:
    im = image.copy()
    d = ImageDraw.Draw(im)
    w = max(3, im.width // 400)
    for det in dets:
        d.rectangle(det["box"], outline=(255, 0, 255), width=w)
        tag = det["label"] if det["score"] is None else f"{det['label']} {det['score']:.2f}"
        d.text((det["box"][0] + 4, det["box"][1] + 4), tag, fill=(255, 255, 0))
    im.thumbnail((1000, 1000))
    im.save(path, quality=80)


def main() -> None:
    name = sys.argv[1]
    modes = sys.argv[2:] or (["grounding", "od"] if name.startswith("florence2") else ["gdino"])
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"detector": name, "repo": DETECTORS[name], "runs": {}}
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        processor, model = load(name)
        if DEVICE == "mps":
            torch.mps.synchronize()
        report["load_s"] = round(time.perf_counter() - t0, 2)
        for mode in modes:
            runs = report["runs"].setdefault(mode, {})
            for car in CARS:
                image = Image.open(SAMPLES / car).convert("RGB")
                timings = []
                for _ in range(3):  # pierwszy przebieg = rozgrzewka
                    t0 = time.perf_counter()
                    if name.startswith("florence2"):
                        dets = detect_florence(processor, model, image, mode)
                    else:
                        dets = detect_gdino(processor, model, image)
                    timings.append(time.perf_counter() - t0)
                runs[car] = {"first_s": round(timings[0], 2), "warm_s": round(min(timings[1:]), 2), "detections": dets}
                draw(image, dets, OUT / f"{name}-{mode}__{Path(car).stem}.jpg")
                print(name, mode, car, len(dets), runs[car]["warm_s"], flush=True)
    report["memory"] = mem.report()
    (OUT / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}))


if __name__ == "__main__":
    main()
