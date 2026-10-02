import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from model import CLASSES, build_model, eval_transform, pick_device

TF = eval_transform()


def perturbations(image):
    return {
        "original": image,
        "darker": ImageEnhance.Brightness(image).enhance(0.65),
        "brighter": ImageEnhance.Brightness(image).enhance(1.35),
        "lower_contrast": ImageEnhance.Contrast(image).enhance(0.65),
        "blur": image.filter(ImageFilter.GaussianBlur(radius=1.5)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="models/robustness.json")
    args = parser.parse_args()

    device = pick_device()
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    image = Image.open(args.image).convert("RGB")
    results = {}

    with torch.inference_mode():
        for name, variant in perturbations(image).items():
            x = TF(variant).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            results[name] = {
                "top_class": CLASSES[idx],
                "confidence": float(probs[idx]),
                "probabilities": {
                    CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))
                },
            }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
