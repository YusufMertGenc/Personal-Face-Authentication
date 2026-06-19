
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from model import build_model


def load_model(config_path: str = "../config.json") -> tuple[nn.Module, torch.device]:
    """Load the best saved checkpoint. PROVIDED — do not modify."""
    with open(config_path) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(cfg["model"]["backbone"], cfg["model"]["num_classes"])
    ckpt   = Path(cfg["data"]["checkpoint_dir"]) / "best_model.pt"

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt}'. "
            "Run training (Section 4 of the notebook) first."
        )

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {ckpt} (device: {device})")
    return model, device


def get_inference_transform(image_size: int = 224):
    """Return the evaluation-time transform. PROVIDED — do not modify."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])



def predict(image: Image.Image,
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
) -> dict:


    image = image.convert("RGB")

    transform = get_inference_transform(image_size)
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)

    probs = probs.squeeze(0).cpu()

    return {
        "NOT YOU": float(probs[0].item()),
        "YOU": float(probs[1].item()),
    }



def build_app(config_path: str = "../config.json"):

    import gradio as gr

    with open(config_path) as f:
        cfg = json.load(f)

    image_size = cfg["data"]["image_size"]

    model, device = load_model(config_path)

    def inference(image):
        if image is None:
            return {"NOT YOU": 0.0, "YOU": 0.0}

        pil_image = Image.fromarray(image).convert("RGB")
        return predict(pil_image, model, device, image_size)

    app = gr.Interface(
        fn=inference,
        inputs=gr.Image(label="Upload your face"),
        outputs=gr.Label(num_top_classes=2, label="Prediction"),
        title="Personal Face Authentication",
        description="Upload a face image and the model predicts whether it is YOU or NOT YOU.",
    )

    return app



if __name__ == "__main__":
    app = build_app()
    app.launch()
