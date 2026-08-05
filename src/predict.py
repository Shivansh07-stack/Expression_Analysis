"""
Run inference on a single image with a trained checkpoint.

Usage:
    python predict.py --image path/to/face.jpg --checkpoint checkpoints/best_fer2013_model.pth
"""

import argparse

import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import EXPRESSIONS, build_transforms
from model import EmotionCNN, NUM_CLASSES


def predict(model, image_path, device):
    _, eval_transform = build_transforms()

    image = Image.open(image_path).convert("L")
    tensor = eval_transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    predicted_idx = int(probs.argmax())
    return image, probs.cpu().numpy(), EXPRESSIONS[predicted_idx], float(probs[predicted_idx])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--no-show", action="store_true", help="Skip the matplotlib window")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    # supports both a bare state_dict and the {"model_state_dict": ...} format train.py saves
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)

    image, probs, label, confidence = predict(model, args.image, device)
    print(f"Predicted: {label} ({confidence * 100:.1f}%)")

    if not args.no_show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(image, cmap="gray")
        ax1.set_title("Input")
        ax1.axis("off")

        ax2.barh([e.capitalize() for e in EXPRESSIONS], probs)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Probability")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
