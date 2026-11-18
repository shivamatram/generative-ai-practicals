"""
Experiment 7: Create artistic images using Neural Style Transfer (NST).

Put two images in the same folder:
    content.jpg  -> image whose objects/layout should remain
    style.jpg    -> painting/photo whose artistic style should be copied

Run:
    pip install torch torchvision pillow
    python experiment_7_neural_style_transfer.py

Output: stylized_output.jpg
"""

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
STEPS = 300
STYLE_WEIGHT = 1_000_000
CONTENT_WEIGHT = 1

def load_image(path):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def gram_matrix(features):
    _, channels, height, width = features.size()
    matrix = features.view(channels, height * width)
    return matrix @ matrix.t() / (channels * height * width)

def main():
    content = load_image("content.jpg")
    style = load_image("style.jpg")

    weights = models.VGG19_Weights.DEFAULT
    vgg = models.vgg19(weights=weights).features.to(DEVICE).eval()
    for parameter in vgg.parameters():
        parameter.requires_grad_(False)

    # Selected VGG19 layers: shallow layers capture style; deeper layer captures content.
    style_layers = {"0", "5", "10", "19", "28"}
    content_layer = "21"

    def extract_features(image):
        features = {}
        x = image
        for name, layer in vgg._modules.items():
            x = layer(x)
            if name in style_layers or name == content_layer:
                features[name] = x
        return features

    with torch.no_grad():
        content_features = extract_features(content)
        style_features = extract_features(style)
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_layers}

    generated = content.clone().requires_grad_(True)
    optimizer = optim.Adam([generated], lr=0.02)

    for step in range(STEPS):
        generated_features = extract_features(generated)
        content_loss = torch.mean(
            (generated_features[content_layer] - content_features[content_layer]) ** 2
        )

        style_loss = 0
        for layer in style_layers:
            generated_gram = gram_matrix(generated_features[layer])
            style_loss += torch.mean((generated_gram - style_grams[layer]) ** 2)

        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        generated.data.clamp_(0, 1)

        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{STEPS}, loss={total_loss.item():.4f}")

    output = transforms.ToPILImage()(generated.detach().cpu().squeeze(0))
    output.save("stylized_output.jpg")
    print("Saved stylized_output.jpg")

if __name__ == "__main__":
    main()
