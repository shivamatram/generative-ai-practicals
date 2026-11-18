"""
Experiment 5: Study and train a simple Deep Convolutional GAN (DCGAN) on MNIST.

Run:
    pip install torch torchvision matplotlib
    python experiment_5_gan_mnist.py

The generator learns to create handwritten-digit-like images. Training can take
several minutes; use a GPU when available.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.0002
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, z):
        return self.network(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.network(image)

def main():
    os.makedirs("gan_output", exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

    for epoch in range(EPOCHS):
        for real_images, _ in loader:
            real_images = real_images.to(DEVICE)
            batch = real_images.size(0)
            real_labels = torch.ones(batch, 1, device=DEVICE)
            fake_labels = torch.zeros(batch, 1, device=DEVICE)

            # Train discriminator: real images -> 1, generated images -> 0
            noise = torch.randn(batch, LATENT_DIM, device=DEVICE)
            fake_images = generator(noise)
            d_real = discriminator(real_images)
            d_fake = discriminator(fake_images.detach())
            d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generator: make discriminator classify fake images as real
            noise = torch.randn(batch, LATENT_DIM, device=DEVICE)
            generated = generator(noise)
            g_loss = criterion(discriminator(generated), real_labels)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] D_loss={d_loss.item():.4f} G_loss={g_loss.item():.4f}")
        with torch.no_grad():
            samples = generator(torch.randn(25, LATENT_DIM, device=DEVICE))
            save_image(samples, f"gan_output/epoch_{epoch + 1}.png",
                       nrow=5, normalize=True, value_range=(-1, 1))

    torch.save(generator.state_dict(), "gan_output/generator.pth")
    print("Training complete. Generated samples are saved in gan_output/.")

if __name__ == "__main__":
    main()
