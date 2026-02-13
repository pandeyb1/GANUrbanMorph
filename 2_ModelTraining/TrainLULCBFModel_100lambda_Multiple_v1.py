import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import rasterio
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import gc

print("Using torch", torch.__version__)

class EncoderBlock(nn.Module):
    """Encoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)

        self.bn=None
        if norm:
            self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        fx = self.lrelu(x)
        fx = self.conv(fx)

        if self.bn is not None:
            fx = self.bn(fx)

        return fx

class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes)

        self.dropout=None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)

    def forward(self, x):
        fx = self.relu(x) # Changed from fx = self.relu(fx) to fx = self.relu(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)

        if self.dropout is not None:
            fx = self.dropout(fx)

        return fx

class BasicBlock(nn.Module):
    """Basic block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm2d(outplanes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fx = self.conv(x)

        if self.isn is not None:
            fx = self.isn(fx)

        fx = self.lrelu(fx)
        return fx

class Generator(nn.Module):
    """Unet-like Encoder-Decoder model with latent vector input"""
    def __init__(self, latent_dim=8): # Added latent_dim parameter
        super().__init__()
        self.latent_dim = latent_dim

        # Modified the first encoder layer to accept latent_dim channels in addition to the input image channel
        self.encoder1 = nn.Conv2d(1 + self.latent_dim, 64, kernel_size=4, stride=2, padding=1)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)

        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(2*512, 512, dropout=True)
        self.decoder6 = DecoderBlock(2*512, 512, dropout=True)
        self.decoder5 = DecoderBlock(2*512, 512)
        self.decoder4 = DecoderBlock(2*512, 256)
        self.decoder3 = DecoderBlock(2*256, 128)
        self.decoder2 = DecoderBlock(2*128, 64)
        self.decoder1 = nn.ConvTranspose2d(2*64, 1, kernel_size=4, stride=2, padding=1)

    # Modified the forward method to accept latent_vector as an additional input
    def forward(self, x, latent_vector):
        # Expand latent vector to match image dimensions and concatenate
        latent_expanded = latent_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))
        x_with_latent = torch.cat([x, latent_expanded], dim=1)

        # encoder forward
        e1 = self.encoder1(x_with_latent)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
        # decoder forward + skip connections
        d8 = self.decoder8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.decoder7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.decoder6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.decoder2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)

        return torch.tanh(d1)

class Discriminator(nn.Module):
    """Conditional Discriminator"""
    def __init__(self,):
        super().__init__()
        self.block1 = BasicBlock(2, 64, norm=False)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, target_images, cond_images):
        x = torch.cat([target_images,cond_images], dim=1)
        # blocks forward
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)

        return fx

class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_files = [f for f in os.listdir(target_dir) if f.endswith('.tif')]
        self.input_images = []
        self.target_images = []
        print("Loading data into memory...")
        for filename in self.input_files:
            input_path = os.path.join(input_dir, filename)
            target_path = os.path.join(target_dir, filename)
            with rasterio.open(input_path) as src:
                input_image = src.read(1)
                input_image = (input_image / 95) * 2 - 1
            with rasterio.open(target_path) as src:
                target_image = src.read(1)
                target_image = (target_image) * 2 - 1
            # Add channel dimension
            input_image = np.expand_dims(input_image, axis=0)
            target_image = np.expand_dims(target_image, axis=0)
            # Move to Tensor and append
            self.input_images.append(torch.from_numpy(input_image))
            self.target_images.append(torch.from_numpy(target_image))
        print("Data loading complete.")
    def __len__(self):
        return len(self.input_files)
    def __getitem__(self, idx):
        return self.input_images[idx], self.target_images[idx]

class DiscriminatorLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.loss_fn(fake_pred, fake_target)
        real_loss = self.loss_fn(real_pred, real_target)
        loss = (fake_loss + real_loss)/2
        return loss

class GeneratorLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha=alpha
        self.bce=nn.BCEWithLogitsLoss()
        self.l1=nn.L1Loss()

    def forward(self, fake, real, fake_pred):
        fake_target = torch.ones_like(fake_pred)
        loss = self.bce(fake_pred, fake_target) + self.alpha* self.l1(fake, real)
        return loss

batchsize = 128
# lr = 0.0002 # Removed single learning rate definition
learning_rates = [0.0001,0.0002,0.0005, 0.001,0.005,0.01] # Defined a list of learning rates
lambda_pixel = 100  # Weight for pixel-wise loss for the generator
num_epochs = 1000
input_dir = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/Archive/central"
target_dir = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TrainingData/Archive/BFrac"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
latent_dim = 8 # Define the dimension of the latent vector
generator = nn.DataParallel(Generator(latent_dim=latent_dim), device_ids=[0]).to(device) # Pass latent_dim
discriminator = nn.DataParallel(Discriminator(), device_ids=[0]).to(device)
dataset = ImageDataset(input_dir, target_dir)
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True,num_workers=12)

step_count = 0
g_criterion = GeneratorLoss(alpha=lambda_pixel)
d_criterion = DiscriminatorLoss()
print("Starting Training Loop")

for lr in learning_rates: # Loop through each learning rate
    print(f"Training with learning rate: {lr}")
    # Re-initialize optimizers for each learning rate
    d_optimizer = torch.optim.Adam(discriminator.parameters(),lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    GLOSS = list()
    DLOSS = list()

    for epoch in range(num_epochs):
        avgdloss =list()
        avggloss = list()
        epoch_start = time.time()
        for i, (input_images, target_images) in enumerate(dataloader):
            cond = input_images.to(device)
            cond = cond.to(torch.float32)
            real = target_images.to(device)
            real = real.to(torch.float32)
            imbatch_size = input_images.size(0)
            real_labels = torch.ones(imbatch_size, 1, device=device)
            fake_labels = torch.zeros(imbatch_size, 1, device=device)

            # Sample latent vectors
            latent_vector = torch.randn(imbatch_size, latent_dim, device=device) # Sample latent vectors

            # Generator`s loss
            fake = generator(cond, latent_vector) # Pass latent_vector to generator
            fake_pred = discriminator(fake, cond)
            g_loss = g_criterion(fake, real, fake_pred)

            # Discriminator`s loss
            fake = generator(cond, latent_vector).detach() # Pass latent_vector to generator
            fake_pred = discriminator(fake, cond)
            real_pred = discriminator(real, cond)
            d_loss = d_criterion(fake_pred, real_pred)

            # Generator`s params update
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Discriminator`s params update
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            avgdloss.append(d_loss.cpu().detach().numpy())
            avggloss.append(g_loss.cpu().detach().numpy())

        if (epoch+1) % 500 == 0 or (epoch+1) == num_epochs: # Save at the last epoch as well
             torch.save(generator.state_dict(), f'/Users/9oy/Documents/Projects/IM3/EvaluationP/LULCCond_BFgenerator_withLatentVector_100L_epoch_{epoch+1}_LR_{lr}.pth')
             #torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')

        avgdloss = np.array(avgdloss).mean()
        avggloss = np.array(avggloss).mean()
        GLOSS.append(avggloss)
        DLOSS.append(avgdloss)
        if (epoch+1) % 50 == 0:
          print(f"Epoch {epoch+1} completed in {time.time() - epoch_start:.2f} sec")
          print(f"Average D Loss: {avgdloss}")
          print(f"Average G Loss: {avggloss}")

    # Save loss lists for each learning rate
    np.savetxt(f'/Users/9oy/Documents/Projects/IM3/EvaluationP/LULCCond_BFgenerator_withLatentVector_100L_LR_{lr}_GLOSS.csv', np.array(GLOSS), delimiter=',')
    np.savetxt(f'/Users/9oy/Documents/Projects/IM3/EvaluationP/LULCCond_BFgenerator_withLatentVector_100L_LR_{lr}_DLOSS.csv', np.array(DLOSS), delimiter=',')

generator_path = '/Users/9oy/Documents/Projects/IM3/EvaluationP/LULCCond_BFgenerator_withLatentVector_100L_epoch_1000_LR_0.0002.pth'
generator_eval = Generator(latent_dim=latent_dim).to(device)

# Remove the 'module.' prefix from the state_dict keys
state_dict = torch.load(generator_path, map_location=device)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

generator_eval.load_state_dict(new_state_dict)

generator_eval.eval()  # Set the generator to evaluation mode
print(f"Generator loaded successfully from {generator_path}")

# Select 4 random indices from the dataset
random_indices = random.sample(range(len(dataset)), 4)

# Get the corresponding input and target images
random_input_images = [dataset[i][0] for i in random_indices]
random_target_images = [dataset[i][1] for i in random_indices]

print(f"Selected {len(random_indices)} random tiles from the dataset.")

## Run Inference
generated_images = []
with torch.no_grad():  # Disable gradient calculation for inference
    for input_image in random_input_images:
        input_image = input_image.unsqueeze(0).to(device).to(torch.float32) # Add batch dimension and move to device
        latent_vector = torch.randn(1, latent_dim, device=device) # Sample a latent vector for each image
        fake_image = generator_eval(input_image, latent_vector).squeeze(0).cpu().numpy() # Generate image, remove batch dimension and move to CPU
        generated_images.append(fake_image)
print(f"Generated images for {len(generated_images)} random tiles.")

# Visualize the results
fig, axes = plt.subplots(4, 3, figsize=(12, 16))

for i in range(4):
    # Input image
    axes[i, 0].imshow(random_input_images[i].squeeze(), cmap='gray')
    axes[i, 0].set_title('Input Image')
    axes[i, 0].axis('off')

    # Target image
    axes[i, 1].imshow(random_target_images[i].squeeze(), cmap='gray')
    axes[i, 1].set_title('Target Image')
    axes[i, 1].axis('off')

    # Generated image
    axes[i, 2].imshow(generated_images[i].squeeze(), cmap='gray')
    axes[i, 2].set_title('Generated Image')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.show()

torch.cuda.empty_cache()
gc.collect()