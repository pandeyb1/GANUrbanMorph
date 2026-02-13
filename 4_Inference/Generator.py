from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
#from torchvision import transforms
import rasterio
import numpy as np
from matplotlib import pyplot as plt
import rasterio as rio
from rasterio.windows import from_bounds
from rasterio.coords import BoundingBox
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

class Generatormulti(nn.Module):
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
    

class MorphInf():
    def __init__(self, params):
        self.window_size = 256
        self.stride = 64
        self.params = params
        with rio.open(self.params.get("ProjLULC"))  as src:
            self.row, self.col = src.shape
            self.crs = src.crs
            self.ProjLULC = src.read(1)
            self.ProjLULC[(self.ProjLULC<0) | (self.ProjLULC>95)] = 0
            self.left, self.bottom, self.right, self.top = src.bounds
            self.bounds = src.bounds
            self.transform= src.transform
        with rio.open(self.params.get("CurLULC")) as src:
            ww = from_bounds(self.left, self.bottom, self.right, self.top, transform=src.transform)
            self.CurLULC = src.read(1,window=ww)
            self.CurLULC[self.CurLULC>95] = 0
            self.CurLULC[self.ProjLULC==0]=0
        with rio.open(self.params.get("BFfile")) as src:
            self.BF2015 = src.read(1)
        with rio.open(self.params.get("BHfile")) as src:
            self.BH2015 = src.read(1) * 0.3048 
            self.BH2015[self.ProjLULC==0] = 0

        print("Inference Parameters Initialized")
    
    def initinference(self):
        padded_h = max(self.row, self.window_size)
        if (padded_h - self.window_size) % self.stride != 0:
            padded_h = padded_h + (self.stride - (padded_h - self.window_size) % self.stride)

        padded_w = max(self.col, self.window_size)
        if (padded_w - self.window_size) % self.stride != 0:
            padded_w = padded_w + (self.stride - (padded_w - self.window_size) % self.stride)
        pad_h_val = padded_h - self.row
        pad_w_val = padded_w - self.col
        self.padded_array = np.pad(self.ProjLULC,((0, pad_h_val), (0, pad_w_val)),mode='reflect')
            
        padded_h, padded_w = self.padded_array.shape
        self.predictions = np.zeros_like(self.padded_array, dtype=np.float32)
        self.counts = np.zeros_like(self.padded_array, dtype=np.float32)
        self.predictions1 = np.zeros_like(self.padded_array, dtype=np.float32)
        self.counts1 = np.zeros_like(self.padded_array, dtype=np.float32)

        # Generate window coordinates
        self.y_coords = range(0, padded_h - self.window_size + 1, self.stride)
        self.x_coords = range(0, padded_w - self.window_size + 1, self.stride)
        print("Inference Ready")
    
    def initmodel(self):
        state_dict = torch.load(self.params.get("model"),map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # Remove 'module.' prefix
            new_state_dict[new_key] = v
        self.generator = Generatormulti()
        self.generator.load_state_dict(new_state_dict)
        self.generator.eval()
        self.generator.to(device)
        print("Generator Ready for Building Footprints")

        state_dict = torch.load(self.params.get("model1"),map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # Remove 'module.' prefix
            new_state_dict[new_key] = v
        self.generator1 = Generatormulti()
        self.generator1.load_state_dict(new_state_dict)
        self.generator1.eval()
        self.generator1.to(device)
        print("Generator Ready for Building Heights")

    def runinference(self):
        torch.manual_seed(self.params.get("seed"))
        with torch.no_grad():
            total_steps = self.row
            with tqdm(total=total_steps, desc="Progress") as pbar:
                pbar.update(self.stride)
                for y in self.y_coords:
                    pbar.update(self.stride)
                    for x in self.x_coords:
                        patch = self.padded_array[y:y + self.window_size, x:x + self.window_size]
                        patch = patch.astype(np.float32)
                        tiledat = (patch/95) * 2 - 1
                        latent_vector = torch.randn(1, 8, device=device) # Sample latent vectors
                        cond = torch.from_numpy(np.expand_dims(tiledat, axis=(0,1))).to(torch.float32).to(device)
                        output = self.generator(cond,latent_vector).cpu().detach().numpy()[0,0,:,:]
                        output = ((output + 1)/2)
                        self.predictions[y:y + self.window_size, x:x + self.window_size] += output
                        self.counts[y:y + self.window_size, x:x + self.window_size] += 1

                        tiledat1 = (output) * 2 - 1
                        cond1 = torch.from_numpy(np.expand_dims(tiledat1, axis=(0,1))).to(torch.float32).to(device)
                        output1 = self.generator1(cond1,latent_vector).cpu().detach().numpy()[0,0,:,:]
                        output1 = np.exp((output1 + 1) * np.log(75)/2)
                        existingout1 = self.predictions1[y:y + self.window_size, x:x + self.window_size]
                        self.predictions1[y:y + self.window_size, x:x + self.window_size] += output1
                        self.counts1[y:y + self.window_size, x:x + self.window_size] += 1
        # Average the predictions in overlapping regions
        self.BF= self.predictions / self.counts
        self.BH= self.predictions1 / self.counts1
        self.BF = self.BF[:self.row, :self.col]
        self.BH = self.BH[:self.row, :self.col]
        self.BH2015[self.BH2015 >75] = 75
        self.BF[self.ProjLULC == self.CurLULC] = self.BF2015[self.ProjLULC == self.CurLULC]
        self.BH[self.ProjLULC == self.CurLULC] = self.BH2015[self.ProjLULC == self.CurLULC]
        self.BH[self.BF == 0] = 0