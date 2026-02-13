import os
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import rasterio
import numpy as np
import scipy
import time
#from matplotlib import pyplot as plt
batchsize = 16

def plot_generator_examples(dataloader, generator, device='cpu', num_examples=5):
    """
    Plots real targets vs generator outputs for a few examples from a dataloader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader yielding (inputs, targets).
        generator (nn.Module): Generator model.
        device (str): 'cpu' or 'cuda' device to run model on.
        num_examples (int): Number of examples to plot.
    """
    generator.eval()
    generator.to(device)

    inputs_batch, targets_batch = next(iter(dataloader))

    inputs_batch = inputs_batch.to(device)
    inputs_batch = inputs_batch.to(torch.float32)
    targets_batch = targets_batch.to(device)
    imbatch_size = inputs_batch.size(0)

    with torch.no_grad():
        latent_vector = torch.randn(imbatch_size, 8, device=device) # Sample latent vectors
        outputs_batch = generator(inputs_batch,latent_vector)

    inputs_batch = inputs_batch.cpu()
    targets_batch = targets_batch.cpu()

    outputs_batch = outputs_batch.cpu()

    fig, axes = plt.subplots(num_examples, 2, figsize=(6, 3 * num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, 2)

    for i in range(num_examples):
        # Real target
        axes[i, 0].imshow(targets_batch[i].squeeze(), cmap='inferno')
        axes[i, 0].set_title('Real Target')
        axes[i, 0].axis('off')
        
        # Generator output
        axes[i, 1].imshow(outputs_batch[i].squeeze(), cmap='inferno')
        axes[i, 1].set_title('Generator Output')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

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

def add_equation_and_r2(ax, slope, intercept, r_value, x_pos=0.05, y_pos_eq=0.95, y_pos_r2=0.90, color='black', fontsize=12):

    r_squared = r_value**2

    # Format the equation string. Handle negative intercept for cleaner display.
    if intercept >= 0:
        equation_text = r'$y = {:.2f}x + {:.2f}$'.format(slope, intercept)
    else:
        # Use abs(intercept) to avoid double negative (e.g., + -5 becomes - 5)
        equation_text = r'$y = {:.2f}x - {:.2f}$'.format(slope, abs(intercept))

    r2_text = r'$R^2 = {:.2f}$'.format(r_squared) # LaTeX for math

    # Add the equation to the plot
    ax.text(x_pos, y_pos_eq, equation_text, transform=ax.transAxes,
            fontsize=fontsize, color=color, verticalalignment='top')

    # Add the R-squared value to the plot
    ax.text(x_pos, y_pos_r2, r2_text, transform=ax.transAxes,
            fontsize=fontsize, color=color, verticalalignment='top')

def shanentropy(ref):
    refhist,_ =np.histogram(ref,bins=128,range=(0,1))
    prob_dist = refhist/np.sum(refhist)
    prob_dist = prob_dist[prob_dist > 0]
    entropy = -np.sum(prob_dist * np.log2(prob_dist))
    return(entropy)

def runtestinference(modelfile,dataloader,device):
    print(os.path.basename(modelfile))
    torch.manual_seed(100)
    state_dict = torch.load(modelfile,map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = v
    generator = Generator()
    generator.load_state_dict(new_state_dict)
    generator.eval()
    
    generator.to(device)

    plot_generator_examples(dataloader, generator, device='cpu', num_examples=4)

    outr = list()
    refsum = list()
    predsum = list()
    refsd = list()
    predsd = list()
    refshannon = list()
    predshannon = list()
    maes = list()
    mses = list()


    for i, (input_images, target_images) in enumerate(dataloader):
        cond = input_images.to(torch.float32).to(device)
        real = target_images.to(torch.float32).to(device)
        latent_vector = torch.randn(input_images.size(0), 8, device=device) # Sample latent vectors
        with torch.no_grad():
            generator.to(device)
            fake = generator(cond,latent_vector)
        imb = real.shape[0]
        for j in range(imb):
            pred= fake[j,0,:,:].cpu().numpy()
            ref= real[j,0,:,:].cpu().numpy()
            mae = np.mean(np.abs(pred - ref))
            mse = np.mean((pred - ref) ** 2)
            maes.append(mae)
            mses.append(mse)
            slp, interc, rval,pval,stnderr = scipy.stats.linregress(pred.flatten(), ref.flatten())
            ref = ((ref + 1)/2)
            pred = ((pred + 1)/2)
            refsum.append(ref.sum())
            predsum.append(pred.sum())
            outr.append(rval)
            predsd.append(np.std(pred))
            refsd.append(np.std(ref))
            refshannon.append(shanentropy(ref))
            predshannon.append(shanentropy(pred))



    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(np.array(refsum)/65536,np.array(predsum)/65536,alpha=0.25,c="k")
    plt.ylabel("Generated",fontsize=14)
    plt.xlabel("Real",fontsize=14)
    plt.title("Total Building Footprint Fraction",fontsize=16)
    plt.axline((0.3, 0.3), slope=1,c='k')
    slp, interc, rval,pval,stnderr = scipy.stats.linregress(np.array(refsum)/65536, np.array(predsum)/65536)
    print(slp,interc,rval,pval)
    add_equation_and_r2(ax, slp, interc, rval, x_pos=0.05, y_pos_eq=0.95, y_pos_r2=0.88, color='k', fontsize=14)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(np.array(refsd),np.array(predsd),alpha=0.25,c="k")
    slp, interc, rval,pval,stnderr = scipy.stats.linregress(np.array(refsd),np.array(predsd))
    print(slp,interc,rval,pval)
    plt.ylabel("Generated",fontsize=14)
    plt.xlabel("Real",fontsize=14)
    plt.axline((0.3, 0.3), slope=1,c='k')
    plt.title("Building Footprint Fraction Heterogeneity",fontsize=16)
    add_equation_and_r2(ax, slp, interc, rval, x_pos=0.05, y_pos_eq=0.95, y_pos_r2=0.88, color='k', fontsize=14)
    plt.show()
    return(maes,mses)

input_dir = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TestNooverlap/central"
target_dir = "/Users/9oy/Documents/Projects/IM3/EvaluationP/Data/TestNooverlap/Frac"
dataset = ImageDataset(input_dir, target_dir)
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False,num_workers=0, pin_memory=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

modelfile1 = r"/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/2_EvalP/v1/BF/LULCCond_BFgenerator_withLatentVector_100L_epoch_1000_LR_0.0001.pth"
modelfile2 = r"/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/2_EvalP/v1/BF/LULCCond_BFgenerator_withLatentVector_100L_epoch_1000_LR_0.0002.pth"
modelfile3 = r"/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/2_EvalP/v1/BF/LULCCond_BFgenerator_withLatentVector_100L_epoch_1000_LR_0.0005.pth"
modelfile4 = r"/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/2_EvalP/v1/BF/LULCCond_BFgenerator_withLatentVector_100L_epoch_1000_LR_0.001.pth"
modelfile5 = r"/Users/9oy/Documents/Projects/IM3/EvaluationP/Script/2_EvalP/v1/BF/LULCCond_BFgenerator_withLatentVector_100L_epoch_1000_LR_0.005.pth"

m11,m21 = runtestinference(modelfile1,dataloader,device)
m12,m22 = runtestinference(modelfile2,dataloader,device)
m13,m23 = runtestinference(modelfile3,dataloader,device)
m14,m24 = runtestinference(modelfile4,dataloader,device)
m15,m25 = runtestinference(modelfile5,dataloader,device)

print(np.round(np.array(m11).mean(),4))
print(np.round(np.array(m12).mean(),4))
print(np.round(np.array(m13).mean(),4))
print(np.round(np.array(m14).mean(),4))
print(np.round(np.array(m15).mean(),4))

print(np.round(np.array(m21).mean(),4))
print(np.round(np.array(m22).mean(),4))
print(np.round(np.array(m23).mean(),4))
print(np.round(np.array(m24).mean(),4))
print(np.round(np.array(m25).mean(),4))

print(np.round(np.median(np.array(m11)),4))
print(np.round(np.median(np.array(m12)),4))
print(np.round(np.median(np.array(m13)),4))
print(np.round(np.median(np.array(m14)),4))
print(np.round(np.median(np.array(m15)),4))


print(np.round(np.median(np.array(m21)),4))
print(np.round(np.median(np.array(m22)),4))
print(np.round(np.median(np.array(m23)),4))
print(np.round(np.median(np.array(m24)),4))
print(np.round(np.median(np.array(m25)),4))