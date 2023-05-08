import random
import time

import numpy as np
from tqdm.auto import tqdm

import torch

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.utils import make_grid, save_image
from torchvision import transforms

import matplotlib.pyplot as plt
from pathlib import Path

from utils.func_tool import save_logs


def fix_experiment_seed(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

fix_experiment_seed()

results_folder = Path("./results_vae")
results_folder.mkdir(exist_ok = True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Training Hyperparameters
train_batch_size = 64   # Batch Size
z_dim = 32        # Latent Dimensionality
lr = 1e-4         # Learning Rate
epochs = 100

# Define Dataset Statistics
image_size = 32
input_channels = 3
data_root = './data'


# Helper Functions
def show_image(image, nrow=8):
  # Input: image
  # Displays the image using matplotlib
  grid_img = make_grid(image.detach().cpu(), nrow=nrow, padding=0)
  plt.imshow(grid_img.permute(1, 2, 0))
  plt.axis('off')


def get_dataloaders(data_root, batch_size):
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    transform = transforms.Compose((
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize))

    train = datasets.SVHN(data_root, split='train', download=True, transform=transform)
    test = datasets.SVHN(data_root, split='test', download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_dataloader, test_dataloader

# Visualize the Dataset
def visualize():
  train_dataloader, _ = get_dataloaders(data_root=data_root, batch_size=train_batch_size)
  imgs, labels = next(iter(train_dataloader))

  save_image((imgs + 1.) * 0.5, './results_vae/orig.png')
  show_image((imgs + 1.) * 0.5)

# if __name__ == '__main__':
#   visualize()

def average_list(input_list):
    return sum(input_list) / len(input_list)

def save_bar(input, i):
    ff = []
    [ff.append(i) for i in range(32)]
    values = input
    fig = plt.figure(figsize=(10, 5))
    plt.bar(ff, values.detach().cpu(), color='blue', width=0.6)
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.title(f"Distribution {i}")
    save_path = Path('results_vae', f"No_{i}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    del fig


def bar_plot_class(dataloader, model):
    with tqdm(dataloader, unit="batch", leave=False) as tepoch:
        model.eval()
        for batch in tepoch:
            tepoch.set_description(f"Epoch: {epoch}")
            imgs, label = batch
            x = imgs.to(device)

            recon, nll, kl, latent_z = model(x)
            C0, C1, C2, C3, C4, C5, C6, C7, C8, C9 = [], [], [], [], [], [], [], [], [], []
            [C0.append(latent_z[idx]) for idx in torch.where(label==0)[0]]
            [C1.append(latent_z[idx]) for idx in torch.where(label==1)[0]]
            [C2.append(latent_z[idx]) for idx in torch.where(label==2)[0]]
            [C3.append(latent_z[idx]) for idx in torch.where(label==3)[0]]
            [C4.append(latent_z[idx]) for idx in torch.where(label==4)[0]]
            [C5.append(latent_z[idx]) for idx in torch.where(label==5)[0]]
            [C6.append(latent_z[idx]) for idx in torch.where(label==6)[0]]
            [C7.append(latent_z[idx]) for idx in torch.where(label==7)[0]]
            [C8.append(latent_z[idx]) for idx in torch.where(label==8)[0]]
            [C9.append(latent_z[idx]) for idx in torch.where(label==9)[0]]
            avg_0 = average_list(C0)
            avg_1 = average_list(C1)
            avg_2 = average_list(C2)
            avg_3 = average_list(C3)
            avg_4 = average_list(C4)
            avg_5 = average_list(C5)
            avg_6 = average_list(C6)
            avg_7 = average_list(C7)
            avg_8 = average_list(C8)
            avg_9 = average_list(C9)
            avg = [avg_0, avg_1, avg_2, avg_3, avg_4, avg_5, avg_6, avg_7, avg_8, avg_9]
            i = 0
            for c_avg in avg:
                save_bar(c_avg, i)
                i = i + 1

class Encoder(nn.Module):
  def __init__(self, nc, nef, nz, isize, device):
    super(Encoder, self).__init__()

    # Device
    self.device = device

    # Encoder: (nc, isize, isize) -> (nef*8, isize//16, isize//16)
    self.encoder = nn.Sequential(
      nn.Conv2d(nc, nef, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef),

      nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef * 2),

      nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef * 4),

      nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
      nn.LeakyReLU(0.2, True),
      nn.BatchNorm2d(nef * 8),
    )

  def forward(self, inputs):
    batch_size = inputs.size(0)
    hidden = self.encoder(inputs)
    hidden = hidden.view(batch_size, -1)
    return hidden

class Decoder(nn.Module):
  def __init__(self, nc, ndf, nz, isize):
    super(Decoder, self).__init__()

    # Map the latent vector to the feature map space
    self.ndf = ndf
    self.out_size = isize // 16
    self.decoder_dense = nn.Sequential(
      nn.Linear(nz, ndf * 8 * self.out_size * self.out_size),
      nn.ReLU(True)
    )

    self.decoder_conv = nn.Sequential(
      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(ndf * 8, ndf * 4, 3, 1, padding=1),
      nn.LeakyReLU(0.2, True),

      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(ndf * 4, ndf * 2, 3, 1, padding=1),
      nn.LeakyReLU(0.2, True),

      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(ndf * 2, ndf, 3, 1, padding=1),
      nn.LeakyReLU(0.2, True),

      nn.UpsamplingNearest2d(scale_factor=2),
      nn.Conv2d(ndf, nc, 3, 1, padding=1)
    )

  def forward(self, input):
    batch_size = input.size(0)
    hidden = self.decoder_dense(input).view(
      batch_size, self.ndf * 8, self.out_size, self.out_size)
    output = self.decoder_conv(hidden)
    return output

class DiagonalGaussianDistribution(object):
  # Gaussian Distribution with diagonal covariance matrix
  def __init__(self, mean, logvar=None):
    super(DiagonalGaussianDistribution, self).__init__()
    # Parameters:
    # mean: A tensor representing the mean of the distribution
    # logvar: Optional tensor representing the log of the standard variance
    #         for each of the dimensions of the distribution

    self.mean = mean
    if logvar is None:
        logvar = torch.zeros_like(self.mean)
    self.logvar = torch.clamp(logvar, -30., 20.)

    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)

  def sample(self):
    # Provide a reparameterized sample from the distribution
    # Return: Tensor of the same size as the mean
    sample = self.mean + self.std * torch.randn_like(self.mean)      # WRITE CODE HERE
    return sample

  def kl(self):
    # Compute the KL-Divergence between the distribution with the standard normal N(0, I)
    # Return: Tensor of size (batch size,) containing the KL-Divergence for each element in the batch
    kl_div = - 0.5 * (1 + self.logvar - self.mean.pow(2) - self.var).sum(dim=1)
    return kl_div


  def nll(self, sample, dims=[1, 2, 3]):
    # Computes the negative log likelihood of the sample under the given distribution
    # Return: Tensor of size (batch size,) containing the log-likelihood for each element in the batch
    negative_ll = (0.5 * (torch.log(2 * torch.pi * self.var) + ((sample - self.mean) * (sample - self.mean)) / self.var)).sum(dim=dims)    # WRITE CODE HERE
    return negative_ll

  def mode(self):
    # Returns the mode of the distribution
    mode = self.mean     # WRITE CODE HERE
    return mode

class VAE(nn.Module):
    def __init__(self, in_channels=3, decoder_features=32, encoder_features=32, z_dim=100, input_size=32,
                 device=torch.device("cuda:0")):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.in_channels = in_channels
        self.device = device

        # Encode the Input
        self.encoder = Encoder(nc=in_channels,
                               nef=encoder_features,
                               nz=z_dim,
                               isize=input_size,
                               device=device
                               )

        # Map the encoded feature map to the latent vector of mean, (log)variance
        out_size = input_size // 16
        self.mean = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim)
        self.logvar = nn.Linear(encoder_features * 8 * out_size * out_size, z_dim)

        # Decode the Latent Representation
        self.decoder = Decoder(nc=in_channels,
                               ndf=decoder_features,
                               nz=z_dim,
                               isize=input_size
                               )

    def encode(self, x):
        # Input:
        #   x: Tensor of shape (batch_size, 3, 32, 32)
        # Returns:
        #   posterior: The posterior distribution q_\phi(z | x)

        # WRITE CODE HERE
        hidden = self.encoder(x)
        mu = self.mean(hidden)
        logvar = self.logvar(hidden)

        return DiagonalGaussianDistribution(mu, logvar)

    def decode(self, z):
        # Input:
        #   z: Tensor of shape (batch_size, z_dim)
        # Returns
        #   conditional distribution: The likelihood distribution p_\theta(x | z)

        # WRITE CODE HERE
        x_r = self.decoder(z)
        mu = x_r
        return DiagonalGaussianDistribution(mu, logvar=torch.zeros_like(mu))

    def sample(self, batch_size):
        # Input:
        #   batch_size: The number of samples to generate
        # Returns:
        #   samples: Generated samples using the decoder
        #            Size: (batch_size, 3, 32, 32)

        # WRITE CODE HERE
        z = torch.randn(batch_size, self.z_dim).to(device)
        distribution_x = self.decode(z)
        samples = distribution_x.mode()
        return samples

    def log_likelihood(self, x, K=100):
        # Approximate the log-likelihood of the data using Importance Sampling
        # Inputs:
        #   x: Data sample tensor of shape (batch_size, 3, 32, 32)
        #   K: Number of samples to use to approximate p_\theta(x)
        # Returns:
        #   ll: Log likelihood of the sample x in the VAE model using K samples
        #       Size: (batch_size,)
        posterior = self.encode(x)
        prior = DiagonalGaussianDistribution(torch.zeros_like(posterior.mean))

        log_likelihood = torch.zeros(x.shape[0], K).to(self.device)
        for i in range(K):
            z = posterior.sample()  # WRITE CODE HERE (sample from q_phi)
            recon = self.decode(z)  # WRITE CODE HERE (decode to conditional distribution)

            p_z = -prior.nll(z, dims=[1])
            p_x_z = -recon.nll(x)
            p_theta_x_z = p_x_z + p_z
            q_phi = -posterior.nll(z, dims=[1])
            log_likelihood[:, i] = (p_theta_x_z - q_phi)# WRITE CODE HERE (log of the summation terms in approximate log-likelihood, that is, log p_\theta(x, z_i) - log q_\phi(z_i | x))

            del z, recon
        ll = torch.logsumexp(log_likelihood, dim=1) + torch.log(torch.tensor([1.0 / K])) # WRITE CODE HERE (compute the final log-likelihood using the log-sum-exp trick)
        return ll

    def forward(self, x):
        # Input: torch.log(recon.sample())
        #   x: Tensor of shape (batch_size, 3, 32, 32)
        # Returns:
        #   reconstruction: The mode of the distribution p_\theta(x | z) as a candidate reconstruction
        #                   Size: (batch_size, 3, 32, 32)
        #   Conditional Negative Log-Likelihood: The negative log-likelihood of the input x under the distribution p_\theta(x | z)
        #                                         Size: (batch_size,)
        #   KL: The KL Divergence between the variational approximate posterior with N(0, I)
        #       Size: (batch_size,)
        posterior = self.encode(x)  # WRITE CODE HERE
        latent_z = posterior.sample()  # WRITE CODE HERE (sample a z)
        recon = self.decode(latent_z)  # WRITE CODE HERE (decode)

        return recon.mode(), recon.nll(x), posterior.kl(), latent_z

if __name__ == '__main__':
  model = VAE(in_channels=input_channels,
            input_size=image_size,
            z_dim=z_dim,
            decoder_features=32,
            encoder_features=32,
            device=device
            )
  model.to(device)
  optimizer = Adam(model.parameters(), lr=lr)
#

# logger = dict()
# logger['train_time'] = [0]
if __name__ == '__main__':
    model = VAE(in_channels=input_channels,
                input_size=image_size,
                z_dim=z_dim,
                decoder_features=32,
                encoder_features=32,
                device=device
                )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    train_dataloader, test_dataloader = get_dataloaders(data_root, batch_size=train_batch_size)
    for epoch in range(epochs):
        # start_time = time.time()
        with tqdm(train_dataloader, unit="batch", leave=False) as tepoch:
            model.train()
            for batch in tepoch:
              tepoch.set_description(f"Epoch: {epoch}")

              optimizer.zero_grad()

              imgs, label = batch
              batch_size = imgs.shape[0]
              x = imgs.to(device)

              recon, nll, kl, latent_z = model(x)
              loss = (nll + kl).mean()

              loss.backward()
              optimizer.step()

              tepoch.set_postfix(loss=loss.item())
    train_batch_size = 128
    bar_plot_class(train_dataloader, model)





        # freq_time = time.time() - start_time
        # logger['train_time'].append(freq_time)

        # samples = model.sample(batch_size=64)
        # save_image((x + 1.) * 0.5, './results_vae/orig.png')
        # save_image((recon + 1.) * 0.5, './results_vae/recon.png')
        # save_image((samples + 1.) * 0.5, f'./results_vae/samples_{epoch}.png')

        # if epoch % 5 == 0:
        #     with torch.no_grad():
        #         with tqdm(test_dataloader, unit="batch", leave=True) as tepoch:
        #             model.eval()
        #             log_likelihood = 0.
        #             num_samples = 0.
        #             for batch in tepoch:
        #                 tepoch.set_description(f"Epoch: {epoch}")
        #                 imgs, _ = batch
        #                 batch_size = imgs.shape[0]
        #                 x = imgs.to(device)
        #                 recon, nll, kl = model(x)
        #
        #                 save_image((x + 1.) * 0.5, './results_vae/testorig.png')
        #                 save_image((recon + 1.) * 0.5, './results_vae/testrecon.png')
        #                 save_image((samples + 1.) * 0.5, f'./results_vae/testsamples_{epoch}.png')

        # show_image(((samples + 1.) * 0.5).clamp(0., 1.))
    # save_logs(logger, "results_vae/log_new", str(1))

# if __name__ == '__main__':
#   _, test_dataloader = get_dataloaders(data_root, batch_size=train_batch_size)
#   with torch.no_grad():
#       with tqdm(test_dataloader, unit="batch", leave=True) as tepoch:
#           model.eval()
#           log_likelihood = 0.
#           num_samples = 0.
#           for batch in tepoch:
#               tepoch.set_description(f"Epoch: {epoch}")
#               imgs,_ = batch
#               batch_size = imgs.shape[0]
#               x = imgs.to(device)
#               log_likelihood += model.log_likelihood(x).sum()
#               num_samples += batch_size
#               tepoch.set_postfix(log_likelihood=log_likelihood / num_samples)

def interpolate(model, z_1, z_2, n_samples):

    # Interpolate between z_1 and z_2 with n_samples number of points, with the first point being z_1 and last being z_2.
    # Inputs:
    #   z_1: The first point in the latent space
    #   z_2: The second point in the latent space
    #   n_samples: Number of points interpolated
    # Returns:
    #   sample: The mode of the distribution obtained by decoding each point in the latent space
    #           Should be of size (n_samples, 3, 32, 32)
    lengths = torch.linspace(0., 1., n_samples).unsqueeze(1).to(device)
    z = torch.stack([z_2 + (z_1 - z_2) * t for t in lengths])    # WRITE CODE HERE (interpolate z_1 to z_2 with n_samples points)
    return model.decode(z).mode()

if __name__ == '__main__':
    z_1 = torch.randn(1, z_dim).to(device)
    z_2 = torch.randn(1, z_dim).to(device)

    interp = interpolate(model, z_1, z_2, 10)
    # show_image((interp + 1.) * 0.5, nrow=10)
    save_image((interp + 1.) * 0.5, './results_vae/interpolate.png')

