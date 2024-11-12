import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getDataLoader(name, batch_size):
    if name == 'mnist':
        # Load MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    elif name == 'cifar':
        # Load CIFAR-10 dataset
        transform = transforms.Compose([
            transforms.ToTensor()
            ])
        
        cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(cifar10, batch_size=batch_size, shuffle=True)
        
        return dataloader

# Training loop
def trainGan(num_epochs, dataloader, generator, discriminator, latent_dim, criterion, g_optimizer, d_optimizer, device, folder_name):
    # if statements due to shape changes b/w mnist and cifar
    G_losses = []
    D_losses = []
    
    # send to device
    generator.to(device)
    discriminator.to(device)
    
    for epoch in tqdm(range(num_epochs)):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            if folder_name == 'Gan_MNIST':
                real_images = real_images.reshape(batch_size, -1).to(device)
            elif folder_name == 'Gan_CIFAR':
                real_images = real_images.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            if folder_name == 'Gan_MNIST':
                real_labels = torch.ones(batch_size, 1).to(device) # let discriminator know all are real
            elif folder_name == 'Gan_CIFAR':
                real_labels = torch.ones(batch_size).to(device) # let discriminator know all are real
            d_output_real = discriminator(real_images)
            d_loss_real = criterion(d_output_real, real_labels)
            
            # Fake images
            if folder_name == 'Gan_MNIST':
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)
            elif folder_name == 'Gan_CIFAR':
                z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
                fake_labels = torch.zeros(batch_size).to(device)
            fake_images = generator(z)
            d_output_fake = discriminator(fake_images.detach()) # let discriminator know all are fake
            d_loss_fake = criterion(d_output_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            
            if folder_name == 'Gan_MNIST':
                z = torch.randn(batch_size, latent_dim).to(device)
            elif folder_name == 'Gan_CIFAR':
                z = torch.randn(batch_size, latent_dim, 1, 1).to(device)
            fake_images = generator(z)
            g_output = discriminator(fake_images)
            g_loss = criterion(g_output, real_labels)
            
            g_loss.backward()
            g_optimizer.step()

        # Save losses for plotting
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        # Print progress and visualize generated images
        if (epoch + 1) % 10 == 0:
            visualize_generator_output(epoch + 1, generator, latent_dim, device, folder_name)
        
    return G_losses, D_losses

def visualize_generator_output(epoch, generator, latent_dim, device, folder_name):
    # if statements due to shape changes b/w mnist and cifar
    
    # send to device
    generator.to(device)
    
    generator.eval()
    
    with torch.no_grad():
        if folder_name == 'Gan_MNIST':
            z = torch.randn(64, latent_dim).to(device)
        elif folder_name == 'Gan_CIFAR':
            z = torch.randn(64, latent_dim, 1, 1).to(device)
        fake_images = generator(z)
        if folder_name == 'Gan_MNIST':
            fake_images = fake_images.reshape(-1, 1, 28, 28)
        grid = make_grid(fake_images, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title(f'Generated Images at Epoch {epoch}')
        plt.axis('off')
        plt.savefig(f'{folder_name}/generated_images_epoch_{epoch}.png')
        plt.close()
    generator.train()

def latent_space_interpolation_gan(generator, latent_dim, device, folder_name, num_steps=10):
    # if statements due to shape changes b/w mnist and cifar
    
    # send to device
    generator.to(device)
    
    generator.eval()
    with torch.no_grad():
        # Generate two random latent vectors
        if folder_name == 'Gan_MNIST':
            z1 = torch.randn(1, latent_dim).to(device)
            z2 = torch.randn(1, latent_dim).to(device)
        elif folder_name == 'Gan_CIFAR':
            z1 = torch.randn(1, latent_dim, 1, 1).to(device)
            z2 = torch.randn(1, latent_dim, 1, 1).to(device)
        
        # Interpolate between z1 and z2
        alphas = np.linspace(0, 1, num_steps)
        interpolated_images = []
        
        for alpha in alphas:
            z = alpha * z1 + (1 - alpha) * z2
            fake_image = generator(z)
            if folder_name == 'Gan_MNIST':
                interpolated_images.append(fake_image.cpu().view(28, 28))
            elif folder_name == 'Gan_CIFAR':
                interpolated_images.append(fake_image.cpu().squeeze(0))
        
        # Visualize the interpolation
        fig, axes = plt.subplots(1, num_steps, figsize=(20, 2))
        for i, img in enumerate(interpolated_images):
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        
        plt.suptitle('Latent Space Interpolation')
        plt.tight_layout()
        plt.savefig(f'{folder_name}/latent_space_interpolation.png')
        plt.close()

# loss for vae
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def trainVAE(num_epochs, dataloader, model, optimizer, device):
    # send to device
    model.to(device)
    
    model.train()
    elbo_losses = []
    kl_divergences = []

    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        kld_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            kld_loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())      
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(dataloader.dataset)
        avg_kld = kld_loss.item() / len(dataloader.dataset)
        elbo_losses.append(avg_loss)
        kl_divergences.append(avg_kld)
    
    return elbo_losses, kl_divergences