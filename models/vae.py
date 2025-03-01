import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, z_mean, z_log_var):
        batch = z_mean.size(0)
        dim = z_mean.size(1)
        epsilon = torch.randn(batch, dim)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class ConditionEncoder(nn.Module):
    """Encode condition images"""

    def __init__(self, cond_dim=(128, 128, 5), latent_dim=32):
        super(ConditionEncoder, self).__init__()
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Here we calculate the size of the input to the fully connected layer
        # The image is reduced to (64, 32, 32) after conv1 and conv2 with stride=2
        self.fc = nn.Linear(65536, self.latent_dim)

    def forward(self, cond_input):
        x = F.relu(self.conv1(cond_input))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class Encoder(nn.Module):
    """Maps images to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, input_dim=(128, 128, 1), cond_z_dim=10, latent_dim=32):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cond_z_dim = cond_z_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        self.fc = nn.Linear(128 * 16 * 16, self.cond_z_dim)

        self.fc_z_mean = nn.Linear(
            2 * self.cond_z_dim, self.latent_dim
        )  # Output for z_mean
        self.fc_z_log_var = nn.Linear(
            2 * self.cond_z_dim, self.latent_dim
        )  # Output for z_log_var

    def forward(self, x, cond_z):
        # Apply convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten the output
        x = x.view(x.size(0), -1)  # Flatten

        # Apply fully connected layer to generate a latent feature space
        x = F.relu(self.fc(x))

        z = torch.cat([x, cond_z], dim=-1)
        z_mean = self.fc_z_mean(z)
        z_log_var = self.fc_z_log_var(z)

        return z_mean, z_log_var


class Decoder(nn.Module):

    def __init__(self, latent_dim=32, cond_dim=32):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.fc1 = nn.Linear(latent_dim + cond_dim, 16 * 16 * 128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z, cond_z):
        x = torch.cat([z, cond_z], dim=-1)
        x = F.relu(self.fc1(x))

        x = x.view(x.size(0), 128, 16, 16)  # Reshape to the channel depth

        x = F.relu(self.deconv1(x))

        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        return x


class VAE(nn.Module):
    def __init__(
        self,
        input_dim=(128, 128, 1),
        cond_dim=(128, 128, 5),
        latent_dim=32,
        cond_latent_dim=10,
        kl_weight=0.1,
    ):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        self.encoder = Encoder(
            input_dim=input_dim, cond_z_dim=cond_latent_dim, latent_dim=latent_dim
        )
        self.cond_encoder = ConditionEncoder(cond_dim=cond_dim, latent_dim=10)
        self.decoder = Decoder(latent_dim=latent_dim, cond_dim=cond_latent_dim)

    def forward(self, x, cond):
        cond_f = self.cond_encoder(cond)
        z_mean, z_log_var = self.encoder(x, cond_f)
        z = Sampling()(z_mean, z_log_var)
        output = self.decoder(z, cond_f)
        return output, z_mean, z_log_var

    def kl_loss(self, z_mean, z_log_var):

        kl_divergence = -0.5 * torch.sum(
            1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1
        )
        kl_loss = torch.mean(kl_divergence)  # Take the mean over the batch
        return kl_loss * self.kl_weight  # Apply the weight to the KL loss

    def compute_loss(self, x, x_reconstructed, z_mean, z_log_var):
        # Reconstruction loss (BCE or MSE)
        recon_loss = F.mse_loss(x_reconstructed, x, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        return recon_loss + self.kl_weight * kl_loss

    def sample_random(self, cond, num):
        z = torch.randn(num, self.latent_dim)
        cond, cond_f = self.cond_encoder(cond)
        output = self.decoder(z, cond_f)
        return output

    def encode_condition(self, cond):
        _, cond_f = self.cond_encoder(cond)
        return cond_f

    def encode(self, x, cond):
        cond, cond_f = self.cond_encoder(cond)
        z_mean, z_log_var = self.encoder(x, cond_f)
        return z_mean, z_log_var
