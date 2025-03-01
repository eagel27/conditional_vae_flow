import os
from models.flow import (
    RegressiveFlow,
    test_log_prob,
    train_flow,
    visualize_generated_samples,
)
from models.vae import VAE
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

MODE = os.getenv("MODE", "flow")

if __name__ == "__main__":
    if MODE == "flow":
        latent_dim = 2
        cond_dim = 3

        model = RegressiveFlow(latent_dim=latent_dim, cond_dim=cond_dim)

        # Generate synthetic data for testing
        num_samples = 100
        x_data = torch.randn(num_samples, latent_dim)
        cond_data = torch.randn(num_samples, cond_dim)

        dataset = TensorDataset(x_data, cond_data)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        train_flow(model, data_loader, num_epochs=10, lr=0.001)

        cond_input = torch.randn(100, cond_dim)
        visualize_generated_samples(model, cond_input, num_samples=100)

        test_log_prob(model, x_data, cond_data)
    else:

        vae = VAE(
            input_dim=(128, 128, 1),
            cond_dim=(128, 128, 5),
            latent_dim=32,
            kl_weight=0.1,
        )
        optimizer = optim.Adam(vae.parameters(), lr=1e-3)

        x = torch.randn(16, 1, 128, 128)
        cond = torch.randn(16, 5, 128, 128)

        reconstructed, z_mean, z_log_var = vae(x, cond)
        print(reconstructed.shape)
