import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class MADE(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_units=[512, 512]):
        super(MADE, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.fc1 = nn.Linear(self.cond_dim, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(
            hidden_units[1], 2 * self.latent_dim
        )  # For both shift and log scale

    def forward(self, cond_input):
        x = torch.relu(self.fc1(cond_input))
        x = torch.relu(self.fc2(x))
        out = self.fc3(x)

        shift, log_scale = torch.chunk(out, 2, dim=-1)
        return shift, log_scale


class RegressiveFlow(nn.Module):
    def __init__(self, latent_dim, cond_dim, encode_condition=False):
        super(RegressiveFlow, self).__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.encode_condition = encode_condition
        self.cond_encoder = None

        if self.encode_condition:
            self.cond_encoder = self.make_cond_encoder()

        self.made = MADE(latent_dim, cond_dim)
        self.permute = torch.randperm(latent_dim)
        self.transform = nn.ModuleList([self.made, self.made, self.made])

    def make_cond_encoder(self):
        return nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 16 * 64, 10)  # Example encoder
        )

    def forward(self, x, cond):
        if self.encode_condition:
            cond = self.cond_encoder(cond)

        for bijector in self.transform:
            shift, log_scale = bijector(cond)
            x = self.apply_bijector(x, shift, log_scale)

        return x

    def apply_bijector(self, x, shift, log_scale):
        batch_size = x.size(0)

        shift = shift.expand(batch_size, -1)
        log_scale = log_scale.expand(batch_size, -1)

        return x * torch.exp(log_scale) + shift

    def log_prob(self, x, cond):
        shift, log_scale = self.made(cond)
        log_2pi = torch.log(torch.tensor(2 * torch.pi, dtype=torch.float32))
        log_prob = -0.5 * (
            log_2pi + log_scale + (x - shift) ** 2 / torch.exp(log_scale)
        )
        return log_prob.sum(dim=-1)

    def sample_flow(self, num_samples, cond):
        samples = torch.randn(num_samples, self.latent_dim)
        for bijector in self.transform:
            shift, log_scale = bijector(cond)
            samples = self.apply_bijector(samples, shift, log_scale)
        return samples


def train_flow(model, data_loader, num_epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for x, cond in data_loader:
            optimizer.zero_grad()
            log_prob = model.log_prob(x, cond)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")


def visualize_generated_samples(model, cond_input, num_samples=100):
    generated_samples = model.sample_flow(num_samples, cond_input)
    generated_samples = generated_samples.detach().numpy()

    plt.scatter(generated_samples[:, 0], generated_samples[:, 1])
    plt.title("Generated Latent Space Samples")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.show()


def test_log_prob(model, x_data, cond_data):
    sample_idx = 0
    x_sample, cond_sample = x_data[sample_idx], cond_data[sample_idx]
    log_prob = model.log_prob(x_sample.unsqueeze(0), cond_sample.unsqueeze(0))
    print("Log-probability of sample:", log_prob.item())
