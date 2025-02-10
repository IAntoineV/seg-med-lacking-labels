import torch
import torch.nn as nn

# Variational Autoencoder
class VariationalAutoEncoder(nn.Module):
    def __init__(self, last_encoder_dim, latent_dim, encoder_model, decoder_model):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = encoder_model
        self.fc_mu = nn.Linear(last_encoder_dim, latent_dim)
        self.fc_logvar = nn.Linear(last_encoder_dim, latent_dim)
        self.decoder = decoder_model


    def forward(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        if self.use_decoder=="decoder_stats":
            adj = self.decoder(x_g, data.stats)
        else:
            adj = self.decoder(x_g)
        return adj

    def encode(self, data):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        return x_g

    def reparameterize(self, mu, logvar, eps_scale=1.):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, mu, logvar):
        x_g = self.reparameterize(mu, logvar)
        adj = self.decoder(x_g)
        return adj

    def decode_mu(self, mu):
        adj = self.decoder(mu)
        return adj

    def loss_function(self, data, beta=0.05):
        x_g = self.encoder(data)
        mu = self.fc_mu(x_g)
        logvar = self.fc_logvar(x_g)
        x_g = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(x_g)
        recon_loss = (reconstructed - data).pow(2).mean()
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kld
        return loss, recon_loss, kld
