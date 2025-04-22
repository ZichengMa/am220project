from sklearn.decomposition import PCA
import numpy as np
def compute_pca(train_data, test_data, n_components=100):
    pca = PCA(n_components=n_components)
    pca.fit(train_data)
    return pca.transform(train_data), pca.transform(test_data)

import umap

def compute_umap(train_data, test_data, n_components=100):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reducer.fit(train_data)
    return reducer.transform(train_data), reducer.transform(test_data)

import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128), nn.ReLU(),
            # nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=100):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# VAE
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=100):
        super().__init__()
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU()
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_fc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

from sklearn.manifold import TSNE

def compute_tsne(train_data, test_data, n_components=100):
    tsne = TSNE(n_components=n_components, random_state=42, verbose=1)
    X_total = np.vstack([train_data, test_data])
    X_embedded = tsne.fit_transform(X_total)
    X_train_embedded = X_embedded[:len(train_data)]
    X_test_embedded = X_embedded[len(train_data):]
    return X_train_embedded, X_test_embedded

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 128, 64], num_classes=10):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3)) 
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)