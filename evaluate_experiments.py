import torch
from representation_models import compute_pca, compute_umap, Autoencoder, VAE, SimpleClassifier, MLPClassifier
from train_eval import train_classifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import time
import random
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(12)
results = {}

latent_dim = 48
epochs_representation = 4
epochs_classifier = 45

DATASET = 'MNIST'

if DATASET == 'CIFAR10':
    from data_loader import get_cifar10 as get_dataset
    input_size = 32 * 32 * 3
elif DATASET == 'MNIST':
    from data_loader import get_mnist as get_dataset
    input_size = 28 * 28


# Load CIFAR-10 once
train_loader, test_loader = get_dataset(batch_size=256)

# Extract features into numpy arrays
def loader_to_numpy(loader):
    data_list, label_list = [], []
    for data, label in loader:
        data_list.append(data.numpy())
        label_list.append(label.numpy())
    return np.vstack(data_list).reshape(len(loader.dataset), -1), np.hstack(label_list)

train_data, train_labels = loader_to_numpy(train_loader)
test_data, test_labels = loader_to_numpy(test_loader)

methods = {
    # "Baseline (Raw)": None,
    # "PCA": compute_pca,
    "UMAP": compute_umap,
    # "Autoencoder": Autoencoder,
    # "VAE": VAE
}



for method_name, method in methods.items():
    print(f"\n{'-'*20} Evaluating: {method_name} {'-'*20}")

    start_time = time.time()

    if method_name == "Baseline (Raw)":
        # Raw data directly
        X_train, X_test = train_data, test_data

    elif method_name in ["PCA", "UMAP"]:
        print(f"Computing {method_name} representation...")
        X_train, X_test = method(train_data, test_data, n_components=latent_dim)

    elif method_name in ["Autoencoder", "VAE"]:
        print(f"Training {method_name}...")
        model_class = method
        model = model_class(input_dim=input_size, latent_dim=latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(torch.tensor(train_data).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        model.train()
        for epoch in tqdm(range(epochs_representation)):
            epoch_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                inputs = batch[0].to(device)
                if method_name == "Autoencoder":
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, inputs)
                else:
                    outputs, mu, logvar = model(inputs)
                    recon_loss = criterion(outputs, inputs)
                    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_div
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        # Get representations
        model.eval()
        with torch.no_grad():
            if method_name == "Autoencoder":
                X_train = model.encoder(torch.tensor(train_data).float().to(device)).cpu().numpy()
                X_test = model.encoder(torch.tensor(test_data).float().to(device)).cpu().numpy()
            elif method_name == "VAE":
                mu_train, _ = model.encode(torch.tensor(train_data).float().to(device))
                mu_test, _ = model.encode(torch.tensor(test_data).float().to(device))
                X_train = mu_train.cpu().numpy()
                X_test = mu_test.cpu().numpy()

    rep_time = time.time()
    print(f"Representation learning done in {rep_time - start_time:.2f}s")

    # Create classifier DataLoader
    clf_train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train).float(), torch.tensor(train_labels))
    clf_test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test).float(), torch.tensor(test_labels))

    clf_train_loader = torch.utils.data.DataLoader(clf_train_dataset, batch_size=256, shuffle=True)
    clf_test_loader = torch.utils.data.DataLoader(clf_test_dataset, batch_size=256, shuffle=False)

    # Train classifier
    input_dim = X_train.shape[1]
    clf_model = MLPClassifier(input_dim=input_dim).to(device)


    print("Training classifier...")
    clf_start_time = time.time()
    acc, f1 = train_classifier(clf_model, clf_train_loader, clf_test_loader,
                               epochs=epochs_classifier, device=device)
    clf_time = time.time()
    total_time = clf_time - start_time

    results[method_name] = {
        'Accuracy': acc,
        'F1 Score': f1,
        'Representation Time (s)': rep_time - start_time,
        'Classifier Training Time (s)': clf_time - clf_start_time,
        'Total Time (s)': total_time
    }

    print(f"{method_name} | Accuracy: {acc:.4f}, F1: {f1:.4f}, Total Time: {total_time:.2f}s")

print("\n\nFinal Results for %s, latent dim: %d, epochs_representation: %d, epochs_classifier: %d" % (DATASET, latent_dim, epochs_representation, epochs_classifier))
print(f"{'Method':<15} | {'Accuracy':<8} | {'F1 Score':<8} | {'Rep Time (s)':<12} | {'Clf Time (s)':<12} | {'Total (s)':<10}")
print('-'*80)
for name, metrics in results.items():
    print(f"{name:<15} | {metrics['Accuracy']:<8.4f} | {metrics['F1 Score']:<8.4f} | "
          f"{metrics['Representation Time (s)']:<12.2f} | {metrics['Classifier Training Time (s)']:<12.2f} | "
          f"{metrics['Total Time (s)']:<10.2f}")
