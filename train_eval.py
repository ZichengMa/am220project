import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import torch

def train_classifier(model, train_loader, test_loader, epochs=20, lr=0.001, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in tqdm(range(epochs)):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        # old_loss = loss.item()
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {old_loss:.4f}")
        # if loss changes < 1%, stop training
        # if epoch > 0 and abs(old_loss - loss.item()) < 0.001 * old_loss:
        #     print("Early stopping")
        #     break

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            preds = model(x)
            y_true.extend(y.numpy())
            y_pred.extend(preds.cpu().argmax(dim=1).numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1
