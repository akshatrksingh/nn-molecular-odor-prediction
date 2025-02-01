import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import StepLR
from model import OdorNN

class MolecularProcessor:
    def __init__(self, radius=2, n_bits=1024):
        self.radius = radius
        self.n_bits = n_bits

    def smiles_to_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
            arr = np.zeros((self.n_bits,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        return np.zeros((self.n_bits,))

class OdorProcessor:
    def __init__(self, df):
        self.df = df
        self.odor_labels = [col for col in df.columns if col not in ['nonStereoSMILES', 'descriptors', 'fingerprint']]

    def get_odor_matrix(self):
        return self.df[self.odor_labels].astype(int).values

df = pd.read_csv("data/Multi-Labelled_Smiles_Odors_dataset.csv", low_memory=False)

processor = MolecularProcessor()
fingerprint_matrix = np.array(df["nonStereoSMILES"].apply(processor.smiles_to_fingerprint).tolist())

odor_processor = OdorProcessor(df)
odor_matrix = odor_processor.get_odor_matrix()

X_train, X_test, y_train, y_test = train_test_split(fingerprint_matrix, odor_matrix, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor, X_test_tensor = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor, y_test_tensor = torch.tensor(y_train, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
model = OdorNN(input_dim, output_dim).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=5, gamma=0.7)

epochs = 30
batch_size = 64
patience, patience_counter, best_val_loss = 10, 0, float("inf")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for i in range(0, X_train_tensor.shape[0], batch_size):
        X_batch, y_batch = X_train_tensor[i:i+batch_size], y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        y_val_pred_probs = model(X_test_tensor)
        val_loss = criterion(y_val_pred_probs, y_test_tensor).item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss, patience_counter = val_loss, 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

torch.save(model.state_dict(), "odor_model.pth")
print("Model saved as `odor_model.pth`")

model.eval()
with torch.no_grad():
    y_test_pred_probs = model(X_test_tensor)
y_test_pred = (y_test_pred_probs.cpu().numpy() > 0.5).astype(int)

print(f"\nTest Accuracy: {(y_test_pred == y_test).mean():.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))