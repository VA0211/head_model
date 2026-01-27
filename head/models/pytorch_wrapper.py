import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sklearn.base

# --- EXACT REPLICAS OF THE PAPER'S MODELS ---
class MLP3(nn.Module):
    def __init__(self, n_in, n_units=512, n_out=12):
        super(MLP3, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_in, n_units),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_units, n_units),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_units, n_out)
        )

    def forward(self, x):
        return self.seq(x)

class MLP4(nn.Module):
    def __init__(self, n_in, n_units=512, n_out=12):
        super(MLP4, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_in, n_units),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_units, n_units),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_units, n_units),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(n_units, n_out)
        )

    def forward(self, x):
        return self.seq(x)

# --- SKLEARN WRAPPER (To make it compatible with the pipeline) ---
class PyTorchClassifier(sklearn.base.BaseEstimator):
    def __init__(self, model_class, gpu=-1, n_epoch=250, n_out=12):
        self.model_class = model_class
        self.gpu = gpu
        self.n_epoch = n_epoch
        self.n_out = n_out
        self.model = None
        self.device = torch.device('cuda' if gpu >= 0 and torch.cuda.is_available() else 'cpu')

    def fit_and_validate(self, train_x, train_y, validate_x=None, validate_y=None):
        # Convert to Tensor
        X_t = torch.tensor(train_x, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(train_y, dtype=torch.long).to(self.device)
        
        # Initialize Model
        self.model = self.model_class(len(train_x[0]), n_out=self.n_out).to(self.device)
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=100, shuffle=True)

        print(f"Training {self.model_class.__name__} on {self.device} for {self.n_epoch} epochs...")
        
        self.model.train()
        for epoch in range(self.n_epoch):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.n_epoch} complete.")

    def fit(self, x, y):
        self.fit_and_validate(x, y)

    def predict_proba(self, x):
        self.model.eval()
        # Process in batches to avoid OOM
        X_t = torch.tensor(x, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=1000, shuffle=False)
        
        probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                out = self.model(batch_x)
                prob = torch.softmax(out, dim=1)
                probs.append(prob.cpu().numpy())
                
        return np.concatenate(probs)

    def predict(self, x):
        prob = self.predict_proba(x)
        return np.argmax(prob, axis=1)