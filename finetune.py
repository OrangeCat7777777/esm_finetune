import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import esm

import argparse
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=1022)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ProteinSequenceDataset(Dataset):
    def __init__(self, df, max_len=1022):
        self.seqs = df.sequence
        self.labels = df.label
        self.max_len = max_len

    def __getitem__(self, idx):
        seq = self.seqs.iloc[idx]
        seq = seq[:self.max_len]
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)
        return seq, label

    def __len__(self):
        return len(self.labels)

class MLP(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out

def tokenize(X, y, alphabet):
    '''
    For ESM tokenization.
    '''
    batch_converter = alphabet.get_batch_converter()
    bs = len(y)
    X = list(zip([""] * bs, X))
    _, _, X = batch_converter(X)
    X = X[:, 1:X.shape[1]-1]
    batch_lens = (X != alphabet.padding_idx).sum(1)
    y = y.reshape(-1, 1)
    return X, y, batch_lens

def train(alphabet, d_emb, pretrained_model, downstream_model, train_loader, pretrained_optimizer, downstream_optimizer, epoch, device):
    pretrained_model.train()
    downstream_model.train()

    train_loss = 0

    for X, y in tqdm(train_loader):
        bs = len(y)
        X, y, batch_lens = tokenize(X, y, alphabet)
        X, y = X.to(device), y.to(device)
        embs_aa = pretrained_model.forward(X, repr_layers=[33])["representations"][33]
        embs_mean = torch.zeros([bs, d_emb]).to(device)
        for i, tokens_len in enumerate(batch_lens):
            embs_mean[i] = embs_aa[i, :tokens_len].mean(0)

        y_pred = downstream_model(embs_mean)
        loss = F.mse_loss(y_pred, y)
        
        # torch.nn.utils.clip_grad_norm_(
        #     parameters=model.parameters(), max_norm=0.1
        # )

        pretrained_optimizer.zero_grad()
        downstream_optimizer.zero_grad()
        loss.backward()
        pretrained_optimizer.step()
        downstream_optimizer.step()

        train_loss += loss.item()
    
    n_samples = len(train_loader)
    train_loss = train_loss / n_samples
    print(f"Training loss epoch {epoch}: {train_loss}")
    print()

def eval(alphabet, d_emb, pretrained_model, downstream_model, test_loader, epoch, device):
    pretrained_model.eval()
    downstream_model.eval()
    
    eval_loss = 0

    preds, labels = [], []

    with torch.no_grad():
        for X, y in test_loader:
            bs = len(y)
            X, y, batch_lens = tokenize(X, y, alphabet)
            X, y = X.to(device), y.to(device)
            embs_aa = pretrained_model.forward(X, repr_layers=[33])["representations"][33]
            embs_mean = torch.zeros([bs, d_emb]).to(device)
            for i, tokens_len in enumerate(batch_lens):
                embs_mean[i] = embs_aa[i, :tokens_len].mean(0)

            y_pred = downstream_model(embs_mean)
            loss = F.mse_loss(y_pred, y)
            eval_loss += loss.item()

            preds.extend(y_pred.cpu().detach()) 
            labels.extend(y.cpu().detach())
        
        preds = [pred.item() for pred in preds]
        labels = [label.item() for label in labels]
        preds = np.array(preds)
        labels = np.array(labels)

        n_samples = len(test_loader)
        eval_loss = eval_loss / n_samples

        print(f"Epoch {epoch}")
        print(f"ValidationLoss: {eval_loss}")
        print(f"MAE {np.mean(np.abs(labels - preds))}")
        print(f"Pearson Correlation {pearsonr(labels, preds)}")  
        print(f"Spearman Correlation {spearmanr(labels, preds)}")     
        print()

        sr = spearmanr(labels, preds)[0]
        pr = pearsonr(labels, preds)[0]
        return sr, pr

def main(args):
    set_random_seed(args.seed)

    device = "cuda"
    task = "gb1"

    train_df = pd.read_csv(f"{task}/splits/train.csv")
    train_ds = ProteinSequenceDataset(train_df)

    valid_df = pd.read_csv(f"{task}/splits/validate.csv")
    valid_ds = ProteinSequenceDataset(valid_df)

    test_df = pd.read_csv(f"{task}/splits/test.csv")
    test_ds = ProteinSequenceDataset(test_df)

    pretrained_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    pretrained_model.to(device)
    d_emb = 1280

    downstream_model = MLP(d_emb)
    downstream_model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.bs)
    valid_loader = DataLoader(valid_ds, batch_size=args.bs)
    test_loader = DataLoader(test_ds, batch_size=args.bs)

    pretrained_optimizer = torch.optim.Adam(params=pretrained_model.parameters(), lr=args.lr/100)
    downstream_optimizer = torch.optim.Adam(params=downstream_model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=N_EPOCHS)

    best_sr = 0
    best_epoch = 0

    for epoch in range(1, args.n_epochs + 1):
        train(alphabet, d_emb, pretrained_model, downstream_model, train_loader, pretrained_optimizer, downstream_optimizer, epoch, device)
        sr, _ = eval(alphabet, d_emb, pretrained_model, downstream_model, valid_loader, epoch, device)
        if sr > best_sr:
            best_epoch = epoch
            torch.save(pretrained_model, "pretrained_model.pt")
            torch.save(downstream_model, "downstream_model.pt")
    
    pretrained_model = torch.load("pretrained_model.pt")
    downstream_model = torch.load("downstream_model.pt") 
    print("Test result for the best model:")
    print("-" * 64)
    eval(alphabet, d_emb, pretrained_model, downstream_model, test_loader, best_epoch, device)    

if __name__ == "__main__":
    args = get_args()
    main(args)
