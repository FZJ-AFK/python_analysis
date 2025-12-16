import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from metrics import evaluate

def train_gan(G_class, D_class, X, y, drug, cell, gene_dim, device, name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics_all, d_metrics_all = [], []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        d_tr, d_te = drug[train_idx], drug[test_idx]
        c_tr, c_te = cell[train_idx], cell[test_idx]

        G = G_class(gene_dim).to(device)
        D = D_class(gene_dim).to(device)

        opt_G = optim.Adam(G.parameters(), lr=1e-3)
        opt_D = optim.Adam(D.parameters(), lr=1e-3)

        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        for _ in range(20):
            for i in range(0, len(X_tr), 32):
                xb, yb = X_tr[i:i+32], y_tr[i:i+32]
                db, cb = d_tr[i:i+32], c_tr[i:i+32]

                # D
                opt_D.zero_grad()
                real = D(yb, db, cb)
                fake = D(G(xb, db, cb).detach(), db, cb)
                d_loss = bce_loss(real, torch.ones_like(real)) + \
                         bce_loss(fake, torch.zeros_like(fake))
                d_loss.backward()
                opt_D.step()

                # G
                opt_G.zero_grad()
                gen = G(xb, db, cb)
                adv = D(gen, db, cb)
                g_loss = mse_loss(gen, yb) + 0.1 * bce_loss(adv, torch.ones_like(adv))
                g_loss.backward()
                opt_G.step()

        with torch.no_grad():
            y_pred = G(X_te, d_te, c_te).cpu().numpy()
            y_true = y_te.cpu().numpy()
            metrics_all.append(evaluate(y_true, y_pred))

            real = D(y_te, d_te, c_te).cpu().numpy().ravel()
            fake = D(torch.tensor(y_pred).to(device), d_te, c_te).cpu().numpy().ravel()
            labels = np.concatenate([np.ones_like(real), np.zeros_like(fake)])
            scores = np.concatenate([real, fake])
            d_metrics_all.append([
                accuracy_score(labels, scores > 0),
                roc_auc_score(labels, scores)
            ])

    return name, np.mean(metrics_all, 0), np.std(metrics_all, 0), np.mean(d_metrics_all, 0)
