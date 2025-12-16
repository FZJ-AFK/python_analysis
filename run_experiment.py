import torch
import csv
from data_utils import load_data
from models import ResidualGenerator, MLPDiscriminator
from train import train_gan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, y, drug, cell, gene_dim = load_data(
    "E:/data/geo_data/p-lung.csv",
    "E:/data/geo_data/p-drug.csv",
    device
)

name = "Res+MLP"
result = train_gan(ResidualGenerator, MLPDiscriminator,
                   X, y, drug, cell, gene_dim, device, name)

names, means, stds, d_means = result

with open("res_mlp_result.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model"] + list(means) + list(stds) + list(d_means))
    writer.writerow([name] + list(means) + list(stds) + list(d_means))
