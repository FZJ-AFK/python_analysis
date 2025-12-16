import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(input_path, output_path, device):
    input_df = pd.read_csv(input_path, header=None)
    output_df = pd.read_csv(output_path, header=None)

    sample_names = input_df.iloc[0, 1:].values
    cell_names   = input_df.iloc[1, 1:].values
    drug_names   = input_df.iloc[2, 1:].values
    gene_names   = input_df.iloc[3:, 0].values

    gene_expr_input  = input_df.iloc[3:, 1:].astype(float).values.T
    gene_expr_output = output_df.iloc[3:, 1:].astype(float).values.T

    le_drug = LabelEncoder()
    le_cell = LabelEncoder()
    drug_ids = le_drug.fit_transform(drug_names)
    cell_ids = le_cell.fit_transform(cell_names)

    scaler_input  = StandardScaler()
    scaler_output = StandardScaler()
    gene_expr_input  = scaler_input.fit_transform(gene_expr_input)
    gene_expr_output = scaler_output.fit_transform(gene_expr_output)

    X = torch.tensor(gene_expr_input, dtype=torch.float32).to(device)
    y = torch.tensor(gene_expr_output, dtype=torch.float32).to(device)
    drug = torch.tensor(drug_ids, dtype=torch.float32).unsqueeze(1).to(device)
    cell = torch.tensor(cell_ids, dtype=torch.float32).unsqueeze(1).to(device)

    return X, y, drug, cell, X.shape[1]

