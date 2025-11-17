# Toxicity Prediction (GAT)

Graph Attention Network (GAT)-based molecular toxicity prediction.
Converts SMILES strings to molecular graphs using RDKit, extracts atom-level features,
and trains a GAT model using PyTorch Geometric. Trained on ClinTox and evaluated on Tox21.

## Repo structure

toxicity-prediction/
├─ notebooks/
│  └─ toxicity_model.ipynb
├─ src/
│  ├─ model.py
│  ├─ data_processing.py
│  ├─ inference.py
│  └─ download_weights.py
├─ models/
│  └─ README.md
├─ assets/
│  └─ placeholder.txt
├─ requirements.txt
├─ .gitignore
├─ LICENSE
└─ README.md
