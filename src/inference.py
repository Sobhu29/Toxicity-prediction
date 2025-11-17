import argparse
import torch
from torch_geometric.data import DataLoader
from src.data_processing import smiles_to_graph
from src.model import GATNet
import os

def predict(smiles, model_path="models/best_model.pt", device="cpu"):
    data = smiles_to_graph(smiles, y_label=None)
    if data is None:
        return {"error": "invalid_smiles", "smiles": smiles}
    # Add a batch dimension
    loader = DataLoader([data], batch_size=1, shuffle=False)
    # load model
    map_location = torch.device(device)
    if not os.path.exists(model_path):
        return {"error": "model_not_found", "message": f"Expected model at {model_path}"}
    # Infer number of node features from data
    num_node_features = data.num_node_features
    model = GATNet(num_node_features=num_node_features)
    state = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state)
    model.to(map_location)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(map_location)
            logits = model(batch)
            prob = torch.sigmoid(logits).cpu().item()
            label = "Toxic" if prob > 0.5 else "Non-Toxic"
            return {"smiles": smiles, "label": label, "score": prob}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict molecular toxicity from a SMILES string.")
    parser.add_argument("--smiles", required=True, help="SMILES string to predict")
    parser.add_argument("--model", default="models/best_model.pt", help="Path to model weights")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()
    out = predict(args.smiles, model_path=args.model, device=args.device)
    if "error" in out:
        print("ERROR:", out)
    else:
        print(f"SMILES: {out['smiles']}")
        print(f"Prediction: {out['label']} (score={out['score']:.4f})")

