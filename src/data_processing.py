import torch
from rdkit import Chem
from torch_geometric.data import Data

def get_atom_features(atom):
    """Return a numeric feature vector for an rdkit atom."""
    # Convert a few properties to numeric values
    try:
        atomic_num = atom.GetAtomicNum()
        degree = atom.GetDegree()
        formal_charge = atom.GetFormalCharge()
        hybrid = int(atom.GetHybridization()) if atom.GetHybridization() is not None else 0
        is_aromatic = int(atom.GetIsAromatic())
        total_hs = atom.GetTotalNumHs()
        num_radicals = atom.GetNumRadicalElectrons()
        in_ring = int(atom.IsInRing())
        chiral_tag = int(atom.GetChiralTag())
    except Exception:
        # fallback zeros if something unexpected appears
        atomic_num = degree = formal_charge = hybrid = 0
        is_aromatic = total_hs = num_radicals = in_ring = chiral_tag = 0

    features = [
        atomic_num,
        degree,
        formal_charge,
        hybrid,
        is_aromatic,
        total_hs,
        num_radicals,
        in_ring,
        chiral_tag
    ]
    return torch.tensor(features, dtype=torch.float)

def smiles_to_graph(smiles_string, y_label=None):
    """
    Convert SMILES string to a torch_geometric.data.Data object.
    If y_label is None, label will be set to 0 (useful for inference).
    Returns None if RDKit cannot parse the SMILES.
    """
    if smiles_string is None:
        return None
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    atom_features_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
    if len(atom_features_list) == 0:
        return None
    x = torch.stack(atom_features_list, dim=0)

    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
    if len(edge_indices) == 0:
        # molecules with no bonds (rare) — create empty edge_index
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    y = torch.tensor([0.0], dtype=torch.float) if y_label is None else torch.tensor([float(y_label)], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

