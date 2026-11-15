# Phase 3.1: Define native SE(3)-Equivariant Architecture

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from prody import parsePDB, fetchPDB

class NativeGVP(nn.Module):
    def __init__(self, s_in, v_in, s_out, v_out):
        super().__init__()
        self.wh = nn.Linear(v_in, v_out, bias=False)
        self.ws = nn.Linear(s_in + v_out, s_out)
        self.wg = nn.Linear(s_out, v_out)

    def forward(self, s, V):
        V_t = V.transpose(-1, -2)
        Vh = self.wh(V_t).transpose(-1, -2)
        Vh_norm = torch.norm(Vh, dim=-1)
        s_concat = torch.cat([s, Vh_norm], dim=-1)
        s_out = F.relu(self.ws(s_concat))
        gate = torch.sigmoid(self.wg(s_out)).unsqueeze(-1)
        V_out = gate * Vh
        return s_out, V_out

class DeepEquivariantPocketGNN(nn.Module):
    def __init__(self, scalar_dim=64, vector_dim=16, num_layers=3):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_embeddings=5, embedding_dim=scalar_dim - 1)

        self.edge_gvps = nn.ModuleList([
            NativeGVP(s_in=scalar_dim + 1, v_in=1, s_out=scalar_dim, v_out=vector_dim) for _ in range(num_layers)
        ])
        self.node_gvps = nn.ModuleList([
            NativeGVP(s_in=scalar_dim * 2, v_in=vector_dim, s_out=scalar_dim, v_out=vector_dim) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(scalar_dim) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(scalar_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, graph, return_embeddings=False):
        row, col = graph.edge_index
        num_nodes = graph.num_nodes

        deg = torch.bincount(col, minlength=num_nodes).float().unsqueeze(-1) / 100.0
        s_init = torch.cat([self.atom_embedding(graph.x), deg], dim=-1)
        if return_embeddings:
            s_init = s_init.detach().requires_grad_(True)

        s = s_init
        edge_v_in = graph.edge_vec.unsqueeze(1)

        for edge_gvp, node_gvp, norm in zip(self.edge_gvps, self.node_gvps, self.layer_norms):
            edge_s_in = torch.cat([s[row], graph.edge_attr], dim=-1)
            msg_s, msg_v = edge_gvp(edge_s_in, edge_v_in)

            deg_inv = 1.0 / (torch.bincount(col, minlength=num_nodes).float().unsqueeze(-1) + 1e-5)
            agg_s = torch.zeros(num_nodes, msg_s.size(-1), device=s.device)
            agg_v = torch.zeros(num_nodes, msg_v.size(1), 3, device=s.device)

            agg_s.index_add_(0, col, msg_s)
            agg_v.index_add_(0, col, msg_v)
            agg_s = agg_s * deg_inv
            agg_v = agg_v * deg_inv.unsqueeze(-1)

            s_upd, _ = node_gvp(torch.cat([s, agg_s], dim=-1), agg_v)
            s = norm(s + s_upd)

        probs = self.classifier(s)
        if return_embeddings:
            return probs, s_init
        return probs

def extract_ground_truth_labels(pdb_id, folder, protein_pos_tensor, cutoff=4.5):
    pdb_path = fetchPDB(pdb_id, folder=folder)
    structure = parsePDB(pdb_path)
    ligand = structure.select('hetero and not water and chain A')
    ligand_coords = torch.tensor(ligand.getCoords(), dtype=torch.float32, device=protein_pos_tensor.device)
    dist_to_ligand = torch.cdist(protein_pos_tensor, ligand_coords, p=2.0)
    min_dist, _ = torch.min(dist_to_ligand, dim=1)
    labels = (min_dist <= cutoff).float().unsqueeze(-1)
    return labels, ligand_coords
  
  
  #==========================================================================================================================================================#
  
  # Phase 3.2: Ground-Truth Labeling & GVP Training

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Graph
    protein_graph = torch.load("../checkpoints/kras_g12d_graph.pt", map_location=device, weights_only=False)

    # 2. Extract Answer Key (Ground Truth)
    y_true, true_ligand_coords = extract_ground_truth_labels(
        pdb_id="6ZLI", folder="../data/raw/", protein_pos_tensor=protein_graph.pos, cutoff=4.5
    )

    # 3. Initialize Upgraded 3-Layer Model & Optimizer
    deep_model = DeepEquivariantPocketGNN(scalar_dim=64, vector_dim=16, num_layers=3).to(device)
    optimizer = torch.optim.AdamW(deep_model.parameters(), lr=0.003, weight_decay=1e-4)

    num_pos = y_true.sum()
    pos_weight = ((len(y_true) - num_pos) / (num_pos + 1e-5)).item()

    print("Training 3-Layer Residual SE(3)-GVP Network...")
    deep_model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        preds = deep_model(protein_graph)

        # Focal-modulated Weighted BCE
        pt = torch.where(y_true == 1.0, preds, 1.0 - preds)
        focal_weight = ((1.0 - pt) ** 2.0) * torch.where(y_true == 1.0, pos_weight, 1.0)
        loss = F.binary_cross_entropy(preds, y_true, weight=focal_weight.detach())

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                pred_bin = (preds > 0.70).float()
                tp = ((pred_bin == 1) & (y_true == 1)).sum().item()
                fp = ((pred_bin == 1) & (y_true == 0)).sum().item()
                recall = tp / (num_pos.item() + 1e-5)
                precision = tp / (tp + fp + 1e-5)
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Recall: {recall*100:.1f}% | Precision: {precision*100:.1f}%")

    # 4. Save Trained Artifacts for Phases 4 & 5
    os.makedirs("../checkpoints", exist_ok=True)
    torch.save(deep_model.state_dict(), "../checkpoints/equivariant_gvp_kras_weights.pt")
    torch.save({"y_true": y_true, "true_ligand_coords": true_ligand_coords}, "../checkpoints/kras_ground_truth.pt")
    print("\nPhase 3.2 Complete. Weights and Ground Truth saved to ../checkpoints/")