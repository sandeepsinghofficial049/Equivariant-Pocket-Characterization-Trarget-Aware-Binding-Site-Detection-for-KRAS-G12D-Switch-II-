#Phase 4: DBSCAN Spatial Segmentation (Matching P > 0.70 Training Threshold)


import os
import torch
import numpy as np
from sklearn.cluster import DBSCAN
from prody import parsePDB, writePDB

def segment_binding_pocket(
    graph_path="../checkpoints/kras_g12d_graph.pt",
    weights_path="../checkpoints/equivariant_gvp_kras_weights.pt",
    raw_pdb_path="../data/raw/6zli.pdb.gz",
    output_pdb_path="../data/raw/kras_6zli_3layer_gvp_pocket.pdb",
    prob_threshold=0.70,
    dbscan_eps=4.5,
    dbscan_min_samples=5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Phase 2 Graph and Phase 3 Trained Weights
    protein_graph = torch.load(graph_path, map_location=device, weights_only=False)
    
    model = DeepEquivariantPocketGNN(scalar_dim=64, vector_dim=16, num_layers=3).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Predict Node-Level Pocket Probabilities
    with torch.no_grad():
        pocket_probs = model(protein_graph)
        
    probs_np = pocket_probs.detach().cpu().numpy().flatten()
    coords_np = protein_graph.pos.detach().cpu().numpy()
    
    # 3. Filter High-Confidence Pocket Nodes
    high_prob_mask = probs_np > prob_threshold
    cand_coords = coords_np[high_prob_mask]
    cand_probs = probs_np[high_prob_mask]
    
    if len(cand_coords) == 0:
        raise RuntimeError(f"No atoms exceeded probability threshold {prob_threshold}")
        
    # 4. DBSCAN Spatial Clustering
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(cand_coords)
    labels = clustering.labels_
    
    valid_clusters = [c for c in set(labels) if c != -1]
    if not valid_clusters:
        raise RuntimeError("DBSCAN classified all high-probability nodes as noise (-1).")
        
    # Select primary pocket cluster by cumulative probability mass
    best_cluster = max(valid_clusters, key=lambda c: cand_probs[labels == c].sum())
    cluster_mask = (labels == best_cluster)
    
    cluster_coords = cand_coords[cluster_mask]
    cluster_weights = cand_probs[cluster_mask]
    
    # 5. Compute Probability-Weighted Center of Mass (Soft-CoM)
    pred_com = np.average(cluster_coords, axis=0, weights=cluster_weights)
    
    # 6. Export B-Factor-Encoded PDB for PyMOL Visualization
    structure = parsePDB(raw_pdb_path)
    chain_a = structure.select('protein and chain A')
    chain_a.setBetas(probs_np * 100.0)
    writePDB(output_pdb_path, chain_a)
    
    # 7. Save Phase 4 Segmentation Artifacts for Phase 5
    phase4_payload = {
        "probs_np": probs_np,
        "coords_np": coords_np,
        "cluster_coords": cluster_coords,
        "cluster_weights": cluster_weights,
        "pred_com": pred_com,
        "num_clusters": len(valid_clusters)
    }
    os.makedirs("../checkpoints", exist_ok=True)
    torch.save(phase4_payload, "../checkpoints/kras_phase4_segmentation.pt")
    
    print("--- Phase 4: DBSCAN Pocket Segmentation Summary ---")
    print(f"Total Protein Atoms Evaluated           : {len(probs_np)}")
    print(f"High-Confidence Atoms (P > {prob_threshold:.2f})        : {len(cand_coords)}")
    print(f"Distinct Spatial Clusters Found         : {len(valid_clusters)}")
    print(f"Primary Segmented Pocket Size           : {len(cluster_coords)} atoms")
    print(f"Predicted Pocket Soft-CoM (X, Y, Z)     : [{pred_com[0]:.2f}, {pred_com[1]:.2f}, {pred_com[2]:.2f}]")
    print(f"Exported PyMOL B-Factor Heatmap PDB     : {output_pdb_path}")
    
    return phase4_payload

if __name__ == "__main__":
    phase4_results = segment_binding_pocket()