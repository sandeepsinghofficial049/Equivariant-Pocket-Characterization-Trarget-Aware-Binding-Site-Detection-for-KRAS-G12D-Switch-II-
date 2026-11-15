#Phase 5: CoM Validation + XAI Attribution

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from prody import parsePDB

def run_validation_and_xai(
    graph_path="../checkpoints/kras_g12d_graph.pt",
    weights_path="../checkpoints/equivariant_gvp_kras_weights.pt",
    gt_path="../checkpoints/kras_ground_truth.pt",
    phase4_path="../checkpoints/kras_phase4_segmentation.pt",
    raw_pdb_path="../data/raw/6zli.pdb.gz",
    figure_out_path="../data/raw/kras_gvp_validation_summary.png"
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Checkpoints from Prior Phases
    protein_graph = torch.load(graph_path, map_location=device, weights_only=False)
    gt_data = torch.load(gt_path, map_location=device, weights_only=False)
    phase4_data = torch.load(phase4_path, map_location='cpu', weights_only=False)

    y_true = gt_data["y_true"]
    true_ligand_coords = gt_data["true_ligand_coords"]

    model = DeepEquivariantPocketGNN(scalar_dim=64, vector_dim=16, num_layers=3).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # =====================================================================
    # A. CENTER-OF-MASS (CoM) BENCHMARK VALIDATION
    # =====================================================================
    pred_com = phase4_data["pred_com"]
    true_com = true_ligand_coords.detach().cpu().numpy().mean(axis=0)
    com_distance = np.linalg.norm(pred_com - true_com)

    print("=== 1. Center-of-Mass (CoM) Benchmark Validation ===")
    print(f"Predicted Pocket CoM          : [{pred_com[0]:.2f}, {pred_com[1]:.2f}, {pred_com[2]:.2f}]")
    print(f"True Co-Crystallized CoM      : [{true_com[0]:.2f}, {true_com[1]:.2f}, {true_com[2]:.2f}]")
    print(f"CoM Euclidean Distance Error  : {com_distance:.2f} Å (Benchmark Target: < 4.00 Å)\n")

    # =====================================================================
    # B. EXPLAINABLE AI (XAI) INPUT-GRADIENT ATTRIBUTION
    # =====================================================================
    pocket_probs, s_init = model(protein_graph, return_embeddings=True)
    pocket_probs[y_true == 1.0].sum().backward()
    atom_importance = torch.norm(s_init.grad, dim=-1).detach().cpu().numpy()

    structure = parsePDB(raw_pdb_path)
    chain_a = structure.select('protein and chain A')
    resnum_to_name = dict(zip(chain_a.getResnums(), chain_a.getResnames()))

    res_ids_np = protein_graph.residue_ids.detach().cpu().numpy()
    unique_res = np.unique(res_ids_np)
    res_scores = {res: atom_importance[res_ids_np == res].mean() for res in unique_res}
    top_residues = sorted(res_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    print("=== 2. Top 10 XAI-Attributed KRAS G12D Residues ===")
    for rank, (res_num, score) in enumerate(top_residues, 1):
        res_name = resnum_to_name.get(res_num, "UNK")
        tag = ""
        if 10 <= res_num <= 18:
            tag = " <-- P-Loop / G12D Active Site!"
        elif 28 <= res_num <= 35:
            tag = " <-- Switch I Loop!"
        elif 57 <= res_num <= 62:
            tag = " <-- Switch II Catalytic Site!"
        elif res_num in [116, 117, 118, 119, 145, 146, 147]:
            tag = " <-- Guanine Recognition Motif"
        print(f"  {rank:2d}. {res_name} {res_num:3d} | Attribution Score: {score:.4f}{tag}")

    # =====================================================================
    # C. SE(3) ROTATION & TRANSLATION INVARIANCE AUDIT
    # =====================================================================
    Q, R_qr = torch.linalg.qr(torch.randn(3, 3, device=device))
    Q = Q * torch.sign(torch.diag(R_qr))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    translation = torch.randn(1, 3, device=device) * 50.0

    rotated_graph = protein_graph.clone()
    rotated_graph.pos = torch.matmul(protein_graph.pos, Q) + translation
    rotated_graph.edge_vec = torch.matmul(protein_graph.edge_vec, Q)

    with torch.no_grad():
        orig_probs = model(protein_graph)
        rot_probs = model(rotated_graph)
        max_diff = torch.max(torch.abs(orig_probs - rot_probs)).item()

    print("\n=== 3. SE(3) Equivariance Verification Audit ===")
    print(f"Proper Rotation Matrix det(R) : {torch.det(Q).item():.4f}")
    print(f"Max Atomic Prob Discrepancy   : {max_diff:.2e} (Target: < 1e-5)\n")

    # =====================================================================
    # D. EXPORT 2-PANEL PORTFOLIO FIGURE
    # =====================================================================
    probs_np = phase4_data["probs_np"]
    coords_np = phase4_data["coords_np"]
    cluster_coords = phase4_data["cluster_coords"]

    fig = plt.figure(figsize=(14, 6), dpi=150)

    # Panel A: 3D Pocket Segmentation & CoM
    ax1 = fig.add_subplot(121, projection='3d')
    non_pocket_mask = probs_np <= 0.70
    ax1.scatter(coords_np[non_pocket_mask, 0], coords_np[non_pocket_mask, 1], coords_np[non_pocket_mask, 2],
                c='lightgrey', alpha=0.25, s=10, label='KRAS Scaffold')
    ax1.scatter(cluster_coords[:, 0], cluster_coords[:, 1], cluster_coords[:, 2],
                c='#00b4d8', alpha=0.85, s=35, edgecolors='k', linewidth=0.3, label=f'GVP Pocket ({len(cluster_coords)} atoms)')
    ax1.scatter(pred_com[0], pred_com[1], pred_com[2],
                c='blue', marker='^', s=180, edgecolors='white', linewidth=1.5, label='Predicted CoM')
    ax1.scatter(true_com[0], true_com[1], true_com[2],
                c='#e63946', marker='*', s=260, edgecolors='black', linewidth=1.0, label=f'True Ligand CoM ({com_distance:.2f} Å)')
    ax1.set_title("Phase 4: 3D Equivariant Pocket Segmentation (6ZLI)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (Å)"); ax1.set_ylabel("Y (Å)"); ax1.set_zlabel("Z (Å)")
    ax1.legend(loc='upper left', fontsize=8.5)

    # Panel B: XAI Attribution Bar Chart
    ax2 = fig.add_subplot(122)
    res_labels = [f"{resnum_to_name.get(r, 'UNK')}{r}" for r, _ in top_residues][::-1]
    scores = [s for _, s in top_residues][::-1]
    colors = ['#e63946' if (10 <= r <= 18) else ('#f4a261' if (28 <= r <= 35) else '#2a9d8f') for r, _ in top_residues][::-1]

    ax2.barh(res_labels, scores, color=colors, edgecolor='black', height=0.65)
    ax2.set_title("Phase 5: XAI Gradient Attribution by KRAS Motif", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Mean Atomic Gradient Attribution Score (||∇_s P||_2)")
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    legend_elements = [
        Patch(facecolor='#e63946', edgecolor='black', label='P-Loop / G12D Active Site (Res 14-18)'),
        Patch(facecolor='#f4a261', edgecolor='black', label='Switch I Loop (Res 29)'),
        Patch(facecolor='#2a9d8f', edgecolor='black', label='Guanine Recognition & Core (Res 116-146)')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=8.5)

    plt.tight_layout()
    plt.savefig(figure_out_path, dpi=300)
    plt.show()
    print(f"Saved Validation & XAI Figure to: {figure_out_path}")

if __name__ == "__main__":
    run_validation_and_xai()