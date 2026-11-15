#Phase 2: Building the Geometric Graph

import torch
from torch_geometric.data import Data

def build_protein_graph(parsed_data, cutoff_angstroms=8.0):
    """
    Constructs a Radius Graph using pure PyTorch tensor operations (torch.cdist).
    Zero external C++ compilation dependencies (no pyg-lib or torch-cluster required).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Active Compute Device: {device}")

    pos = parsed_data['positions'].to(device)
    node_features = parsed_data['node_features'].to(device)

    print(f"Building Native PyTorch Radius Graph (Cutoff: {cutoff_angstroms} Å)...")

    # 1. Compute pairwise Euclidean distance matrix: Shape [N_atoms, N_atoms]
    dist_matrix = torch.cdist(pos, pos, p=2.0)

    # 2. Create adjacency mask: distance <= cutoff AND distance > 0 (excludes self-loops)
    adjacency_mask = (dist_matrix <= cutoff_angstroms) & (dist_matrix > 1e-5)

    # 3. Extract edge_index in standard PyG format: Shape [2, num_edges]
    edge_index = adjacency_mask.nonzero(as_tuple=False).t().contiguous()

    # 4. Compute directional edge vectors (x_j - x_i) and scalar distances for SE(3) message passing
    row, col = edge_index
    edge_vectors = pos[col] - pos[row]                       # Shape: [num_edges, 3]
    edge_lengths = dist_matrix[row, col].unsqueeze(-1)       # Shape: [num_edges, 1]
    edge_unit_vectors = edge_vectors / (edge_lengths + 1e-8) # Normalized 3D directional vectors

    # Pack everything into the PyTorch Geometric Data object
    graph_data = Data(
        x=node_features,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_lengths,
        edge_vec=edge_unit_vectors,
        residue_ids=parsed_data['residue_ids'].to(device)
    )

    print("\n--- Graph Construction Summary ---")
    print(f"Total Nodes (Atoms): {graph_data.num_nodes}")
    print(f"Total Edges (Interactions): {graph_data.num_edges}")
    print(f"Average neighbors per atom: {graph_data.num_edges / graph_data.num_nodes:.2f}")

    return graph_data

if __name__ == "__main__":
    protein_graph = build_protein_graph(extracted_tensors, cutoff_angstroms=8.0)

    # Save the graph for Phase 3
    os.makedirs("../checkpoints", exist_ok=True)
    torch.save(protein_graph, "../checkpoints/kras_g12d_graph.pt")