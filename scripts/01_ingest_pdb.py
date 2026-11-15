#Phase 1: Phase 1: Structural Parsing & Coordinate Extraction

import os
import numpy as np
import torch
from prody import parsePDB, fetchPDB

def process_kras_structure(pdb_id, output_dir):
    """
    Downloads and parses a target PDB file, extracting atomic coordinates
    and element types for GNN featurization.
    """
    print(f"Fetching PDB: {pdb_id}...")

    # Ensure raw data directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download the PDB file directly into the data folder
    pdb_path = fetchPDB(pdb_id, folder=output_dir)

    # Parse the biomolecular structure
    structure = parsePDB(pdb_path)

    # Filter: Select only protein atoms (ignore waters, salts, and artifacts)
    # We also isolate Chain A to avoid duplicating multi-mer assemblies
    protein = structure.select('protein and chain A')

    if protein is None:
        raise ValueError("Failed to extract protein chain A.")

    print(f"Successfully isolated {len(protein)} atoms from {pdb_id}.")

    # 1. Extract Continuous Vector Features (3D Spatial Coordinates)
    # Shape: [N_atoms, 3]
    coords = protein.getCoords()
    pos_tensor = torch.tensor(coords, dtype=torch.float32)

    # 2. Extract Discrete Scalar Features (Atom Types/Elements)
    # We convert standard biological elements (C, N, O, S) into categorical integers
    elements = protein.getElements()
    element_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3}

    # Map elements to integers, defaulting to 4 for uncommon heavy atoms
    encoded_elements = [element_map.get(el, 4) for el in elements]
    node_features = torch.tensor(encoded_elements, dtype=torch.long)

    # 3. Extract Residue Information (Useful for mapping the Switch II loop later)
    res_indices = torch.tensor(protein.getResnums(), dtype=torch.long)

    # Compile the extracted data into a dictionary payload
    parsed_data = {
        'positions': pos_tensor,       # The 3D geometry
        'node_features': node_features,# The chemical identity
        'residue_ids': res_indices     # The sequence mapping
    }

    print("\n--- Tensor Extraction Summary ---")
    print(f"Coordinate Tensor Shape: {pos_tensor.shape}")
    print(f"Node Feature Tensor Shape: {node_features.shape}")
    print("Ready for K-NN Graph Construction.")

    return parsed_data

if __name__ == "__main__":
    # KRAS G12D structure highly relevant for cryptic pocket modeling
    target_pdb = "6ZLI"
    raw_data_path = "../data/raw/"

    extracted_tensors = process_kras_structure(target_pdb, raw_data_path)