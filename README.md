# Equivariant Pocket Characterization & Target-Aware Binding Site Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-Enabled-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Overview
An advanced, graph-based deep learning pipeline for detecting cryptic ligand-binding pockets on challenging oncology targets. 

Inspired by foundational grid-based methods like *DeepPocket*, this repository upgrades the paradigm of 3D binding site detection by replacing traditional rigid 3D Convolutional Neural Networks (CNNs) with **SE(3)-Equivariant Graph Neural Networks (GNNs)** and **Geometric Vector Perceptrons (GVPs)**. 

This model extracts exact continuous spatial coordinates and multi-modal atom features to perform precise, target-aware conditioning—specifically benchmarked against the dynamic Switch II cryptic pocket of **mutant KRAS G12D** (PDB: 6ZLI / 7EW9).

## 🧬 The Biological Problem: Cryptic Pockets & KRAS G12D
Traditional structure-based drug design (SBDD) often relies on static crystal structures. However, high-value oncology targets like KRAS G12D possess "cryptic pockets"—transient cavities (like the Switch II pocket) that only open via dynamic backbone shifts. Voxel-based 3D CNNs struggle to capture these subtle spatial coordinate changes due to rigid grid approximations and the loss of precise vector data.

## 💻 The Computational Solution
To solve this, we treat the protein as a mathematically continuous spatial graph rather than a discretized 3D image.
* **Continuous Precision:** We retain exact Cartesian coordinates ($x, y, z$) without volumetric grid approximations.
* **SE(3)-Equivariance:** Utilizing GVPs guarantees that the model's predictions are perfectly invariant to 3D spatial transformations (rotations and translations) of the input protein.
* **Multi-Modal Featurization:** The architecture simultaneously processes continuous spatial geometries (bond vectors) and discrete biological attributes (atom types, charges).

---

## ⚙️ System Architecture

### 1. Data Ingestion & Structural Parsing
* **ProDy / Biopython:** Parses raw `.pdb` files, filtering out co-crystallized artifacts and water molecules, isolating the mutated KRAS sequences.

### 2. Graph Construction & Featurization
* **K-Nearest Neighbors (K-NN):** Constructs a radius graph via **PyTorch Geometric (PyG)**, connecting atom nodes within a strict Angstrom cutoff to capture local micro-environments.
* **Multi-Modal Data Handling:** Extracts scalar features (pharmacophores, partial charges) and vector features (relative 3D coordinates, orientations).

### 3. The Deep Learning Engine
* **Geometric Vector Perceptrons (GVPs):** Specialized layers process both scalars and 3D vectors independently, preserving directional geometry without collapsing the mathematical space.

### 4. Pocket Segmentation & Clustering
* **Node-Level Probability:** Outputs a continuous probability score (0.0 - 1.0) for every atom indicating pocket membership.
* **DBSCAN Clustering:** Unsupervised spatial clustering groups high-probability nodes into cohesive, biologically viable active site volumes.

### 5. Interpretability & XAI Validation
* **GNNExplainer:** Validates that the network's attention maps correctly focus on the Asp12 mutation and Switch II loop, ensuring the AI learned true physical biochemistry.
* **Center-of-Mass (CoM) Metrics:** Evaluates accuracy against known co-crystallized ligand bounding boxes (target: < 4 Å RMSD).

---

## 📂 Repository Structure

```text
Equivariant-Pocket-Detection/
│
├── data/                    # Raw PDB files (KRAS G12D) and processed PyG graphs
├── models/                  # GVP layer definitions and SE(3)-GNN architecture
├── notebooks/               # Colab tutorials for pocket prediction and visualization
├── scripts/                 
│   ├── parse_pdb.py         # ProDy/Biopython structural ingestion
│   ├── build_graphs.py      # K-NN graph construction and multi-modal featurization
│   ├── train.py             # Model training loop
│   └── explain.py           # XAI extraction using GNNExplainer
├── requirements.txt         # Environment dependencies
└── README.md                # Project documentation
```

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/Equivariant-Pocket-Detection.git
   cd Equivariant-Pocket-Detection
   ```

2. **Set up a Conda environment (Recommended):**
   ```bash
   conda create -n eq_pocket python=3.10
   conda activate eq_pocket
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Note: Ensure PyTorch Geometric and ProDy are installed according to your CUDA version.
   ```

## 🧪 Usage Example

To parse a target PDB (e.g., KRAS G12D) and predict the cryptic binding pocket:

```bash
python scripts/predict_pocket.py --pdb_id 6ZLI --output_dir ./results/
```
*This script will generate a PyMOL-compatible session highlighting the predicted high-probability pocket residues.*

---
*Developed as a foundational module for De Novo Structure-Based Drug Design via Flow Matching.*
