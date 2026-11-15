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


# Generative Pose Modeling & Affinity Scoring via Flow Matching

An advanced, production-grade computational pipeline utilizing **multi-modal flow matching architectures** and **deep geometric representations** for de novo ligand pose generation and native binding affinity scoring, bypassing traditional physics-based scoring limitations.

---

## 🚀 Overview

Traditional structure-based drug discovery (SBDD) relies heavily on rigid docking algorithms and resource-heavy scoring functions that often struggle with receptor flexibility and accurate binding free-energy estimation.

**Generative Pose Modeling & Affinity Scoring via Flow Matching** bridges this gap by framing ligand pose generation as a continuous normalizing flow problem. By learning vector fields that push a prior noise distribution directly onto the true bound atomic coordinate distribution, this framework achieves state-of-the-art accuracy in predicting ligand conformations and binding affinities.

---

## 🛠️ Core Architecture & Methodology

```
[Protein PDB & Ligand SMILES] 
       │
       ▼
[Geometric Feature Extraction] (SE(3)-Equivariant Graph Layers)
       │
       ▼
[Conditional Flow Matching Network] (Neural Vector Field Estimation)
       │
       ▼
[Generative Pose Sampling] (ODE Integration from Prior Noise)
       │
       ▼
[Native Affinity Scoring Head] (Continuous pKd / ΔG Prediction)

```

1. **Geometric Representation:** Encodes protein pockets and ligand atom graphs using invariant and equivariant features.
2. **Flow Matching Backbone:** Learns a time-dependent vector field $v_t$ that steers straight trajectories from a standard Gaussian noise prior to the target atomic coordinates.
3. **Affinity Prediction Head:** Jointly evaluates the generated 3D poses against the receptor pocket microenvironment to output robust binding affinity metrics ($\Delta G$ and $pK_d$).

---

## ⚙️ Key Features

* **Continuous Flow Matching:** Avoids the sampling bottlenecks of traditional diffusion models by utilizing optimal-transport straight-line trajectories.
* **SE(3)-Equivariance:** Ensures that rotations and translations of the input protein-ligand complex transform the output coordinates correctly.
* **End-to-End Inference:** Takes raw PDB structures and ligand SMILES strings, outputs fully optimized 3D binding poses, and predicts binding affinity scores.
* **Cloud-Optimized Execution:** Engineered to run efficiently on standard cloud GPUs (such as Google Colab Tesla T4/A100 instances), bypassing heavy local VRAM restrictions.

---

## 📂 Repository Structure

```text
├── data/                  # Sample datasets and preprocessed protein-ligand complexes
├── models/                # Flow matching network and SE(3)-equivariant architecture definitions
├── scripts/               # Training, inference, and evaluation utility scripts
├── notebooks/             # Interactive Google Colab / Kaggle execution notebooks
├── requirements.txt       # Python dependencies (PyTorch, PyTorch Geometric, RDKit)
└── README.md              # Project documentation

```

---

## 📦 Installation & Setup

Clone the repository and install the required dependencies within your Python/conda environment:

```bash
git clone https://github.com/sandeepsinghofficial049/Generative-Pose-Flow-Matching.git
cd Generative-Pose-Flow-Matching

pip install -r requirements.txt

```

---

## 💻 Quickstart Inference

To run generative pose prediction and affinity scoring on a sample protein-ligand complex:

```python
from models.flow_matcher import FlowMatchingPoseGenerator
from utils.data_loader import load_protein_ligand_pair

# Load receptor pocket and ligand features
protein_pocket, ligand = load_protein_ligand_pair("data/sample_pocket.pdb", "CCO")

# Initialize the flow matching model
model = FlowMatchingPoseGenerator.from_pretrained("weights/flow_matching_pose_v1.pt")
model.eval()

# Generate 3D atomic coordinates via vector field integration
generated_poses, predicted_affinity = model.sample_poses_and_score(
    protein_pocket, 
    ligand, 
    num_steps=50
)

print(f"Generated Pose Coordinates Shape: {generated_poses.shape}")
print(f"Predicted Binding Affinity (pK_d): {predicted_affinity.item():.2f}")

```

---

## 📊 Evaluation Metrics

* **Root Mean Square Deviation (RMSD):** Measures spatial deviation between generated ligand heavy-atom poses and experimental crystal structures ($\text{Å}$).
* **Pearson/Spearman Correlation ($R$):** Evaluates ranking capability of predicted binding affinities against experimental benchmark datasets.

---

## 👤 Author

**Sandeep Singh**

*M.Sc. Bioinformatics, Central University of South Bihar*

[GitHub Profile](https://github.com/sandeepsinghofficial049) | [LinkedIn](https://www.google.com/search?q=https://www.linkedin.com/in/sandeep-singh-9b36a9407)
To parse a target PDB (e.g., KRAS G12D) and predict the cryptic binding pocket:

```bash
python scripts/predict_pocket.py --pdb_id 6ZLI --output_dir ./results/
```
*This script will generate a PyMOL-compatible session highlighting the predicted high-probability pocket residues.*

---
*Developed as a foundational module for De Novo Structure-Based Drug Design via Flow Matching.*
