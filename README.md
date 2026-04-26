# Δ-MBE PhysNet Models

Neural network models for delta-learning corrections to water cluster energies using the Many-Body Expansion (MBE). Separate [PhysNet](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181) models are trained for each MBE order (1–4 body) to predict the difference (Δ) between a low-level (DFTB+) and high-level (BH&H-LYP/aug-cc-pVTZ) description of each n-body interaction.

## Scientific Background

The total energy correction is decomposed via MBE:

$$\Delta E = \sum_i \Delta E_i^{(1)} + \sum_{i<j} \Delta E_{ij}^{(2)} + \sum_{i<j<k} \Delta E_{ijk}^{(3)} + \cdots$$

where each term is the difference between BH&H-LYP/aTZ and DFTB+ for that n-body subsystem. A PhysNet model is trained independently for each order, enabling scalable high-accuracy molecular dynamics at DFTB+ speed with corrections toward the DFT reference.

## Repository Structure

```
delta-mbe/
├── format_data/
│   ├── format_data.py        # Convert ORCA/DFTB+ outputs → PhysNet .npz datasets
│   ├── order_1.npz           # 1-body delta training data
│   ├── order_2.npz           # 2-body delta training data
│   ├── order_3.npz           # 3-body delta training data
│   └── order_4.npz           # 4-body delta training data
for 1-body model
├── order1/
│   └── config.txt            # PhysNet hyperparameters for 2-body model
├── order2/
│   └── config.txt            # PhysNet hyperparameters for 3-body model
├── order3/
│   └── config.txt            # PhysNet hyperparameters 
for 4-body model
├── order4/
│   └── config.txt            # PhysNet hyperparameters 
PhysNet_f32f64/
├── f32/                  # Single-precision PhysNet (training + evaluation)
│   ├── train/
│   └── eval/
└── f64/                  # Double-precision PhysNet (training + evaluation)
    ├── train/
    └── eval/
```

## Dependencies

```bash
conda create --name physnet_env python=3.6
conda activate physnet_env
pip install ase==3.19.1
pip install tensorflow==1.12
```

## Workflow

### 1. Prepare Training Data

`format_data/format_data.py` parses ORCA (BH&H-aTZ) and DFTB+ output files, applies the MBE inclusion-exclusion principle, and writes per-order `.npz` datasets with arrays `R` (coordinates, Å), `Z` (atomic numbers), `E` (delta energies, eV), `F` (delta forces, eV/Å), `D` (delta dipoles), `Q` (charges), and `N` (atom counts).

```bash
python format_data.py <system_name> <system_path> <start_frame> <end_frame> <n_fragments>
```

Output files (`order_1.npz` – `order_4.npz`) are written to the `format_data/` directory.

### 2. Train PhysNet Models

Copy or symlink the desired `order_N.npz` dataset into the training directory, then launch training from the corresponding `orderN/` directory:

```bash
cd PhysNet_f32f64/f32/train
./train.py @../../order1/config.txt
```

Training creates a timestamped run directory (e.g. `20260421191822_ddVwFlab_F128K64b5a2i3o1.../`) containing checkpoints and a `best/` subdirectory with the best model by validation loss. To resume a previous run, set `--restart=<run_directory_name>` in the config file.

#### Key Hyperparameters (per `config.txt`)

| Parameter | Value | Description |
|---|---|---|
| `num_features` | 128 | Atom embedding dimension |
| `num_basis` | 64 | Radial basis functions |
| `num_blocks` | 5 | Interaction blocks |
| `cutoff` | 20.0 Å | Interaction cutoff radius |
| `force_weight` | 50.0 | Relative weight of force loss |
| `max_steps` | 500,000 | Training steps |
| `learning_rate` | 0.001 | Initial learning rate |
| `decay_steps` | 10,000 | Steps between LR decay |
| `decay_rate` | 0.5 | LR decay factor |
| `use_shift` | 1 (order 1 only) | Per-element energy shift |

Dataset sizes by order:

| Order | Training | Validation |
|---|---|---|
| 1-body | 2,740 | 305 |
| 2-body | 8,221 | 914 |
| 3-body | 13,702 | 1,523 |
| 4-body | 13,702 | 1,523 |

### 3. Evaluate Models

From the `eval/` directory (using the best checkpoint from a training run):

```bash
# Predict energy, forces, dipole for a single structure
./predict_mol.py -i structure.xyz

# Geometry optimization (BFGS via ASE)
./optimize.py -i structure.xyz

# Harmonic vibrational frequencies
./ase_vibrations.py -i opt_structure.xyz

# Potential energy along a bond stretch
./predict_stretch.py -i opt_structure.xyz
```

## References

**PhysNet architecture:**
Oliver T. Unke and Markus Meuwly, "PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments, and Partial Charges," *J. Chem. Theory Comput.* 2019, 15(6), 3678–3693.

**Single- vs. double-precision PES accuracy:**
Silvan Käser and Markus Meuwly, "Numerical Accuracy Matters: Applications of Machine Learned Potential Energy Surfaces," *J. Phys. Chem. Lett.* 2024, 15(12), 3419–3424.
