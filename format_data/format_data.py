import numpy as np
import itertools
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

def parse_energy(out_path: Path) -> float:
    """Extract energy (Ha) from ORCA output."""
    with out_path.open() as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                return float(line.split()[-1])
    raise RuntimeError("Energy line not found in {}".format(out_path))

def parse_dftb_energy(out_path: Path) -> float:
    """Extract energy (Ha) from DFTB+ output."""
    with out_path.open() as f:
        for line in f:
            if "total_energy" in line:
                # return total energy from next line
                while True:
                    line2 = next(f)
                    if line2.strip() and not line2.strip().startswith('#'):
                        try:
                            return float(line2.strip())
                        except ValueError:
                            raise RuntimeError(f"Malformed energy line in {out_path}: {line2.strip()}")
    raise RuntimeError("Energy line not found in {}".format(out_path))

def parse_engrad_file(path: Path) -> np.ndarray:
    """Extract gradient (Eh/bohr) from ORCA .engrad file as array shape (n_atoms,3)."""
    with path.open() as f:
        natm: Optional[int] = None
        gradients = []
        # first find number of atoms
        for line in f:
            if "Number of atoms" in line:
                # next non-comment numeric line is natm
                while True:
                    line2 = next(f)
                    if line2.strip() and not line2.strip().startswith('#'):
                        natm = int(line2.strip())
                        break
                break
        if natm is None:
            raise RuntimeError(f"Cannot find atom count in {path}")
        # find gradient block
        for line in f:
            if "current gradient" in line:
                # read 3*natm float lines
                count = 3 * natm
                while len(gradients) < count:
                    line3 = next(f)
                    if line3.strip() and not line3.strip().startswith('#'):
                        gradients.append(float(line3.strip()))
                break
        if len(gradients) != 3 * natm:
            print(f"Unexpected gradient entries in {path}")
        return np.array(gradients).reshape(natm, 3)
    
def parse_dftb_gradient(path: Path) -> np.ndarray:
    """Extract gradient (Eh/bohr) from DFTB+ results.tag file as array shape (n_atoms,3)."""
    with path.open() as f:
        forces = []
        in_forces_block = False
        for line in f:
            if line.strip().startswith("forces"):
                in_forces_block = True
                continue
            if in_forces_block:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                # Check if we've moved to the next data block (non-numeric line starting with a letter)
                if stripped and not stripped[0].isdigit() and not stripped[0] in "+-." :
                    break
                try:
                    forces.extend(float(x) for x in stripped.split())
                except ValueError:
                    break
        if not forces:
            raise RuntimeError(f"Forces not found in {path}")
        # Convert flat list to (n_atoms, 3) and negate (gradient = -force)
        forces_array = np.array(forces).reshape(-1, 3)
        return -forces_array

def parse_dipole(path: Path) -> np.ndarray:
    """Extract dipole moment (eA) from ORCA output."""
    with path.open() as f:
        for line in f:
            if line.strip().startswith("Total Dipole Moment"):
                return np.array([float(x) for x in line.split()[4:7]])
    raise RuntimeError("Dipole moment line not found in {}".format(path))

def read_xyz(path: Path) -> Tuple[List[str], np.ndarray]:
    """Return (symbols, coordinates[Å]) from a standard XYZ file."""
    with path.open() as f:
        try:
            natm = int(f.readline())
        except ValueError:
            raise RuntimeError("First line must contain number of atoms")
        _ = f.readline()  # comment
        symbols, xyz = [], []
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                raise RuntimeError("Malformed XYZ line: " + line)
            symbols.append(parts[0])
            xyz.append([float(x) for x in parts[1:4]])
        if len(symbols) != natm:
            raise RuntimeError(f"Expected {natm} atoms, found {len(symbols)}")
        return symbols, np.array(xyz)

def read_xyz_from_out(path: Path) -> Tuple[List[str], np.ndarray]:
    """Read XYZ data from an ORCA output file. Take coordinates from INPUT FILE section of the output."""
    with path.open() as f:
        symbols, xyz = [], []
        coord_lines = []
        for line in f:
            parts = line.split()
            input_lines = []
            if len(parts) > 0:
                if parts[0] == "|" and "%" not in line and "!" not in line:
                    if "****END OF INPUT****" in line:
                        break
                    input_lines.append(line)
                    
            for line in input_lines:
                if len(line.split()) > 2 and "*" not in line:
                    coord_lines.append(line)
        for line in coord_lines:
            parts = line.split()
            symbols.append(parts[2])
            xyz.append([float(x) for x in parts[3:6]])

    return symbols, np.array(xyz)

def read_total_charge_from_out(path: Path) -> int:
    """Read total charge from an ORCA output file."""
    with path.open() as f:
        for line in f:
            parts = line.split()
            input_lines = []
            if len(parts) > 0:
                if parts[0] == "|" and "%" not in line and "!" not in line:
                    if "****END OF INPUT****" in line:
                        break
                    input_lines.append(line)
                    
            for line in input_lines:
                parts = line.split()
                if "*xyz" in line:
                    total_charge = int(parts[3])
                    return total_charge
    raise RuntimeError("Total charge line not found in {}".format(path))

def get_atomic_number(symbol: str) -> int:
    if symbol == "H":
        return 1
    elif symbol == "He":
        return 2
    elif symbol == "Li":
        return 3
    elif symbol == "Be":
        return 4
    elif symbol == "B":
        return 5
    elif symbol == "C":
        return 6
    elif symbol == "N":
        return 7
    elif symbol == "O":
        return 8
    elif symbol == "Na":
        return 11
    elif symbol == "Cl":
        return 17
    else:
        raise ValueError(f"Unknown element: {symbol}")

def generate_combinations(n_frag: int, order: int) -> List[Tuple[int, ...]]:
    combos: List[Tuple[int, ...]] = []
    for k in range(1, order + 1):
        combos.extend(itertools.combinations(range(n_frag), k))
    return combos

# ---- New helpers --------------------------------------------------------------

def subsets_of(combo: Tuple[int, ...]):
    """All non-empty subsets of combo (including combo itself)."""
    for k in range(1, len(combo) + 1):
        for s in itertools.combinations(combo, k):
            yield s

def embed_to_sup(coords_sup: np.ndarray, coords_sub: np.ndarray, vec_sub: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Embed per-atom vector (natm_sub,3) into superset shape (natm_sup,3)
    by matching coordinates. Assumes single-point jobs (identical coords across levels).
    """
    out = np.zeros((len(coords_sup), 3), dtype=float)
    for j, r in enumerate(coords_sub):
        d = np.linalg.norm(coords_sup - r, axis=1)
        k = int(np.argmin(d))
        if d[k] > tol:
            raise RuntimeError("Failed to match atom by coordinates during embedding")
        out[k] = vec_sub[j]
    return out

def delta_scalar_for_combo(energies: Dict[Tuple[int, ...], float], combo: Tuple[int, ...]) -> float:
    """ΔE_|combo| via inclusion-exclusion on scalar energies."""
    total = 0.0
    m = len(combo)
    for s in subsets_of(combo):
        sign = -1 if ((m - len(s)) % 2 == 1) else 1
        total += sign * energies[s]
    return total

def delta_vector_for_combo(vectors: Dict[Tuple[int, ...], np.ndarray],
                           coords: Dict[Tuple[int, ...], np.ndarray],
                           combo: Tuple[int, ...]) -> np.ndarray:
    """ΔG_|combo| via inclusion-exclusion, embedded into combo's atom order."""
    coords_C = coords[combo]
    acc = np.zeros_like(coords_C, dtype=float)  # shape (nC,3)
    m = len(combo)
    for s in subsets_of(combo):
        sign = -1 if ((m - len(s)) % 2 == 1) else 1
        g_s = vectors[s]  # shape (ns,3)
        cs  = coords[s]
        acc += sign * embed_to_sup(coords_C, cs, g_s)
    return acc

def delta_dipole_for_combo(dips: Dict[Tuple[int, ...], np.ndarray], combo: Tuple[int, ...]) -> np.ndarray:
    """Δμ_|combo| via inclusion-exclusion on 3-vectors."""
    total = np.zeros(3, dtype=float)
    m = len(combo)
    for s in subsets_of(combo):
        sign = -1 if ((m - len(s)) % 2 == 1) else 1
        total += sign * dips[s]
    return total

def main(system, systempath, frames, n_frag=7, frame_suffix=""):
    """
    Format .npz files to train delta-learning MBE model with PhysNet.

    Data should be stored as python dictionary in a compressed numpy binary file (.npz). The dictionary contains seven numpy arrays:

    R (num_data, max_atoms, 3): Cartesian coordinates of nuclei (in Angstrom [A])
    Q (num_data,):              Total charge (in elementary charges [e])
    D (num_data, 3):            Dipole moment vector with respect to the origin (in elementary charges times Angstrom [eA])
    E (num_data,):              Potential energy with respect to free atoms (in electronvolt [eV])
    F (num_data, max_atoms, 3): Forces acting on the nuclei (in electronvolt per Angstrom [eV/A])
    Z (num_data, max_atoms):    Nuclear charges/atomic numbers of nuclei
    N (num_data,):              Number of atoms in each structure (structures consisting of less than max_atoms entries are zero-padded)

    For each order up to order 4, create a separate .npz file consisting of the PhysNet training data for that order ONLY, i.e., for order 1,
    only data for the one-body calculations should be included.
    """
    conv_eVA = 27.211386246/0.52917721  # Hartree/Bohr -> eV/A
    conv_eV = 27.211386246             # Hartree -> eV
    conv_eA = 0.2081943/0.3934303      # a.u. -> eA
    conv_kcal = 23.0609                # eV -> kcal/mol

    levels = ["dftb", "bhandh-aTZ"]

    for order in range(1, 5):
        data: Dict[str, np.ndarray] = { "R": [], "Q": [], "D": [], "E": [], "F": [], "Z": [], "N": [] }

        # choose frames as needed
        for i in frames:
            if frame_suffix:
                frame = f"frame_{i:03d}_{frame_suffix}"
            else:
                frame = f"frame_{i:03d}"

            # caches across all subsets up to `order` for this frame
            energies: Dict[str, Dict[Tuple[int, ...], float]] = {lvl: {} for lvl in levels}
            grads:    Dict[str, Dict[Tuple[int, ...], np.ndarray]] = {lvl: {} for lvl in levels}
            dips:     Dict[str, Dict[Tuple[int, ...], np.ndarray]] = {lvl: {} for lvl in levels}
            coords:   Dict[Tuple[int, ...], np.ndarray] = {}

            # collect all subsets up to `order`
            all_combos = generate_combinations(n_frag, order)  # includes 1..order
            for combo in all_combos:
                combo_str = "_".join(map(str, combo))
                # coordinates identical across levels; read once (from high)
                if combo not in coords:
                    _, coords[combo] = read_xyz_from_out(
                        Path(f"{systempath}_mbe_{levels[1]}_simplefrags/{frame}/high/_mbe_tmp/{combo_str}/frag.out")
                    )
                for level in levels:
                    if level == 'dftb':
                        out_path = Path(f"{systempath}_mbe_{level}_simplefrags/{frame}/low/_mbe_tmp/{combo_str}/results.tag")
                        engrad_path = Path(f"{systempath}_mbe_{level}_simplefrags/{frame}/low/_mbe_tmp/{combo_str}/results.tag")
                        energies[level][combo] = parse_dftb_energy(out_path)
                        grads[level][combo]    = parse_dftb_gradient(engrad_path)
                    elif level == 'bhandh-aTZ':
                        out_path = Path(f"{systempath}_mbe_{level}_simplefrags/{frame}/high/_mbe_tmp/{combo_str}/frag.out")
                        engrad_path = Path(f"{systempath}_mbe_{level}_simplefrags/{frame}/high/_mbe_tmp/{combo_str}/frag.engrad")
                        energies[level][combo] = parse_energy(out_path)  # Ha
                        grads[level][combo]    = parse_engrad_file(engrad_path)           # Eh/bohr
                    else:
                        raise ValueError(f"Unknown level: {level}")
                    
                    # set dipole moment to zero for simplicity since we don't need it
                    dips[level][combo]     = np.zeros(3, dtype=float)  # eA

            # now generate training rows only for combos of this exact order
            for combo in (c for c in all_combos if len(c) == order):
                combo_str = "_".join(map(str, combo))
                symbols, coords_C = read_xyz_from_out(
                    Path(f"{systempath}_mbe_{levels[1]}_simplefrags/{frame}/high/_mbe_tmp/{combo_str}/frag.out")
                )
                total_charge = read_total_charge_from_out(
                    Path(f"{systempath}_mbe_{levels[1]}_simplefrags/{frame}/high/_mbe_tmp/{combo_str}/frag.out")
                )

                # Δ at each level
                dE_low  = delta_scalar_for_combo(energies["dftb"],  combo)
                dE_high = delta_scalar_for_combo(energies["bhandh-aTZ"], combo)
                dG_low  = delta_vector_for_combo(grads["dftb"],  coords, combo)
                dG_high = delta_vector_for_combo(grads["bhandh-aTZ"], coords, combo)
                dMu_low  = delta_dipole_for_combo(dips["dftb"],  combo)
                dMu_high = delta_dipole_for_combo(dips["bhandh-aTZ"], combo)

                # ΔΔ = high − low, convert units
                delta_energy   = (dE_high - dE_low) * conv_eV                 # eV
                print(delta_energy*conv_kcal)
                delta_gradients = (dG_high - dG_low) * conv_eVA               # eV/A
                delta_forces    = -delta_gradients                            # eV/A
                delta_dipole    = (dMu_high - dMu_low) * conv_eA              # eA

                atomic_numbers = [get_atomic_number(s) for s in symbols]

                data["R"].append(coords_C)
                data["Q"].append(total_charge)
                data["D"].append(delta_dipole)
                data["E"].append(delta_energy)
                data["F"].append(delta_forces)
                data["Z"].append(atomic_numbers)
                data["N"].append(len(atomic_numbers))

        # Find max atoms across all structures in this order
        max_atoms = max(data["N"]) if data["N"] else 0
        
        # Pad R, F, and Z to (num_data, max_atoms, ...)
        data["R"] = np.array([np.vstack([r, np.zeros((max_atoms - len(r), 3))]) for r in data["R"]])
        data["F"] = np.array([np.vstack([f, np.zeros((max_atoms - len(f), 3))]) for f in data["F"]])
        data["Z"] = np.array([z + [0] * (max_atoms - len(z)) for z in data["Z"]])
        
        # Convert other fields to numpy arrays
        data["Q"] = np.array(data["Q"])
        data["D"] = np.array(data["D"])
        data["E"] = np.array(data["E"])
        data["N"] = np.array(data["N"])

        print("Shapes -> R:", np.array(data['R']).shape, "Q:", np.array(data['Q']).shape,
              "D:", np.array(data['D']).shape, "E:", np.array(data['E']).shape,
              "F:", np.array(data['F']).shape, "Z:", np.array(data['Z']).shape)
        np.savez_compressed(f"{system}_order_{order}.npz", **data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Format .npz training data for delta-MBE PhysNet models."
    )
    parser.add_argument("system",
                        help="System name (e.g. h2o21, ohmh2o20)")
    parser.add_argument("systempath",
                        help="Base path for the system data directory (without trailing _mbe_... suffix)")
    parser.add_argument("--frames", nargs="+", type=int, required=True,
                        help="Frame indices to process (e.g. --frames 0 2 4 6)")
    parser.add_argument("--n-frags", type=int, default=7,
                        help="Number of fragments (default: 7)")
    parser.add_argument("--frame-suffix", default="",
                        help="Optional suffix appended after the frame index in directory names "
                             "(e.g. --frame-suffix ohm_opt gives frame_001_ohm_opt)")
    args = parser.parse_args()

    print(f"Processing {args.system}...")
    main(args.system, args.systempath, args.frames,
         n_frag=args.n_frags, frame_suffix=args.frame_suffix)

            