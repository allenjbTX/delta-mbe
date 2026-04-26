<p align="center">
Single- and double-precision Codes for PhysNet<br>
Meuwly Group, University of Basel
</p>

# General
The present repository provides access to two implementations of the PhysNet [1] codes (single (F32)- and double (F64)-precision), which can be used to learn molecular potential energy surfaces (PESs). The resulting single- and double-precision PESs were assessed in Reference [2]. The required installation and dependencies are outlined below and are followed by examples for training and using the neural network-based PES for H<sub>2</sub>CO. Therefore, the repository also contains the ab initio reference MP2/aug-cc-pVTZ level data for H<sub>2</sub>CO [3], as well as ready-to-use models [2].


# Installations & dependencies

The following installation steps were tested on a Ubuntu 20.04 workstation and using
Conda 23.7.2 (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

a) If not installed already, install Miniconda on your machine (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

b) Create an environment named (e.g.) physnet_env, install Python 3.6:

    conda create --name physnet_env python=3.6

   Activate it:

    conda activate physnet_env

    (deactivating it by typing: conda deactivate)


c) With activated environment, all dependencies can be installed.

    pip install ase==3.19.1
    pip install tensorflow==1.12


# Examples
Training and using the single- and double-precision PESs follows the same procedure with the required adaptations made in the f32 and f64 folders and corresponding source codes.

### PhysNet training
In either of the f32 or f64 folders, training can be started with activated conda environment by running

    ./train.py @run_ch2o_mp2.inp
    
Once the training is converged, the models can be extracted from the "best/" folder, which can be found in a newly creaded folder with a timestamp.

### Evaluation
In either of the f32/eval or f64/eval folders, scripts for exemplary evaluations are given. These can for example be used to predict the energy of a given structure in .xyz format (predict_mol.py)

    ./predict_mol.py -i h2co.xyz
    
or to optimize a given structure in .xyz format (optimize.py)

    ./optimize.py -i h2co.xyz
    
or to calculate the harmonic frequencies of an optimized molecule (ase_vibrations.py)

    ./ase_vibrations.py -i opt_h2co.xyz
    
or to calculate the energy along a stretch of the C-H bond (predict_stretch.py) - see Fig. 1 of Ref [2].

    ./predict_stretch.py -i opt_h2co.xyz


# How to cite 

When using the PhysNet or the H<sub>2</sub>CO PES, please cite the following papers:

#### For PhysNet:
Oliver T. Unke and Markus Meuwly "PhysNet: A Neural Network for Predicting Energies,
Forces, Dipole Moments, and Partial Charges", J. Chem. Theory Comput., 2019,
15, 6, 3678–3693

#### For the H<sub>2</sub>CO dataset/PES:
Silvan Käser, Debasish Koner, Anders S. Christensen, O. Anatole von Lilienfeld, and Markus Meuwly "Machine Learning Models of Vibrating H<sub>2</sub>CO: Comparing Reproducing Kernels, FCHL, and PhysNet"
J. Phys. Chem. A 2020, 124(42), 8853-8865, DOI: 10.1021/acs.jpca.0c05979

#### For the double-precision PES:
Silvan Käser and Markus Meuwly "Numerical Accuracy Matters: Applications of Machine Learned Potential Energy Surfaces", J. Phys. Chem. Lett., 2024, 15(12), 3419-3424 

# References


[1] Oliver T. Unke and Markus Meuwly "PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments, and Partial Charges", J. Chem. Theory Comput., 2019, 15, 6, 3678–3693

[2] Silvan Käser and Markus Meuwly "Numerical Accuracy Matters: Applications of Machine Learned Potential Energy Surfaces", 2023, arXiv e-prints, DOI: 10.48550/arXiv.2311.17398

[3] Silvan Käser, Debasish Koner, Anders S. Christensen, O. Anatole von Lilienfeld, and Markus Meuwly "Machine Learning Models of Vibrating H<sub>2</sub>CO: Comparing Reproducing Kernels, FCHL, and PhysNet"
J. Phys. Chem. A 2020, 124(42), 8853-8865, DOI: 10.1021/acs.jpca.0c05979

# Contact

If you have any questions about the PESs free to contact Silvan Käser (silvan.kaeser@unibas.ch) or Markus Meuwly (m.meuwly@unibas.ch)


