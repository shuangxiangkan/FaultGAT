# FaultGAT: Graph Attention-Based Approach for Intermittent Fault Diagnosis in Multiprocessor Systems

This repository contains the implementation of FaultGAT, Graph Attention-Based Approach for Intermittent Fault Diagnosis in Multiprocessor Systems

## 🚀 Quick Start

### Requirements

- **Python 3.10.x** (Strictly required - other versions will not work)
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0.0

### Installation

⚠️ **IMPORTANT**: Please follow these installation steps exactly as shown below. The core FaultGAT model is currently provided as compiled bytecode (`FaultGAT.pyc`) which requires precise Python version matching.

```bash
# Clone the repository
git clone https://github.com/shuangxiangkan/FaultGAT.git
cd FaultGAT

# Create virtual environment with Python 3.10 (REQUIRED)
python3.10 -m venv pyg_env
source pyg_env/bin/activate  # On Windows: pyg_env\Scripts\activate

# Verify Python version
python --version  # Should show Python 3.10.x

# Install dependencies
pip install -r requirements.txt
```

## 📋 Usage

### Running Experiments

```bash
# Research Question 3: Theoretical diagnosability
python RQ1.py

# Research Question 1: Fault ratio comparison 
python RQ2.py

# Research Question 2: Partial symptom analysis
python RQ3.py
```

### Available Graph Types

- `bc`: BC (Hypercube) Network
- `augmented_k_ary_n_cube`: Augmented K-ary N-cube

## ⚠️ Important Notes

### Core Model Protection

The `FaultGAT` model implementation is currently provided as compiled bytecode (`FaultGAT.pyc`) for intellectual property protection. **The full source code will be open-sourced after paper acceptance.**

- **Current Status**: Available as `FaultGAT.pyc` (compiled bytecode)
- **Python Version**: Requires exactly **Python 3.10.x** (bytecode is version-specific)
- **Future Release**: Complete source code (`FaultGAT.py`) will be available after paper publication
- **No Fallback**: Other Python versions are not supported with current bytecode

### Version Compatibility

If you encounter import errors:

```python
RuntimeError: Cannot load FaultGAT model!
Your Python version: X.Y
Supported versions: 3.10
```

**Solutions:**
1. Use Python 3.10: `pyenv install 3.10.12 && pyenv local 3.10.12`
2. Use conda: `conda create -n faultgat python=3.10`
3. Contact authors for source code access


