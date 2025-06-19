# FaultGAT: Graph Attention-Based Approach for Intermittent Fault Diagnosis in Multiprocessor Systems

This repository contains the implementation of FaultGAT, Graph Attention-Based Approach for Intermittent Fault Diagnosis in Multiprocessor Systems

## 🚀 Quick Start

### Option 1: Docker (Recommended) 🐳

**Easiest way - No Python version worries!**

```bash
# Clone the repository
git clone https://github.com/shuangxiangkan/FaultGAT.git
cd FaultGAT

# Pull the pre-built image (when available)
docker pull ksx/faultgat:latest

# Or build locally
./build_docker.sh

# Run experiments using docker-compose
docker-compose run rq1  # Theoretical diagnosability
docker-compose run rq2  # Fault ratio comparison  
docker-compose run rq3  # Partial symptom analysis

# Or run interactively
docker run -it ksx/faultgat:latest bash
```

### Option 2: Local Installation

⚠️ **IMPORTANT**: Requires exact Python version due to compiled bytecode (`FaultGAT.pyc`)

**Requirements:**
- **Python 3.10.x** (Strictly required - other versions will not work)
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0.0

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

### Docker Usage (Recommended)

**Method 1: Using docker-compose (Easiest)**
```bash
# Run experiments with docker-compose
docker-compose run rq1  # Research Question 1: Theoretical diagnosability
docker-compose run rq2  # Research Question 2: Fault ratio comparison
docker-compose run rq3  # Research Question 3: Partial symptom analysis
```

**Method 2: Using Docker directly**
```bash
# Create result directories first
mkdir -p results_RQ1 results_RQ2 results_RQ3

# Run experiments
docker run --rm -v $(pwd)/results_RQ1:/app/results_RQ1 ksx/faultgat:latest python RQ1.py
docker run --rm -v $(pwd)/results_RQ2:/app/results_RQ2 ksx/faultgat:latest python RQ2.py
docker run --rm -v $(pwd)/results_RQ3:/app/results_RQ3 ksx/faultgat:latest python RQ3.py
```

**Interactive mode (for debugging):**
```bash
docker run -it ksx/faultgat:latest bash
# Inside container: python RQ1.py
```

Results will be saved to `results_RQ*/` directories on your host machine.

### Local Python Usage

```bash
# Research Question 1: Theoretical diagnosability
python RQ1.py

# Research Question 2: Fault ratio comparison 
python RQ2.py

# Research Question 3: Partial symptom analysis
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


