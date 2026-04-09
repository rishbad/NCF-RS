"""
=============================================================
 NCF Replication — Setup Script
 Run this ONCE before anything else to verify your environment
=============================================================

Usage:
  python setup.py
"""

import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

print("="*55)
print(" NCF Replication — Environment Setup")
print("="*55)

# ── Install required packages ────────────────────────────────
required = {
    'torch'     : 'torch',
    'numpy'     : 'numpy',
    'pandas'    : 'pandas',
    'matplotlib': 'matplotlib',
    'sklearn'   : 'scikit-learn',
    'tqdm'      : 'tqdm',
}

print("\n[1] Checking/installing packages...")
for import_name, pip_name in required.items():
    try:
        __import__(import_name)
        print(f"  ✅ {pip_name} already installed")
    except ImportError:
        print(f"  ⏳ Installing {pip_name}...")
        install(pip_name)
        print(f"  ✅ {pip_name} installed")

# ── Create directory structure ───────────────────────────────
print("\n[2] Creating project directories...")
dirs = [
    'data/raw',
    'data/processed',
    'models',
    'utils',
    'results',
    'plots',
]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"  📁 {d}/")

# ── Check raw data files ─────────────────────────────────────
print("\n[3] Checking raw data files...")
expected_files = {
    'data/raw/ratings.dat'             : 'MovieLens 1M ratings',
    'data/raw/Pinterest-posts.csv'     : 'Pinterest posts',
    'data/raw/Pinterest-profiles.csv'  : 'Pinterest profiles',
}
for path, desc in expected_files.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"  ✅ {path} ({size:.1f} MB) — {desc}")
    else:
        print(f"  ❌ MISSING: {path} — {desc}")
        print(f"     → Copy this file into data/raw/ before running main.py")

# ── PyTorch GPU check ────────────────────────────────────────
print("\n[4] PyTorch GPU check...")
try:
    import torch
    print(f"  PyTorch version : {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"     GPU memory     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("  ⚠️  CUDA not available — will run on CPU (slower but works)")
        print("     Estimated runtime on CPU: ~20–40 min")
except ImportError:
    print("  ❌ PyTorch not found — please install with:")
    print("     pip install torch")

print("\n" + "="*55)
print(" Setup complete! Next step:")
print("   python data/data_preprocessing.py")
print("   python main.py")
print("="*55)
