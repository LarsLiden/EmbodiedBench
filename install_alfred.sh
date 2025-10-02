#!/bin/bash
set -e  # Exit on any error

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"
export EMBODIED_BENCH_ROOT=$(pwd)

echo "Setting up EmbodiedBench environments..."
echo "Current directory: $(pwd)"

# Environment for Habitat and Alfred
echo "Creating embench environment..."
conda env create -f conda_envs/environment.yaml
conda activate embench
pip install -e .

# Install Git LFS
echo "Installing Git LFS..."
conda activate embench
conda install -y -c conda-forge git-lfs
git lfs install
# Only pull LFS files if we're in a git repository
if git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Git repository detected, pulling LFS files..."
    git lfs pull
else
    echo "Not in a Git repository, skipping git lfs pull"
fi

# Install EB-ALFRED
echo "Installing EB-ALFRED dataset..."
conda activate embench
git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0

echo "Alfred Installation completed successfully!"
