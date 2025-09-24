# EmbodiedBench Development Instructions

**ALWAYS follow these instructions first and fallback to additional search and context gathering only when the information here is incomplete or found to be in error.**

EmbodiedBench is a comprehensive benchmark for Multi-modal Large Language Models (MLLMs) as embodied agents. It evaluates models across four environments: EB-Alfred, EB-Habitat, EB-Navigation, and EB-Manipulation with 1,128 testing tasks.

## Working Effectively

### **CRITICAL: Installation Environment Requirements**
**WARNING: Pip installs consistently fail due to network timeouts. Conda environments work but pip dependencies require manual retry handling.**

### Bootstrap and Build Commands (NEVER CANCEL - Measured Timings)

**Step 1: Create Conda Environments**
- `conda env create -f conda_envs/environment.yaml` -- **FAILS due to pip timeout issues**. Create conda-only environment instead:
- `conda env create -f conda_envs/environment_conda_only.yaml` -- takes 7 seconds. NEVER CANCEL.
- `conda env create -f conda_envs/environment_eb-nav.yaml` -- **FAILS due to pip timeout issues**
- `conda env create -f conda_envs/environment_eb-man.yaml` -- **FAILS due to pip timeout issues**

**Step 2: Manual pip Installation (NETWORK ISSUES)**
**WARNING: All pip installs fail with ReadTimeoutError after 5+ minutes. Network limitations in CI environments.**
```bash
conda activate embench
pip install --timeout=600 --retries=5 -r requirements.txt  # FAILS - network timeout
```

**Step 3: Basic Package Installation (WORKS)**
```bash
conda activate embench
pip install -e .  # takes 5 seconds - WORKS
```

**Step 4: Git LFS Setup (WORKS)**
```bash
git lfs install  # takes 1 second - WORKS
git lfs pull     # timing depends on dataset size
```

### **Essential Commands That Work**
- `conda env create -f conda_envs/environment_conda_only.yaml` -- 7 seconds
- `conda activate embench`
- `pip install -e .` -- 5 seconds
- `git lfs install` -- 1 second

### **Commands That FAIL (Network Issues)**
- `bash install.sh` -- FAILS after 5 minutes due to pip timeout
- `conda env create -f conda_envs/environment.yaml` -- FAILS due to pip dependencies
- `pip install hydra-core` -- FAILS with ReadTimeoutError
- Any pip install of external packages

### Build and Test Timing Expectations
- **Conda environment creation (conda-only)**: 7 seconds - NEVER CANCEL
- **Pip dependencies installation**: **FAILS consistently** - network timeout after 5+ minutes
- **Local package installation**: 5 seconds - NEVER CANCEL
- **Git LFS operations**: varies by dataset size, can be 10+ minutes - NEVER CANCEL

## Environment Setup and Validation

### **Three Required Conda Environments**
1. **embench** - for EB-Alfred and EB-Habitat
2. **embench_nav** - for EB-Navigation  
3. **embench_man** - for EB-Manipulation

### **X Server Setup (Headless Display)**
```bash
# Start headless X server - REQUIRED for all environments
python -m embodiedbench.envs.eb_alfred.scripts.startx 1
export DISPLAY=:1
```

### **Dataset Requirements (Large Downloads)**
**WARNING: Requires Git LFS for large datasets**
- EB-Alfred dataset: `git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED`
- EB-Habitat dataset: YCB and ReplicaCAD via `habitat_sim.utils.datasets_download`
- EB-Manipulation dataset: CoppeliaSim + `git clone https://huggingface.co/datasets/EmbodiedBench/EB-Manipulation`

## Validation and Testing

### **Manual Validation Requirements**
**CRITICAL: Without full pip dependencies, main functionality cannot be tested**

**What WORKS in basic conda environment:**
- `import embodiedbench` - basic package import
- `pip install -e .` - local package installation
- Git LFS operations
- X server components (Xvfb available)

**What FAILS without pip dependencies:**
- `import embodiedbench.main` - requires hydra-core (fails with ModuleNotFoundError)
- `python -m embodiedbench.main` - requires full pip dependencies
- All model evaluation commands
- Environment-specific functionality

### **Network Limitations Documentation**
**CRITICAL: This environment has severe network limitations:**
- All pip installs to PyPI fail with ReadTimeoutError after 5+ minutes
- HTTPSConnectionPool timeouts consistently occur
- Retries and increased timeouts do not resolve the issue
- Only local package installations work

## Common Tasks and File Locations

### **Repository Structure**
```
EmbodiedBench/
├── embodiedbench/           # Main package
│   ├── envs/               # Environment implementations
│   │   ├── eb_alfred/      # Alfred environment
│   │   ├── eb_habitat/     # Habitat environment  
│   │   ├── eb_navigation/  # Navigation environment
│   │   └── eb_manipulation/ # Manipulation environment
│   ├── evaluator/          # Evaluation logic
│   ├── planner/            # Planning components
│   └── main.py             # Main entry point
├── conda_envs/             # Conda environment definitions
├── install.sh              # Installation script (FAILS due to network)
└── setup.py                # Package setup
```

### **Configuration Files**
- `embodiedbench/configs/` - Main configuration files
- `embodiedbench/envs/*/config/` - Environment-specific configs

### **Key Commands for Development**
```bash
# Environment activation
conda activate embench      # for EB-Alfred, EB-Habitat
conda activate embench_nav  # for EB-Navigation  
conda activate embench_man  # for EB-Manipulation

# Model evaluation (REQUIRES FULL PIP DEPENDENCIES)
python -m embodiedbench.main env=eb-alf model_name=gpt-4o-mini exp_name='baseline'
python -m embodiedbench.main env=eb-hab model_name=gpt-4o-mini exp_name='baseline'
python -m embodiedbench.main env=eb-nav model_name=gpt-4o exp_name='baseline'
python -m embodiedbench.main env=eb-man model_name=claude-3-5-sonnet-20241022 exp_name='baseline'
```

## Environment-Specific Setup

### **EB-Alfred**
- Requires: ai2thor, headless X server
- Dataset: `git clone https://huggingface.co/datasets/EmbodiedBench/EB-ALFRED`
- Test: `python -m embodiedbench.envs.eb_alfred.EBAlfEnv` (REQUIRES pip dependencies)

### **EB-Habitat** 
- Requires: habitat-sim==0.3.0, habitat-lab
- Installation: `conda install -y habitat-sim==0.3.0 withbullet headless -c conda-forge -c aihabitat`
- Dataset: `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`

### **EB-Navigation**
- Uses embench_nav environment
- Test: `python -m embodiedbench.envs.eb_navigation.EBNavEnv` (REQUIRES pip dependencies)

### **EB-Manipulation**
- Requires: CoppeliaSim V4.1.0, PyRep
- CoppeliaSim download: `wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz`
- Environment variables required:
  ```bash
  export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
  export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
  ```

## Troubleshooting

### **Network Issues**
- **Symptom**: ReadTimeoutError, pip install failures
- **Cause**: Network firewall/timeout limitations in CI environment
- **Workaround**: Use pre-installed conda packages only, manual dependency installation required

### **X Server Issues**
- **Setup**: Always run `python -m embodiedbench.envs.eb_alfred.scripts.startx 1` before testing
- **Display**: Set `export DISPLAY=:1`
- **Components**: Xvfb is available at `/usr/bin/Xvfb`

### **Import Errors**
- **Missing hydra**: Install fails due to network - limits functionality testing
- **Missing ai2thor**: Install fails due to network - EB-Alfred cannot be tested
- **Missing torch**: Install fails due to network - model functionality unavailable

## Development Workflow

1. **Always start with basic conda environment creation** (works reliably)
2. **Install local package** with `pip install -e .` (works)  
3. **Set up Git LFS** for dataset access (works)
4. **Document network limitations** - pip dependencies must be installed in working environments
5. **Test only what works** - avoid running commands that require external pip packages
6. **Use environment variables** for configuration rather than relying on full installs

**Remember: This environment has severe network limitations. Full functionality requires working pip installs of 200+ Python packages.**