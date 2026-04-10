# NV Experiment Machine Profile

**Hostname:** `2u1g-b650-0079`  
**Motherboard:** ASRockRack B650D4U (v4.01)  
**OS:** Ubuntu 25.10 (Questing Quokka)  
**Kernel:** `6.17.0-20-generic`  
**User home:** `/localhome/local-pvellaisamy`  
**Project path:** `/localhome/local-pvellaisamy/prabhu/SODA---System-Offload-Dynamics-Analyzer-NV`

---

## CPU

| Property         | Value                              |
|------------------|------------------------------------|
| Model            | AMD Ryzen 7 7800X3D 8-Core Processor |
| Architecture     | x86_64 (Zen 4, 3D V-Cache)        |
| Cores / Threads  | 8 cores / 16 threads (SMT)         |
| Sockets          | 1                                  |
| Base / Max clock | 426 MHz – 5053 MHz                 |
| L1d cache        | 256 KiB (8 × 32 KiB)              |
| L1i cache        | 256 KiB (8 × 32 KiB)              |
| L2 cache         | 8 MiB (8 × 1 MiB)                 |
| L3 cache         | 96 MiB (1 instance — 3D V-Cache)  |
| NUMA nodes       | 1 (CPUs 0–15)                      |

---

## RAM

| Property   | Value    |
|------------|----------|
| Total      | 128 GiB  |
| Available  | ~119 GiB |
| Swap       | None     |

---

## Storage

| Mount | Filesystem      | Total | Used | Available |
|-------|-----------------|-------|------|-----------|
| `/`   | `/dev/nvme0n1p2`| 2.9 TB| 28 GB | 2.8 TB  |

---

## GPU

| Property          | Value                          |
|-------------------|--------------------------------|
| Model             | NVIDIA GeForce RTX 5050        |
| Architecture      | Blackwell (GB207)              |
| PCI slot          | `01:00.0`                      |
| VRAM              | 8151 MiB (~8 GB)               |
| Compute Capability| 12.0                           |
| SM clock (max)    | 3090 MHz                       |
| Memory clock (max)| 10001 MHz                      |
| TDP / Power limit | 130 W                          |
| NVIDIA driver     | 590.48.01                      |
| CUDA (driver)     | 13.1                           |

> **Note:** The nvidia kernel module must be loaded before use.
> It auto-loads after first `modprobe nvidia` per session (persistent across reboots via DKMS).
> Verify with: `nvidia-smi`

---

## Software Environment

### System Python
| Property | Value          |
|----------|----------------|
| Python   | 3.13.7         |
| Path     | `/usr/bin/python3` |
| nvcc     | Not installed (CUDA toolkit is inside conda env) |

### Conda
| Property        | Value                                         |
|-----------------|-----------------------------------------------|
| Miniconda path  | `/localhome/local-pvellaisamy/miniconda3`     |
| Conda version   | 26.1.1                                        |
| Python (base)   | 3.13.12                                       |
| Solver          | libmamba (default)                            |
| Envs directory  | `/localhome/local-pvellaisamy/miniconda3/envs`|

### Conda Environments

#### `base`
Default Miniconda base. Python 3.13.12. Not used for experiments.

#### `soda-311` ← primary experiment env
| Package              | Version    |
|----------------------|------------|
| Python               | 3.11.15    |
| PyTorch              | 2.11.0     |
| CUDA (torch runtime) | 13.0       |
| Triton               | 3.6.0      |
| Transformers         | 5.5.3      |
| Accelerate           | 1.13.0     |
| NumPy                | 2.4.4      |
| Matplotlib           | 3.10.8     |
| tqdm                 | 4.67.3     |
| cuda-toolkit         | 13.0.2     |
| cuda-bindings        | 13.2.0     |
| cuda-pathfinder      | 1.5.2      |
| nvidia-cuda-runtime  | 13.0.96    |
| nvidia-cuda-nvrtc    | 13.0.88    |
| nvidia-cuda-cupti    | 13.0.85    |
| soda                 | 0.1.0 (editable install from project root) |

**PyTorch CUDA check:**
```
PyTorch CUDA available: True
CUDA device: NVIDIA GeForce RTX 5050
CUDA version (torch): 13.0
```

---

## Quick Setup (Fresh Machine)

### 1. Install NVIDIA Driver
```bash
sudo apt install -y nvidia-driver-590-open
sudo modprobe nvidia
nvidia-smi   # verify
```

### 2. Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 3. Recreate `soda-311` Environment
```bash
conda create -n soda-311 python=3.11 -y
conda activate soda-311
pip install torch==2.11.0 triton==3.6.0 transformers==5.5.3 accelerate==1.13.0 \
    numpy==2.4.4 matplotlib==3.10.8 tqdm==4.67.3 \
    cuda-python cuda-toolkit nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cuda-cupti
pip install -e /path/to/SODA---System-Offload-Dynamics-Analyzer-NV
```

### 4. Set HuggingFace Token
```bash
# Store token in ~/.hf_token (auto-loaded by env.sh on every source)
echo "hf_YOUR_TOKEN_HERE" > ~/.hf_token
chmod 600 ~/.hf_token
```

Required for gated models (Llama-3.x). Also accept the license on
[huggingface.co/meta-llama](https://huggingface.co/meta-llama) with your HF account.

### 5. Verify GPU + CUDA
```bash
nvidia-smi
conda activate soda-311
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Sourcing / Session Setup

Every new terminal session needs these three steps before running any sweep:

```bash
# 1. Clear any stale HF_HOME that may be set from a previous/system shell
#    (env.sh sets it unconditionally, but an exported value in ~/.bashrc can
#    survive across shells — unset it first to be safe)
unset HF_HOME

# 2. Source SODA environment (sets SODA_ENV_LOADED, PYTHONPATH, HF_HOME, HF_TOKEN, etc.)
source env.sh

# 3. Activate the experiment conda env
conda activate soda-311
```

Verify the environment is clean before running:
```bash
echo "HF_HOME  = $HF_HOME"       # must be /localhome/local-pvellaisamy/.cache/huggingface
echo "HF_TOKEN = ${HF_TOKEN:0:8}..."  # must start with hf_
echo "SODA_ENV_LOADED = $SODA_ENV_LOADED"  # must be 1
nvidia-smi                        # must show RTX 5050
```

### Known Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `PermissionError at /scratch` | `HF_HOME` still set to old `/scratch/...` value | `unset HF_HOME && source env.sh` |
| `nvidia-smi: failed to communicate` | `nvidia` kernel module not loaded | `sudo modprobe nvidia` |
| `SODA environment not loaded` | `env.sh` not sourced | `source env.sh` |
| `OSError: gated repo` (Llama) | HF token missing or license not accepted | Set `~/.hf_token`; accept license on HF |

---

## NICC CLI (NVIDIA Intelligent Coding Companion)

NICC installs NVCARPS coding rules and agent skills into `.cursor/rules/` and `.cursor/skills/`.

### This machine (NVIDIA farm / VNC)

NICC is preinstalled at `/home/nv/utils/nicc-cli/latest`. `env.sh` adds it to `PATH` automatically — no extra install needed.

### One-time setup (run once per machine, once per repo)

```bash
source env.sh           # adds nicc to PATH
bash scripts/setup_nicc.sh
```

The script:
1. Detects farm vs local machine and installs nicc if needed
2. Runs `nicc login` (NVIDIA SSO) — stores token in `~/.nicc-cli/`
3. Pulls Python + CUDA rules → `.cursor/rules/`
4. Pulls all available skills → `.cursor/skills/`

After running, **restart Cursor** so it re-scans the new rules and skills.

### On a new machine

```bash
# Clone repo and enter it
git clone <repo-url>
cd SODA---System-Offload-Dynamics-Analyzer-NV

# Source env (adds nicc to PATH — farm or ~/.local/bin)
source env.sh

# Run setup (detects farm vs local, installs if needed, logs in, pulls rules+skills)
bash scripts/setup_nicc.sh

# Re-authenticate (token is personal — never committed to the repo)
nicc login
```

**Local machine (non-farm) pre-requisites:**
- Node.js 18+ (`node --version`)
- Access to NVIDIA Artifactory (`it-aws-artifactory-users` group)
- First-time npm login: username = NVIDIA AD name (not full email)

### Common commands

```bash
nicc --version                                        # check version
nicc login                                            # re-authenticate
nicc rules list python --all --target cursor --location project   # refresh Python rules
nicc skills list --all --target cursor --location project         # refresh all skills
nicc skills check-updates                             # check for skill updates
```

### Installed skills (`.cursor/skills/`)

| Skill | Purpose |
|---|---|
| `nsys-diag` | Nsight Systems diagnostics — directly relevant to SODA trace analysis |
| `nvtx-engine-analysis` | NVTX engine trace analysis |
| `cuda-trace-validate` | CUDA trace validation |
| `perf-auditor` | Performance auditing |
| `smart-profiler` | Smart profiling guidance |
| `code-review` | Structured code review |
| `dead-code-detector` | Unused code detection |
| `dependency-auditor` | Dependency security + license audit |
| `coding-pipeline` | Structured 6-phase coding orchestrator |
| `adversarial-plan-review` | Adversarial plan reviewer before implementation |
| `github-bug-investigation` | Bug investigation workflow |

> **Note on `--all`:** Do NOT use `nicc skills list --all` — it pulls 800+ skills including hardware/VLSI/Perforce skills that are inaccessible on this machine and produce hundreds of `⚠ Skill not found` errors. The setup script uses `nicc skills pull <name>` with a curated list instead.

### Notes for Agents

- `nicc` is on `PATH` after `source env.sh` — no extra steps needed on this machine
- Token lives in `~/.nicc-cli/` — personal, not in the repo, must re-auth on each new machine
- On VNC/headless: if browser can't reach SSO callback, copy the final URL from the browser and paste it into the terminal when prompted
- `.cursor/rules/` and `.cursor/skills/` **are** committed to the repo — rules/skills are shared, tokens are not
- Never use `nicc skills list --all` — use `nicc skills pull <name>` for targeted installs

---

## Running the NV Sweep

```bash
# Full prefill sweep — all models
bash experiments/nv/run_nv_sweep.sh

# Decode sweep
bash experiments/nv/run_nv_sweep.sh --mode decode

# Specific models
bash experiments/nv/run_nv_sweep.sh --models gpt2_small,gpt2_medium

# Quick debug run
bash experiments/nv/run_nv_sweep.sh \
    --models gpt2_small \
    --batch-sizes 1,2 --seq-lens 128,256 \
    --warmup 2 --runs 5
```

Available model keys: `gpt2_small`, `gpt2_medium`, `gpt2_large`, `gpt2_xl`,
`llama_3_2_1b`, `llama_3_2_3b`, `llama_3_1_8b`, `llama_3_1_70b`

Results land in `output/` at repo root (one subdirectory per model).

---

## Notes for Agents

- Always run `unset HF_HOME && source env.sh && conda activate soda-311` at the start of every session
- `HF_HOME` is set **unconditionally** in `env.sh` to `$HOME/.cache/huggingface` — `/scratch` does not exist on this machine
- HF token is read from `~/.hf_token` by `env.sh`; required for Llama models
- CUDA toolkit is **inside the conda env**, not system-wide (`nvcc` not on system PATH)
- No swap configured — OOM will hard-kill processes; the sweep driver writes OOM reports and continues
- NVMe root drive has 2.8 TB free — safe for large checkpoint/dataset storage
- GPU is Blackwell (compute cap 12.0) — requires driver ≥ 570 and recent PyTorch/Triton builds
- Reboot is not required after `modprobe nvidia`; DKMS ensures module persists across reboots
