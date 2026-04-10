#!/usr/bin/env bash
# =============================================================================
# setup_nicc.sh — One-time NICC CLI setup for any machine
#
# What this does:
#   1. Detects install path (NVIDIA farm vs local machine)
#   2. Installs nicc-cli via npm if not on farm
#   3. Verifies nicc is on PATH
#   4. Runs `nicc login` (SSO — browser or URL paste)
#   5. Installs Python + CUDA rules into .cursor/rules (project-scoped)
#   6. Installs all available skills into .cursor/skills (project-scoped)
#
# Usage (from repo root, after sourcing env.sh):
#   source env.sh
#   bash scripts/setup_nicc.sh
#
# Safe to re-run (idempotent). Skips steps already done.
#
# On a new machine:
#   1. Clone the repo
#   2. source env.sh
#   3. bash scripts/setup_nicc.sh
#   4. nicc login  (re-authenticate — token is personal, not in the repo)
# =============================================================================

set -euo pipefail

NICC_FARM="/home/nv/utils/nicc-cli/latest"
NICC_LOCAL="$HOME/.local/bin"
NICC_NPM_REGISTRY="https://artifactory.nvidia.com/artifactory/api/npm/hw-nicc-npm-local/"

# ---------------------------------------------------------------------------
# Step 1: Locate / install nicc
# ---------------------------------------------------------------------------
echo ""
echo "=== [1/5] Locating nicc CLI ==="

if command -v nicc &>/dev/null; then
    NICC_PATH="$(command -v nicc)"
    echo "  ✓ nicc already on PATH: $NICC_PATH"

elif [ -f "$NICC_FARM/nicc" ]; then
    echo "  Detected NVIDIA farm. Adding preinstalled nicc to PATH."
    export PATH="$NICC_FARM:$PATH"
    echo "  ✓ nicc from farm: $NICC_FARM/nicc"
    echo ""
    echo "  NOTE: env.sh already adds this automatically on future sessions."

else
    echo "  nicc not found. Installing via npm..."

    # Ensure Node.js 18+ is available
    if ! command -v node &>/dev/null; then
        echo ""
        echo "  ERROR: Node.js is required but not installed."
        echo "  Install it first:"
        echo "    # Ubuntu / Debian:"
        echo "    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -"
        echo "    sudo apt-get install -y nodejs"
        echo "    # macOS:"
        echo "    brew install node"
        exit 1
    fi

    NODE_VER=$(node --version | sed 's/v//' | cut -d. -f1)
    if [ "$NODE_VER" -lt 18 ]; then
        echo "  ERROR: Node.js 18+ required (found v$NODE_VER)."
        exit 1
    fi
    echo "  Node.js $(node --version) ✓"

    # Install to ~/.local so no sudo needed
    mkdir -p "$NICC_LOCAL"
    npm config set prefix "$HOME/.local"

    echo "  Logging into NVIDIA Artifactory npm registry..."
    echo "  (username = your NVIDIA AD name, not full email)"
    npm login --registry="$NICC_NPM_REGISTRY"

    echo "  Installing nicc-cli..."
    npm install -g nicc-cli --registry="$NICC_NPM_REGISTRY"

    export PATH="$NICC_LOCAL:$PATH"
    echo "  ✓ nicc installed to $NICC_LOCAL/nicc"
    echo ""
    echo "  NOTE: env.sh already adds ~/.local/bin to PATH on future sessions."
fi

# ---------------------------------------------------------------------------
# Step 2: Verify nicc
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/5] Verifying nicc ==="
nicc --version
echo "  ✓ nicc is working"

# ---------------------------------------------------------------------------
# Step 3: Authenticate (nicc login)
# ---------------------------------------------------------------------------
echo ""
echo "=== [3/5] Authenticating with NVCARPS ==="

TOKEN_DIR="$HOME/.nicc-cli"
if [ -d "$TOKEN_DIR" ] && [ -n "$(ls -A "$TOKEN_DIR" 2>/dev/null)" ]; then
    echo "  Token directory exists: $TOKEN_DIR"
    echo "  Skipping login (already authenticated)."
    echo "  To re-authenticate run: nicc login"
else
    echo "  No existing token found. Starting SSO login..."
    echo ""
    echo "  A browser window will open. Log in with NVIDIA SSO."
    echo "  On VNC/headless: copy the final callback URL from the browser"
    echo "  and paste it back into this terminal when prompted."
    echo ""
    nicc login
    echo "  ✓ Authenticated"
fi

# ---------------------------------------------------------------------------
# Step 4: Install rules → .cursor/rules/
# ---------------------------------------------------------------------------
echo ""
echo "=== [4/5] Installing coding rules into .cursor/rules/ ==="

if [ ! -f "pyproject.toml" ]; then
    echo "  WARNING: Not in repo root (pyproject.toml not found)."
    echo "  Rules will be installed in: $(pwd)"
fi

# Python standard rules
echo ""
echo "  → Python rules"
nicc rules list python \
    --all \
    --target cursor \
    --location project \
    || echo "  WARNING: Python rules install failed (may need login or network)"

# CUDA rules (if available)
echo ""
echo "  → CUDA rules"
nicc rules list cuda \
    --all \
    --target cursor \
    --location project \
    || echo "  (No CUDA rules found or install failed — continuing)"

echo ""
echo "  Rules installed:"
ls .cursor/rules/ 2>/dev/null || echo "  (none yet)"

# ---------------------------------------------------------------------------
# Step 5: Install skills → .cursor/skills/
# Curated set relevant to ML/systems/CUDA work.
# Using `nicc skills pull <name>` instead of --all to avoid pulling
# hardware/VLSI/Perforce skills that aren't accessible or relevant here.
# ---------------------------------------------------------------------------
echo ""
echo "=== [5/5] Installing skills into .cursor/skills/ ==="

# Skills relevant to this project (profiling, CUDA, Python, code quality)
_SKILLS=(
    "nsys-diag"               # Nsight Systems diagnostics (core SODA tooling)
    "nvtx-engine-analysis"    # NVTX engine trace analysis
    "cuda-trace-validate"     # CUDA trace validation
    "perf-auditor"            # Performance auditing
    "smart-profiler"          # Smart profiling guidance
    "code-review"             # Code review
    "dead-code-detector"      # Dead code detection
    "dependency-auditor"      # Dependency security + license audit
    "coding-pipeline"         # Structured 6-phase coding orchestrator
    "adversarial-plan-review" # Adversarial plan reviewer before implementation
    "github-bug-investigation" # Bug investigation workflow
)

for skill in "${_SKILLS[@]}"; do
    echo "  → $skill"
    nicc skills pull "$skill" \
        --target cursor \
        --location project \
        2>&1 | grep -E "Saved:|Pulled|WARNING|ERROR" || true
done

echo ""
echo "  Skills installed:"
ls .cursor/skills/ 2>/dev/null || echo "  (none yet)"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " NICC setup complete!"
echo ""
echo " What was installed:"
echo "   .cursor/rules/   — coding standard rules (Python, CUDA)"
echo "   .cursor/skills/  — agent skills (NVCARPS)"
echo ""
echo " Next steps:"
echo "   1. Restart Cursor so it re-scans .cursor/rules/ and .cursor/skills/"
echo "   2. Verify:  ls .cursor/rules  &&  ls .cursor/skills"
echo ""
echo " On a NEW machine, run:"
echo "   source env.sh"
echo "   bash scripts/setup_nicc.sh"
echo "   nicc login   # token is personal — not stored in the repo"
echo "============================================================"
