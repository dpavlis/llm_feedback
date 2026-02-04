#!/bin/bash
# =============================================================================
# CloverDX LLM Chat - Installation Script
# =============================================================================
# This script sets up the Python virtual environment and installs dependencies.
#
# Usage:
#   ./install.sh          # Install with default Python
#   ./install.sh python3.11  # Install with specific Python version
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
PYTHON_CMD="${1:-python3}"

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         CloverDX LLM Chat - Installation Script              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if Python is available
echo -e "${YELLOW}Checking Python installation...${NC}"
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo -e "${RED}Error: $PYTHON_CMD not found. Please install Python 3.9+ first.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$("$PYTHON_CMD" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$("$PYTHON_CMD" -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$("$PYTHON_CMD" -c 'import sys; print(sys.version_info.minor)')

echo -e "Found Python ${GREEN}$PYTHON_VERSION${NC}"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}Error: Python 3.9 or higher is required. Found $PYTHON_VERSION${NC}"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}Error: $REQUIREMENTS_FILE not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at ./$VENV_DIR${NC}"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
        echo -e "${GREEN}Creating new virtual environment...${NC}"
        "$PYTHON_CMD" -m venv "$VENV_DIR"
    fi
else
    echo -e "${GREEN}Creating virtual environment...${NC}"
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${GREEN}Installing dependencies from $REQUIREMENTS_FILE...${NC}"
pip install -r "$REQUIREMENTS_FILE"

# Check if CUDA is available (optional)
echo -e "${YELLOW}Checking GPU availability...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}')" 2>/dev/null || echo "Could not check GPU (torch may still be installing)"

# Create .env file if it doesn't exist
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo -e "${GREEN}Creating .env file from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env to configure your settings.${NC}"
fi

# Create data directory
mkdir -p data/conversations

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 Installation Complete!                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  To activate the environment:                                ║"
echo "║    source venv/bin/activate                                  ║"
echo "║                                                              ║"
echo "║  To configure settings:                                      ║"
echo "║    Edit .env file                                            ║"
echo "║                                                              ║"
echo "║  To start the server:                                        ║"
echo "║    python run.py                                             ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
