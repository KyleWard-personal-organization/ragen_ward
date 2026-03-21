import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Log directory
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# Data directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Models directory
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Checkpoints directory
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
