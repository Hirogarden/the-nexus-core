"""
Shared pytest configuration for The Nexus Core test suite.
Adds the repo root to sys.path so tests can import project modules directly.
"""

import sys
from pathlib import Path

# Ensure the repo root (parent of this tests/ dir) is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
