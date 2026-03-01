"""
config.py

Centralized configuration for collision-time simulation.

This file defines all experimental parameters in one location to ensure:
- Reproducibility
- Transparency
- Clean separation from logic
- Ease of modification

All parameters were selected to balance statistical stability
with reasonable runtime for demonstration purposes.
"""

# ---------------------------
# Global Experiment Settings
# -----------------------------

# Number of Monte Carlo trials per configuration.
# 300 provides stable averages without excessive runtime.
TRIALS_PER_CONFIG = 300

# Fixed seed for reproducibility.
RNG_SEED = 42

# Fixed input message length in bytes.
# 32 bytes (256 bits) aligns naturally with SHA-256 structure.
BYTE_LENGTH = 32


# --------------------------
# Hash Truncation Parameters
# ---------------------------

# Truncation sizes (n_bits) to evaluate.
# Limited to <= 20 bits to ensure runtime feasibility.
N_VALUES = [8, 10, 12, 14, 16, 18, 20]


# =------------------------------
# Reduced-Support Entropy Mapping
# -------------------------------

# Explicit entropy sweeps for each truncation level.
# This makes the experimental design fully transparent.
K_VALUES_BY_N = {
    8:  [8, 6, 4],
    10: [10, 8, 6],
    12: [12, 10, 8],
    14: [14, 12, 10],
    16: [16, 14, 12],
    18: [18, 16, 14],
    20: [20, 18, 16],
}


# --------------------------------
# Skewed Distribution Parameters
# --------------------------------

# Heavy-point probabilities used in skewed models.
# These values illustrate increasing probability concentration
# while keeping distributions non-degenerate.
SKEW_PROBABILITIES = [0.1, 0.2, 0.4]


# ----------------
# Output Settings
# ---------------

# Directory where analysis plots will be saved.
OUTPUT_DIRECTORY = "figures"
