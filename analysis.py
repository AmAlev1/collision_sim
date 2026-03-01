"""
analysis.py

Purpose
-------
Interpret experimental collision-time results and generate
publication-quality plots comparing empirical data to
theoretical birthday-bound predictions.

This module:
- Computes theoretical collision time scaling.
- Computes relative deviation metrics.
- Produces and saves figures to disk.
- Doesn't perform experiments.
- Doesn't modify raw data.

"""

from __future__ import annotations

import math
import os
from typing import List, Dict
import matplotlib.pyplot as plt

from experiment import ExperimentResult


# Theoretical Model

def theoretical_collision_time(bits: int) -> float:
    """
    Birthday bound approximation:

    Expected collision time ≈ sqrt(pi / 2 * 2^bits)
    """
    return math.sqrt((math.pi / 2.0) * (2 ** bits))


def relative_deviation(empirical: float, theoretical: float) -> float:
    """
    Compute relative deviation:
    (empirical - theoretical) / theoretical
    """
    if theoretical == 0:
        return 0.0
    return (empirical - theoretical) / theoretical


# Data Extraction Helpers

def _group_results_by_model(results: List[ExperimentResult]) -> Dict[str, List[ExperimentResult]]:
    """
    Group experiment results by model_name.
    """
    grouped: Dict[str, List[ExperimentResult]] = {}
    for r in results:
        grouped.setdefault(r.model_name, []).append(r)
    return grouped


def _effective_entropy(result: ExperimentResult) -> int:
    """
    Determine effective entropy for plotting.

    Uniform → entropy = n_bits
    Reduced_support → entropy = k_bits
    Skewed → entropy = k_bits (nominal support)
    """
    if result.model_name == "uniform":
        return result.n_bits
    else:
        return result.k_bits if result.k_bits is not None else result.n_bits


# Plot 1: Collision Time vs Entropy

def plot_collision_vs_entropy(results: List[ExperimentResult], save_path: str):
    grouped = _group_results_by_model(results)

    plt.figure(figsize=(8, 6))

    # Plot theoretical curve for reference
    entropy_values = sorted(
        set(_effective_entropy(r) for r in results)
    )
    theoretical_vals = [theoretical_collision_time(b) for b in entropy_values]
    plt.plot(
        entropy_values,
        theoretical_vals,
        color="black",
        linewidth=2.5,
        label="Theoretical Birthday Bound"
    )

    # Plot empirical results
    style_map = {
        "uniform": {"linestyle": "-", "linewidth": 2},
        "reduced_support": {"linestyle": "--", "linewidth": 2},
        "skewed": {"linestyle": ":", "linewidth": 2},
    }

    for model_name, model_results in grouped.items():
        sorted_results = sorted(model_results, key=_effective_entropy)
        x = [_effective_entropy(r) for r in sorted_results]
        y = [r.mean_collision_time for r in sorted_results]

        plt.plot(
            x,
            y,
            label=f"{model_name}",
            **style_map.get(model_name, {})
        )

    plt.xlabel("Effective Entropy (bits)")
    plt.ylabel("Mean Collision Time")
    plt.title("Collision Time vs Effective Entropy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "collision_vs_entropy.png"))
    plt.close()


# Plot 2: Empirical vs Theoretical
def plot_empirical_vs_theoretical(results: List[ExperimentResult], save_path: str):
    plt.figure(figsize=(8, 6))

    entropy_vals = []
    empirical_vals = []
    theoretical_vals = []

    for r in results:
        entropy = _effective_entropy(r)
        entropy_vals.append(entropy)
        empirical_vals.append(r.mean_collision_time)
        theoretical_vals.append(theoretical_collision_time(entropy))

    plt.scatter(theoretical_vals, empirical_vals, alpha=0.7)
    max_val = max(max(theoretical_vals), max(empirical_vals))

    plt.plot(
        [0, max_val],
        [0, max_val],
        color="black",
        linestyle="--",
        label="Ideal Agreement"
    )

    plt.xlabel("Theoretical Collision Time")
    plt.ylabel("Empirical Mean Collision Time")
    plt.title("Empirical vs Theoretical Collision Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "empirical_vs_theoretical.png"))
    plt.close()



# Plot 3: Relative Deviation vs Entropy
def plot_relative_deviation(results: List[ExperimentResult], save_path: str):
    grouped = _group_results_by_model(results)

    plt.figure(figsize=(8, 6))

    for model_name, model_results in grouped.items():
        sorted_results = sorted(model_results, key=_effective_entropy)
        x = []
        y = []

        for r in sorted_results:
            entropy = _effective_entropy(r)
            theory = theoretical_collision_time(entropy)
            dev = relative_deviation(r.mean_collision_time, theory)
            x.append(entropy)
            y.append(dev)

        plt.plot(
            x,
            y,
            label=model_name,
            linewidth=2
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.xlabel("Effective Entropy (bits)")
    plt.ylabel("Relative Deviation")
    plt.title("Relative Deviation from Birthday Bound")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "relative_deviation.png"))
    plt.close()



# Main Entry
def run_analysis(results: List[ExperimentResult], save_directory: str = "figures", show_plots: bool = False):
    """
    Generate all plots and save to disk.

    Parameters
    ----------
    results : list of ExperimentResult
    save_directory : directory for output figures
    show_plots : whether to display plots interactively
    """

    os.makedirs(save_directory, exist_ok=True)

    plot_collision_vs_entropy(results, save_directory)
    plot_empirical_vs_theoretical(results, save_directory)
    plot_relative_deviation(results, save_directory)

    if show_plots:
        plt.show()
