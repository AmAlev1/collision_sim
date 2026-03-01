"""
main.py

Entry point for the collision-time simulation project.

This script would load the experiment configuration, run the Monte Carlo
collision experiments, Generates analysis plots, then saves the results to disk.

All parameters are defined in config.py.
All computation logic is contained in experiment.py.
All visualization logic is contained in analysis.py.
"""

import time

import config
from experiment import run_experiment
from analysis import run_analysis


def main() -> None:
    """
    Execute full experiment and analysis pipeline.
    """

    print("=" * 60)
    print("Collision-Time Simulation")
    print("=" * 60)

    print(f"Truncation levels: {config.N_VALUES}")
    print(f"Trials per configuration: {config.TRIALS_PER_CONFIG}")
    print(f"Random seed: {config.RNG_SEED}")
    print()

    start_time = time.time()

    # ---------------------------
    # Run Monte Carlo Experiments
    # ---------------------------

    print("Running experiments...")
    results = run_experiment(
        n_values=config.N_VALUES,
        trials_per_config=config.TRIALS_PER_CONFIG,
        rng_seed=config.RNG_SEED,
        byte_length=config.BYTE_LENGTH,
        k_values_by_n=config.K_VALUES_BY_N,
        skew_probabilities=config.SKEW_PROBABILITIES,
    )

    print("Experiments complete.")
    print(f"Total configurations executed: {len(results)}")
    print()

    # -------------------------------
    # Run Analysis and Generate Plots
    # --------------------------------

    print("Generating analysis plots...")
    run_analysis(
        results,
        save_directory=config.OUTPUT_DIRECTORY,
        show_plots=False,
    )

    print(f"Plots saved to '{config.OUTPUT_DIRECTORY}/'")
    print()

    end_time = time.time()
    runtime = end_time - start_time

    print(f"Total runtime: {runtime:.2f} seconds")
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
