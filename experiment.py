"""
experiment.py

Purpose
-------
Run Monte Carlo collision-time experiments across parameter configurations and
return structured results suitable for later analysis and plotting.

This module orchestrates:
- Parameter sweeps over truncation size (n_bits)
- Controlled input distribution regimes (uniform, reduced support, skewed)
- Repeated independent trials per configuration
- Summary statistics (mean, variance, standard error)

Design Principles
---------------
- No plotting and no file I/O (handled by analysis.py / main.py).
- No theoretical curve computation (handled by analysis.py).
- Reproducibility via injected, seeded RNG.
- Results returned as a list of record-like dataclasses for easy filtering,
  grouping, and plotting.

Notes on RNG 
------------------
A singke random.Random(seed) instance is used to generate and pass the variable into models. This ensures
full reproducibility. Independence between samples is preserved because each call
to sample() draws new random results from the RNG. The RNG stream is
continuous, but trials remain statistically well-defined under the model (and
remain reproducible).
"""

from __future__ import annotations

import math
import statistics
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from hash_engine import SHA256Truncated
from input_models import UniformModel, ReducedSupportModel, SkewedModel
from collision_engine import run_single_trial


@dataclass(frozen=True)
class ExperimentResult:
    #Record of summary statistics for a single experimental configuration.
    model_name: str
    n_bits: int
    k_bits: Optional[int]
    heavy_probability: Optional[float]
    trials: int
    mean_collision_time: float
    variance: float
    std_error: float


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _run_trials(hash_engine: SHA256Truncated,
                input_model,
                trials: int) -> List[int]:
    #Run repeated independent trials for a fixed (hash_engine, input_model) pair.
    #Returns the list of stopping times (sample counts including the duplicate draw).
    _validate_positive_int("trials", trials)

    stopping_times: List[int] = []
    for _ in range(trials):
        t = run_single_trial(hash_engine=hash_engine, input_model=input_model)
        stopping_times.append(t)
    return stopping_times


def _summarize(stopping_times: Sequence[int]) -> Tuple[float, float, float]:
    """
    Compute mean, population variance, and standard error of the mean.

    Population variance (pvariance) is used for describing the variance
    of the observed stopping-time distribution under repeated sampling.
    """
    if len(stopping_times) == 0:
        raise ValueError("stopping_times must be non-empty")

    mean_t = float(statistics.mean(stopping_times))

    # For robustness: pvariance requires at least 1 data point; it returns 0 for 1 point.
    var_t = float(statistics.pvariance(stopping_times))

    # Standard error of mean (SEM) = sqrt(Var / n).
    sem_t = math.sqrt(var_t / len(stopping_times))

    return mean_t, var_t, sem_t


def run_experiment(
    n_values: Sequence[int],
    *,
    trials_per_config: int,
    rng_seed: int,
    byte_length: int = 32,
    # Reduced-support entropy levels k are typically swept relative to n.
    # If not provided, we default to k = n, n-2, n-4, ... down to max(1, n-10).
    k_values_by_n: Optional[dict[int, Sequence[int]]] = None,
    # Skewed model probabilities (kept small in number to avoid overcomplication).
    skew_probabilities: Sequence[float] = (0.1, 0.2, 0.4),
) -> List[ExperimentResult]:
    """
    Run the full experiment suite.

    Parameters
    ----------
    n_values: Truncation sizes (in bits) to evaluate.

    trials_per_config: Number of Monte Carlo trials per configuration.

    rng_seed: Seed for reproducibility.

    byte_length: Fixed message length in bytes produced by input models.

    k_values_by_n:Optional mapping from n_bits to the sequence of k_bits values to test.
                   If None, a default decreasing sequence is used for each n.

    skew_probabilities: Probabilities p for the SkewedModel heavy point.

    Returns
    -------
    List[ExperimentResult]: One record per configuration (model, n_bits, k_bits, p).
    """
    _validate_positive_int("trials_per_config", trials_per_config)
    _validate_positive_int("rng_seed", rng_seed)
    _validate_positive_int("byte_length", byte_length)

    # Validate n_values early.
    if not isinstance(n_values, (list, tuple)) and not hasattr(n_values, "__iter__"):
        raise TypeError("n_values must be an iterable of ints")

    n_values_list: List[int] = []
    for n in n_values:
        if not isinstance(n, int):
            raise TypeError(f"n_values must contain ints, got {type(n).__name__}")
        if n < 1 or n > 256:
            raise ValueError("Each n_bits must satisfy 1 <= n_bits <= 256")
        n_values_list.append(n)

    # Validate skew probabilities.
    for p in skew_probabilities:
        if not isinstance(p, (float, int)):
            raise TypeError("skew_probabilities must contain floats")
        if not (0.0 < float(p) < 1.0):
            raise ValueError("Each skew probability must satisfy 0 < p < 1")

    # Single RNG stream for reproducibility; passed into models.
    rng = random.Random(rng_seed)

    results: List[ExperimentResult] = []

    for n_bits in n_values_list:
        # Hash engine fixed per truncation length.
        hash_engine = SHA256Truncated(n_bits=n_bits)

        # Regime A: Uniform baseline
        uniform_model = UniformModel(byte_length=byte_length, rng=rng)
        uniform_times = _run_trials(hash_engine, uniform_model, trials_per_config)
        mean_t, var_t, sem_t = _summarize(uniform_times)
        results.append(
            ExperimentResult(
                model_name="uniform",
                n_bits=n_bits,
                k_bits=None,
                heavy_probability=None,
                trials=trials_per_config,
                mean_collision_time=mean_t,
                variance=var_t,
                std_error=sem_t,
            )
        )

        # Regime B: Reduced-support entropy sweep
        if k_values_by_n is not None and n_bits in k_values_by_n:
            k_values = list(k_values_by_n[n_bits])
        else:
            # Default: k = n, n-2, n-4, ... down to max(1, n-10).
            # This keeps the sweep modest while still showing clear scaling.
            k_min = max(1, n_bits - 10)
            k_values = list(range(n_bits, k_min - 1, -2))

        for k_bits in k_values:
            if not isinstance(k_bits, int):
                raise TypeError(f"k_bits must be int, got {type(k_bits).__name__}")
            if k_bits < 1:
                raise ValueError("k_bits must be >= 1")
            if k_bits > 8 * byte_length:
                raise ValueError(
                    "k_bits cannot exceed input bit-length (8*byte_length). "
                    "Increase byte_length or reduce k_bits."
                )

            reduced_model = ReducedSupportModel(k_bits=k_bits, byte_length=byte_length, rng=rng)
            reduced_times = _run_trials(hash_engine, reduced_model, trials_per_config)
            mean_t, var_t, sem_t = _summarize(reduced_times)
            results.append(
                ExperimentResult(
                    model_name="reduced_support",
                    n_bits=n_bits,
                    k_bits=k_bits,
                    heavy_probability=None,
                    trials=trials_per_config,
                    mean_collision_time=mean_t,
                    variance=var_t,
                    std_error=sem_t,
                )
            )

        # Regime C: Skewed full-support stress tests
        # Keep this optional but included by default; it adds depth without
        # exploding the parameter grid.
        for p in skew_probabilities:
            p = float(p)
            # Herethe nominal support size is tied to k_bits = n_bits for interpretability.
            # You can change this later (e.g., fix k_bits at a constant) if desired.
            k_bits = n_bits

            skewed_model = SkewedModel(
                k_bits=k_bits,
                heavy_probability=p,
                heavy_value=0,
                byte_length=byte_length,
                rng=rng,
            )
            skewed_times = _run_trials(hash_engine, skewed_model, trials_per_config)
            mean_t, var_t, sem_t = _summarize(skewed_times)
            results.append(
                ExperimentResult(
                    model_name="skewed",
                    n_bits=n_bits,
                    k_bits=k_bits,
                    heavy_probability=p,
                    trials=trials_per_config,
                    mean_collision_time=mean_t,
                    variance=var_t,
                    std_error=sem_t,
                )
            )

    return results
