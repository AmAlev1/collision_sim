"""
collision_engine.py

Purpose
------
Provide the core Monte Carlo primitive for the collision experiment.

This module defines the logic for running a single collision trial.
When given a hash engine and an input model, this module repeatedly
sample inputs, hashes them, and stops at the first repeated truncated output.

The returned stopping time counts the total number of samples drawn,
including the duplicate that triggers the collision.

Design Principles
-----------------
- No experiment orchestration.
- No statistics aggregation.
- No plotting.
- No global state.
- Deterministic behaviour given deterministic RNG in input model.
- Clear separation between hashing, sampling, and collision detection.
"""

from __future__ import annotations

from typing import Protocol


class HashEngineProtocol(Protocol):
    """
    Minimal protocol for hash engines used in this module.

    The object must implement:
        hash(input_bytes: bytes) -> int
    """
    def hash(self, input_bytes: bytes) -> int:
        ...


class InputModelProtocol(Protocol):
    """
    Minimal protocol for input models used in this module.

    The object must implement:
        sample() -> bytes
    """
    def sample(self) -> bytes:
        ...


def run_single_trial(hash_engine: HashEngineProtocol,
                     input_model: InputModelProtocol) -> int:
    """
    Run a single collision trial.

    Parameters
    ----------
    hash_engine :
        Object providing a hash(input_bytes: bytes) -> int method.
        Typically an instance of SHA256Truncated.

    input_model :
        Object providing a sample() -> bytes method.
        Typically an instance of UniformModel, ReducedSupportModel,
        or SkewedModel.

    Returned values
    ----------------
        Stopping time: Interger value for number of samples drawn
        until the first repeated hash output is observed (including the duplicate).

    Notes
    -----
    - Each call to this function represents an independent trial.
    - Internal state (the set of seen outputs) is reset on every call.
    - Complexity per trial is O(T), where T is the stopping time.
    """

    seen_hashes = set()
    sample_count = 0

    while True:
        # Draw a fresh input sample from the model.
        message = input_model.sample()

        # Hash and truncate via the provided hash engine.
        digest = hash_engine.hash(message)

        sample_count += 1

        # If we've seen this digest before, collision occurs.
        if digest in seen_hashes:
            return sample_count

        # Otherwise record and continue.
        seen_hashes.add(digest)
