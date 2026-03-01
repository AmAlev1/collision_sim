"""
input_models.py

This module defines the input sampling models used to generate message bytes for the
collision-time experiments. This module is where the input distribution is controlled,
enabling clean tests of how collision time responds to entropy reduction and probability
mass concentration.

Design Notes
------------
- This module produces input bytes, it doesn't Hash.
- All models expose a single method: sample() -> bytes.
- Sampling is performed using an injected random. Random instance for full
  reproducibility (no hidden global RNG usage).
- Inputs are represented canonically as integers (bitstrings), then converted to
  fixed-length big-endian byte strings.


All generated messages are fixed-length (default 32 bytes = 256 bits). This avoids
mixing "length effects" into the study and keeps the experiment aligned with
bit-level entropy modeling. Zero-padding is used when the sampled integer has
fewer bits than the fixed length.

Models Implemented
------------------
1) UniformModel:
   - High-entropy baseline; samples uniformly from the full 256-bit space.

2) ReducedSupportModel(k_bits):
   - Uniform over a reduced support of size 2^k_bits; models clean entropy
     reduction (effective dimension reduction).

3) SkewedModel(k_bits, heavy_probability):
   - Full nominal support of size 2^k_bits, but with probability mass concentrated
     on a single "heavy" value. This probes sensitivity to concentration effects
     (e.g., reduced min-entropy) while retaining a large support.

"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


def int_to_fixed_bytes(value: int, length: int) -> bytes:
    """
    Convert a non-negative integer to a fixed-length big-endian byte string.

    Parameters
    ---------
    value : int
        Non-negative integer to convert.
    length : int
        Target number of bytes in output.

    Returned values
    ---------------
    bytes
        Big-endian representation of `value`, zero-padded to `length` bytes.

    Notes
    ------
    This function enforces a fixed-length representation to keep the experiment
    focused on entropy/distribution effects rather than message-length variation.
    """
    if not isinstance(value, int):
        raise TypeError(f"value must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError("value must be non-negative")
    if not isinstance(length, int):
        raise TypeError(f"length must be int, got {type(length).__name__}")
    if length < 1:
        raise ValueError("length must be >= 1")

    # to_bytes will raise OverflowError if value doesn't fit in the specified length.
    try:
        return value.to_bytes(length, byteorder="big", signed=False)
    except OverflowError as e:
        raise ValueError(
            f"value={value} does not fit in {length} bytes "
            f"({8*length} bits). Increase byte_length or reduce value range."
        ) from e


@dataclass(frozen=True)
class BaseInputModel:
    """
    Base class for input sampling models.

    Subclasses must implement:
        sample() -> bytes

    Attributes include the following:
    
    byte_length : int
        Fixed number of bytes in each generated message.
    rng : random.Random
        Random number generator instance (injected for reproducibility).
    """
    byte_length: int = 32
    rng: random.Random = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.byte_length, int):
            raise TypeError(f"byte_length must be int, got {type(self.byte_length).__name__}")
        if self.byte_length < 1:
            raise ValueError("byte_length must be >= 1")
        if not isinstance(self.rng, random.Random):
            raise TypeError("rng must be an instance of random.Random (inject for reproducibility)")

    def sample(self) -> bytes:
        raise NotImplementedError("Subclasses must implement sample().")


@dataclass(frozen=True)
class UniformModel(BaseInputModel):
    """
    Uniform high-entropy baseline model.

    Samples uniformly from the full space of 8*byte_length-bit messages by
    drawing that many random bits and converting to a fixed-length byte string.
    """

    def sample(self) -> bytes:
        # Draw exactly 8*byte_length bits, yielding a uniform integer in [0, 2^(8L)-1].
        bits = 8 * self.byte_length
        value = self.rng.getrandbits(bits)
        return int_to_fixed_bytes(value, self.byte_length)


@dataclass(frozen=True)
class ReducedSupportModel(BaseInputModel):
    """
    Uniform model over a reduced support of size 2^k_bits.

    Parameters
    ----------
    k_bits : int
        Effective entropy in bits. Samples uniformly from [0, 2^k_bits - 1].

    Notes
    -----
    This model implements *clean* entropy reduction: all outcomes in the reduced
    support have equal probability, so Shannon entropy is exactly k_bits.
    """
    k_bits: int = 16

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.k_bits, int):
            raise TypeError(f"k_bits must be int, got {type(self.k_bits).__name__}")
        if self.k_bits < 1:
            raise ValueError("k_bits must be >= 1")
        if self.k_bits > 8 * self.byte_length:
            raise ValueError(
                "k_bits cannot exceed the message bit-length (8*byte_length). "
                "Increase byte_length or reduce k_bits."
            )

    def sample(self) -> bytes:
        # Uniform over reduced support: value in [0, 2^k_bits - 1].
        value = self.rng.getrandbits(self.k_bits)
        return int_to_fixed_bytes(value, self.byte_length)


@dataclass(frozen=True)
class SkewedModel(BaseInputModel):
    """
    Skewed distribution with a single heavy-probability point.

    Parameters
    ----------
    k_bits : int
        Nominal support size parameter (support is subset of [0, 2^k_bits - 1]).
        Keeping k_bits large preserves a large support while allowing concentration.
    heavy_probability : float
        Probability assigned to the heavy value in each draw (0 < p < 1).
    heavy_value : Optional[int]
        The integer that receives heavy probability mass. Default is 0.
        Keeping this deterministic simplifies reproducibility and interpretation.

    Sampling Rule
    -------------
    With probability p:
        return heavy_value
    Otherwise:
        return a uniform sample from [0, 2^k_bits - 1] excluding heavy_value.

    Notes
    -----
    This preserves independence between samples while introducing probability mass
    concentration, allowing the experiment to probe collision-time sensitivity to
    second-moment effects beyond "clean" support reduction.
    """
    k_bits: int = 16
    heavy_probability: float = 0.2
    heavy_value: Optional[int] = 0

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.k_bits, int):
            raise TypeError(f"k_bits must be int, got {type(self.k_bits).__name__}")
        if self.k_bits < 1:
            raise ValueError("k_bits must be >= 1")
        if self.k_bits > 8 * self.byte_length:
            raise ValueError(
                "k_bits cannot exceed the message bit-length (8*byte_length). "
                "Increase byte_length or reduce k_bits."
            )

        if not isinstance(self.heavy_probability, (float, int)):
            raise TypeError("heavy_probability must be a float in (0, 1)")
        p = float(self.heavy_probability)
        if not (0.0 < p < 1.0):
            raise ValueError("heavy_probability must satisfy 0 < p < 1")

        if self.heavy_value is None:
            raise ValueError("heavy_value must not be None")
        if not isinstance(self.heavy_value, int):
            raise TypeError(f"heavy_value must be int, got {type(self.heavy_value).__name__}")
        if self.heavy_value < 0:
            raise ValueError("heavy_value must be non-negative")

        # Ensure heavy_value lies within the nominal support [0, 2^k_bits - 1].
        max_val = (1 << self.k_bits) - 1
        if self.heavy_value > max_val:
            raise ValueError(
                f"heavy_value must be within [0, 2^k_bits - 1]; got heavy_value={self.heavy_value}, "
                f"max={max_val}"
            )

    def sample(self) -> bytes:
        p = float(self.heavy_probability)
        max_val = (1 << self.k_bits) - 1
        hv = int(self.heavy_value)

        # With probability p, return the heavy point.
        if self.rng.random() < p:
            return int_to_fixed_bytes(hv, self.byte_length)
        """
        Otherwise sample uniformly from the remaining support excluding hv.
        We do this without rejection sampling to avoid subtle performance effects
        and to keep the distribution exactly uniform over the excluded set.
        
        Strategy:
        - Draw u uniformly from [0, 2^k_bits - 2] (one fewer element).
        - Map u to [0, 2^k_bits - 1] skipping hv:
              if u < hv: value = u
              else:      value = u + 1
        
        This yields a perfect uniform draw over all values except hv.
        """
        u = self.rng.randrange(0, max_val)  # size = 2^k_bits - 1
        value = u if u < hv else u + 1
        return int_to_fixed_bytes(value, self.byte_length)
