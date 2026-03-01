"""
hash_engine.py

The purpose of this module is to provide a small, reliable hashing primitive for the collision-time experiments.

This module implements SHA-256 hashing (via Python's standard 'hashlib') and returns a
truncated n-bit output as an integer in the range [0, 2^n - 1].

Design Notes
------------
- This module contains *no* experiment logic and *no* randomness.
- Truncation is applied to the final digest only; SHA-256 itself is not modified.
- Returning integers makes collision detection fast and unambiguous (set membership).
- We use a precomputed bitmask to keep the lowest n bits of the digest integer.
  Under standard random-oracle-style heuristics, any fixed subset of digest bits
  should be close to uniformly distributed when inputs are high-entropy.
"""

from __future__ import annotations

import hashlib


class SHA256Truncated:
    """
    SHA-256 wrapper that returns an n-bit truncated output as an integer.

    Parameters
    ----------
    n_bits : int
        Number of output bits to keep (1 <= n_bits <= 256).

    Notes: The returned value is the lowest n_bits of the SHA-256 digest interpreted
    as a big-endian integer.
    """

    def __init__(self, n_bits: int) -> None:
        # Validate configuration early to prevent subtle downstream errors.
        if not isinstance(n_bits, int):
            raise TypeError(f"n_bits must be an int, got {type(n_bits).__name__}")
        if n_bits < 1 or n_bits > 256:
            raise ValueError("n_bits must satisfy 1 <= n_bits <= 256")

        self.n_bits: int = n_bits

        # Precompute the output mask once. This avoids reallocating or recomputing
        # on every call to hash(), and makes the truncation step explicit.
        self.mask: int = (1 << n_bits) - 1

        # Range size is often useful for sanity checks and later theoretical comparisons.
        self.range_size: int = 1 << n_bits

    def hash(self, input_bytes: bytes) -> int:
        """
        Hash bytes with SHA-256 and truncate the result to n_bits.

        Parameters
        ----------
        input_bytes : bytes
            Raw input message bytes.

        Returned values
        -------
        int
            Truncated digest as an integer in [0, 2^n_bits - 1].
        """
        # Keep this module strict: hashing operates on bytes only.
        # Encoding decisions belong in the input model layer.
        if not isinstance(input_bytes, (bytes, bytearray, memoryview)):
            raise TypeError(
                "input_bytes must be bytes-like (bytes/bytearray/memoryview), "
                f"got {type(input_bytes).__name__}"
            )

        # Compute the SHA-256 digest as bytes (32 bytes = 256 bits).
        digest: bytes = hashlib.sha256(bytes(input_bytes)).digest()

        # Convert digest bytes to an integer for efficient masking and set membership.
        digest_int: int = int.from_bytes(digest, byteorder="big", signed=False)

        # Truncate to n bits by applying the precomputed mask.
        # This yields an integer in the n-bit output space.
        return digest_int & self.mask

    def get_range_size(self) -> int:
        #Return the size of the truncated output space, i.e., 2^n_bits.
        return self.range_size

    def __repr__(self) -> str:
        # Helpful for debugging without printing excessive information.
        return f"{self.__class__.__name__}(n_bits={self.n_bits})"
