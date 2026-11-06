"""Shared geometry for physics and playback. Units: meters, kilograms, seconds."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Track:
    """Flat rectangle; length and width measure the clear space inside the walls."""
    length: float = 6.0
    width: float = 4.0
    barrier_height: float = 0.3
    barrier_thickness: float = 0.12

    def __post_init__(self):
        dimensions = (self.length, self.width, self.barrier_height, self.barrier_thickness)
        if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
            raise ValueError("Track dimensions must be finite and positive")

    def barriers(self):
        """Position and full dimensions for four fixed, axis-aligned box walls."""
        length, width = self.length, self.width
        thickness, height = self.barrier_thickness, self.barrier_height
        for side in (-1, 1):
            yield (0, side * (width + thickness) / 2, height / 2), (length + 2 * thickness, thickness, height)
            yield (side * (length + thickness) / 2, 0, height / 2), (thickness, width, height)


@dataclass(frozen=True)
class Vehicle:
    length: float = 0.52
    width: float = 0.26
    height: float = 0.12
    mass: float = 3.2
    friction: float = 0.6

    def __post_init__(self):
        if not np.isfinite((self.length, self.width, self.height, self.mass, self.friction)).all():
            raise ValueError("Vehicle parameters must be finite")
        if min(self.length, self.width, self.height, self.mass) <= 0 or self.friction < 0:
            raise ValueError("Vehicle dimensions/mass must be positive; friction nonnegative")

    @property
    def dimensions(self):
        return self.length, self.width, self.height
