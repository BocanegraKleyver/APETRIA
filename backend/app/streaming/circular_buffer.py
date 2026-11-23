
"""
APETRIA - Buffer circular de telemetría.

Construye ventanas temporales incrementales para
LSTM Autoencoder utilizando observaciones reales.

No genera datos sintéticos.
No entrena modelos.
No modifica umbrales.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CircularBufferConfig:
    window_size: int = 30
    expected_features: int = 38
    stride: int = 1

    def __post_init__(self):
        if self.window_size < 1:
            raise ValueError(
                "window_size debe ser positivo."
            )

        if self.expected_features < 1:
            raise ValueError(
                "expected_features debe ser positivo."
            )

        if self.stride < 1:
            raise ValueError(
                "stride debe ser positivo."
            )


@dataclass(frozen=True)
class TemporalWindow:
    machine: str
    point_index: int
    window_start: int
    window_end_exclusive: int
    values: np.ndarray


class CircularBuffer:
    """
    Buffer circular independiente por máquina.

    Recibe observaciones individuales y genera
    ventanas temporales cuando se cumple el tamaño
    y el stride configurados.

    Los índices comienzan en cero.
    """

    def __init__(
        self,
        machine: str,
        config: Optional[CircularBufferConfig] = None,
    ):
        if not isinstance(machine, str) or not machine.strip():
            raise ValueError(
                "Debe especificarse una máquina válida."
            )

        self.machine = machine
        self.config = config or CircularBufferConfig()

        self._buffer = deque(
            maxlen=self.config.window_size
        )

        self._received = 0
        self._emitted = 0

    @property
    def observations_received(self) -> int:
        return self._received

    @property
    def windows_emitted(self) -> int:
        return self._emitted

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) == self.config.window_size

    def add(
        self,
        observation: np.ndarray,
    ) -> Optional[TemporalWindow]:
        """
        Agrega una observación al buffer.

        Devuelve una ventana cuando corresponde
        emitirla según window_size y stride.

        Si todavía no hay suficientes observaciones,
        devuelve None.
        """

        values = np.asarray(
            observation,
            dtype=np.float32,
        )

        if values.ndim != 1:
            raise ValueError(
                "La observación debe ser un vector."
            )

        if len(values) != self.config.expected_features:
            raise ValueError(
                f"Se esperaban "
                f"{self.config.expected_features} variables, "
                f"se recibieron {len(values)}."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "La observación contiene valores no finitos."
            )

        self._buffer.append(values.copy())

        point_index = self._received
        self._received += 1

        if not self.is_ready:
            return None

        first_valid_index = self.config.window_size - 1

        if (
            point_index - first_valid_index
        ) % self.config.stride != 0:
            return None

        window_start = (
            point_index - self.config.window_size + 1
        )

        window = np.stack(
            tuple(self._buffer),
            axis=0,
        )

        self._emitted += 1

        return TemporalWindow(
            machine=self.machine,
            point_index=point_index,
            window_start=window_start,
            window_end_exclusive=point_index + 1,
            values=window,
        )

    def reset(self):
        """
        Reinicia el buffer y sus contadores.
        """
        self._buffer.clear()
        self._received = 0
        self._emitted = 0

    def info(self) -> dict:
        return {
            "project": "APETRIA",
            "component": "circular_buffer",
            "machine": self.machine,
            "window_size": self.config.window_size,
            "expected_features": (
                self.config.expected_features
            ),
            "stride": self.config.stride,
            "observations_received": self._received,
            "windows_emitted": self._emitted,
            "ready": self.is_ready,
        }
