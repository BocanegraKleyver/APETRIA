
"""
APETRIA - Preparación de ventanas temporales para LSTM Autoencoder.

Trabaja exclusivamente con observaciones existentes y ordenadas
cronológicamente.

No genera datos sintéticos.
No realiza normalización ni entrenamiento.
No mezcla máquinas ni particiones de datos.
"""

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np


LabelPolicy = Literal["any", "last", "all"]


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 30
    stride: int = 1
    expected_features: int = 38

    def __post_init__(self) -> None:
        for name in (
            "window_size",
            "stride",
            "expected_features",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(
                    f"{name} debe ser un entero positivo."
                )

            if value < 1:
                raise ValueError(
                    f"{name} debe ser mayor que cero."
                )


class TemporalWindowing:
    """
    Convierte observaciones consecutivas en ventanas temporales.

    Forma de entrada:
        (n_observaciones, n_variables)

    Forma de salida:
        (n_ventanas, window_size, n_variables)

    Cada instancia trabaja sobre una sola partición de una
    sola máquina. Para evitar cruces, se deben crear ventanas
    independientemente para X_train, X_val y X_test.
    """

    def __init__(
        self,
        config: WindowConfig | None = None,
    ) -> None:
        self.config = config or WindowConfig()

    def validate_features(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Valida datos reales ya preprocesados."""

        matrix = np.asarray(X)

        if matrix.ndim != 2:
            raise ValueError(
                "X debe tener forma (observaciones, variables)."
            )

        if matrix.shape[1] != self.config.expected_features:
            raise ValueError(
                f"Se esperaban {self.config.expected_features} "
                f"variables; se recibieron {matrix.shape[1]}."
            )

        if not np.issubdtype(matrix.dtype, np.number):
            raise TypeError("X debe contener valores numéricos.")

        if not np.isfinite(matrix).all():
            raise ValueError(
                "X contiene valores NaN o infinitos."
            )

        return matrix

    def count_windows(self, n_observations: int) -> int:
        """Calcula la cantidad de ventanas posibles."""

        if (
            isinstance(n_observations, bool)
            or not isinstance(
                n_observations, (int, np.integer)
            )
            or n_observations < 0
        ):
            raise ValueError(
                "n_observations debe ser un entero no negativo."
            )

        size = self.config.window_size
        stride = self.config.stride

        if n_observations < size:
            return 0

        return 1 + (n_observations - size) // stride

    def window_indices(
        self,
        n_observations: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retorna índices [inicio, fin) de cada ventana.

        El índice fin es exclusivo.
        """

        count = self.count_windows(n_observations)

        starts = (
            np.arange(count, dtype=np.int64)
            * self.config.stride
        )

        ends = starts + self.config.window_size

        return starts, ends

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Construye todas las ventanas de una partición.

        Para conjuntos grandes, preferir iter_batches().
        """

        matrix = self.validate_features(X)

        starts, _ = self.window_indices(len(matrix))

        size = self.config.window_size
        features = self.config.expected_features

        if len(starts) == 0:
            return np.empty(
                (0, size, features),
                dtype=matrix.dtype,
            )

        offsets = np.arange(size, dtype=np.int64)

        indices = starts[:, None] + offsets[None, :]

        return matrix[indices]

    def iter_batches(
        self,
        X: np.ndarray,
        batch_size: int = 256,
    ) -> Iterator[np.ndarray]:
        """
        Entrega ventanas por lotes para limitar memoria.

        No altera el orden cronológico.
        """

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError(
                "batch_size debe ser un entero positivo."
            )

        matrix = self.validate_features(X)

        starts, _ = self.window_indices(len(matrix))

        offsets = np.arange(
            self.config.window_size,
            dtype=np.int64,
        )

        for batch_start in range(
            0,
            len(starts),
            batch_size,
        ):
            batch_starts = starts[
                batch_start:batch_start + batch_size
            ]

            indices = (
                batch_starts[:, None]
                + offsets[None, :]
            )

            yield matrix[indices]

    def transform_labels(
        self,
        y: np.ndarray,
        policy: LabelPolicy = "any",
    ) -> np.ndarray:
        """
        Convierte etiquetas puntuales en etiquetas por ventana.

        any:
            Anomalía si al menos un instante es anómalo.
        last:
            Utiliza la etiqueta del último instante.
        all:
            Anomalía si todos los instantes son anómalos.

        No debe utilizarse para entrenar el LSTM no supervisado.
        Se reserva para evaluación y análisis.
        """

        if policy not in ("any", "last", "all"):
            raise ValueError(
                "La política debe ser any, last o all."
            )

        labels = np.asarray(y)

        if labels.ndim != 1:
            raise ValueError(
                "y debe ser un vector unidimensional."
            )

        if not np.isin(labels, (0, 1)).all():
            raise ValueError(
                "Las etiquetas deben ser binarias: 0 o 1."
            )

        starts, _ = self.window_indices(len(labels))

        if len(starts) == 0:
            return np.empty(0, dtype=np.int8)

        offsets = np.arange(
            self.config.window_size,
            dtype=np.int64,
        )

        indices = starts[:, None] + offsets[None, :]
        window_labels = labels[indices]

        if policy == "any":
            result = np.any(window_labels == 1, axis=1)

        elif policy == "all":
            result = np.all(window_labels == 1, axis=1)

        else:
            result = window_labels[:, -1] == 1

        return result.astype(np.int8)
