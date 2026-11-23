
"""
APETRIA - Calibración de umbrales para detección de anomalías.

Métodos:
    - Z-score fijo calibrado con errores de validación.
    - Z-score adaptativo con actualización controlada.

Este módulo:
    - No entrena modelos de IA.
    - No genera datos sintéticos.
    - No accede a etiquetas ni datos de prueba.
    - No asume normalidad estadística de los errores.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ThresholdConfig:
    z_threshold: float = 3.0
    min_std: float = 1e-12
    adaptive: bool = False
    learning_rate: float = 0.01
    min_calibration_samples: int = 30

    def __post_init__(self):
        if not np.isfinite(self.z_threshold):
            raise ValueError("z_threshold debe ser finito.")

        if self.z_threshold <= 0:
            raise ValueError("z_threshold debe ser positivo.")

        if not np.isfinite(self.min_std) or self.min_std <= 0:
            raise ValueError("min_std debe ser positivo y finito.")

        if (
            not np.isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 1
        ):
            raise ValueError(
                "learning_rate debe estar entre 0 y 1."
            )

        if self.min_calibration_samples < 2:
            raise ValueError(
                "min_calibration_samples debe ser al menos 2."
            )


@dataclass(frozen=True)
class ThresholdResult:
    scores: np.ndarray
    z_scores: np.ndarray
    predictions: np.ndarray
    threshold: float
    mean: float
    std: float

    @property
    def anomaly_count(self):
        return int(np.sum(self.predictions))

    @property
    def anomaly_rate(self):
        if len(self.predictions) == 0:
            return 0.0
        return float(np.mean(self.predictions))


class AdaptiveThreshold:
    """
    Detector estadístico basado en errores de reconstrucción.

    El ajuste inicial se realiza con errores de calibración.

    Por defecto el umbral es fijo. La actualización adaptativa
    requiere adaptive=True y una llamada explícita a update().
    """

    def __init__(
        self,
        config: Optional[ThresholdConfig] = None,
    ):
        self.config = config or ThresholdConfig()

        self.mean_ = None
        self.variance_ = None
        self.n_calibration_ = 0
        self.n_updates_ = 0

    @staticmethod
    def _validate_errors(errors, allow_empty=False):
        values = np.asarray(errors, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                "Los errores deben ser un vector 1D."
            )

        if not allow_empty and len(values) == 0:
            raise ValueError(
                "El vector de errores está vacío."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "Los errores contienen NaN o infinitos."
            )

        if (values < 0).any():
            raise ValueError(
                "Los errores de reconstrucción no pueden ser negativos."
            )

        return values

    @property
    def is_fitted(self):
        return (
            self.mean_ is not None
            and self.variance_ is not None
        )

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "Primero debe ejecutarse fit()."
            )

    @property
    def mean(self):
        self._check_fitted()
        return float(self.mean_)

    @property
    def std(self):
        self._check_fitted()
        return float(
            max(
                np.sqrt(max(self.variance_, 0.0)),
                self.config.min_std,
            )
        )

    @property
    def threshold(self):
        self._check_fitted()

        return float(
            self.mean + self.config.z_threshold * self.std
        )

    def fit(self, calibration_errors):
        """
        Calibra media y desviación estándar poblacional.

        Los datos deben provenir de una partición de
        calibración, nunca de X_test.
        """
        errors = self._validate_errors(calibration_errors)

        if len(errors) < self.config.min_calibration_samples:
            raise ValueError(
                "Cantidad insuficiente de errores "
                "para calibración."
            )

        self.mean_ = float(np.mean(errors))
        self.variance_ = float(np.var(errors, ddof=0))

        self.n_calibration_ = len(errors)
        self.n_updates_ = 0

        return self

    def score(self, errors):
        self._check_fitted()

        values = self._validate_errors(
            errors,
            allow_empty=True,
        )

        return (values - self.mean) / self.std

    def predict(self, errors):
        self._check_fitted()

        values = self._validate_errors(
            errors,
            allow_empty=True,
        )

        z_scores = self.score(values)

        predictions = (
            values > self.threshold
        ).astype(np.int8)

        return ThresholdResult(
            scores=values.copy(),
            z_scores=z_scores,
            predictions=predictions,
            threshold=self.threshold,
            mean=self.mean,
            std=self.std,
        )

    def update(self, error, accepted_as_normal=False):
        """
        Actualización secuencial opcional.

        Solo actualiza la referencia cuando:
            1. adaptive=True;
            2. accepted_as_normal=True;
            3. la observación no supera el umbral actual.

        accepted_as_normal representa una autorización
        externa explícita. No implica que la observación
        haya sido verificada como normal.

        Retorna True si se actualizó la referencia.
        """
        self._check_fitted()

        if not self.config.adaptive:
            return False

        values = self._validate_errors([error])
        value = float(values[0])

        if not accepted_as_normal:
            return False

        # Evaluar antes de actualizar.
        if value > self.threshold:
            return False

        alpha = self.config.learning_rate

        old_mean = self.mean_
        old_variance = self.variance_

        delta = value - old_mean

        new_mean = old_mean + alpha * delta

        # Varianza exponencial ponderada.
        new_variance = (
            (1.0 - alpha)
            * (
                old_variance
                + alpha * delta * delta
            )
        )

        self.mean_ = float(new_mean)
        self.variance_ = float(max(new_variance, 0.0))
        self.n_updates_ += 1

        return True

    def predict_sequential(
        self,
        errors,
        accepted_as_normal=None,
    ):
        """
        Predice en orden cronológico.

        Si adaptive=True, las estadísticas pueden cambiar
        después de cada observación aceptada.

        Sin autorizaciones explícitas, no se actualiza
        ninguna estadística.

        Devuelve umbrales y Z-scores usados en cada instante.
        """
        self._check_fitted()

        values = self._validate_errors(
            errors,
            allow_empty=True,
        )

        if accepted_as_normal is None:
            accepted = np.zeros(
                len(values),
                dtype=bool,
            )
        else:
            accepted = np.asarray(accepted_as_normal)

            if (
                accepted.ndim != 1
                or len(accepted) != len(values)
                or accepted.dtype.kind != "b"
            ):
                raise ValueError(
                    "accepted_as_normal debe ser un "
                    "vector booleano de igual longitud."
                )

        predictions = np.zeros(
            len(values),
            dtype=np.int8,
        )

        z_scores = np.zeros(
            len(values),
            dtype=np.float64,
        )

        thresholds = np.zeros(
            len(values),
            dtype=np.float64,
        )

        for i, value in enumerate(values):
            current_threshold = self.threshold
            current_z = (
                (value - self.mean) / self.std
            )

            thresholds[i] = current_threshold
            z_scores[i] = current_z

            predictions[i] = int(
                value > current_threshold
            )

            if accepted[i]:
                self.update(
                    float(value),
                    accepted_as_normal=True,
                )

        return {
            "scores": values.copy(),
            "z_scores": z_scores,
            "predictions": predictions,
            "thresholds": thresholds,
            "updates": self.n_updates_,
        }

    def get_state(self):
        self._check_fitted()

        return {
            "method": "z_score",
            "adaptive": self.config.adaptive,
            "z_threshold": self.config.z_threshold,
            "mean": self.mean,
            "std": self.std,
            "variance": float(self.variance_),
            "threshold": self.threshold,
            "calibration_samples": self.n_calibration_,
            "updates": self.n_updates_,
            "learning_rate": self.config.learning_rate,
            "min_std": self.config.min_std,
        }
