
"""
APETRIA - Detector híbrido de anomalías.

Combina las predicciones de Isolation Forest y
LSTM Autoencoder sobre observaciones temporales
alineadas.

Políticas:
    dual: conserva las decisiones individuales.
    or: alerta si al menos un detector alerta.
    and: alerta si ambos detectores alertan.

No entrena modelos.
No modifica umbrales.
No genera datos sintéticos.

La política dual no produce una decisión binaria
fusionada: combined_predictions será None.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from backend.app.ml.inference import (
    APETRIAInference,
    InferenceResult,
)


HybridPolicy = Literal["dual", "or", "and"]


@dataclass(frozen=True)
class HybridDetectionResult:
    machine: str
    policy: str

    point_indices: np.ndarray

    isolation_forest_predictions: np.ndarray
    lstm_predictions: np.ndarray

    isolation_forest_scores: np.ndarray
    lstm_scores: np.ndarray

    combined_predictions: Optional[np.ndarray]

    isolation_forest_threshold: float
    lstm_threshold: float

    observations_received: int
    observations_aligned: int
    initial_observations_excluded: int

    @property
    def isolation_forest_alerts(self) -> int:
        return int(
            np.sum(self.isolation_forest_predictions)
        )

    @property
    def lstm_alerts(self) -> int:
        return int(np.sum(self.lstm_predictions))

    @property
    def combined_alerts(self) -> Optional[int]:
        if self.combined_predictions is None:
            return None

        return int(
            np.sum(self.combined_predictions)
        )

    @property
    def alerts_from_both(self) -> int:
        return int(np.sum(
            (self.isolation_forest_predictions == 1)
            & (self.lstm_predictions == 1)
        ))

    @property
    def alerts_only_isolation_forest(self) -> int:
        return int(np.sum(
            (self.isolation_forest_predictions == 1)
            & (self.lstm_predictions == 0)
        ))

    @property
    def alerts_only_lstm(self) -> int:
        return int(np.sum(
            (self.isolation_forest_predictions == 0)
            & (self.lstm_predictions == 1)
        ))


class HybridDetector:
    """
    Combina resultados de inferencia ya calculados.

    Las predicciones LSTM se asignan al último
    punto temporal de cada ventana.

    Para window_size=30 y stride=1:
        IF:   índices 0, 1, 2, ..., N-1
        LSTM: índices 29, 30, ..., N-1

    El resultado se alinea desde el índice 29.
    """

    VALID_POLICIES = ("dual", "or", "and")

    def __init__(
        self,
        inference_service: APETRIAInference,
        policy: HybridPolicy = "dual",
    ):
        if policy not in self.VALID_POLICIES:
            raise ValueError(
                f"Política inválida: {policy}. "
                f"Opciones: {self.VALID_POLICIES}"
            )

        self.inference_service = inference_service
        self.policy = policy

    @staticmethod
    def _validate_result(
        result: InferenceResult,
        expected_model: str,
    ):
        if result.model != expected_model:
            raise ValueError(
                f"Se esperaba {expected_model}, "
                f"se recibió {result.model}."
            )

        scores = np.asarray(result.scores)
        predictions = np.asarray(result.predictions)

        if scores.ndim != 1:
            raise ValueError(
                "Los scores deben ser unidimensionales."
            )

        if predictions.ndim != 1:
            raise ValueError(
                "Las predicciones deben ser unidimensionales."
            )

        if len(scores) != len(predictions):
            raise ValueError(
                "Cantidad de scores y predicciones "
                "inconsistente."
            )

        if not np.isfinite(scores).all():
            raise ValueError(
                "Los scores contienen valores inválidos."
            )

        if not np.isin(predictions, [0, 1]).all():
            raise ValueError(
                "Las predicciones deben ser binarias."
            )

        if not np.isfinite(result.threshold):
            raise ValueError(
                "El umbral no es finito."
            )

    def combine(
        self,
        isolation_forest_result: InferenceResult,
        lstm_result: InferenceResult,
        observations_received: int,
        policy: Optional[HybridPolicy] = None,
    ) -> HybridDetectionResult:
        """
        Alinea y combina predicciones ya calculadas.

        No ejecuta inferencia.
        """

        selected_policy = (
            self.policy if policy is None else policy
        )

        if selected_policy not in self.VALID_POLICIES:
            raise ValueError(
                f"Política inválida: {selected_policy}"
            )

        self._validate_result(
            isolation_forest_result,
            "isolation_forest",
        )

        self._validate_result(
            lstm_result,
            "lstm_autoencoder",
        )

        if (
            isolation_forest_result.machine
            != lstm_result.machine
        ):
            raise ValueError(
                "Los resultados corresponden "
                "a máquinas diferentes."
            )

        if (
            isolation_forest_result.machine
            != self.inference_service.machine
        ):
            raise ValueError(
                "La máquina no coincide con "
                "el servicio de inferencia."
            )

        if (
            not isinstance(observations_received, int)
            or observations_received < 1
        ):
            raise ValueError(
                "observations_received debe ser "
                "un entero positivo."
            )

        window_size = (
            self.inference_service.lstm_config.window_size
        )

        stride = self.inference_service.windowing.config.stride

        if stride < 1:
            raise ValueError(
                "El stride debe ser positivo."
            )

        if observations_received < window_size:
            raise ValueError(
                "No hay suficientes observaciones "
                "para generar ventanas LSTM."
            )

        if len(isolation_forest_result.predictions) != (
            observations_received
        ):
            raise ValueError(
                "La cantidad de predicciones IF "
                "no coincide con las observaciones."
            )

        expected_indices = np.arange(
            window_size - 1,
            observations_received,
            stride,
            dtype=np.int64,
        )

        if len(lstm_result.predictions) != len(
            expected_indices
        ):
            raise ValueError(
                "La cantidad de predicciones LSTM "
                "no coincide con las ventanas esperadas."
            )

        if_predictions = np.asarray(
            isolation_forest_result.predictions,
            dtype=np.int8,
        )[expected_indices]

        if_scores = np.asarray(
            isolation_forest_result.scores,
            dtype=np.float64,
        )[expected_indices]

        lstm_predictions = np.asarray(
            lstm_result.predictions,
            dtype=np.int8,
        )

        lstm_scores = np.asarray(
            lstm_result.scores,
            dtype=np.float64,
        )

        if selected_policy == "or":
            combined = np.logical_or(
                if_predictions,
                lstm_predictions,
            ).astype(np.int8)

        elif selected_policy == "and":
            combined = np.logical_and(
                if_predictions,
                lstm_predictions,
            ).astype(np.int8)

        else:
            combined = None

        return HybridDetectionResult(
            machine=self.inference_service.machine,
            policy=selected_policy,
            point_indices=expected_indices,
            isolation_forest_predictions=if_predictions,
            lstm_predictions=lstm_predictions,
            isolation_forest_scores=if_scores,
            lstm_scores=lstm_scores,
            combined_predictions=combined,
            isolation_forest_threshold=float(
                isolation_forest_result.threshold
            ),
            lstm_threshold=float(
                lstm_result.threshold
            ),
            observations_received=observations_received,
            observations_aligned=len(expected_indices),
            initial_observations_excluded=window_size - 1,
        )

    def predict(
        self,
        X: np.ndarray,
        policy: Optional[HybridPolicy] = None,
        batch_size: int = 128,
    ) -> HybridDetectionResult:
        """
        Ejecuta ambos modelos mediante inference.py
        y combina sus predicciones.

        X debe estar previamente normalizada
        con el preprocesamiento de la máquina SMD.
        """

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "Se esperaba una matriz bidimensional."
            )

        if_result = (
            self.inference_service.predict_isolation_forest(X)
        )

        lstm_result = (
            self.inference_service.predict_lstm(
                X,
                batch_size=batch_size,
            )
        )

        return self.combine(
            isolation_forest_result=if_result,
            lstm_result=lstm_result,
            observations_received=len(X),
            policy=policy,
        )

    def info(self) -> dict:
        return {
            "project": "APETRIA",
            "component": "hybrid_detector",
            "machine": self.inference_service.machine,
            "default_policy": self.policy,
            "supported_policies": list(
                self.VALID_POLICIES
            ),
            "lstm_window_size": (
                self.inference_service.lstm_config.window_size
            ),
            "lstm_stride": (
                self.inference_service.windowing.config.stride
            ),
            "training_during_inference": False,
            "threshold_updates": False,
        }
