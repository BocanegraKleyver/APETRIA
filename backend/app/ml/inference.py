
"""
APETRIA - Servicio de inferencia de modelos.

Carga modelos entrenados de Isolation Forest y
LSTM Autoencoder, junto con sus umbrales calibrados.

No entrena modelos.
No modifica umbrales.
No utiliza etiquetas de prueba.
No genera datos sintéticos.

Las entradas deben estar previamente transformadas
con el preprocesamiento correspondiente a la máquina.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits, threadpool_info

from backend.app.ml.isolation_forest import SMDIsolationForest
from backend.app.ml.lstm_autoencoder import (
    LSTMAutoencoder,
    LSTMAutoencoderConfig,
)
from backend.app.preprocessing.windowing import (
    TemporalWindowing,
    WindowConfig,
)


ROOT = Path(__file__).resolve().parents[3]

MODEL_DIR = ROOT / "models/lstm_autoencoder"
CALIBRATION_DIR = ROOT / "results/calibration"


@dataclass(frozen=True)
class InferenceResult:
    machine: str
    model: str
    scores: np.ndarray
    predictions: np.ndarray
    threshold: float
    observations: int


class APETRIAInference:
    """
    Servicio reutilizable de inferencia.

    Recibe datos numéricos previamente escalados
    mediante el preprocesamiento original de SMD.
    """

    def __init__(
        self,
        machine: str,
        device: str = "cpu",
        numeric_threads: int | None = None,
    ):
        if not machine:
            raise ValueError(
                "Debe especificarse una máquina."
            )

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA no está disponible."
            )

        if device not in ("cpu", "cuda"):
            raise ValueError(
                "Dispositivo no admitido."
            )

        if numeric_threads is not None:
            if isinstance(numeric_threads, bool) or not isinstance(numeric_threads, int) or numeric_threads < 1:
                raise ValueError("numeric_threads debe ser un entero positivo o None.")
            if device != "cpu":
                raise ValueError("numeric_threads solo se admite con device='cpu'.")

        self.machine = machine
        self.device = device
        self.numeric_threads = numeric_threads

        # El limite afecta los pools nativos de TODO el proceso, no solo esta
        # instancia. Se conserva el controlador para mantenerlo vigente.
        # No crear servicios con limites distintos en el mismo proceso.
        self._numeric_thread_controller = (
            threadpool_limits(limits=numeric_threads)
            if numeric_threads is not None
            else None
        )

        self.isolation_forest = SMDIsolationForest(
            machine=machine
        )

        self.lstm_model = None
        self.lstm_config = None
        self.lstm_threshold = None
        self.windowing = None

        self._load_lstm()

    def _load_lstm(self):
        checkpoint_path = (
            MODEL_DIR / f"{self.machine}.pt"
        )

        calibration_path = (
            CALIBRATION_DIR
            / f"{self.machine}_lstm_threshold.json"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )

        if not calibration_path.exists():
            raise FileNotFoundError(
                calibration_path
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        with calibration_path.open(
            encoding="utf-8"
        ) as file:
            calibration_document = json.load(file)

        if checkpoint["machine"] != self.machine:
            raise ValueError(
                "El checkpoint corresponde a otra máquina."
            )

        if calibration_document["machine"] != self.machine:
            raise ValueError(
                "La calibración corresponde a otra máquina."
            )

        config = LSTMAutoencoderConfig(
            **checkpoint["model_config"]
        )

        if (
            calibration_document["model_config"]
            != checkpoint["model_config"]
        ):
            raise ValueError(
                "Configuraciones LSTM inconsistentes."
            )

        calibration = calibration_document["calibration"]

        if calibration["adaptive"] is not False:
            raise ValueError(
                "Solo se admiten umbrales fijos calibrados."
            )

        if calibration["partition"] != "X_val":
            raise ValueError(
                "El umbral debe provenir de X_val."
            )

        threshold = float(
            calibration["threshold"]
        )

        if not np.isfinite(threshold):
            raise ValueError(
                "Umbral LSTM inválido."
            )

        model = LSTMAutoencoder(config)

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        model.to(self.device)
        model.eval()

        self.lstm_model = model
        self.lstm_config = config
        self.lstm_threshold = threshold

        self.windowing = TemporalWindowing(
            WindowConfig(
                window_size=config.window_size,
                stride=1,
                expected_features=config.input_size,
            )
        )

    def predict_isolation_forest(
        self,
        X: np.ndarray,
    ) -> InferenceResult:
        """
        Predicción pointwise con Isolation Forest.
        """

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "Se esperaba una matriz bidimensional."
            )

        if X.shape[1] != 38:
            raise ValueError(
                "Se esperaban 38 variables."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "La entrada contiene valores inválidos."
            )

        result = self.isolation_forest.predict(
            X,
            policy="validation_p99",
        )


        return InferenceResult(
            machine=self.machine,
            model="isolation_forest",
            scores=np.asarray(result.anomaly_scores),
            predictions=np.asarray(result.predictions),
            threshold=float(result.threshold),
            observations=len(X),
        )



    def predict_lstm(
        self,
        X: np.ndarray,
        batch_size: int = 128,
    ) -> InferenceResult:
        """
        Predicción LSTM sobre ventanas temporales.

        Cada resultado corresponde al último punto
        de su ventana.
        """

        X = np.asarray(X, dtype=np.float32)

        self.windowing.validate_features(X)

        if batch_size < 1:
            raise ValueError(
                "batch_size debe ser positivo."
            )

        if len(X) < self.lstm_config.window_size:
            raise ValueError(
                "No hay suficientes observaciones "
                "para formar una ventana."
            )

        errors = []

        for batch in self.windowing.iter_batches(
            X,
            batch_size=batch_size,
        ):
            _, batch_errors = self.lstm_model.predict(
                batch,
                batch_size=batch_size,
                device=self.device,
            )

            errors.append(
                np.asarray(
                    batch_errors,
                    dtype=np.float64,
                )
            )

        scores = np.concatenate(errors)

        predictions = (
            scores > self.lstm_threshold
        ).astype(np.int8)

        return InferenceResult(
            machine=self.machine,
            model="lstm_autoencoder",
            scores=scores,
            predictions=predictions,
            threshold=self.lstm_threshold,
            observations=len(scores),
        )



    def predict_lstm_window(
        self,
        window: np.ndarray,
    ) -> InferenceResult:
        """
        Ejecuta inferencia LSTM sobre una ventana temporal.

        Entrada:
            window: matriz (window_size, input_size).

        Salida:
            Una puntuación MSE y una predicción binaria.

        La ventana debe estar previamente normalizada
        mediante el preprocesamiento correspondiente
        a la máquina SMD.
        """

        window = np.asarray(
            window,
            dtype=np.float32,
        )

        expected_shape = (
            self.lstm_config.window_size,
            self.lstm_config.input_size,
        )

        if window.shape != expected_shape:
            raise ValueError(
                f"Dimensión de ventana inválida. "
                f"Esperada: {expected_shape}. "
                f"Recibida: {window.shape}."
            )

        if not np.isfinite(window).all():
            raise ValueError(
                "La ventana contiene valores no finitos."
            )

        batch = window[np.newaxis, :, :]

        _, errors = self.lstm_model.predict(
            batch,
            batch_size=1,
            device=self.device,
        )

        scores = np.asarray(
            errors,
            dtype=np.float64,
        )

        if scores.shape != (1,):
            raise RuntimeError(
                "La inferencia LSTM debe devolver "
                "exactamente una puntuación."
            )

        predictions = (
            scores > self.lstm_threshold
        ).astype(np.int8)

        return InferenceResult(
            machine=self.machine,
            model="lstm_autoencoder",
            scores=scores,
            predictions=predictions,
            threshold=float(self.lstm_threshold),
            observations=1,
        )



    def info(self) -> dict:
        return {
            "project": "APETRIA",
            "machine": self.machine,
            "device": self.device,
            "numeric_threads_requested": self.numeric_threads,
            "native_threadpools": (
                [
                    {
                        "user_api": pool.get("user_api"),
                        "internal_api": pool.get("internal_api"),
                        "num_threads": pool.get("num_threads"),
                    }
                    for pool in threadpool_info()
                ]
                if self.numeric_threads is not None
                else None
            ),
            "models": [
                "isolation_forest",
                "lstm_autoencoder",
            ],
            "isolation_forest_policy": "validation_p99",
            "lstm_policy": "validation_z3_fixed",
            "lstm_window_size": (
                self.lstm_config.window_size
            ),
            "lstm_threshold": self.lstm_threshold,
            "training_during_inference": False,
            "threshold_updates": False,
        }
