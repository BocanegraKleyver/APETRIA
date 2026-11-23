from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

from backend.app.ml.lstm_autoencoder import (
    LSTMAutoencoder,
    LSTMAutoencoderConfig,
)

import json


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class NASAInferenceResult:
    scores: np.ndarray
    predictions: np.ndarray
    threshold: float


class NASAInference:
    """
    Inferencia NASA SMAP/MSL con modelos previamente entrenados.

    - Isolation Forest por observacion.
    - LSTM Autoencoder por ventana temporal.
    - Umbrales fijos calibrados sobre X_val.
    - Sin entrenamiento ni recalibracion.
    """

    def __init__(
        self,
        machine: str,
        device: str = "cpu",
    ):
        self.machine = machine
        self.device = torch.device(device)

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA no esta disponible")

        if_dir = ROOT / "models/nasa/isolation_forest"
        lstm_dir = ROOT / "models/nasa/lstm_autoencoder"
        calibration_dir = ROOT / "results/calibration/nasa"

        if_metadata_path = if_dir / f"{machine}_metadata.json"
        if_model_path = if_dir / f"{machine}.joblib"
        lstm_path = lstm_dir / f"{machine}.pt"
        calibration_path = (
            calibration_dir / f"{machine}_lstm_threshold.json"
        )

        for path in (
            if_metadata_path,
            if_model_path,
            lstm_path,
            calibration_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        if_metadata = json.loads(
            if_metadata_path.read_text(encoding="utf-8")
        )

        if if_metadata["channel"] != machine:
            raise ValueError("Metadata IF de otro canal")

        if if_metadata["threshold_policy"] != "validation_quantile_0.99":
            raise ValueError("Politica IF no compatible")

        self.if_model = joblib.load(if_model_path)
        self.if_threshold = float(if_metadata["threshold"])

        checkpoint = torch.load(
            lstm_path,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["machine"] != machine:
            raise ValueError("Checkpoint LSTM de otro canal")

        self.lstm_config = LSTMAutoencoderConfig(
            **checkpoint["model_config"]
        )

        self.lstm_model = LSTMAutoencoder(self.lstm_config)
        self.lstm_model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.lstm_model.to(self.device)
        self.lstm_model.eval()

        calibration = json.loads(
            calibration_path.read_text(encoding="utf-8")
        )

        if calibration["machine"] != machine:
            raise ValueError("Calibracion LSTM de otro canal")

        if calibration["model_config"] != checkpoint["model_config"]:
            raise ValueError("Configuracion LSTM incompatible")

        self.lstm_threshold = float(
            calibration["calibration"]["threshold"]
        )

        if self.if_model.n_features_in_ != self.lstm_config.input_size:
            raise ValueError("Dimensiones IF y LSTM incompatibles")

        if if_metadata["input_size"] != self.lstm_config.input_size:
            raise ValueError("Metadata IF incompatible")

        if if_metadata["window_size"] != self.lstm_config.window_size:
            raise ValueError("Ventanas IF/LSTM incompatibles")

    def predict_isolation_forest(
        self,
        X: np.ndarray,
    ) -> NASAInferenceResult:
        values = np.asarray(X, dtype=np.float32)

        if values.ndim != 2 or values.shape[1] != self.lstm_config.input_size:
            raise ValueError("Dimensiones IF incorrectas")

        if not np.isfinite(values).all():
            raise ValueError("Valores IF no finitos")

        scores = -self.if_model.score_samples(values)
        predictions = (scores > self.if_threshold).astype(np.int8)

        return NASAInferenceResult(
            scores=scores,
            predictions=predictions,
            threshold=self.if_threshold,
        )

    def predict_lstm_window(
        self,
        window: np.ndarray,
    ) -> NASAInferenceResult:
        values = np.asarray(window, dtype=np.float32)

        expected_shape = (
            self.lstm_config.window_size,
            self.lstm_config.input_size,
        )

        if values.shape != expected_shape:
            raise ValueError(
                f"Ventana incorrecta: {values.shape}, "
                f"esperada: {expected_shape}"
            )

        if not np.isfinite(values).all():
            raise ValueError("Valores LSTM no finitos")

        _, errors = self.lstm_model.predict(
            values[np.newaxis, :, :],
            batch_size=1,
            device=str(self.device),
        )

        predictions = (
            errors > self.lstm_threshold
        ).astype(np.int8)

        return NASAInferenceResult(
            scores=errors,
            predictions=predictions,
            threshold=self.lstm_threshold,
        )
