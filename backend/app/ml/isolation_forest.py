
"""
APETRIA - Componente de inferencia Isolation Forest.

Carga modelos previamente entrenados sobre SMD y sus metadatos.

Políticas disponibles:
- validation_p99: umbral provisional calibrado con X_val.
- isolation_forest_native: decisión nativa de scikit-learn.

Este módulo:
- NO entrena modelos.
- NO recalibra umbrales.
- NO accede a etiquetas de prueba.
- NO aplica preprocesamiento sobre datos crudos.

Las entradas deben estar preprocesadas con los parámetros
correspondientes a la máquina SMD utilizada.
"""

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "isolation_forest"

EXPECTED_FEATURES = 38

POLICY_P99 = "validation_p99"
POLICY_NATIVE = "isolation_forest_native"

SUPPORTED_POLICIES = {POLICY_P99, POLICY_NATIVE}


@dataclass(frozen=True)
class IsolationForestResult:
    machine: str
    policy: str
    threshold: float
    anomaly_scores: np.ndarray
    predictions: np.ndarray

    @property
    def total_observations(self) -> int:
        return int(len(self.predictions))

    @property
    def total_anomalies(self) -> int:
        return int(np.sum(self.predictions))


class SMDIsolationForest:
    """
    Motor de inferencia para una máquina SMD.

    Cada instancia utiliza el modelo entrenado y el umbral
    correspondiente a una máquina específica.
    """

    def __init__(
        self,
        machine: str,
        model_dir: Path | str | None = None,
    ) -> None:

        if not isinstance(machine, str):
            raise TypeError("machine debe ser una cadena.")

        if not machine.startswith("machine-"):
            raise ValueError(
                "El identificador debe tener formato machine-X-Y."
            )

        if not all(
            part.isdigit()
            for part in machine.removeprefix("machine-").split("-")
        ):
            raise ValueError(
                "Identificador de máquina SMD inválido."
            )

        self.machine = machine

        self.model_dir = Path(
            model_dir if model_dir is not None else DEFAULT_MODEL_DIR
        ).resolve()

        self.model_path = self.model_dir / f"{machine}.joblib"

        self.metadata_path = (
            self.model_dir / f"{machine}_metadata.json"
        )

        self.metadata = self._load_metadata()
        self.model = self._load_model()

        self.threshold_p99 = float(
            self.metadata["threshold"]
        )

        self.threshold_native = float(
            -self.model.offset_
        )

    def _load_metadata(self) -> dict:

        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"No existen metadatos: {self.metadata_path}"
            )

        with self.metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        if not isinstance(metadata, dict):
            raise ValueError("Formato de metadatos inválido.")

        if metadata.get("machine") != self.machine:
            raise ValueError(
                "El identificador del modelo y sus metadatos "
                "no coinciden."
            )

        if metadata.get("dataset") != "SMD":
            raise ValueError("Dataset inesperado.")

        if metadata.get("algorithm") != "IsolationForest":
            raise ValueError("Algoritmo inesperado.")

        if metadata.get("features") != EXPECTED_FEATURES:
            raise ValueError(
                "Cantidad de variables incompatible."
            )

        if metadata.get("test_data_used") is not False:
            raise ValueError(
                "Los metadatos no garantizan la separación "
                "del conjunto de prueba."
            )

        if (
            metadata.get("preprocessing_fit_partition")
            != "X_train_only"
        ):
            raise ValueError(
                "Preprocesamiento de entrenamiento inesperado."
            )

        if (
            metadata.get("threshold_method")
            != "validation_score_quantile"
        ):
            raise ValueError(
                "Método de calibración inesperado."
            )

        if not np.isclose(
            float(metadata["threshold_quantile"]),
            0.99,
        ):
            raise ValueError(
                "El umbral no corresponde al percentil 99."
            )

        threshold = float(metadata["threshold"])

        if not np.isfinite(threshold):
            raise ValueError("El umbral P99 no es finito.")

        return metadata

    def _load_model(self) -> IsolationForest:

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"No existe el modelo: {self.model_path}"
            )

        # Los archivos joblib deben provenir exclusivamente
        # de nuestro entrenamiento y ser de confianza.
        model = joblib.load(self.model_path)

        if not isinstance(model, IsolationForest):
            raise TypeError(
                "El archivo no contiene un IsolationForest."
            )

        if model.n_features_in_ != EXPECTED_FEATURES:
            raise ValueError(
                "El modelo tiene una cantidad inesperada "
                "de variables."
            )

        return model

    @staticmethod
    def _validate_input(X: np.ndarray) -> np.ndarray:

        matrix = np.asarray(X, dtype=np.float64)

        if matrix.ndim != 2:
            raise ValueError(
                "La entrada debe tener forma "
                "(observaciones, 38)."
            )

        if matrix.shape[0] == 0:
            raise ValueError(
                "La entrada no contiene observaciones."
            )

        if matrix.shape[1] != EXPECTED_FEATURES:
            raise ValueError(
                f"Se esperaban {EXPECTED_FEATURES} variables; "
                f"se recibieron {matrix.shape[1]}."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "La entrada contiene NaN o valores infinitos."
            )

        return matrix

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula puntajes continuos de anomalía.

        Mayor puntaje = mayor anomalía.
        """

        matrix = self._validate_input(X)

        return -self.model.score_samples(matrix)

    def predict(
        self,
        X: np.ndarray,
        policy: str = POLICY_P99,
    ) -> IsolationForestResult:
        """
        Ejecuta inferencia sin modificar el modelo.

        Retorna:
        - Puntajes de anomalía.
        - Predicciones binarias: 0 normal, 1 anomalía.
        - Umbral correspondiente a la política.
        """

        if policy not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Política inválida: {policy}. "
                f"Opciones: {sorted(SUPPORTED_POLICIES)}"
            )

        matrix = self._validate_input(X)

        scores = -self.model.score_samples(matrix)

        if policy == POLICY_P99:

            threshold = self.threshold_p99

            predictions = (
                scores > threshold
            ).astype(np.int8)

        else:

            threshold = self.threshold_native

            predictions = (
                self.model.predict(matrix) == -1
            ).astype(np.int8)

        return IsolationForestResult(
            machine=self.machine,
            policy=policy,
            threshold=threshold,
            anomaly_scores=scores,
            predictions=predictions,
        )

    def info(self) -> dict:
        """
        Devuelve información del modelo cargado.
        """

        return {
            "machine": self.machine,
            "dataset": "SMD",
            "algorithm": "IsolationForest",
            "features": EXPECTED_FEATURES,
            "threshold_p99": self.threshold_p99,
            "threshold_native": self.threshold_native,
            "threshold_status": self.metadata.get(
                "threshold_status"
            ),
            "model_file": self.model_path.name,
            "trained_at_utc": self.metadata.get(
                "trained_at_utc"
            ),
        }
