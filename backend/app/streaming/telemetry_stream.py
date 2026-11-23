
"""
APETRIA - Procesamiento incremental de telemetria.

Procesa observaciones reales previamente normalizadas.

- Isolation Forest: una inferencia por observacion.
- LSTM Autoencoder: una inferencia por ventana completa.
- Politicas: dual, or, and.
- Identificacion de discontinuidades temporales.
- Preservacion de identificadores originales.
- Medicion separada del tiempo de inferencia.

No entrena modelos.
No recalibra umbrales.
No genera telemetria sintetica.
"""

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Literal, Optional

import numpy as np

from backend.app.ml.inference import APETRIAInference
from backend.app.streaming.circular_buffer import (
    CircularBuffer,
    CircularBufferConfig,
)


DetectionPolicy = Literal["dual", "or", "and"]


@dataclass(frozen=True)
class TelemetryResult:
    machine: str
    point_index: int
    policy: DetectionPolicy

    isolation_forest_score: float
    isolation_forest_prediction: int
    isolation_forest_threshold: float

    lstm_score: Optional[float]
    lstm_prediction: Optional[int]
    lstm_threshold: Optional[float]

    combined_prediction: Optional[int]

    window_ready: bool
    window_start: Optional[int]

    isolation_forest_latency_ns: int
    lstm_latency_ns: Optional[int]
    processing_latency_ns: int

    # Metadatos de continuidad temporal.
    sequence_id: Optional[int] = None
    sequence_gap_detected: bool = False
    missing_observations: int = 0
    temporal_segment: int = 0


class TelemetryStream:
    """
    Procesador incremental para una maquina SMD.

    Las entradas deben tener 38 caracteristicas y
    haber sido transformadas previamente con el
    preprocesamiento correspondiente a la maquina.

    Puede recibir identificadores originales de
    secuencia para detectar discontinuidades.

    No gestiona transporte de red ni colas.
    """

    def __init__(
        self,
        machine: str,
        policy: DetectionPolicy = "dual",
        device: str = "cpu",
        inference_service: Optional[APETRIAInference] = None,
    ):
        if policy not in ("dual", "or", "and"):
            raise ValueError(
                "Politica invalida. Use dual, or o and."
            )

        self.machine = machine
        self.policy = policy

        self.inference = (
            inference_service
            if inference_service is not None
            else APETRIAInference(
                machine=machine,
                device=device,
            )
        )

        if self.inference.machine != machine:
            raise ValueError(
                "El servicio de inferencia corresponde "
                "a otra maquina."
            )

        self.buffer = CircularBuffer(
            machine=machine,
            config=CircularBufferConfig(
                window_size=(
                    self.inference.lstm_config.window_size
                ),
                expected_features=(
                    self.inference.lstm_config.input_size
                ),
                stride=1,
            ),
        )

        self._processed = 0
        self._if_alerts = 0
        self._lstm_alerts = 0

        self._last_sequence_id = None
        self._sequence_gaps = 0
        self._missing_observations = 0
        self._temporal_segment = 0
        self._segment_start_sequence_id = None

        # Contador acumulado, porque el buffer
        # reinicia sus contadores al detectar huecos.
        self._total_windows_emitted = 0

    def process(
        self,
        observation: np.ndarray,
        sequence_id: Optional[int] = None,
    ) -> TelemetryResult:
        """
        Procesa una observacion individual.

        Si sequence_id se omite, utiliza el indice
        interno consecutivo para mantener
        compatibilidad con llamadas anteriores.

        Si se detecta un salto positivo en la
        secuencia, reinicia el buffer LSTM.

        Los identificadores duplicados o anteriores
        al ultimo procesado se rechazan.

        Los tiempos son locales y no incluyen
        transporte de red ni espera en colas.
        """

        values = np.asarray(
            observation,
            dtype=np.float32,
        )

        expected_features = (
            self.inference.lstm_config.input_size
        )

        if values.shape != (expected_features,):
            raise ValueError(
                f"Se esperaba un vector de "
                f"{expected_features} caracteristicas."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "La observacion contiene "
                "valores no finitos."
            )

        if sequence_id is None:
            # Compatible con el modo secuencial
            # original sin identificadores explicitos.
            if self._last_sequence_id is None:
                current_sequence_id = 0
            else:
                current_sequence_id = (
                    self._last_sequence_id + 1
                )
        else:
            if (
                isinstance(sequence_id, bool)
                or not isinstance(sequence_id, int)
                or sequence_id < 0
            ):
                raise ValueError(
                    "sequence_id debe ser un "
                    "entero no negativo."
                )

            current_sequence_id = sequence_id

        sequence_gap_detected = False
        missing_observations = 0

        if self._last_sequence_id is not None:
            expected_sequence_id = (
                self._last_sequence_id + 1
            )

            if current_sequence_id < expected_sequence_id:
                raise ValueError(
                    "TelemetryStream requiere eventos "
                    "ordenados y sin duplicados. "
                    "Utilice TemporalReorder antes "
                    "de procesarlos."
                )

            if current_sequence_id > expected_sequence_id:
                sequence_gap_detected = True

                missing_observations = (
                    current_sequence_id
                    - expected_sequence_id
                )

                # El LSTM no puede construir ventanas
                # que atraviesen observaciones faltantes.
                self.buffer.reset()

                self._sequence_gaps += 1
                self._missing_observations += (
                    missing_observations
                )
                self._temporal_segment += 1

                self._segment_start_sequence_id = (
                    current_sequence_id
                )

        if self._last_sequence_id is None:
            self._segment_start_sequence_id = (
                current_sequence_id
            )

        processing_start = perf_counter_ns()

        if_start = perf_counter_ns()

        if_result = (
            self.inference.predict_isolation_forest(
                values[np.newaxis, :]
            )
        )

        if_latency = (
            perf_counter_ns() - if_start
        )

        if_score = float(
            if_result.scores[0]
        )

        if_prediction = int(
            if_result.predictions[0]
        )

        window_event = self.buffer.add(values)

        lstm_score = None
        lstm_prediction = None
        lstm_threshold = None
        lstm_latency = None
        window_start = None
        combined_prediction = None

        if window_event is not None:
            lstm_start = perf_counter_ns()

            lstm_result = (
                self.inference.predict_lstm_window(
                    window_event.values
                )
            )

            lstm_latency = (
                perf_counter_ns() - lstm_start
            )

            lstm_score = float(
                lstm_result.scores[0]
            )

            lstm_prediction = int(
                lstm_result.predictions[0]
            )

            lstm_threshold = float(
                lstm_result.threshold
            )

            # Convertir el indice relativo del
            # buffer al indice original SMD.
            window_start = (
                self._segment_start_sequence_id
                + window_event.window_start
            )

            self._total_windows_emitted += 1

            if self.policy == "or":
                combined_prediction = int(
                    bool(if_prediction)
                    or bool(lstm_prediction)
                )

            elif self.policy == "and":
                combined_prediction = int(
                    bool(if_prediction)
                    and bool(lstm_prediction)
                )

        point_index = self._processed

        self._processed += 1

        self._last_sequence_id = (
            current_sequence_id
        )

        self._if_alerts += if_prediction

        if lstm_prediction is not None:
            self._lstm_alerts += lstm_prediction

        processing_latency = (
            perf_counter_ns()
            - processing_start
        )

        return TelemetryResult(
            machine=self.machine,
            point_index=point_index,
            policy=self.policy,
            isolation_forest_score=if_score,
            isolation_forest_prediction=if_prediction,
            isolation_forest_threshold=float(
                if_result.threshold
            ),
            lstm_score=lstm_score,
            lstm_prediction=lstm_prediction,
            lstm_threshold=lstm_threshold,
            combined_prediction=combined_prediction,
            window_ready=(
                window_event is not None
            ),
            window_start=window_start,
            isolation_forest_latency_ns=if_latency,
            lstm_latency_ns=lstm_latency,
            processing_latency_ns=processing_latency,
            sequence_id=current_sequence_id,
            sequence_gap_detected=(
                sequence_gap_detected
            ),
            missing_observations=(
                missing_observations
            ),
            temporal_segment=(
                self._temporal_segment
            ),
        )

    def reset(self):
        """
        Reinicia contadores y continuidad temporal.

        Conserva modelos y umbrales calibrados.
        """

        self.buffer.reset()

        self._processed = 0
        self._if_alerts = 0
        self._lstm_alerts = 0

        self._last_sequence_id = None
        self._sequence_gaps = 0
        self._missing_observations = 0
        self._temporal_segment = 0
        self._segment_start_sequence_id = None

        self._total_windows_emitted = 0

    def info(self) -> dict:
        return {
            "project": "APETRIA",
            "component": "telemetry_stream",
            "machine": self.machine,
            "policy": self.policy,
            "observations_processed": (
                self._processed
            ),
            "windows_emitted": (
                self._total_windows_emitted
            ),
            "isolation_forest_alerts": (
                self._if_alerts
            ),
            "lstm_alerts": (
                self._lstm_alerts
            ),
            "window_size": (
                self.buffer.config.window_size
            ),
            "expected_features": (
                self.buffer.config.expected_features
            ),
            "last_sequence_id": (
                self._last_sequence_id
            ),
            "sequence_gaps": (
                self._sequence_gaps
            ),
            "missing_observations": (
                self._missing_observations
            ),
            "temporal_segment": (
                self._temporal_segment
            ),
            "models_retrained": False,
            "thresholds_updated": False,
        }
