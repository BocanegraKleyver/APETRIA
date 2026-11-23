
"""
APETRIA - Control temporal mediante watermarks.

Clasifica observaciones por indice de secuencia.

Responsabilidades:
- Identificar eventos en orden.
- Identificar eventos fuera de orden.
- Rechazar eventos tardios.
- Detectar duplicados recientes.
- Mantener memoria acotada.

No reordena eventos.
No ejecuta inferencia.
No inventa timestamps.

La reconstruccion cronologica corresponde
a TemporalReorder.
"""

from dataclasses import dataclass
from typing import Literal, Optional


EventStatus = Literal[
    "in_order",
    "out_of_order",
    "late",
    "duplicate",
]


@dataclass(frozen=True)
class WatermarkConfig:
    allowed_lateness: int = 2

    def __post_init__(self):
        if (
            isinstance(self.allowed_lateness, bool)
            or not isinstance(self.allowed_lateness, int)
            or self.allowed_lateness < 0
        ):
            raise ValueError(
                "allowed_lateness debe ser "
                "un entero no negativo."
            )


@dataclass(frozen=True)
class WatermarkResult:
    sequence_id: int
    status: EventStatus
    watermark: int
    max_sequence_seen: int
    accepted: bool


class WatermarkTracker:
    """
    Seguimiento temporal por maquina.

    La watermark se calcula como:

        max_sequence_seen - allowed_lateness

    Un evento nuevo con indice inferior a la
    watermark se clasifica como tardio.

    Los identificadores recientes se conservan
    para detectar duplicados.

    Los identificadores antiguos se eliminan
    del conjunto de seguimiento para limitar
    el consumo de memoria.

    accepted=True significa que el evento puede
    pasar a una etapa de reordenamiento.

    No significa que pueda enviarse directamente
    al LSTM.
    """

    def __init__(
        self,
        machine: str,
        config: Optional[WatermarkConfig] = None,
    ):
        if (
            not isinstance(machine, str)
            or not machine.strip()
        ):
            raise ValueError(
                "Debe especificarse una maquina."
            )

        self.machine = machine
        self.config = config or WatermarkConfig()

        self._max_sequence_seen = -1

        # Solo conserva indices recientes.
        self._seen = set()

        self._received = 0
        self._in_order = 0
        self._out_of_order = 0
        self._late = 0
        self._duplicates = 0

        self._peak_tracked = 0

    @property
    def max_sequence_seen(self) -> int:
        return self._max_sequence_seen

    @property
    def watermark(self) -> int:
        if self._max_sequence_seen < 0:
            return -1

        return max(
            -1,
            self._max_sequence_seen
            - self.config.allowed_lateness,
        )

    def _cleanup_seen(self):
        """
        Descarta identificadores anteriores
        a la watermark.

        Un identificador antiguo que vuelva
        a llegar sera rechazado como tardio.

        No se garantiza distinguir si ese
        identificador ya habia sido recibido.
        """

        current_watermark = self.watermark

        self._seen = {
            sequence_id
            for sequence_id in self._seen
            if sequence_id >= current_watermark
        }

    def observe(
        self,
        sequence_id: int,
    ) -> WatermarkResult:
        """
        Registra y clasifica un indice.

        Estados:
        - in_order
        - out_of_order
        - late
        - duplicate

        Los eventos tardios y duplicados
        no son admitidos.
        """

        if (
            isinstance(sequence_id, bool)
            or not isinstance(sequence_id, int)
            or sequence_id < 0
        ):
            raise ValueError(
                "sequence_id debe ser un entero "
                "no negativo."
            )

        self._received += 1

        current_watermark = self.watermark

        # Primero verificar duplicados recientes.
        if sequence_id in self._seen:
            self._duplicates += 1

            return WatermarkResult(
                sequence_id=sequence_id,
                status="duplicate",
                watermark=current_watermark,
                max_sequence_seen=self._max_sequence_seen,
                accepted=False,
            )

        # Identificadores en el limite o anteriores
        # ya no son admisibles.
        if (
            self._max_sequence_seen >= 0
            and sequence_id <= current_watermark
        ):
            self._late += 1

            return WatermarkResult(
                sequence_id=sequence_id,
                status="late",
                watermark=current_watermark,
                max_sequence_seen=self._max_sequence_seen,
                accepted=False,
            )

        if sequence_id > self._max_sequence_seen:
            status: EventStatus = "in_order"

            self._max_sequence_seen = sequence_id
            self._in_order += 1

        else:
            status = "out_of_order"
            self._out_of_order += 1

        self._seen.add(sequence_id)

        self._cleanup_seen()

        self._peak_tracked = max(
            self._peak_tracked,
            len(self._seen),
        )

        return WatermarkResult(
            sequence_id=sequence_id,
            status=status,
            watermark=self.watermark,
            max_sequence_seen=self._max_sequence_seen,
            accepted=True,
        )

    def reset(self):
        """
        Reinicia el seguimiento temporal.
        """

        self._max_sequence_seen = -1
        self._seen.clear()

        self._received = 0
        self._in_order = 0
        self._out_of_order = 0
        self._late = 0
        self._duplicates = 0

        self._peak_tracked = 0

    def info(self) -> dict:
        return {
            "project": "APETRIA",
            "component": "watermarks",
            "machine": self.machine,
            "allowed_lateness": (
                self.config.allowed_lateness
            ),
            "max_sequence_seen": (
                self._max_sequence_seen
            ),
            "watermark": self.watermark,
            "received": self._received,
            "in_order": self._in_order,
            "out_of_order": self._out_of_order,
            "late": self._late,
            "duplicates": self._duplicates,
            "unique_indices_stored": len(self._seen),
            "peak_indices_stored": self._peak_tracked,
        }
