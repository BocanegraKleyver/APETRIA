
"""
APETRIA - Reordenamiento temporal de observaciones.

Reconstruye secuencias a partir de observaciones
que pueden llegar fuera de orden.

Utiliza indices originales de secuencia.
No inventa timestamps ni observaciones.
No ejecuta inferencia ni modifica modelos.

La salida timeline conserva el orden cronologico
entre observaciones liberadas y discontinuidades.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np


@dataclass(frozen=True)
class TemporalReorderConfig:
    expected_features: int = 38
    max_out_of_order: int = 2
    initial_sequence_id: int = 0

    def __post_init__(self):
        for name, value, minimum in (
            ("expected_features", self.expected_features, 1),
            ("max_out_of_order", self.max_out_of_order, 0),
            ("initial_sequence_id", self.initial_sequence_id, 0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                raise ValueError(
                    f"{name} debe ser un entero >= {minimum}."
                )


@dataclass(frozen=True)
class ReorderedObservation:
    sequence_id: int
    values: np.ndarray


@dataclass(frozen=True)
class SequenceGap:
    start_sequence_id: int
    end_sequence_id_exclusive: int

    @property
    def missing_count(self) -> int:
        return (
            self.end_sequence_id_exclusive
            - self.start_sequence_id
        )


TemporalItem = Union[
    ReorderedObservation,
    SequenceGap,
]


@dataclass(frozen=True)
class TemporalReorderResult:
    released: List[ReorderedObservation]
    gaps: List[SequenceGap]
    accepted: bool
    status: str

    # Salida cronologica unificada.
    timeline: List[TemporalItem] = field(
        default_factory=list
    )


class TemporalReorder:
    """
    Reordenador temporal por maquina.

    Recibe observaciones con sequence_id original.

    Libera observaciones consecutivas y declara
    huecos cuando una posicion queda fuera del
    margen de desorden permitido.

    El consumidor debe procesar timeline en orden.
    """

    def __init__(
        self,
        machine: str,
        config: Optional[TemporalReorderConfig] = None,
    ):
        if (
            not isinstance(machine, str)
            or not machine.strip()
        ):
            raise ValueError(
                "Debe especificarse una maquina."
            )

        self.machine = machine
        self.config = config or TemporalReorderConfig()

        self.reset()

    @property
    def next_expected(self) -> int:
        return self._next_expected

    @property
    def max_seen(self) -> int:
        return self._max_seen

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _release_ready(
        self,
    ) -> List[ReorderedObservation]:
        released = []

        while self._next_expected in self._pending:
            item = self._pending.pop(
                self._next_expected
            )

            released.append(item)

            self._released += 1
            self._next_expected += 1

        return released

    def _build_result(
        self,
        timeline: List[TemporalItem],
        accepted: bool,
        status: str,
    ) -> TemporalReorderResult:
        released = [
            item
            for item in timeline
            if isinstance(item, ReorderedObservation)
        ]

        gaps = [
            item
            for item in timeline
            if isinstance(item, SequenceGap)
        ]

        return TemporalReorderResult(
            released=released,
            gaps=gaps,
            accepted=accepted,
            status=status,
            timeline=timeline,
        )

    def _advance(self) -> List[TemporalItem]:
        timeline = []

        cutoff = (
            self._max_seen
            - self.config.max_out_of_order
        )

        while self._next_expected <= cutoff:
            ready = self._release_ready()

            if ready:
                timeline.extend(ready)
                continue

            gap_start = self._next_expected

            while (
                self._next_expected <= cutoff
                and self._next_expected not in self._pending
            ):
                self._next_expected += 1

            gap = SequenceGap(
                start_sequence_id=gap_start,
                end_sequence_id_exclusive=self._next_expected,
            )

            timeline.append(gap)

            self._gaps += 1
            self._missing += gap.missing_count

        # Tambien liberar los eventos consecutivos
        # disponibles despues del ultimo hueco.
        timeline.extend(self._release_ready())

        return timeline

    def add(
        self,
        sequence_id: int,
        observation,
    ) -> TemporalReorderResult:

        if (
            isinstance(sequence_id, bool)
            or not isinstance(sequence_id, int)
            or sequence_id < 0
        ):
            raise ValueError(
                "sequence_id debe ser un entero no negativo."
            )

        values = np.asarray(
            observation,
            dtype=np.float32,
        )

        if values.shape != (
            self.config.expected_features,
        ):
            raise ValueError(
                "Forma de observacion incorrecta: "
                f"{values.shape}"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "La observacion contiene valores no finitos."
            )

        self._received += 1

        # Un identificador ya declarado perdido o liberado
        # no puede reingresar a la secuencia cronologica.
        if sequence_id < self._next_expected:
            self._late += 1

            return self._build_result(
                timeline=[],
                accepted=False,
                status="late_or_duplicate",
            )

        if sequence_id in self._pending:
            self._duplicates += 1

            return self._build_result(
                timeline=[],
                accepted=False,
                status="duplicate",
            )

        # Regla explicita de frontera temporal, coherente
        # con WatermarkTracker: un evento situado en la
        # watermark (o por debajo) ya es demasiado tardio.
        # _advance() normalmente ya lo excluye mediante
        # _next_expected; esta guarda documenta y refuerza
        # el contrato de admision sin cambiar las interfaces.
        if (
            self._max_seen >= self.config.initial_sequence_id
            and sequence_id
            <= self._max_seen - self.config.max_out_of_order
        ):
            self._late += 1

            return self._build_result(
                timeline=[],
                accepted=False,
                status="late_or_duplicate",
            )

        self._pending[sequence_id] = ReorderedObservation(
            sequence_id=sequence_id,
            values=values.copy(),
        )

        self._accepted += 1

        self._max_seen = max(
            self._max_seen,
            sequence_id,
        )

        self._peak_pending = max(
            self._peak_pending,
            len(self._pending),
        )

        timeline = self._advance()

        return self._build_result(
            timeline=timeline,
            accepted=True,
            status="accepted",
        )

    def flush(
        self,
        declare_missing: bool = False,
    ) -> TemporalReorderResult:
        """
        Finaliza o drena una secuencia.

        declare_missing=False:
            Libera solamente eventos consecutivos.

        declare_missing=True:
            Declara huecos pendientes hasta el
            maximo indice recibido.

        No utilizar declare_missing=True mientras
        puedan llegar eventos pendientes.
        """

        if not declare_missing:
            return self._build_result(
                timeline=self._release_ready(),
                accepted=True,
                status="flushed",
            )

        timeline = []

        while self._pending:
            ready = self._release_ready()

            if ready:
                timeline.extend(ready)
                continue

            next_available = min(self._pending)

            gap = SequenceGap(
                start_sequence_id=self._next_expected,
                end_sequence_id_exclusive=next_available,
            )

            timeline.append(gap)

            self._gaps += 1
            self._missing += gap.missing_count

            self._next_expected = next_available

        return self._build_result(
            timeline=timeline,
            accepted=True,
            status="flushed",
        )

    def reset(self):
        self._pending = {}

        self._next_expected = (
            self.config.initial_sequence_id
        )

        self._max_seen = (
            self.config.initial_sequence_id - 1
        )

        self._received = 0
        self._accepted = 0
        self._released = 0
        self._duplicates = 0
        self._late = 0
        self._gaps = 0
        self._missing = 0
        self._peak_pending = 0

    def info(self) -> dict:
        return {
            "project": "APETRIA",
            "component": "temporal_reorder",
            "machine": self.machine,
            "max_out_of_order": (
                self.config.max_out_of_order
            ),
            "next_expected": self._next_expected,
            "max_seen": self._max_seen,
            "pending": len(self._pending),
            "received": self._received,
            "accepted": self._accepted,
            "released": self._released,
            "duplicates": self._duplicates,
            "late": self._late,
            "gaps": self._gaps,
            "missing": self._missing,
            "peak_pending": self._peak_pending,
        }
