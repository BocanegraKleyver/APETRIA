from collections import deque
from dataclasses import dataclass
from threading import Condition, RLock
from time import monotonic
from typing import Literal, Optional
import numpy as np

OverflowPolicy = Literal['reject', 'drop_newest', 'drop_oldest']

class BackpressureError(RuntimeError):
    """La cola alcanzo su capacidad maxima."""

class BackpressureClosedError(RuntimeError):
    """La cola fue cerrada y no admite nuevos eventos."""

@dataclass(frozen=True)
class BackpressureConfig:
    max_queue_size: int = 1000
    expected_features: int = 38
    overflow_policy: OverflowPolicy = 'reject'
    high_watermark: float = 0.80
    low_watermark: float = 0.50

    def __post_init__(self):
        if isinstance(self.max_queue_size, bool) or not isinstance(self.max_queue_size, int) or self.max_queue_size < 1:
            raise ValueError('max_queue_size debe ser positivo.')
        if isinstance(self.expected_features, bool) or not isinstance(self.expected_features, int) or self.expected_features < 1:
            raise ValueError('expected_features debe ser positivo.')
        if self.overflow_policy not in ('reject', 'drop_newest', 'drop_oldest'):
            raise ValueError('Politica de saturacion invalida.')
        if not (0 <= self.low_watermark < self.high_watermark <= 1):
            raise ValueError('Se requiere 0 <= low_watermark < high_watermark <= 1.')

@dataclass(frozen=True)
class QueuedObservation:
    sequence_id: int
    values: np.ndarray

class BackpressureQueue:
    """Cola FIFO acotada y sincronizada para una maquina.

    enqueue/dequeue mantienen comportamiento no bloqueante original.
    put/get son bloqueantes opcionales, con timeout y cierre coordinado.
    put espera recuperacion de backpressure (<= low_watermark), no solo espacio.
    close() impide nuevas admisiones y permite drenar eventos pendientes.
    No hay persistencia ni garantia de entrega end-to-end.
    """

    def __init__(self, machine: str, config: Optional[BackpressureConfig] = None):
        if not isinstance(machine, str) or not machine.strip():
            raise ValueError('Debe especificarse una maquina.')
        self.machine = machine
        self.config = config or BackpressureConfig()
        self._condition = Condition(RLock())
        self._queue = deque()
        self._next_sequence_id = 0
        self._accepted = 0
        self._dequeued = 0
        self._dropped = 0
        self._rejected = 0
        self._backpressure_active = False
        self._activations = 0
        self._recoveries = 0
        self._peak_queue_size = 0
        self._closed = False
        self._producer_waits = 0
        self._consumer_waits = 0
        self._producer_wait_seconds = 0.0
        self._consumer_wait_seconds = 0.0
        self._producer_timeouts = 0
        self._consumer_timeouts = 0

    @property
    def size(self) -> int:
        with self._condition:
            return len(self._queue)

    @property
    def capacity(self) -> int:
        return self.config.max_queue_size

    @property
    def occupancy(self) -> float:
        with self._condition:
            return len(self._queue) / self.capacity

    @property
    def is_full(self) -> bool:
        with self._condition:
            return len(self._queue) >= self.capacity

    @property
    def backpressure_active(self) -> bool:
        with self._condition:
            return self._backpressure_active

    @property
    def should_throttle(self) -> bool:
        with self._condition:
            return self._backpressure_active

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def _update_backpressure(self):
        # Requiere self._condition adquirido.
        occupancy = len(self._queue) / self.capacity
        if not self._backpressure_active and occupancy >= self.config.high_watermark:
            self._backpressure_active = True
            self._activations += 1
        elif self._backpressure_active and occupancy <= self.config.low_watermark:
            self._backpressure_active = False
            self._recoveries += 1

    def _prepare(self, observation, sequence_id):
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (self.config.expected_features,):
            raise ValueError('Dimensiones de observacion invalidas.')
        if not np.isfinite(values).all():
            raise ValueError('La observacion contiene valores no finitos.')
        if sequence_id is not None and (
            isinstance(sequence_id, bool) or not isinstance(sequence_id, int) or sequence_id < 0
        ):
            raise ValueError('sequence_id debe ser un entero no negativo.')
        return values.copy()

    def _enqueue_locked(self, values, sequence_id):
        if self._closed:
            raise BackpressureClosedError(f'Cola cerrada para {self.machine}.')
        assigned = self._next_sequence_id if sequence_id is None else sequence_id
        if len(self._queue) >= self.capacity:
            if self.config.overflow_policy == 'reject':
                self._rejected += 1
                raise BackpressureError(f'Cola saturada para {self.machine}.')
            if self.config.overflow_policy == 'drop_newest':
                self._dropped += 1
                if sequence_id is None:
                    self._next_sequence_id += 1
                return False
            self._queue.popleft()
            self._dropped += 1
        self._queue.append(QueuedObservation(sequence_id=assigned, values=values))
        if sequence_id is None:
            self._next_sequence_id += 1
        self._accepted += 1
        self._peak_queue_size = max(self._peak_queue_size, len(self._queue))
        self._update_backpressure()
        self._condition.notify_all()
        return True

    def enqueue(self, observation: np.ndarray, sequence_id: Optional[int] = None) -> bool:
        """Admision inmediata original; no espera por backpressure."""
        values = self._prepare(observation, sequence_id)
        with self._condition:
            return self._enqueue_locked(values, sequence_id)

    def dequeue(self) -> Optional[QueuedObservation]:
        """Extraccion inmediata original; None si no hay elementos."""
        with self._condition:
            if not self._queue:
                return None
            item = self._queue.popleft()
            self._dequeued += 1
            self._update_backpressure()
            self._condition.notify_all()
            return item

    @staticmethod
    def _deadline(timeout):
        if timeout is None:
            return None
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not np.isfinite(timeout) or timeout < 0:
            raise ValueError('timeout debe ser un numero finito >= 0 o None.')
        return monotonic() + timeout

    def put(self, observation: np.ndarray, sequence_id: Optional[int] = None, timeout: Optional[float] = None) -> bool:
        """Espera hasta que no haya backpressure y exista capacidad.

        Retorna False al vencer timeout, sin descartar el evento.
        Lanza BackpressureClosedError si se cierra durante la espera.
        No aplica las politicas de overflow: regula antes de admitir.
        """
        deadline = self._deadline(timeout)
        values = self._prepare(observation, sequence_id)
        with self._condition:
            while True:
                if self._closed:
                    raise BackpressureClosedError(f'Cola cerrada para {self.machine}.')
                if not self._backpressure_active and len(self._queue) < self.capacity:
                    return self._enqueue_locked(values, sequence_id)
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    self._producer_timeouts += 1
                    return False
                self._producer_waits += 1
                start = monotonic()
                try:
                    self._condition.wait(remaining)
                finally:
                    self._producer_wait_seconds += monotonic() - start

    def get(self, timeout: Optional[float] = None) -> Optional[QueuedObservation]:
        """Espera por un elemento. Retorna None ante timeout o cierre drenado."""
        deadline = self._deadline(timeout)
        with self._condition:
            while True:
                if self._queue:
                    return self.dequeue()
                if self._closed:
                    return None
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    self._consumer_timeouts += 1
                    return None
                self._consumer_waits += 1
                start = monotonic()
                try:
                    self._condition.wait(remaining)
                finally:
                    self._consumer_wait_seconds += monotonic() - start

    def close(self) -> None:
        """Cierre idempotente: despierta esperas y permite drenar la cola."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def info(self) -> dict:
        with self._condition:
            occupancy = len(self._queue) / self.capacity
            return {
                'project': 'APETRIA', 'component': 'backpressure', 'machine': self.machine,
                'capacity': self.capacity, 'queue_size': len(self._queue),
                'occupancy': occupancy, 'occupancy_percent': round(occupancy * 100, 2),
                'overflow_policy': self.config.overflow_policy,
                'high_watermark': self.config.high_watermark,
                'low_watermark': self.config.low_watermark,
                'backpressure_active': self._backpressure_active,
                'should_throttle': self._backpressure_active,
                'activations': self._activations, 'recoveries': self._recoveries,
                'peak_queue_size': self._peak_queue_size,
                'accepted': self._accepted, 'dequeued': self._dequeued,
                'dropped': self._dropped, 'rejected': self._rejected,
                'next_sequence_id': self._next_sequence_id,
                'closed': self._closed,
                'producer_waits': self._producer_waits,
                'consumer_waits': self._consumer_waits,
                'producer_wait_seconds': self._producer_wait_seconds,
                'consumer_wait_seconds': self._consumer_wait_seconds,
                'producer_timeouts': self._producer_timeouts,
                'consumer_timeouts': self._consumer_timeouts,
            }
