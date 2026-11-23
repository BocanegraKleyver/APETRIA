
"""
APETRIA - LSTM Autoencoder para detección de anomalías.

Arquitectura neuronal para reconstrucción de secuencias temporales
multivariadas provenientes de datasets reales, como SMD.

Este módulo:
- Define el encoder y decoder LSTM.
- Reconstruye ventanas temporales.
- Calcula errores de reconstrucción.
- No entrena ni calibra modelos.
- No utiliza etiquetas ni genera datos sintéticos.

El entrenamiento y la evaluación se implementan por separado.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class LSTMAutoencoderConfig:
    input_size: int = 38
    window_size: int = 30
    hidden_size: int = 64
    latent_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "input_size",
            "window_size",
            "hidden_size",
            "latent_size",
            "num_layers",
        ):
            value = getattr(self, name)

            if isinstance(value, bool) or not isinstance(
                value, int
            ):
                raise TypeError(
                    f"{name} debe ser un entero positivo."
                )

            if value < 1:
                raise ValueError(
                    f"{name} debe ser mayor que cero."
                )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout debe estar entre 0 y 1, "
                "sin incluir 1."
            )

        if self.num_layers == 1 and self.dropout != 0.0:
            raise ValueError(
                "Con num_layers=1, dropout debe ser 0.0."
            )


class LSTMAutoencoder(nn.Module):
    """
    Autoencoder LSTM multivariado.

    Entrada:
        (batch_size, window_size, input_size)

    Salida:
        (batch_size, window_size, input_size)

    El encoder comprime la secuencia en un vector latente.
    El decoder utiliza ese vector para reconstruir la ventana.
    """

    def __init__(
        self,
        config: LSTMAutoencoderConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or LSTMAutoencoderConfig()

        self.encoder = nn.LSTM(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout,
        )

        self.to_latent = nn.Linear(
            self.config.hidden_size,
            self.config.latent_size,
        )

        self.from_latent = nn.Linear(
            self.config.latent_size,
            self.config.hidden_size,
        )

        self.decoder = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout,
        )

        self.output_layer = nn.Linear(
            self.config.hidden_size,
            self.config.input_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        self._validate_tensor(x)

        _, (hidden, _) = self.encoder(x)

        # Estado oculto final de la última capa.
        encoded = hidden[-1]

        latent = self.to_latent(encoded)

        decoded_context = self.from_latent(latent)

        # El decoder recibe el contexto latente repetido
        # durante todos los pasos temporales.
        repeated_context = decoded_context.unsqueeze(1).expand(
            -1,
            self.config.window_size,
            -1,
        )

        decoded, _ = self.decoder(repeated_context)

        reconstruction = self.output_layer(decoded)

        return reconstruction

    def _validate_tensor(
        self,
        x: torch.Tensor,
    ) -> None:

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "La entrada debe ser un tensor PyTorch."
            )

        if x.ndim != 3:
            raise ValueError(
                "La entrada debe tener forma "
                "(batch, tiempo, variables)."
            )

        if x.shape[0] == 0:
            raise ValueError(
                "El lote no puede estar vacío."
            )

        if x.shape[1] != self.config.window_size:
            raise ValueError(
                f"Se esperaban {self.config.window_size} "
                f"pasos temporales; se recibieron {x.shape[1]}."
            )

        if x.shape[2] != self.config.input_size:
            raise ValueError(
                f"Se esperaban {self.config.input_size} "
                f"variables; se recibieron {x.shape[2]}."
            )

        if not torch.is_floating_point(x):
            raise TypeError(
                "El tensor debe utilizar un tipo flotante."
            )

        if not torch.isfinite(x).all().item():
            raise ValueError(
                "La entrada contiene NaN o infinitos."
            )

    @staticmethod
    def reconstruction_error(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Error cuadrático medio por ventana.

        Devuelve un valor por cada ventana del lote.
        """

        if original.shape != reconstructed.shape:
            raise ValueError(
                "Las dimensiones originales y reconstruidas "
                "no coinciden."
            )

        if original.ndim != 3:
            raise ValueError(
                "Se esperan tensores tridimensionales."
            )

        return torch.mean(
            (original - reconstructed) ** 2,
            dim=(1, 2),
        )

    @torch.no_grad()
    def predict(
        self,
        windows: np.ndarray,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reconstruye ventanas existentes y calcula sus errores.

        Devuelve:
            reconstructed: (n_ventanas, tiempo, variables)
            errors: (n_ventanas,)

        IMPORTANTE:
        Antes de entrenar, los resultados no son útiles
        para detectar anomalías.
        """

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError(
                "batch_size debe ser un entero positivo."
            )

        matrix = np.asarray(windows)

        if matrix.ndim != 3:
            raise ValueError(
                "windows debe ser una matriz tridimensional."
            )

        if matrix.shape[0] == 0:
            raise ValueError(
                "No se recibieron ventanas."
            )

        if matrix.shape[1:] != (
            self.config.window_size,
            self.config.input_size,
        ):
            raise ValueError(
                "Dimensiones incompatibles con el modelo."
            )

        if not np.issubdtype(matrix.dtype, np.number):
            raise TypeError(
                "Las ventanas deben ser numéricas."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "Las ventanas contienen NaN o infinitos."
            )

        target_device = torch.device(device)
        self.to(target_device)
        self.eval()

        reconstructions = []
        errors = []

        for start in range(0, len(matrix), batch_size):
            batch = matrix[start:start + batch_size]

            tensor = torch.as_tensor(
                np.asarray(batch, dtype=np.float32),
                device=target_device,
            )

            reconstructed = self.forward(tensor)

            batch_errors = self.reconstruction_error(
                tensor,
                reconstructed,
            )

            reconstructions.append(
                reconstructed.cpu().numpy()
            )

            errors.append(
                batch_errors.cpu().numpy()
            )

        return (
            np.concatenate(reconstructions, axis=0),
            np.concatenate(errors, axis=0),
        )

    def get_config(self) -> dict:
        """Configuración serializable del modelo."""
        return asdict(self.config)
