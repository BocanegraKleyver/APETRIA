"""Reproduccion historica BGL por evento de log; sin inferencia online."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS = (
    ROOT / "results" / "hybrid" / "bgl"
    / "bgl_hybrid_predictions.csv.gz"
)


class BGLReplay:
    """Lectura de predicciones offline BGL auditadas."""

    POLICIES = ("if", "sgd", "and", "or")

    def __init__(self):
        if not PREDICTIONS.is_file():
            raise FileNotFoundError(
                f"Predicciones BGL ausentes: {PREDICTIONS}"
            )

        self.predictions = pd.read_csv(PREDICTIONS)

        required = {
            "line_number", "label", "if_pred",
            "sgd_pred", "and_pred", "or_pred",
        }

        missing = required - set(self.predictions.columns)

        if missing:
            raise ValueError(
                f"Columnas BGL ausentes: {sorted(missing)}"
            )

    def replay(
        self,
        start_index: int = 0,
        limit: int = 100,
        policy: str = "sgd",
    ):
        if policy not in self.POLICIES:
            raise ValueError(f"Politica BGL invalida: {policy}")

        if start_index < 0 or limit < 1 or limit > 10000:
            raise ValueError("Rango de reproduccion BGL invalido")

        end = start_index + limit

        if end > len(self.predictions):
            raise ValueError(
                f"Rango [{start_index}, {end}) excede "
                f"los {len(self.predictions)} eventos BGL"
            )

        selected = self.predictions.iloc[start_index:end]
        results = []

        for index, row in enumerate(
            selected.itertuples(index=False),
            start=start_index,
        ):
            label = int(row.label)
            pred_if = int(row.if_pred)
            pred_sgd = int(row.sgd_pred)
            pred_and = int(row.and_pred)
            pred_or = int(row.or_pred)

            if any(
                value not in (0, 1)
                for value in (
                    label, pred_if, pred_sgd,
                    pred_and, pred_or,
                )
            ):
                raise ValueError(
                    f"Prediccion BGL invalida: {index}"
                )

            if pred_and != (pred_if & pred_sgd):
                raise ValueError(
                    f"AND inconsistente: {index}"
                )

            if pred_or != (pred_if | pred_sgd):
                raise ValueError(
                    f"OR inconsistente: {index}"
                )

            predictions = {
                "if": pred_if,
                "sgd": pred_sgd,
                "and": pred_and,
                "or": pred_or,
            }

            results.append({
                "dataset": "bgl",
                "partition": "historical_test",
                "line_number": int(row.line_number),
                "point_index": index,
                "sequence_id": index,
                "ground_truth": label,
                "isolation_forest_prediction": pred_if,
                "isolation_forest_score": None,
                "sgd_prediction": pred_sgd,
                "sgd_score": None,
                "and_prediction": pred_and,
                "or_prediction": pred_or,
                "combined_prediction": predictions[policy],
                "policy": policy,
                "historical_replay": True,
                "inference_online": False,
                "unit": "log_event",
            })

        return results
