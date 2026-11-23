"""Reproduccion historica HDFS por sesion BlockId; sin inferencia online."""

from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS = ROOT / "results" / "hybrid" / "hdfs" / "hdfs_hybrid_predictions.csv"


class HDFSReplay:
    """Lectura y validacion de predicciones offline HDFS."""

    POLICIES = ("if", "lstm", "and", "or")

    def __init__(self):
        if not PREDICTIONS.is_file():
            raise FileNotFoundError(
                f"Predicciones HDFS ausentes: {PREDICTIONS}"
            )

        self.predictions = pd.read_csv(
            PREDICTIONS,
            dtype={"BlockId": "string"},
        )

        required = {
            "BlockId", "label", "if_pred", "lstm_pred",
            "and_pred", "or_pred", "lstm_score",
        }

        missing = required - set(self.predictions.columns)
        if missing:
            raise ValueError(f"Columnas HDFS ausentes: {sorted(missing)}")

        if self.predictions["BlockId"].isna().any():
            raise ValueError("HDFS contiene BlockId vacios")

        if self.predictions["BlockId"].duplicated().any():
            raise ValueError("HDFS contiene BlockId duplicados")

    def replay(
        self,
        start_index: int = 0,
        limit: int = 100,
        policy: str = "lstm",
    ):
        if policy not in self.POLICIES:
            raise ValueError(f"Politica HDFS invalida: {policy}")

        if start_index < 0 or limit < 1 or limit > 10000:
            raise ValueError("Rango de reproduccion HDFS invalido")

        end = start_index + limit

        if end > len(self.predictions):
            raise ValueError(
                f"Rango [{start_index}, {end}) excede "
                f"las {len(self.predictions)} sesiones HDFS"
            )

        selected = self.predictions.iloc[start_index:end]
        results = []

        for index, row in enumerate(
            selected.itertuples(index=False),
            start=start_index,
        ):
            label = int(row.label)
            pred_if = int(row.if_pred)
            pred_lstm = int(row.lstm_pred)
            pred_and = int(row.and_pred)
            pred_or = int(row.or_pred)
            score_lstm = float(row.lstm_score)

            if any(
                value not in (0, 1)
                for value in (
                    label, pred_if, pred_lstm, pred_and, pred_or
                )
            ):
                raise ValueError(f"Prediccion HDFS invalida: {index}")

            if not math.isfinite(score_lstm):
                raise ValueError(f"Score LSTM no finito: {index}")

            if pred_and != (pred_if & pred_lstm):
                raise ValueError(f"AND inconsistente: {index}")

            if pred_or != (pred_if | pred_lstm):
                raise ValueError(f"OR inconsistente: {index}")

            predictions = {
                "if": pred_if,
                "lstm": pred_lstm,
                "and": pred_and,
                "or": pred_or,
            }

            results.append({
                "dataset": "hdfs",
                "partition": "historical_test",
                "block_id": str(row.BlockId),
                "point_index": index,
                "sequence_id": index,
                "ground_truth": label,
                "isolation_forest_prediction": pred_if,
                "isolation_forest_score": None,
                "lstm_prediction": pred_lstm,
                "lstm_score": score_lstm,
                "and_prediction": pred_and,
                "or_prediction": pred_or,
                "combined_prediction": predictions[policy],
                "policy": policy,
                "historical_replay": True,
                "inference_online": False,
                "unit": "block_session",
            })

        return results
