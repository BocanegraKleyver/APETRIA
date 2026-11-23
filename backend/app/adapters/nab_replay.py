from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "processed" / "nab" / "aws_v1"
PREDICTIONS = ROOT / "results" / "hybrid" / "nab" / "nab_hybrid_predictions.csv.gz"


class NABReplay:
    """Reproduccion de predicciones historicas NAB AWS, sin inferencia en vivo."""

    def __init__(self):
        if not PREDICTIONS.is_file():
            raise FileNotFoundError(f"Predicciones NAB ausentes: {PREDICTIONS}")

        self.predictions = pd.read_csv(PREDICTIONS)
        self.series = sorted(self.predictions["series"].unique().tolist())

    def replay(self, series: str, start_index: int = 0,
               limit: int = 100, policy: str = "or"):

        if series not in self.series:
            raise ValueError(f"Serie NAB no disponible: {series}")
        if policy not in ("and", "or", "dual"):
            raise ValueError(f"Politica NAB invalida: {policy}")
        if start_index < 0 or limit < 1 or limit > 10000:
            raise ValueError("Rango de reproduccion NAB invalido")

        source_path = DATA_DIR / series
        if not source_path.is_file():
            raise FileNotFoundError(f"CSV NAB ausente: {series}")

        history = self.predictions.loc[
            self.predictions["series"] == series
        ].sort_values("point_index")

        end = start_index + limit
        if end > len(history):
            raise ValueError(
                f"Rango [{start_index}, {end}) excede "
                f"las {len(history)} observaciones evaluadas"
            )

        selected = history.iloc[start_index:end]
        source = pd.read_csv(source_path)

        results = []

        for row in selected.itertuples(index=False):
            index = int(row.point_index)

            if index < 0 or index >= len(source):
                raise ValueError(f"Indice NAB fuera de rango: {index}")

            observation = source.iloc[index]

            if (
                str(observation["timestamp"]) != str(row.timestamp)
                or int(observation["label_window"]) != int(row.label_window)
            ):
                raise ValueError(
                    f"Desalineacion entre datos y predicciones: {series}, {index}"
                )

            pred_if = int(row.pred_if)
            pred_robust = int(row.pred_robust)
            pred_and = int(row.pred_and)
            pred_or = int(row.pred_or)

            if_score = float(row.if_score)
            if_threshold = float(row.if_threshold)
            robust_score = float(row.robust_score)
            robust_threshold = float(row.robust_threshold)

            if not all(math.isfinite(v) for v in (
                if_score, if_threshold, robust_score, robust_threshold
            )):
                raise ValueError(f"Score NAB no finito: {series}, {index}")

            if int(if_score > if_threshold) != pred_if:
                raise ValueError(f"Score IF inconsistente: {series}, {index}")
            if int(robust_score > robust_threshold) != pred_robust:
                raise ValueError(f"Score Robust inconsistente: {series}, {index}")

            if pred_and != (pred_if & pred_robust):
                raise ValueError(f"AND inconsistente: {series}, {index}")
            if pred_or != (pred_if | pred_robust):
                raise ValueError(f"OR inconsistente: {series}, {index}")

            combined = (
                pred_and if policy == "and"
                else pred_or if policy == "or"
                else None
            )

            results.append({
                "dataset": "nab",
                "series": series,
                "partition": "historical_test",
                "point_index": index,
                "sequence_id": index,
                "timestamp": str(row.timestamp),
                "value": float(observation["value"]),
                "ground_truth": int(row.label_window),
                "label_point": int(observation["label_point"]),
                "isolation_forest_prediction": pred_if,
                "isolation_forest_score": if_score,
                "isolation_forest_threshold": if_threshold,
                "robust_prediction": pred_robust,
                "robust_score": robust_score,
                "robust_threshold": robust_threshold,
                "and_prediction": pred_and,
                "or_prediction": pred_or,
                "combined_prediction": combined,
                "historical_replay": True,
            })

        return results
