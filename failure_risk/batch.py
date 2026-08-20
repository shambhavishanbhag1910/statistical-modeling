from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from failure_risk.config import DEFAULT_MODEL_PATH
from failure_risk.inference import FailureRiskService


def score_file(input_path: Path, output_path: Path, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    service = FailureRiskService(model_path)
    df = pd.read_csv(input_path)
    predictions = []
    for _, row in df.iterrows():
        payload = row.to_dict()
        pred = service.predict_one(payload)
        predictions.append(pred.__dict__)
    result = pd.concat([df.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    score_file(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
