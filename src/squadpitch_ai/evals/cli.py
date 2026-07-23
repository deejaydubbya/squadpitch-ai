from __future__ import annotations

import argparse
import json

from squadpitch_ai.evals.models import EvalRunConfig
from squadpitch_ai.evals.runner import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SquadPitch AI eval datasets")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--baseline-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--run-id", default="local")
    parser.add_argument("--release-level", default="offline")
    parser.add_argument("--adapter-mode", default="mock", choices=["mock", "metadata_only"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    report = run_eval(
        EvalRunConfig(
            run_id=args.run_id,
            dataset_path=args.dataset,
            dataset_version=args.dataset_version,
            node_baseline_sha=args.baseline_sha,
            candidate_sha=args.candidate_sha,
            adapter_mode=args.adapter_mode,
            release_level=args.release_level,
        ),
        output_dir=args.output_dir,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
