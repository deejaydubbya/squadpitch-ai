from __future__ import annotations

import argparse
import json

from squadpitch_ai.retrieval.benchmark import run_retrieval_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SquadPitch retrieval benchmark")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    report = run_retrieval_benchmark(k=args.k, output_dir=args.output_dir)
    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
