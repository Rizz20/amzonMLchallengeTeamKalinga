import argparse
import sys
from pathlib import Path

from src.config import DATASET_DIR, MODELS_DIR, OUTPUT_DIR, TEST_DIR, TRAIN_DIR
from src.inference import run_inference
from src.train import train_pipeline
from utils.validate_submission import validate_submission


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Pipeline for Business Entity Resolution ML Challenge."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference", "validate", "all"],
        default="all",
        help="Pipeline execution mode.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(TRAIN_DIR),
        help="Path to training directory.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(TEST_DIR),
        help="Path to test directory.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Path to save/load models.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Path to output predictions.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional manual probability threshold override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)

    print(f"Executing Pipeline in mode: [{args.mode.upper()}]")

    if args.mode in ["train", "all"]:
        print("\n>>> STAGE 1: TRAINING & THRESHOLD OPTIMIZATION <<<")
        train_pipeline(train_dir=train_dir, models_dir=models_dir)

    if args.mode in ["inference", "all"]:
        print("\n>>> STAGE 2: INFERENCE & SUBMISSION GENERATION <<<")
        cand_path, match_path = run_inference(
            test_dir=test_dir,
            models_dir=models_dir,
            output_dir=output_dir,
            custom_threshold=args.threshold,
        )

    if args.mode in ["validate", "all"]:
        print("\n>>> STAGE 3: SUBMISSION VERIFICATION <<<")
        match_path = output_dir / "matching_results.tsv"
        cand_path = output_dir / "candidate_pairs.tsv"
        try:
            validate_submission(
                matching_file=str(match_path),
                candidate_file=str(cand_path),
                test_dir=str(test_dir),
            )
        except SystemExit as e:
            if e.code != 0:
                print(f"[ERROR] Submission validation failed with code {e.code}")
                sys.exit(e.code)

    print("\n[SUCCESS] Pipeline executed successfully.")


if __name__ == "__main__":
    main()
