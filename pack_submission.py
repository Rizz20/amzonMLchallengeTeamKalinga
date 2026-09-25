import argparse
import os
from pathlib import Path
import zipfile


from src.config import OUTPUT_DIR, WORKING_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Package ER pipeline submission into required ZIP format.")
    parser.add_argument("--team-name", type=str, default="TeamKalinga", help="Team name prefix for zip archive.")
    return parser.parse_args()


def create_submission_zip(team_name: str = "TeamKalinga"):
    base_dir = Path(__file__).resolve().parent
    zip_filename = f"{team_name}_submission.zip"
    zip_path = WORKING_DIR / zip_filename

    print(f"Creating submission package: {zip_path}...")

    # Files to include
    files_to_pack = []

    # 1. Output folder
    for f in ["candidate_pairs.tsv", "matching_results.tsv"]:
        p = OUTPUT_DIR / f
        if not p.exists():
            p = base_dir / "output" / f
        if p.exists():
            files_to_pack.append((p, f"output/{f}"))
        else:
            print(f"[WARN] {p} not found! Make sure to run inference first.")

    # 2. Documentation template
    doc_file = base_dir / "Documentation_template.md"
    if doc_file.exists():
        files_to_pack.append((doc_file, "Documentation_template.md"))

    # 3. Code package under code/business_entity_resolution/
    code_base = base_dir
    code_items = [
        ("requirements.txt", "code/business_entity_resolution/requirements.txt"),
        ("README.md", "code/business_entity_resolution/README.md"),
        ("run_pipeline.py", "code/business_entity_resolution/run_pipeline.py"),
    ]
    for src_rel, dest_rel in code_items:
        src_path = code_base / src_rel
        if src_path.exists():
            files_to_pack.append((src_path, dest_rel))

    # Add src/ and utils/
    for sub in ["src", "utils"]:
        sub_dir = code_base / sub
        if sub_dir.exists():
            for root, _, files in os.walk(sub_dir):
                for f in files:
                    if f.endswith((".py", ".json")):
                        full_p = Path(root) / f
                        rel_in_sub = full_p.relative_to(code_base)
                        files_to_pack.append((full_p, f"code/business_entity_resolution/{rel_in_sub.as_posix()}"))

    # Create ZIP archive
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_f:
        for src_path, arcname in files_to_pack:
            zip_f.write(src_path, arcname=arcname)
            print(f"  + Added: {arcname}")

    print(f"\n[SUCCESS] Packaged {len(files_to_pack)} files into {zip_filename}")
    print(f"File size: {zip_path.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    args = parse_args()
    create_submission_zip(args.team_name)
