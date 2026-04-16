import csv
import subprocess
import shutil
import os

# =========================
# CONFIG
# =========================
INPUT_CSV = "data/chatgpt_roles.csv"
PROGRESS_CSV = "data/progress.csv"
OUTPUT_DIR = "output"
PROMPTS_FILE = "prompts/prompt.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD INPUT CSV
# Raw prompts, no header — one prompt per row
# =========================
def load_prompts(csv_file):
    prompts = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip():
                prompts.append(row[0].strip())
    return prompts


# =========================
# LOAD PROGRESS CSV
# Returns a dict of {index: status}
# =========================
def load_progress(progress_file):
    progress = {}
    if not os.path.exists(progress_file):
        return progress
    with open(progress_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            progress[int(row["index"])] = row["status"]
    return progress


# =========================
# SAVE PROGRESS ENTRY
# Appends or updates one row in progress CSV
# =========================
def save_progress(progress_file, index, prompt_text, status, output_path=""):
    # Load existing rows
    rows = []
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Update existing row or append new one
    updated = False
    for row in rows:
        if int(row["index"]) == index:
            row["status"] = status
            row["output_file"] = output_path
            updated = True
            break
    if not updated:
        rows.append({
            "index": index,
            "prompt": prompt_text[:100],   # store first 100 chars for reference
            "status": status,
            "output_file": output_path
        })

    # Write back
    tmp = progress_file + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "prompt", "status", "output_file"]
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, progress_file)


# =========================
# MAIN LOOP
# =========================
def run_pipeline():
    print(f"Loading prompts from {INPUT_CSV}...")
    prompts = load_prompts(INPUT_CSV)
    total = len(prompts)
    print(f"Total prompts: {total}")

    progress = load_progress(PROGRESS_CSV)
    already_done = sum(1 for s in progress.values() if s == "done")
    print(f"Already done : {already_done}")
    print(f"Remaining    : {total - already_done}\n")

    for i, prompt_text in enumerate(prompts):

        # Skip already completed
        if progress.get(i) == "done":
            print(f"[{i+1}/{total}] Skipping (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {prompt_text[:80]}...")
        print(f"{'='*60}")

        try:
            # Step 1 — write prompt
            with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
                f.write(prompt_text)

            # Step 2 — generate dataset
            print(f"  [1/3] Generating dataset...")
            subprocess.run(["python", "generate_dataset.py"], check=True)

            # Step 3 — optimize
            print(f"  [2/3] Running optimizer...")
            subprocess.run(["prompt-ops", "migrate"], check=True)

            # Step 4 — save result
            print(f"  [3/3] Saving result...")
            result_files = sorted([
                f for f in os.listdir("results/")
                if f.startswith("config_") and f.endswith(".yaml")
            ])

            output_path = ""
            if result_files:
                latest = os.path.join("results", result_files[-1])
                output_path = os.path.join(OUTPUT_DIR, f"prompt_{i+1:04d}_optimized.yaml")
                shutil.copy(latest, output_path)
                print(f"  Saved → {output_path}")
            else:
                print(f"  WARNING: No result file found")

            # Mark done in progress CSV
            save_progress(PROGRESS_CSV, i, prompt_text, "done", output_path)
            print(f"  Progress saved")

        except subprocess.CalledProcessError as e:
            print(f"  ERROR: {e}")
            save_progress(PROGRESS_CSV, i, prompt_text, "error")
            continue

        except Exception as e:
            print(f"  UNEXPECTED ERROR: {e}")
            save_progress(PROGRESS_CSV, i, prompt_text, "error")
            continue

    # Final summary
    progress = load_progress(PROGRESS_CSV)
    done = sum(1 for s in progress.values() if s == "done")
    errors = sum(1 for s in progress.values() if s == "error")
    print(f"\n{'='*60}")
    print(f"FINISHED — Done: {done} | Errors: {errors} | Total: {total}")
    print(f"Results  → {OUTPUT_DIR}/")
    print(f"Progress → {PROGRESS_CSV}")


if __name__ == "__main__":
    run_pipeline()