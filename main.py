"""
main.py
Master orchestrator for the VDR Chemical Space Profiling project.
Executes research pipeline steps sequentially or individually based on user selection,
outputs summary reports, and prints a step-by-step pipeline summary table.
"""

import os
import sys
import time
import datetime
import subprocess
import pandas as pd

from research_steps.step01_data_curation import run_curation_pipeline
from research_steps.step02_physicochemical import run_physicochemical_step
from research_steps.step03_clustering import run_clustering_step
from research_steps.step04_chemical_space import execute_step04
from ML_generation.validate_generation import validate_model
from ML_generation.prepare_splits import create_scaffold_split


def setup_environment():
    """Creates the necessary directory structure for the pipeline."""
    directories = [
        "data/raw",
        "data/processed",
        "data/processed/dunn_posthoc_matrices",
        "data/figures",
        "data/generated_csv",
        "data/reports",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def print_pipeline_summary_table(pipeline_summary: list):
    """
    Prints a formatted summary table linking research steps to their
    associated CSV files, metrics, and runtime.
    """
    print("\n" + "=" * 115)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] VDR PIPELINE EXECUTION & FILE MAPPING SUMMARY")
    print("=" * 115)

    header = f"{'Step':<30} | {'Output Files':<45} | {'Time (s)':<8} | {'Key Metric':<22}"
    print(header)
    print("-" * 115)

    for record in pipeline_summary:
        step_name = record["Step"]
        time_str = f"{record['Runtime_sec']:.2f}s"
        metric_str = record["Key_Metric"]

        out_files = record["Output_Files"]
        if isinstance(out_files, list):
            first_file = out_files[0]
            print(f"{step_name:<30} | {first_file:<45} | {time_str:<8} | {metric_str:<22}")
            for extra_file in out_files[1:]:
                print(f"{'':<30} | {extra_file:<45} | {'':<8} | {'':<22}")
        else:
            print(f"{step_name:<30} | {out_files:<45} | {time_str:<8} | {metric_str:<22}")
        print("-" * 115)

    total_time = sum(record["Runtime_sec"] for record in pipeline_summary)
    print(f"TOTAL PIPELINE RUNTIME: {total_time:.2f} seconds")
    print("=" * 115)


def export_execution_log(pipeline_summary: list, pipeline_summary_path: str):
    """Appends execution records to the summary CSV."""
    export_records = []
    for row in pipeline_summary:
        file_list = row["Output_Files"]
        file_str = "; ".join(file_list) if isinstance(file_list, list) else str(file_list)
        export_records.append({
            "Step": row["Step"],
            "Input_File": row["Input_File"],
            "Output_Files": file_str,
            "Runtime_sec": row["Runtime_sec"],
            "Key_Metric": row["Key_Metric"],
        })

    df = pd.DataFrame(export_records)
    if os.path.exists(pipeline_summary_path):
        df.to_csv(pipeline_summary_path, mode="a", header=False, index=False)
    else:
        df.to_csv(pipeline_summary_path, index=False)
    print(f"\nSaved execution summary log: {pipeline_summary_path}\n")


# =============================================================================
# EXECUTOR FUNCTIONS (ORIGINAL DATASET)
# =============================================================================

def execute_step_1(raw_csv, curated_csv, curated_sdf) -> dict:
    print("\n>>> Executing Step 1: Data Curation & Standardisation...")
    start_time = time.time()
    try:
        curated_df = run_curation_pipeline(raw_csv, curated_csv, curated_sdf)
        elapsed = time.time() - start_time
        print("[Success] Step 1 completed successfully.")
        return {
            "Step": "Step 1: Curation",
            "Input_File": raw_csv,
            "Output_Files": ["vdr_curated_library.csv", "vdr_curated_library.sdf"],
            "Runtime_sec": round(elapsed, 2),
            "Key_Metric": f"{len(curated_df)} Unique Compounds",
        }
    except Exception as e:
        print(f"\n[Error] Step 1 failed: {e}")
        return None


def execute_step_2(curated_csv) -> dict:
    print("\n>>> Executing Step 2: Physicochemical Profiling & Filtering...")
    start_time = time.time()
    try:
        run_physicochemical_step()
        elapsed = time.time() - start_time
        print("[Success] Step 2 completed successfully.")
        return {
            "Step": "Step 2: Physicochemical",
            "Input_File": curated_csv,
            "Output_Files": [
                "vdr_descriptors_full.csv",
                "vdr_descriptors_filtered.csv",
                "vdr_druglikeness_np.csv",
            ],
            "Runtime_sec": round(elapsed, 2),
            "Key_Metric": "149 Retained Features",
        }
    except Exception as e:
        print(f"\n[Error] Step 2 failed: {e}")
        return None


def execute_step_3(curated_csv) -> dict:
    print("\n>>> Executing Step 3: Fingerprints, Similarity & Clustering...")
    start_time = time.time()
    try:
        run_clustering_step()
        elapsed = time.time() - start_time
        print("[Success] Step 3 completed successfully.")
        return {
            "Step": "Step 3: Fingerprints & Clusters",
            "Input_File": curated_csv,
            "Output_Files": [
                "vdr_fingerprint_evaluation.csv",
                "vdr_fingerprint_redundancy.csv",
                "vdr_butina_clusters.csv",
                "vdr_cluster_stability.csv",
                "vdr_scaffold_assignments.csv",
                "vdr_scaffold_summary.csv",
            ],
            "Runtime_sec": round(elapsed, 2),
            "Key_Metric": "12 FP Panel Evaluated",
        }
    except Exception as e:
        print(f"\n[Error] Step 3 failed: {e}")
        return None


def execute_step_4() -> dict:
    print("\n>>> Executing Step 4: Chemical Space Visualisation & Correlation Analysis...")
    start_time = time.time()
    try:
        execute_step04()
        elapsed = time.time() - start_time
        print("[Success] Step 4 completed successfully.")
        return {
            "Step": "Step 4: Chemical Space",
            "Input_File": "Processed Data (Steps 1-3)",
            "Output_Files": [
                "vdr_chemical_space_coordinates.csv",
                "vdr_pca_variance.csv",
                "vdr_descriptor_spearman_matrix.csv",
                "vdr_kruskal_wallis_results.csv",
                "dunn_posthoc_matrices/*.csv",
                "data/figures/*.png",
            ],
            "Runtime_sec": round(elapsed, 2),
            "Key_Metric": "PCA/t-SNE/UMAP + Stats",
        }
    except Exception as e:
        print(f"\n[Error] Step 4 failed: {e}")
        return None


# =============================================================================
# EXECUTOR FUNCTIONS (ML GENERATION)
# =============================================================================

def execute_ml_generation(model_name: str, script_path: str, output_csv: str) -> dict:
    """Executes a Bash pipeline script for ML molecule generation."""
    print(f"\n>>> Executing Generation for {model_name}...")
    start_time = time.time()

    if not os.path.exists(script_path):
        print(f"\n[Error] Bash script not found: {script_path}")
        return None

    try:
        subprocess.run(["bash", script_path], check=True)
        elapsed = time.time() - start_time
        print(f"[Success] {model_name} generation completed.")
        return {
            "Step": f"ML: {model_name} Generation",
            "Input_File": "Prior Model + Configs",
            "Output_Files": [output_csv],
            "Runtime_sec": round(elapsed, 2),
            "Key_Metric": "Sampling Executed",
        }
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] ML Generation failed with error code: {e.returncode}")
        return None


# =============================================================================
# MENUS
# =============================================================================

def menu_original_dataset(raw_csv, curated_csv, curated_sdf, pipeline_summary_path):
    while True:
        print("\n" + "-" * 40)
        print("  1) ORIGINAL DATASET PIPELINE")
        print("-" * 40)
        print("  [1] Run Step 1: Data Curation & Standardisation")
        print("  [2] Run Step 2: Physicochemical Profiling & Filtering")
        print("  [3] Run Step 3: Fingerprints, Similarity & Clustering")
        print("  [4] Run Step 4: Chemical Space Visualisation & Correlation Analysis")
        print("  [5] Run ENTIRE Pipeline (Steps 1 through 4)")
        print("  [0] Go back")

        choice = input("\nEnter choice [0-5]: ").strip()
        pipeline_summary = []

        if choice == "1":
            res = execute_step_1(raw_csv, curated_csv, curated_sdf)
            if res:
                pipeline_summary.append(res)
        elif choice == "2":
            res = execute_step_2(curated_csv)
            if res:
                pipeline_summary.append(res)
        elif choice == "3":
            res = execute_step_3(curated_csv)
            if res:
                pipeline_summary.append(res)
        elif choice == "4":
            res = execute_step_4()
            if res:
                pipeline_summary.append(res)
        elif choice == "5":
            r1 = execute_step_1(raw_csv, curated_csv, curated_sdf)
            if r1:
                pipeline_summary.append(r1)
            r2 = execute_step_2(curated_csv)
            if r2:
                pipeline_summary.append(r2)
            r3 = execute_step_3(curated_csv)
            if r3:
                pipeline_summary.append(r3)
            r4 = execute_step_4()
            if r4:
                pipeline_summary.append(r4)
        elif choice == "0":
            break
        else:
            print("[Warning] Invalid selection.")
            continue

        if pipeline_summary:
            print_pipeline_summary_table(pipeline_summary)
            export_execution_log(pipeline_summary, pipeline_summary_path)


def menu_reinvent(pipeline_summary_path):
    while True:
        print("\n" + "-" * 40)
        print("  2.1) MODEL: REINVENT 4")
        print("-" * 40)
        print("  [1] Generate (Run TL & Sampling Pipeline)")
        print("  [2] Validate (Benchmark Metrics)")
        print("  [0] Go back")

        choice = input("\nEnter choice [0-2]: ").strip()
        pipeline_summary = []

        if choice == "1":
            res = execute_ml_generation(
                model_name="REINVENT4",
                script_path="ML_generation/reinvent4/run_pipeline.sh",
                output_csv="data/generated_csv/reinvent4_tl_v1.csv",
            )
            if res:
                pipeline_summary.append(res)
        elif choice == "2":
            print("\n>>> Executing Validation for REINVENT 4...")
            start_time = time.time()
            try:
                train_file = (
                    "data/processed/vdr_train.smi"
                    if os.path.exists("data/processed/vdr_train.smi")
                    else "data/processed/vdr_binders.smi"
                )
                metrics = validate_model(
                    model_name="REINVENT4_TL_v1",
                    generated_csv="data/generated_csv/reinvent4_tl_v1.csv",
                    train_smi=train_file,
                    test_smi="data/processed/vdr_test_holdout.smi",
                )
                elapsed = time.time() - start_time
                pipeline_summary.append({
                    "Step": "ML: REINVENT4 Validation",
                    "Input_File": "data/generated_csv/reinvent4_tl_v1.csv",
                    "Output_Files": [
                        "data/processed/ml_generation_benchmarks.csv",
                        "data/reports/reinvent4_tl_v1_validation.txt",
                        "data/figures/reinvent4_tl_v1_validation.png",
                    ],
                    "Runtime_sec": round(elapsed, 2),
                    "Key_Metric": f"Regen: {metrics['Exact_Regen_2D_%']}% | GuacaMol: {metrics['GuacaMol_Rediscovery_Score']} | AlertFree: {metrics['Alert_Free_%']}%",
                })
            except Exception as e:
                print(f"\n[Error] Validation failed: {e}")
        elif choice == "0":
            break
        else:
            print("[Warning] Invalid selection.")

        if pipeline_summary:
            print_pipeline_summary_table(pipeline_summary)
            export_execution_log(pipeline_summary, pipeline_summary_path)


def menu_mol2mol(pipeline_summary_path):
    while True:
        print("\n" + "-" * 40)
        print("  2.3) MODEL: MOL2MOL (REINVENT 4)")
        print("-" * 40)
        print("  [1] Generate (Run Pair Prep, TL & Sampling)")
        print("  [2] Validate (Benchmark Metrics)")
        print("  [0] Go back")

        choice = input("\nEnter choice [0-2]: ").strip()
        pipeline_summary = []

        if choice == "1":
            res = execute_ml_generation(
                model_name="Mol2Mol",
                script_path="ML_generation/Mol2Mol/run_pipeline.sh",
                output_csv="data/generated_csv/mol2mol_tl_v1.csv",
            )
            if res:
                pipeline_summary.append(res)
        elif choice == "2":
            print("\n>>> Executing Validation for Mol2Mol...")
            start_time = time.time()
            try:
                train_file = (
                    "data/processed/vdr_train.smi"
                    if os.path.exists("data/processed/vdr_train.smi")
                    else "data/processed/vdr_binders.smi"
                )
                metrics = validate_model(
                    model_name="Mol2Mol_TL_v1",
                    generated_csv="data/generated_csv/mol2mol_tl_v1.csv",
                    train_smi=train_file,
                    test_smi="data/processed/vdr_test_holdout.smi",
                )
                elapsed = time.time() - start_time
                pipeline_summary.append({
                    "Step": "ML: Mol2Mol Validation",
                    "Input_File": "data/generated_csv/mol2mol_tl_v1.csv",
                    "Output_Files": [
                        "data/processed/ml_generation_benchmarks.csv",
                        "data/reports/mol2mol_tl_v1_validation.txt",
                        "data/figures/mol2mol_tl_v1_validation.png",
                    ],
                    "Runtime_sec": round(elapsed, 2),
                    "Key_Metric": f"Regen: {metrics['Exact_Regen_2D_%']}% | GuacaMol: {metrics['GuacaMol_Rediscovery_Score']} | AlertFree: {metrics['Alert_Free_%']}%",
                })
            except Exception as e:
                print(f"\n[Error] Validation failed: {e}")
        elif choice == "0":
            break
        else:
            print("[Warning] Invalid selection.")

        if pipeline_summary:
            print_pipeline_summary_table(pipeline_summary)
            export_execution_log(pipeline_summary, pipeline_summary_path)


def menu_ml_generation(pipeline_summary_path: str):
    """Tier 2 Menu: De Novo Molecule Generation & Benchmarking."""
    if not os.path.exists("data/processed/vdr_train.smi") or not os.path.exists("data/processed/vdr_test_holdout.smi"):
        print("\n[Info] Generating reproducible 80/20 scaffold-aware train/test splits...")
        create_scaffold_split()

    while True:
        print("\n" + "=" * 60)
        print("            ML GENERATION & MODEL BENCHMARKING")
        print("=" * 60)
        print("  [0] Prepare / Refresh 80/20 Scaffold Train-Test Split")
        print("  [1] REINVENT 4 (Transfer Learning & Sampling)")
        print("  [2] Transmol (Queued for Integration)")
        print("  [3] Mol2Mol (Transformer Analogue Generation)")
        print("  [b] Back to Main Menu")
        print("=" * 60)

        choice = input("Select a generative model [0-3, b]: ").strip().lower()
        pipeline_summary = []

        if choice == "0":
            print("\n>>> Generating 80/20 Scaffold-Aware Train/Test Splits...")
            start_time = time.time()
            try:
                create_scaffold_split()
                elapsed = time.time() - start_time
                pipeline_summary.append({
                    "Step": "ML: Scaffold Train/Test Split",
                    "Input_File": "data/processed/vdr_curated_library.csv",
                    "Output_Files": [
                        "data/processed/vdr_train.smi",
                        "data/processed/vdr_test_holdout.smi",
                    ],
                    "Runtime_sec": round(elapsed, 2),
                    "Key_Metric": "80/20 Murcko Scaffold Split (Seed=42)",
                })
            except Exception as e:
                print(f"\n[Error] Split generation failed: {e}")
        elif choice == "1":
            menu_reinvent(pipeline_summary_path)
        elif choice == "2":
            print("\n[Info] Transmol module is queued for integration.")
        elif choice == "3":
            menu_mol2mol(pipeline_summary_path)
        elif choice == "b":
            break
        else:
            print("Invalid selection. Please enter 0-3 or b.")

        if pipeline_summary:
            print_pipeline_summary_table(pipeline_summary)
            export_execution_log(pipeline_summary, pipeline_summary_path)


def main():
    setup_environment()

    raw_csv = "data/raw/VDR.csv"
    curated_csv = "data/processed/vdr_curated_library.csv"
    curated_sdf = "data/processed/vdr_curated_library.sdf"
    pipeline_summary_path = "data/processed/vdr_pipeline_execution_summary.csv"

    if not os.path.exists(raw_csv):
        print(f"\n[Error] Target dataset not found at '{raw_csv}'.")
        sys.exit(1)

    while True:
        print("\n" + "=" * 70)
        print("   VDR CHEMICAL SPACE PROFILING: MASTER ORCHESTRATOR")
        print("=" * 70)
        print("  [1] Original dataset")
        print("  [2] ML generation")
        print("  [0] Exit")

        choice = input("\nEnter choice [0-2]: ").strip()

        if choice == "1":
            menu_original_dataset(raw_csv, curated_csv, curated_sdf, pipeline_summary_path)
        elif choice == "2":
            menu_ml_generation(pipeline_summary_path)
        elif choice == "0":
            print("\nExiting orchestrator...")
            sys.exit(0)
        else:
            print("[Warning] Invalid selection.")


if __name__ == "__main__":
    main()
