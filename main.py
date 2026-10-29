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
import pandas as pd

from research_steps.step01_data_curation import run_curation_pipeline
from research_steps.step02_physicochemical import run_physicochemical_step
from research_steps.step03_clustering import run_clustering_step
from research_steps.step04_chemical_space import execute_step04


def setup_environment():
    """Creates the necessary directory structure for the pipeline."""
    directories = [
        "data/raw",
        "data/processed",
        "data/processed/dunn_posthoc_matrices",
        "data/figures"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def print_pipeline_summary_table(pipeline_summary: list):
    """
    Prints a formatted, summary table linking research steps to their
    associated CSV files, metrics, and runtime.
    """
    print("\n" + "=" * 115)
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] VDR PIPELINE EXECUTION & FILE MAPPING SUMMARY")
    print("=" * 115)
    
    # Format table for terminal output
    header = f"{'Step':<30} | {'Output Files':<45} | {'Time (s)':<8} | {'Key Metric':<22}"
    print(header)
    print("-" * 115)

    for record in pipeline_summary:
        step_name = record["Step"]
        time_str = f"{record['Runtime_sec']:.2f}s"
        metric_str = record["Key_Metric"]
        
        # Format multiple files if present
        out_files = record["Output_Files"]
        if isinstance(out_files, list):
            first_file = out_files[0]
            print(f"{step_name:<30} | {first_file:<45} | {time_str:<8} | {metric_str:<22}")
            for extra_file in out_files[1:]:
                print(f"{'':<30} | {extra_file:<45} | {'':<8} | {'':<22}")
        else:
            print(f"{step_name:<30} | {out_files:<45} | {time_str:<8} | {metric_str:<22}")
        print("-" * 115)
        
    total_time = sum(record['Runtime_sec'] for record in pipeline_summary)
    print(f"TOTAL PIPELINE RUNTIME: {total_time:.2f} seconds")
    print("=" * 115)


def get_user_choice() -> int:
    """Prompts the user to select pipeline execution mode."""
    print("\nSelect an action to execute:")
    print("  [0] Run ENTIRE Pipeline (Steps 1 through 4)")
    print("  [1] Run Step 1: Data Curation & Standardisation")
    print("  [2] Run Step 2: Physicochemical Profiling & Filtering")
    print("  [3] Run Step 3: Fingerprints, Similarity & Clustering")
    print("  [4] Run Step 4: Chemical Space Visualisation & Correlation Analysis")
    
    while True:
        try:
            choice = input("\nEnter choice [0-4]: ").strip()
            if choice in ["0", "1", "2", "3", "4"]:
                return int(choice)
            print("[Warning] Invalid selection. Please enter a number between 0 and 4.")
        except KeyboardInterrupt:
            print("\n[INFO] Execution cancelled by user.")
            sys.exit(0)


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
            "Key_Metric": f"{len(curated_df)} Unique Compounds"
        }
    except Exception as e:
        print(f"\n[Error] Step 1 failed: {e}")
        sys.exit(1)


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
                "vdr_druglikeness_np.csv"
            ],
            "Runtime_sec": round(elapsed, 2),
            "Key_Metric": "125 Retained Features"
        }
    except Exception as e:
        print(f"\n[Error] Step 2 failed: {e}")
        sys.exit(1)


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
                "vdr_scaffold_summary.csv"
            ],
            "Runtime_sec": round(elapsed, 2),
            "Key_Metric": "12 FP Panel Evaluated"
        }
    except Exception as e:
        print(f"\n[Error] Step 3 failed: {e}")
        sys.exit(1)


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
                "data/figures/*.png"
            ],
            "Runtime_sec": round(elapsed, 2),
            "Key_Metric": "PCA/t-SNE/UMAP + Stats"
        }
    except Exception as e:
        print(f"\n[Error] Step 4 failed: {e}")
        sys.exit(1)


def main():
    print("=" * 70)
    print("   VDR CHEMICAL SPACE PROFILING: PIPELINE ORCHESTRATOR")
    print("=" * 70)

    setup_environment()

    raw_csv = "data/raw/VDR.csv"
    curated_csv = "data/processed/vdr_curated_library.csv"
    curated_sdf = "data/processed/vdr_curated_library.sdf"
    pipeline_summary_path = "data/processed/vdr_pipeline_execution_summary.csv"

    if not os.path.exists(raw_csv):
        print(f"\n[Error] Target dataset not found at '{raw_csv}'.")
        sys.exit(1)

    choice = get_user_choice()
    pipeline_summary = []

    # Execute selected logic
    if choice == 0:
        pipeline_summary.append(execute_step_1(raw_csv, curated_csv, curated_sdf))
        pipeline_summary.append(execute_step_2(curated_csv))
        pipeline_summary.append(execute_step_3(curated_csv))
        pipeline_summary.append(execute_step_4())
    elif choice == 1:
        pipeline_summary.append(execute_step_1(raw_csv, curated_csv, curated_sdf))
    elif choice == 2:
        pipeline_summary.append(execute_step_2(curated_csv))
    elif choice == 3:
        pipeline_summary.append(execute_step_3(curated_csv))
    elif choice == 4:
        pipeline_summary.append(execute_step_4())

    # Print and export summary table
    print_pipeline_summary_table(pipeline_summary)

    # Export execution log dataframe
    export_records = []
    for row in pipeline_summary:
        file_list = row["Output_Files"]
        file_str = "; ".join(file_list) if isinstance(file_list, list) else str(file_list)
        export_records.append({
            "Step": row["Step"],
            "Input_File": row["Input_File"],
            "Output_Files": file_str,
            "Runtime_sec": row["Runtime_sec"],
            "Key_Metric": row["Key_Metric"]
        })
    
    # Save log (appending if file exists, or creating fresh)
    pd.DataFrame(export_records).to_csv(pipeline_summary_path, index=False)
    print(f"\nSaved execution summary log: {pipeline_summary_path}\n")


if __name__ == "__main__":
    main()
