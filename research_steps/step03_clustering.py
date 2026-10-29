"""
research_steps/step03_clustering.py
Executes Step 3: Fingerprint Panel, Similarity, Evaluation & Butina Clustering.
Evaluates category recovery ARI/NMI using curated structural categories.
"""

import os
import pandas as pd
from algorithms.fingerprint_generator import FingerprintGenerator
from algorithms.fingerprint_evaluator import FingerprintEvaluator
from algorithms.butina_clustering import ButinaClusterer, ClusterStabilityAnalyzer
from algorithms.scaffold_analysis import ScaffoldAnalyzer


def run_clustering_step(
    input_csv_path: str = "data/processed/vdr_curated_library.csv",
    out_fp_eval_path: str = "data/processed/vdr_fingerprint_evaluation.csv",
    out_fp_redundancy_path: str = "data/processed/vdr_fingerprint_redundancy.csv",
    out_clusters_path: str = "data/processed/vdr_butina_clusters.csv",
    out_stability_path: str = "data/processed/vdr_cluster_stability.csv",
    out_scaffolds_path: str = "data/processed/vdr_scaffold_assignments.csv",
    out_scaffold_summary_path: str = "data/processed/vdr_scaffold_summary.csv"
):
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Curated library not found at '{input_csv_path}'. Please run Step 1 first.")

    print(f"Loading curated library from '{input_csv_path}'...")
    curated_df = pd.read_csv(input_csv_path)
    total_compounds = len(curated_df)

    # 1. Fingerprint Panel Generation (12 Representations)
    print("Generating 12-fingerprint multi-family panel for all compounds...")
    fp_gen = FingerprintGenerator(fp_size=2048)
    dataset_fps = fp_gen.profile_dataframe(curated_df, smiles_col="std_smiles", id_col="compound_id")

    # 2. Category Recovery Evaluation (ARI / NMI)
    print("Evaluating information content, category recovery (ARI/NMI), and inter-fingerprint redundancy...")
    category_labels = curated_df["category"].values if "category" in curated_df.columns else None

    fp_evaluator = FingerprintEvaluator()
    eval_df, redundancy_df, dist_matrices = fp_evaluator.evaluate_panel(dataset_fps, category_labels=category_labels)

    os.makedirs(os.path.dirname(out_fp_eval_path), exist_ok=True)
    eval_df.to_csv(out_fp_eval_path, index=False)
    redundancy_df.to_csv(out_fp_redundancy_path)
    print(f"Saved fingerprint evaluation: {out_fp_eval_path}")
    print(f"Saved fingerprint redundancy matrix: {out_fp_redundancy_path}")

    # Select retained non-redundant fingerprints (|r| <= 0.90)
    retained_fps = fp_evaluator.select_retained_fingerprints(eval_df, redundancy_df, max_correlation=0.90)
    print(f"Retained {len(retained_fps)} non-redundant fingerprints: {retained_fps}")

    # 3. Butina Clustering
    print("Executing Butina clustering across distance cutoffs [0.2, 0.3, 0.4, 0.5]...")
    clusterer = ButinaClusterer()
    merged_cluster_mapping = pd.DataFrame({"compound_id": list(dataset_fps.keys())})

    for fp_name in retained_fps:
        sum_df, map_df = clusterer.cluster_dataset(dataset_fps, fp_name=fp_name, cutoffs=[0.2, 0.3, 0.4, 0.5])
        merged_cluster_mapping = merged_cluster_mapping.merge(map_df, on="compound_id", how="left")

    merged_cluster_mapping.to_csv(out_clusters_path, index=False)
    print(f"Saved Butina cluster assignments: {out_clusters_path}")

    # 4. Cluster Stability & Bootstrap Resampling
    print("Evaluating cluster agreement and bootstrap stability (50 resamples)...")
    stability_analyzer = ClusterStabilityAnalyzer()

    agreement_df = stability_analyzer.evaluate_fingerprint_agreement(merged_cluster_mapping, retained_fps, cutoff=0.3)

    primary_fp = "ecfp4" if "ecfp4" in retained_fps else retained_fps[0]
    cmpd_ids = list(dataset_fps.keys())
    primary_fp_list = [dataset_fps[cid][primary_fp] for cid in cmpd_ids]
    mean_ari, std_ari = stability_analyzer.bootstrap_resample_stability(primary_fp_list, n_bootstraps=50, dist_cutoff=0.3)

    agreement_df.to_csv(out_stability_path, index=False)
    print(f"Saved cluster stability report: {out_stability_path}")

    # 5. Bemis-Murcko Scaffold Extraction
    print("Extracting Bemis-Murcko scaffolds and generic carbon frameworks...")
    scaffold_analyzer = ScaffoldAnalyzer()
    cmpd_scaffolds_df, bm_counts_df, gen_counts_df, scaffold_stats = scaffold_analyzer.analyze_dataframe(
        curated_df, smiles_col="std_smiles", id_col="compound_id"
    )

    cmpd_scaffolds_df.to_csv(out_scaffolds_path, index=False)

    summary_records = [{
        "total_compounds": scaffold_stats["total_compounds"],
        "unique_bm_scaffolds": scaffold_stats["unique_bm_scaffolds"],
        "unique_generic_frameworks": scaffold_stats["unique_generic_frameworks"],
        "bm_diversity_ratio": scaffold_stats["bm_scaffold_diversity_ratio"],
        "generic_diversity_ratio": scaffold_stats["generic_framework_diversity_ratio"],
        "bm_singletons": scaffold_stats["bm_singleton_scaffolds"],
        "bm_singleton_pct": scaffold_stats["bm_singleton_percentage"]
    }]
    pd.DataFrame(summary_records).to_csv(out_scaffold_summary_path, index=False)
    print(f"Saved scaffold assignments: {out_scaffolds_path}")
    print(f"Saved scaffold summary: {out_scaffold_summary_path}")

    # 6. Audit Trail Summary
    print("\n" + "=" * 55)
    print("       STEP 3 FINGERPRINTS & CLUSTERING AUDIT")
    print("=" * 55)
    print(f"Total Compounds Processed:       {total_compounds}")
    print(f"Fingerprint Panel Evaluated:     12 representations")
    print(f"Retained Fingerprints:           {', '.join(retained_fps)}")
    print(f"Primary FP ({primary_fp}) Clusters (Cutoff 0.3): {merged_cluster_mapping[f'cluster_{primary_fp}_cutoff_0.3'].nunique()}")
    print(f"Primary FP Bootstrap Stability:  ARI = {mean_ari:.3f} ± {std_ari:.3f}")
    print(f"Unique Bemis-Murcko Scaffolds:   {scaffold_stats['unique_bm_scaffolds']}")
    print(f"Unique Generic Frameworks:       {scaffold_stats['unique_generic_frameworks']}")
    print(f"Scaffold Diversity Ratio:        {scaffold_stats['bm_scaffold_diversity_ratio']:.3f}")
    print("=" * 55)


if __name__ == "__main__":
    run_clustering_step()
