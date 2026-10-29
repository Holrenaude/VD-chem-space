import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as smm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
from rdkit import Chem
from rdkit.Chem import AllChem

def generate_morgan_fingerprints(smiles_series: pd.Series, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Generates a binary Morgan fingerprint matrix from SMILES."""
    fps = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
        else:
            fps.append(np.zeros((n_bits,), dtype=int))
    return np.array(fps)

def run_pca_analysis(descriptors_df: pd.DataFrame, n_components: int = 10) -> tuple[pd.DataFrame, PCA]:
    """Applies StandardScaler and fits PCA on continuous filtered descriptors."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(descriptors_df)
    
    pca = PCA(n_components=n_components, random_state=42)
    pca_coords = pca.fit_transform(scaled_data)
    
    col_names = [f"PCA_PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_coords, index=descriptors_df.index, columns=col_names)
    
    return pca_df, pca

def run_tsne_analysis(fps_array: np.ndarray, index: pd.Index, perplexity: float = 30.0) -> pd.DataFrame:
    """Runs t-SNE on the binary/count fingerprint matrix using Jaccard distance."""
    tsne = TSNE(
        n_components=2, metric='jaccard', perplexity=perplexity,
        random_state=42, init='pca', learning_rate='auto'
    )
    coords = tsne.fit_transform(fps_array)
    return pd.DataFrame(coords, index=index, columns=['tSNE_1', 'tSNE_2'])

def run_umap_analysis(fps_array: np.ndarray, index: pd.Index, n_neighbors: int = 15, min_dist: float = 0.1) -> pd.DataFrame:
    """Runs UMAP on the fingerprint space using Jaccard metric."""
    reducer = umap.UMAP(
        n_components=2, metric='jaccard', n_neighbors=n_neighbors,
        min_dist=min_dist, random_state=42
    )
    coords = reducer.fit_transform(fps_array)
    return pd.DataFrame(coords, index=index, columns=['UMAP_1', 'UMAP_2'])

def compute_descriptor_spearman(descriptors_df: pd.DataFrame) -> pd.DataFrame:
    """Computes pairwise Spearman rank correlation matrix across orthogonal descriptors."""
    corr, _ = stats.spearmanr(descriptors_df, axis=0)
    return pd.DataFrame(corr, index=descriptors_df.columns, columns=descriptors_df.columns)

def run_kruskal_dunn_tests(data_df: pd.DataFrame, feature_cols: list[str], group_col: str) -> pd.DataFrame:
    """Performs Kruskal-Wallis tests per feature with Benjamini-Hochberg (FDR) correction."""
    results = []
    groups = [group for _, group in data_df.groupby(group_col)]
    
    for feature in feature_cols:
        group_series = [g[feature].dropna() for g in groups]
        if len(group_series) > 1:
            kw_stat, kw_p = stats.kruskal(*group_series)
            results.append({'Feature': feature, 'Kruskal_Stat': kw_stat, 'Kruskal_pvalue': kw_p})
        
    res_df = pd.DataFrame(results)
    
    # Benjamini-Hochberg FDR correction
    reject, pvals_corrected, _, _ = smm.multipletests(res_df['Kruskal_pvalue'], alpha=0.05, method='fdr_bh')
    res_df['Kruskal_pvalue_FDR'] = pvals_corrected
    res_df['Significant_Difference'] = reject
    
    return res_df.sort_values(by='Kruskal_pvalue_FDR')
