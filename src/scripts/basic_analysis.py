"""

Runs a generalized analysis on a dataset from an excel file
oputputs to a pdf file specified by user

---------- Preprocessing ----------

1) remove nan features
    features where nan_count is > set threshold
2) impute remaining nan
    imputation methods:
            hm   =   Half-Minimum
            knn  =   K-nearest neighbor
            mice =   Multiple Imputation by Chained Equations
            rf   =   Random Forest
3) log2 Transform
    centers distribution (metabolomics is typically right-shifted)
4) autoscale
    ensures each feature has mean = 0 stdev = 1 so high features do not dominate

---------- Analysis ----------

1) PCA, Skree plot, top n most varaint columns table
2) by-group Bar Plot of all columns (mean + stdev)
3) kruskall-wallis w/ dunn posthoc ()

---------- Excel Structure ----------

Requires the excel file to have this format and already normalized:

column 1, row >= 2: genotype/group of each sample
column 2, row >= 2: sex (optional)
column 3, row >= 2: sample Name

row 1, column >= 4: feature name

column >= 4, row >= 3: data matrix

"""

# region Imports

import logging, sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer   # needed to enable IterativeImputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.ensemble import RandomForestRegressor
from scikit_posthocs import posthoc_dunn

from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal, spearmanr, pearsonr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

from PyQt5.QtWidgets import QFileDialog, QApplication

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# endregion

logger = logging.getLogger(__name__)

# region Preprocessing

def find_sparse_features(matrix: np.ndarray, group_indices: dict, threshold: int = 0.5):
    """
    Drops features if ALL groups have >threshold missingness 

    Params
    ------
        matrix                      data matrix to handle
        group_indices               dict of grp_name: list of indices
        threshold                   missingness threshold to drop feature
    
    Returns
    -------
        keep                        list of feature indices to keep
    """
    missing = np.isnan(matrix)

    keep = []
    for i in range(missing.shape[1]):
        for _, indices in group_indices.items():
            grp_missing = missing[indices, i]
            if np.mean(grp_missing) <= threshold:
                keep.append(i)
                break
    
    return keep

def impute_nans(matrix: np.ndarray, method: str = "hm", neighbors: int = 5):
    """
    imputes nan values of non-sparse columns using method sepcified, half-minimum is sufficient for
    most metaboloimcs applications
    DOES NOT copy matrix, edits in place - do that yourself befor using

    Params
    ------
        matrix                      matrix we are imputing
        method                      "hm" or "knn" based on what imputation method to use
        neighbors                   number of neighbors for knn imputation
    """

    if method not in ["hm", "knn","mice","rf"]:
        logger.warning(
            "Imputation method must be" \
            "\nhm   =   Half-Minimum" \
            "\nknn  =   K-nearest neighbor" \
            "\nmice =   Multiple Imputation by Chained Equations" \
            "\nrf   =   Random Forest" \
            "\nDefaulting to half-minimum")
        method = "hm"

    # half minimum
    if method == "hm":
        for j in range(matrix.shape[1]):
            col = matrix[:,j]
            min_val = np.nanmin(col)
            col[np.isnan(col)] = min_val / 2
            matrix[:,j] = col

    # k-nearest neightbor
    elif method == "knn":
        imputer = KNNImputer(n_neighbors = neighbors)
        matrix = imputer.fit_transform(matrix)

    # multiple imputation by chained equations
    elif method == "mice":
        imputer = IterativeImputer(max_iter=10, random_state=0)
        matrix = imputer.fit_transform(matrix)

    # random forest
    elif method == "rf":
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100), max_iter=10, random_state=0)
        matrix = imputer.fit_transform(matrix)

    return matrix

def log2_transform(matrix: np.ndarray):
    """
    log2 transforms a given matrix
    DOES NOT copy matrix, edits in place

    Params
    ------
        matrix                      data matrix with sparse features dropped and nan imputed

    Returns
    -------
        log2 transformed matrix
    """
    return np.log2(matrix + 1e-10) # 1e-10 to prevent log(0)

def autoscale(matrix: np.ndarray):
    """
    autoscales matrix, making each feature have mean 0 and std 1 to prevent high abundance metaoblits
    from dominating pca
    DOES NOT copy matrix, edits in place

    Params
    ------
        matrix                      data matrix withs parse features dropped, nan imputed, and log2 transformed

    Returns
    -------
        autoscaled matrix
    """
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0)
    stds[stds == 0] = 1                     # prevent divide by 0 for constant columns
    return (matrix - means) / stds

def full_preprocess(matrix: np.ndarray, group_indices: dict,
                    sparseness_threshold=None,
                    imputation_method=None,
                    impute_knns=None
                    ):
    """
    fully preproccesses a matrix, finds and drops sparse features, imputes nan values, log2 transforms data, then
    autoscales distributions

    Params
    ------
    matrix                      matrix to process
    group_indices               dict of group_name: list of index values in matrix for samples in that group (rows)
    Returns
    -------
    mat                         preprocessed matrix
    keep                        array of indices to keep
    """

    # get requried data
    if not sparseness_threshold:
        logger.warning("No sparseness threshold detected")
        sparseness_threshold = 0.5

    if not imputation_method:
        logger.warning("No imputation method detected")
        imputation_method = 'hm'
        
    if not impute_knns:
        logger.warning("No nearest neighbors setting detected")
        impute_knns = 5
    
    # copy input matrix
    mat = matrix.copy()

    # figure out which features to drop and drop
    keep = find_sparse_features(matrix, group_indices, sparseness_threshold)
    mat = mat[:,keep]

    # impute remaining nan values
    mat = impute_nans(mat, imputation_method, impute_knns)

    # log2 transform to correct typical right-skewing of metabolomic distributions
    mat = log2_transform(mat)

    # autoscale to remove dominance from high abundance features
    mat = autoscale(mat)

    return mat, keep

# endregion

# region Analysis

def pca(matrix: np.ndarray, exp_var: float = 0.8, n_components=None):
    """
    Performs PCA, unsuperviesd dimensionality reduction to inspect overall data structure

    Params
    ------
        matrix                      matrix of values to test
        exp_var                     amount of variance you wnat to explain with PCs (just gives idx of pc where this value is surpassed)
        n_components                number of components to calculate, None means do max
        
    Returns
    -------
        {
        scores: (n_samples,n_components) - position of samples in reduced space for plotting
        loadings: (n_components,n_features) - loading arrays showing which features drive seperation
        explained_variance: (n_components) - fraciton of total varaince each PC explains
        }
    """
    if exp_var > 1 or exp_var < 0:
        logger.warning("Explained Varaince must be in range [0,1]")
        exp_var = 0.80

    model = PCA(n_components=n_components)
    scores = model.fit_transform(matrix)

    cumulative = np.cumsum(model.explained_variance_ratio_)
    n_for_exp_var = int(np.searchsorted(cumulative, exp_var)) + 1

    return {
        'scores': scores,
        'loadings': model.components_,
        'explained_variance': model.explained_variance_ratio_,
        'cumulative_variance': cumulative,
        'n_for_exp_var': n_for_exp_var
    }

def kruskal_wallis(matrix: np.ndarray, group_indices: dict, posthoc: bool = True):
    """
    Performs Kruskal-Wallis test, non-parametric extension of mann-whitney to 3+ groups with same
    relationship to one-way anova as mann-whitney has to t-test - still need post-hoc tests
    DOES NOT assume groups are normally distributed

    Params
    ------
        matrix                      matrix of values to test (samples x features)
        group_indices               dict of group: list of indices for each group (dm.group_indices)
        posthoc                     wether or not to run posthoc analysis

    Returns
    -------
        result{
            pvals: array of p values from one way anova
            posthoc_pvals: 3D array (n_features, n_groups, n_groups) for all group comparisons from dunn posthoc
            group_names: list of group_names for matching to n_groups indices in posthoc_pvals
        }
    """
    p_values = np.zeros(matrix.shape[1])
    for j in range(matrix.shape[1]):
        groups = [matrix[idxs,j] for idxs in group_indices.values()]
        try:
            _,p = kruskal(*groups)
        except:
            p = np.nan
        p_values[j] = p

    result = {'p_values': p_values}

    if posthoc:
        posthoc_pvals, group_names = _dunn_posthoc(matrix, group_indices)
        result['posthoc_pvals'] = posthoc_pvals
        result['group_names'] = group_names

    return result

def _dunn_posthoc(matrix: np.ndarray, group_indices: dict):
    """
    Runs pairwise Dunn test, used IF kruskall wallis returns signficiant resutls to determine
    which groups are contributing to this significance

    Params
    ------
        matrix                      matrix to analyze
        group_indices               dict of group indices lists

    Returns
    -------
        group_names                 list of group names, index matched to pvals 3D array
        pvals                       3D array (n_features, n_groups, n_groups), pvals[j] gives full p-value matrix
                                    (n_groups, n_groups) for feature j, pvals[:,i,k] gives you the pvalue for 
                                    group i vs group k across all features at once, pvals[j,i,k] gives you a single
                                    p-value for one feature j between groups i and k
    """
    n_groups = len(group_indices)
    n_features = matrix.shape[1]
    pvals = np.full((n_features,n_groups,n_groups), np.nan)
    group_names = list(group_indices.keys())

    for j in range(n_features):
        data = [matrix[indices,j] for indices in group_indices.values()]
        result = posthoc_dunn(data, p_adjust='fdr_bh')
        pvals[j] = result.values

    return pvals, group_names

def fdr_correction(p_values: np.ndarray, method: str = 'fdr_bh'):
    """
    Performs fdr correction on a set of p values to reduce fasle positives.  Default 'bh' is Benhamini-Hotchburg
    which is good for discovery metabolomics where false positives are preferable to missing real hits, could
    also do Bonferroni ('bonferroni') for a more strict correction

    Params
    ------
        p_values                    array of p_values for correction
        method                      how to perform correction, 'bh' for discovery 'bonferronni' for more strict
    
    Returns
    -------
        corrected                   FDR corrected array
    """
    _, corrected, _, _ = multipletests(p_values, method=method)
    return corrected

# endregion

# region Plotting

def plot_pca(ax: Axes, pca_data: dict, group_indices: list, title: str = '', pc_x: int = 1, pc_y: int = 2):
    """
    Scatter of sample scores, one point per sample colored by group label.
    Works for both PCA (pass explained_variance) and PLS-DA (pass None).

    Params
    ------
    scores              (n_samples, n_components) from pca()['scores'] or pls_da()['scores']
    group_indices       dict of group_name: list of idxs
    explained_variance  (n_components,) from pca()['explained_variance'], or None for PLS-DA
    """
    
    # setup group coloring
    group_names = list(group_indices.keys())    
    if len(group_names) > 7:
        cmap = plt.colormaps['tab20b'].resampled(len(group_names))
    else:
        cmap = plt.colormaps['Dark2'].resampled(max(len(group_names), 2))
    colors = [cmap(i) for i in range(len(group_names))]

    # retreive explained variance and/or label x and y axes
    explained_variance = pca_data['explained_variance']
    if explained_variance is not None:
        xlabel = f"PC{pc_x} ({explained_variance[pc_x-1] * 100:.1f}%)"
        ylabel = f"PC{pc_y} ({explained_variance[pc_y-1] * 100:.1f}%)"
    else:
        xlabel = f"PC{pc_x}"
        ylabel = f"PC{pc_y}"
    
    # plot points
    scores = pca_data['scores']
    for i, (name,idxs) in enumerate(group_indices.items()):
        color = colors[i]
        ax.scatter(scores[idxs, pc_x-1], scores[idxs, pc_y-1], label=name, color=color, s=55, alpha=0.85)
    
    # generate Title if needed
    if not title:
        title = f"PC{pc_x} vs PC{pc_y} Plot"

    # cfg graph
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.5)

def plot_scree_bar(ax: Axes, values, title: str = "Explained Varaince", cutoff=None):
    """
    Plots vertically aligned bar plot for scree plot visualization

    Params
    ------
    ax                      axis to plot on
    values                  values to plot
    title                   title
    """

    labels = np.arange(1,len(values)+1)

    ax.barh(labels,values)
    if cutoff is not None:
        ax.axvline(cutoff, color='red', linestyle='--', linewidth=1)

    ax.set_yticks(labels)
    ax.set_yticklabels([f"PC{i}" for i in labels])
    ax.set_title(title)

def plot_pdf_table(rows, row_labels=None, col_labels=None, title='', rows_per_page=25):

    tables = [] 
          
    # column widths
    col_max_chars = [len(col_labels[j]) for j in range(len(col_labels))]
    for row in rows:
        for j,cell in enumerate(row):
            col_max_chars[j] = max(col_max_chars[j], len(str(cell)))
    total = sum(col_max_chars)
    col_widths = [c/total for c in col_max_chars]

    for chunk_start in range(0,len(rows), rows_per_page):

        chunk_end = chunk_start + rows_per_page

        chunk = rows[chunk_start:chunk_end]
        chunk_labels = row_labels[chunk_start:chunk_end] if row_labels else None

        fig,ax = plt.subplots(figsize=(11,8.5))
        ax.axis('off')

        table = ax.table(
            cellText=chunk,
            rowLabels=chunk_labels,
            colLabels=col_labels,
            loc='center',
            cellLoc='left'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        
        # set widths
        for (row,col), cell in table.get_celld().items():
            if col >= 0:
                cell.set_width(col_widths[col])

        # find header height
        max_len = max(len(c) for c in col_labels)
        header_height = min(0.05 + max_len *0.006, 0.3)

        # rotate headers
        for j in range(len(col_labels)):
            cell = table[0,j]
            cell.get_text().set_rotation(60)
            cell.get_text().set_ha('left')
            cell.set_height(header_height)

        ax.set_title(title,pad=20)
        tables.append(fig)

    return tables

def plot_bar(ax: Axes, data, labels, ylabel='', title='', cutoff=None, sorted_desc=True):
    """
    Plots a barplot with error bars, optionally include cutoff horizontal lines at specified values
    and allows sorting of bars in descending order of magnitude

    Params
    ------
    ax                          ax to plot on
    data                        list of arrays to plot avg of
    labels                      index matched labels for values
    ylabel/title                annotation
    cutoff                      values to draw horizontal lines on (optional)
    sorted_desc                 T/F wether or not to sort values in desc order on plot
    """
    values = []
    errs = []
    for arr in data:
        values.append(np.nanmean(arr))
        errs.append(np.nanstd(arr))
    lower_err = [0] * len(values)

    if sorted_desc:
        paris = sorted(zip(values, errs, labels), key=lambda x: x[0] if not np.isnan(x[0]) else -np.inf, reverse=True)
        values,errs,labels = zip(*paris)

    label_map = {label:i for i,label in enumerate(labels)}


    ax.bar(range(len(values)), values, yerr=[lower_err,errs], capsize=4)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)

    if cutoff is not None:
        for h in cutoff:
            ax.axhline(h, color='red', linestyle='--', label=f'{h:3f}')

    ax.set_ylabel(ylabel)
    ax.set_yscale('log')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')

    ax.set_title(title)
    ax.figure.tight_layout()

    return label_map

def add_significance(ax: Axes, x1, x2, y, p):
    if p < 0.001:   symbol = '***'
    elif p < 0.01:  symbol = '**'
    elif p < 0.05:  symbol = '*'
    else:           return
    ax.plot([x1, x2], [y, y], color='black', linewidth=1)
    ax.text((x1+x2)/2, y, symbol, ha='center', va='bottom', fontsize=10)

def format_val(val):
    if val == 0:
        return "0"
    if abs(val) > 1000000 or abs(val) < 0.001:
        return f"{val:.4e}"
    return f"{val:.4f}"

# endregion

# region Data Loading

def pick_file():

    app = QApplication.instance() or QApplication(sys.argv)

    path, _ = QFileDialog.getOpenFileName(None, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")

    if not path:
        return None, None

    excel_path = Path(path)
    pdf_path = excel_path.with_suffix('.pdf')

    return excel_path, pdf_path

def load_data(excel_path):

    # load file
    df = pd.read_excel(excel_path, header=0)

    # sample information
    groups = df.iloc[:,0].tolist()
    sex = df.iloc[:,1].tolist()
    sample_names = df.iloc[:,2].tolist()

    # feature information
    feature_names = df.columns[3:].tolist()

    # data matrix
    matrix = df.iloc[:,3:].to_numpy(dtype=float)

    # group indices (by sex and not)
    group_indices = {}
    sex_group_indices = {}
    for i,(group,s) in enumerate(zip(groups,sex)):
        if group not in group_indices:
            group_indices[group] = []
        group_indices[group].append(i)
        
        # stratify by sex
        if s:
            key = f"{group}_{s}"
            if key not in sex_group_indices:
                sex_group_indices[key] = []
            sex_group_indices[key].append(i)

    # sample map
    sample_map = {}
    for i,sample in enumerate(sample_names):
        sample_map[sample] = i

    # feature map
    feature_map = {}
    for i,feature in enumerate(feature_names):
        feature_map[feature] = i

    return matrix, sample_map, feature_map, group_indices, sex_group_indices

# endregion

def main():

    # ----------------------------- Settings -----------------------------
    # true to sratify groups by sex, false to not
    stratify_by_sex = True
    # fraction of missing values per-group at which a column will be dropped (0.5 is 50%), only drops if ALL groups have more nan values than this
    sparseness_threshold = 0.5
    # how to impute, see top docstring
    imputation_method = 'hm'
    # number of neighbors for knn impuation (if you are using knn imputaion)
    impute_knns = 5
    # method for fdr correction, 'fdr_bh' is Benhamini-Hotchburg or 'bonferroni' for Bonferroni
    fdr_method = 'fdr_bh'
    # number of plots per page for Significance testing (4 default)
    n_per_page = 4
    # --------------------------------------------------------------------

    # get excel path
    excel_path, pdf_path = pick_file()
    if not excel_path:
        return
    
    # stup logger
    log_path = excel_path.with_suffix('.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=log_path
    )
    
    # load data
    raw_matrix, sample_map, raw_feature_map, group_indices, sex_group_indices = load_data(excel_path)
    logger.info("Data Loaded")

    # get group indices to use
    if stratify_by_sex and not sex_group_indices:
        logger.warning("stratify_by_sex is True but no sex data found, falling back to group_indices")
        indices_to_use = group_indices
    else:
        indices_to_use = sex_group_indices if stratify_by_sex else group_indices

    # preprocess matrix
    matrix, keep = full_preprocess(raw_matrix, indices_to_use,
                                   sparseness_threshold,
                                   imputation_method,
                                   impute_knns)
    
    # genearte new feature map and stuff
    raw_feature_names = list(raw_feature_map.keys())
    feature_map = {raw_feature_names[old_idx]:new_idx for new_idx,old_idx in enumerate(keep)}
    feature_list = list(feature_map.keys())
    dropped_features = []
    for entry in raw_feature_names:
        if entry not in feature_list:
            dropped_features.append(entry)
    logger.info("Data Preprocessed")

    # run pca
    pca_data = pca(matrix,0.8)
    logger.info("PCA complete")

    # run kw w/ dunn
    kw_results = kruskal_wallis(matrix, indices_to_use)
    fdr_corrected = fdr_correction(kw_results['p_values'], fdr_method)
    sig_indices = np.where(fdr_corrected < 0.05)[0]
    logger.info(f"Kruskal-Wallic with Dunn posthoc complete\nSignificant Features Found: {len(sig_indices)}")

    # create document
    with PdfPages(pdf_path) as pdf:

        # ---------- Page 1: pca plot ----------
        fig,ax = plt.subplots(figsize=(11,8.5))
        plot_pca(ax,pca_data,indices_to_use)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Page 2: plot skree/cumvar plot ----------
        fig = plt.figure(figsize=(11,8.5))
        gs = gridspec.GridSpec(1,2,figure=fig)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        plot_scree_bar(ax1, pca_data['explained_variance'], 'Explianed Variance')
        plot_scree_bar(ax2, pca_data['cumulative_variance'], 'Cumulative Variance')

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Page 3: barplot of all features ----------
        data = []
        for j in range(raw_matrix.shape[1]):
            col = raw_matrix[:,j]
            data.append(col)
        fig, ax = plt.subplots(figsize=(11,8.5))
        plot_bar(ax,data,list(raw_feature_map.keys()),ylabel='Signal',title='Per-Molecule Signal')

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Page 4+: Significance Testing ----------

        # Table of results
        group_names = kw_results['group_names']
        posthoc_pvals = kw_results['posthoc_pvals']

        rows = []
        for j in sig_indices:
            feature = feature_list[j]
            for gi,g1 in enumerate(group_names):
                for gj in range(gi+1, len(group_names)):
                    g2 = group_names[gj]
                    p = posthoc_pvals[j,gi,gj]
                    if p < 0.05:
                        rows.append([feature,g1,g2,format_val(p)])

        col_labels = ['Feature','Group 1','Group 2', 'Dunn P (BH)']

        for fig in plot_pdf_table(rows, col_labels=col_labels, title='Significant Pariwise Comparisons'):
            pdf.savefig(fig)
            plt.close(fig)


        # barplots of results
        for page_start in range(0,len(sig_indices), n_per_page):
            fig = plt.figure(figsize=(11,8.5))
            gs = gridspec.GridSpec(2, 2, figure=fig)

            for k,j in enumerate(sig_indices[page_start : page_start + n_per_page]):
                ax = fig.add_subplot(gs[k//2, k%2])
                data = [matrix[indices,j] for indices in indices_to_use.values()]
                labels = list(indices_to_use.keys())
                label_positions = plot_bar(ax,data,labels,title=feature_list[j])

                # significance brackets for sig Dunn pairs
                group_names = kw_results['group_names']
                max_val = max(np.nanmean(d) + np.nanstd(d) for d in data)
                level = 0
                for gi,g1 in enumerate(group_names):
                    for gj in range(gi+1, len(group_names)):
                        g2 = group_names[gj]
                        p = kw_results['posthoc_pvals'][j,gi,gj]
                        if p < 0.05:
                            y = max_val * (1.05 + 0.08 * level)
                            add_significance(ax,label_positions[g1], label_positions[g2], y, p)
                            level += 1

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    logger.info("PDF file generated")

if __name__ == '__main__':
    main()
    


