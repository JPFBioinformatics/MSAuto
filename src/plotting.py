"""

Plotting functions for QC metrics, statistical analysis visualization, and chromatogram display.
All functions return matplotlib Figure objects and never call plt.show().
The caller decides whether to embed in GUI or save to PDF.

Usage for PDF:
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages("report.pdf") as pdf:
        fig = plot_something(...)
        pdf.savefig(fig)
        plt.close(fig)

Usage for PyQt5 GUI:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    canvas = FigureCanvas(fig)
    layout.addWidget(canvas)

Holds functions for:
    - Heatmaps (missingness, outliers, clustering, correlation)
    - Line/scatter plots (injection order drift, PCA/PLS-DA scores)
    - Dot/strip plots (feature distributions per group)
    - Histograms (permutation test)
    - Box-and-whisker (group comparisons)
    - Scree plots
    - Volcano plots
    - VIP bar charts
    - Confusion matrix
    - Chromatogram traces
    - Single peak visualization

"""

# region Imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram

import logging
logger = logging.getLogger(__name__)

# endregion

# region                 ---------- Internal Helpers ----------

_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

def _color_map(names: list):
    return {name: _COLORS[i % len(_COLORS)] for i, name in enumerate(names)}

# endregion

# region                 ---------- QC: Heatmaps ----------

def plot_missing_heatmap(missing_matrix: np.ndarray, sample_names: list, mol_names: list,
                         title: str = "Missingness"):
    """
    Heatmap of missing values (samples x features). Black = missing, white = present.

    Params
    ------
    missing_matrix      bool (n_samples, n_features) from DataMatrix.missing
    sample_names        row labels, index matched to matrix rows
    mol_names           column labels, index matched to matrix columns
    """
    n_rows, n_cols = missing_matrix.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.4), max(4, n_rows * 0.3)))
    ax.imshow(missing_matrix.astype(float), aspect='auto', cmap='Greys',
              vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(mol_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(sample_names, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    return fig

def plot_outlier_heatmap(outlier_matrix: np.ndarray, sample_names: list, mol_names: list,
                         metric: str = "", title: str = ""):
    """
    Heatmap of outlier flags (samples x features). Red = outlier, white = normal.

    Params
    ------
    outlier_matrix      bool (n_samples, n_features) from DataMatrix.outliers[metric]
    metric              metric name shown in title if no explicit title given
    """
    if not title:
        title = f"Outliers — {metric}" if metric else "Outliers"
    n_rows, n_cols = outlier_matrix.shape
    cmap = ListedColormap(['white', 'crimson'])
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.4), max(4, n_rows * 0.3)))
    ax.imshow(outlier_matrix.astype(float), aspect='auto', cmap=cmap,
              vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(mol_names, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(sample_names, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Sample")
    fig.tight_layout()
    return fig

def plot_heatmap(matrix: np.ndarray, row_labels: list, col_labels: list,
                 title: str = "", xlabel: str = "", ylabel: str = "",
                 cmap: str = 'viridis'):
    """
    General-purpose heatmap for any (rows x cols) numeric matrix.
    """
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(6, n_cols * 0.4), max(4, n_rows * 0.3)))
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig

# endregion

# region                 ---------- QC: Injection Order ----------

def plot_injection_order(values: np.ndarray, injection_orders: np.ndarray,
                         sample_names: list, title: str = "", ylabel: str = ""):
    """
    Line + scatter of a QC metric vs injection order. Useful for detecting run-order drift in
    RT, S/N, theoretical plates, etc.

    Params
    ------
    values              (n_samples,) array of metric values, one per sample
    injection_orders    (n_samples,) integer injection order, index matched to values
    sample_names        sample name labels, index matched to values
    """
    sort_idx = np.argsort(injection_orders)
    x = injection_orders[sort_idx]
    y = values[sort_idx]
    names = [sample_names[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(max(6, len(x) * 0.35), 4))
    ax.plot(x, y, color='steelblue', linewidth=1.2, zorder=1)
    ax.scatter(x, y, color='steelblue', s=40, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("Injection Order")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

# endregion

# region                 ---------- QC: Box / Strip Plots ----------

def plot_boxplot(values: np.ndarray, group_indices: dict,
                 title: str = "", xlabel: str = "", ylabel: str = ""):
    """
    Box-and-whisker for a single feature across groups.

    Params
    ------
    values              (n_samples,) array for one feature (one column of DataMatrix.data)
    group_indices       dict of group_name -> list of row indices (DataMatrix.group_indices)
    """
    group_names = list(group_indices.keys())
    data = [values[idxs] for idxs in group_indices.values()]
    cmap = _color_map(group_names)

    fig, ax = plt.subplots(figsize=(max(4, len(group_names) * 0.9), 5))
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
    for patch, name in zip(bp['boxes'], group_names):
        patch.set_facecolor(cmap[name])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(group_names) + 1))
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

def plot_stripplot(values: np.ndarray, group_indices: dict,
                   title: str = "", xlabel: str = "", ylabel: str = ""):
    """
    Strip / dot plot with jitter for a single feature across groups.
    Horizontal median bar overlaid on each group.

    Params
    ------
    values              (n_samples,) array for one feature
    group_indices       dict of group_name -> list of row indices (DataMatrix.group_indices)
    """
    group_names = list(group_indices.keys())
    cmap = _color_map(group_names)

    fig, ax = plt.subplots(figsize=(max(4, len(group_names) * 0.9), 5))
    for pos, (name, idxs) in enumerate(group_indices.items(), start=1):
        y = values[idxs]
        x = np.random.normal(pos, 0.08, size=len(y))
        ax.scatter(x, y, color=cmap[name], alpha=0.75, s=40, zorder=3)
        ax.plot([pos - 0.2, pos + 0.2], [np.nanmedian(y)] * 2,
                color='black', linewidth=2, zorder=4)
    ax.set_xticks(range(1, len(group_names) + 1))
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    ax.set_xlim(0.5, len(group_names) + 0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

# endregion

# region                 ---------- Analysis: Volcano ----------

def plot_volcano(fold_changes: np.ndarray, p_values: np.ndarray, mol_names: list,
                 p_thresh: float = 0.05, fc_thresh: float = 1.0,
                 title: str = "Volcano Plot"):
    """
    Volcano plot: log2 fold change (x) vs -log10 p-value (y).
    Significant points (above p_thresh and beyond fc_thresh) are colored red and labeled.

    Params
    ------
    fold_changes        (n_features,) from log2_fold_change()
    p_values            (n_features,) corrected p-values from fdr_correction()
    mol_names           feature name labels, index matched to arrays
    """
    safe_p = np.where(p_values == 0, 1e-300, p_values)
    neg_log_p = -np.log10(safe_p)
    sig = (p_values < p_thresh) & (np.abs(fold_changes) >= fc_thresh)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(fold_changes[~sig], neg_log_p[~sig], color='lightgrey',
               s=20, alpha=0.7, label='Not significant')
    ax.scatter(fold_changes[sig], neg_log_p[sig], color='crimson',
               s=30, zorder=3, label='Significant')

    for i in np.where(sig)[0]:
        ax.annotate(mol_names[i], (fold_changes[i], neg_log_p[i]),
                    fontsize=6, ha='center', va='bottom',
                    xytext=(0, 3), textcoords='offset points')

    ax.axhline(-np.log10(p_thresh), color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(fc_thresh, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(-fc_thresh, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlabel("Log₂ Fold Change")
    ax.set_ylabel("-Log₁₀ p-value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

# endregion

# region                 ---------- Analysis: PCA / PLS-DA Scores ----------

def plot_scores(scores: np.ndarray, labels: list, explained_variance: np.ndarray = None,
                title: str = "Scores Plot",
                xlabel: str = "Component 1", ylabel: str = "Component 2"):
    """
    Scatter of sample scores, one point per sample colored by group label.
    Works for both PCA (pass explained_variance) and PLS-DA (pass None).

    Params
    ------
    scores              (n_samples, n_components) from pca()['scores'] or pls_da()['scores']
    labels              group label per sample, index matched (used for color grouping)
    explained_variance  (n_components,) from pca()['explained_variance'], or None for PLS-DA
    """
    group_names = list(dict.fromkeys(labels))
    cmap = _color_map(group_names)

    if explained_variance is not None:
        xlabel = f"PC1 ({explained_variance[0] * 100:.1f}%)"
        ylabel = f"PC2 ({explained_variance[1] * 100:.1f}%)"

    fig, ax = plt.subplots(figsize=(6, 5))
    for name in group_names:
        idx = [i for i, l in enumerate(labels) if l == name]
        ax.scatter(scores[idx, 0], scores[idx, 1], label=name,
                   color=cmap[name], s=55, alpha=0.85)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.5)
    fig.tight_layout()
    return fig

def plot_scree(scree: dict, title: str = "Scree Plot"):
    """
    Bar chart of variance explained per component with cumulative line and 80% reference.

    Params
    ------
    scree       dict from scree_data() with keys 'explained', 'cumulative', 'n_for_80pct'
    """
    explained = scree['explained']
    cumulative = scree['cumulative']
    n = len(explained)
    x = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(max(5, n * 0.5), 4))
    ax1.bar(x, explained * 100, color='steelblue', alpha=0.7, label='Per component')
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_xticks(x)
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(x, cumulative * 100, color='crimson', marker='o',
             markersize=4, linewidth=1.5, label='Cumulative')
    ax2.axhline(80, color='grey', linestyle='--', linewidth=0.8, label='80% threshold')
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_ylim(0, 105)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc='center right')
    fig.tight_layout()
    return fig

# endregion

# region                 ---------- Analysis: Permutation Test ----------

def plot_permutation(perm_result: dict, title: str = "PLS-DA Permutation Test"):
    """
    Histogram of permuted R² values with a vertical line at the real R².

    Params
    ------
    perm_result     dict from permutation_test() with keys 'real_r2', 'perm_r2s', 'p_value'
    """
    real_r2 = perm_result['real_r2']
    perm_r2s = perm_result['perm_r2s']
    p_value = perm_result['p_value']

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(perm_r2s, bins=30, color='steelblue', alpha=0.7,
            edgecolor='white', label='Permuted R²')
    ax.axvline(real_r2, color='crimson', linewidth=2,
               label=f'Real R² = {real_r2:.3f}')
    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    ax.set_title(f"{title}  (p = {p_value:.4f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

# endregion

# region                 ---------- Analysis: Clustering Heatmap ----------

def plot_clustering_heatmap(matrix: np.ndarray, clust: dict,
                             row_labels: list, col_labels: list,
                             title: str = "Clustering Heatmap",
                             cmap: str = 'RdBu_r'):
    """
    Clustered heatmap with row (sample) and column (feature) dendrograms.

    Params
    ------
    matrix          (n_samples, n_features) data matrix to display
    clust           dict from hierarchical_clustering() with keys
                    'row_linkage', 'col_linkage', 'row_order', 'col_order'
    row_labels      sample name labels, index matched to matrix rows
    col_labels      feature name labels, index matched to matrix columns
    """
    row_order = clust['row_order']
    col_order = clust['col_order']
    row_linkage = clust['row_linkage']
    col_linkage = clust['col_linkage']

    ordered = matrix[np.ix_(row_order, col_order)]
    ordered_rows = [row_labels[i] for i in row_order]
    ordered_cols = [col_labels[i] for i in col_order]

    n_rows, n_cols = ordered.shape
    fig = plt.figure(figsize=(max(8, n_cols * 0.4), max(6, n_rows * 0.3)))
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 5], height_ratios=[1, 5],
                           hspace=0.01, wspace=0.01)

    ax_col_dend = fig.add_subplot(gs[0, 1])
    dendrogram(col_linkage, ax=ax_col_dend, no_labels=True, color_threshold=0)
    ax_col_dend.axis('off')

    ax_row_dend = fig.add_subplot(gs[1, 0])
    dendrogram(row_linkage, ax=ax_row_dend, orientation='left',
               no_labels=True, color_threshold=0)
    ax_row_dend.axis('off')

    ax_heat = fig.add_subplot(gs[1, 1])
    vmax = np.nanpercentile(np.abs(ordered), 95)
    im = ax_heat.imshow(ordered, aspect='auto', cmap=cmap,
                        vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax_heat.set_xticks(range(n_cols))
    ax_heat.set_xticklabels(ordered_cols, rotation=90, fontsize=6)
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(ordered_rows, fontsize=6)
    fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.01)
    fig.suptitle(title)
    return fig

# endregion

# region                 ---------- Analysis: Correlation Heatmap ----------

def plot_correlation_heatmap(corr_matrix: np.ndarray, labels: list,
                              title: str = "Correlation Matrix"):
    """
    Heatmap of a correlation matrix (feature-feature or sample-sample). Diverging colormap
    centered at 0, range [-1, 1].

    Params
    ------
    corr_matrix     square (n x n) array from correlation_matrix()['corr']
    labels          row/column labels, index matched
    """
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.4), max(5, n * 0.4)))
    im = ax.imshow(corr_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

# endregion

# region                 ---------- Analysis: VIP Scores ----------

def plot_vip(vip_scores: np.ndarray, mol_names: list,
             n: int = 20, title: str = "VIP Scores"):
    """
    Horizontal bar chart of top-n VIP scores from PLS-DA, highest at top.
    Bars above the VIP > 1.0 importance threshold are colored red.

    Params
    ------
    vip_scores      (n_features,) array from vip_scores()
    mol_names       feature name labels, index matched to vip_scores
    n               number of top features to show
    """
    sort_idx = np.argsort(vip_scores)[::-1][:n]
    scores = vip_scores[sort_idx]
    names = [mol_names[i] for i in sort_idx]
    colors = ['crimson' if s > 1.0 else 'steelblue' for s in scores]

    fig, ax = plt.subplots(figsize=(6, max(4, n * 0.3)))
    ax.barh(range(len(scores)), scores, color=colors)
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(1.0, color='grey', linestyle='--', linewidth=0.8, label='VIP = 1.0 threshold')
    ax.set_xlabel("VIP Score")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

# endregion

# region                 ---------- Analysis: Confusion Matrix ----------

def plot_confusion_matrix(cv_result: dict, group_names: list,
                           title: str = "PLS-DA Cross-Validation"):
    """
    Heatmap of the confusion matrix from PLS-DA cross-validation with per-cell counts.

    Params
    ------
    cv_result       dict from pls_da_cv() with keys 'confusion', 'accuracy'
    group_names     original group label strings in the same order as the confusion matrix
    """
    cm = cv_result['confusion']
    accuracy = cv_result['accuracy']
    n = len(group_names)

    fig, ax = plt.subplots(figsize=(max(4, n), max(4, n)))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(n))
    ax.set_xticklabels(group_names, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(group_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title}\nAccuracy = {accuracy:.1%}")

    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=10,
                    color='white' if cm[i, j] > thresh else 'black')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

# endregion

# region                 ---------- Chromatogram ----------

def plot_chromatogram(time_array: np.ndarray, intensity_array: np.ndarray,
                      peaks: list = None, title: str = "",
                      xlabel: str = "Retention Time (min)",
                      ylabel: str = "Intensity"):
    """
    Single ion chromatogram trace with optional peak bound overlays.
    Peak regions are shaded orange, bounds marked with dashed lines, apex with a red vertical line.

    Params
    ------
    time_array          (n_timepoints,) retention time values
    intensity_array     (n_timepoints,) intensity values for one ion
    peaks               list of peak dicts with 'left_bound', 'right_bound', 'center' (scan indices)
                        and optionally 'molecule' for annotation labels
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_array, intensity_array, color='steelblue', linewidth=1.0)

    if peaks:
        for peak in peaks:
            lb = peak.get('left_bound')
            rb = peak.get('right_bound')
            center = peak.get('center')
            if lb is not None and rb is not None:
                ax.axvspan(time_array[lb], time_array[rb], alpha=0.15, color='orange')
                ax.axvline(time_array[lb], color='grey', linewidth=0.7, linestyle='--')
                ax.axvline(time_array[rb], color='grey', linewidth=0.7, linestyle='--')
            if center is not None:
                ax.axvline(time_array[center], color='crimson', linewidth=0.8)
                mol = peak.get('molecule')
                if mol:
                    ax.annotate(mol, xy=(time_array[center], intensity_array[center]),
                                fontsize=6, ha='center', va='bottom',
                                xytext=(0, 4), textcoords='offset points')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_peak(time_array: np.ndarray, intensity_array: np.ndarray, peak: dict,
              title: str = "", ylabel: str = "Intensity"):
    """
    Zoomed view of a single peak with reconstructed baseline and key metrics annotated.
    Baseline is reconstructed from the stored slope/y-intercept.

    Params
    ------
    time_array          (n_timepoints,) retention time values
    intensity_array     (n_timepoints,) intensity values for one ion
    peak                peak dict with 'left_bound', 'right_bound', 'center',
                        'bl_slope', 'bl_yint', 'rt', 'area', 'fwhh', 'sn_ratio'
    """
    lb = peak['left_bound']
    rb = peak['right_bound']
    center = peak['center']

    pad = max(5, (rb - lb))
    view_l = max(0, lb - pad)
    view_r = min(len(time_array) - 1, rb + pad)

    t = time_array[view_l:view_r + 1]
    y = intensity_array[view_l:view_r + 1]
    scan_indices = np.arange(view_l, view_r + 1)
    baseline = peak['bl_slope'] * scan_indices + peak['bl_yint']

    rel_lb = lb - view_l
    rel_rb = rb - view_l

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, y, color='steelblue', linewidth=1.2, label='Signal')
    ax.plot(t, baseline, color='grey', linewidth=0.8, linestyle='--', label='Baseline')
    ax.fill_between(t[rel_lb:rel_rb + 1], baseline[rel_lb:rel_rb + 1],
                    y[rel_lb:rel_rb + 1], alpha=0.2, color='steelblue')
    ax.axvline(time_array[lb], color='grey', linewidth=0.7, linestyle=':')
    ax.axvline(time_array[rb], color='grey', linewidth=0.7, linestyle=':')
    ax.axvline(time_array[center], color='crimson', linewidth=0.8, linestyle='--')

    info = (f"RT={peak.get('rt', float('nan')):.3f}  "
            f"Area={peak.get('area', float('nan')):.0f}  "
            f"FWHH={peak.get('fwhh', float('nan')):.4f}  "
            f"S/N={peak.get('sn_ratio', float('nan')):.1f}")
    ax.set_xlabel(f"Retention Time (min)\n{info}")
    ax.set_ylabel(ylabel)
    ax.set_title(title or peak.get('molecule', 'Peak'))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

# endregion
