"""

Plotting functions for QC metrics, statistical analysis visualization, and chromatogram display.
All functions return matplotlib Figure objects and never call plt.show().

"""

# region Imports

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib import cm
from scipy.cluster.hierarchy import dendrogram

import logging
logger = logging.getLogger(__name__)

# endregion

# region                 ---------- Primitives ----------

DARK_MODE = True

def plot_boxplot(ax: Axes, data, labels, ylabel='', title='', cutoff=None, color=True):
    """
    makes a set of boxplots from the data given, optionally including a horizontal cutoff line

    Params
    ------
    ax                          ax to plot on
    data                        list of arrays of data to make boxplots of
    labels                      list of str labels index matched to data
    ylabel/title                for annotating axis/title
    cutoff                      value to plot a cutoff line at
    """
    clean_data = [arr[~np.isnan(arr)] for arr in data]

    bp = ax.boxplot(clean_data,labels=labels, patch_artist=True)

    if color:
        if len(data) > 7:
            cmap = cm.get_cmap('tab20b', len(data))
        else:
            cmap = cm.get_cmap('Dark2', max(len(data), 2))

        for i, patch in enumerate(bp['boxes']):
            c = cmap(i / len(data))
            patch.set_facecolor(c)
            patch.set_alpha(0.5)

    if cutoff is not None:
        ax.axhline(cutoff, color='red', linestyle='--')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.figure.tight_layout()
    apply_dark_theme(ax.figure, ax)

def plot_scatter(ax: Axes, data, labels=None, ylabel='', xlabel='', title='', fit=True, 
                 color=True, avg=False, legend=True, annotate_fit=False, series_labels=None):
    """
    Plots a scatter plot with optional labelling, horizontal/vertical lines, and linear fit

    Params
    ------
    ax                          ax to plot on
    data                        list of arrays to plot, either two lists of arrays with equal count or
                                data[0] is a list of arrays to average and data[1] is the corresponding
                                x values (so array at data[0[0] averages and gets x value at data[1][0])
    point_labels                str labels index matched to x,y
    ylabel/xlabel/title         str axis/title labels
    fit                         True/False to calculate a linear fit for x,y
    color                       T/F color your points?
    avg                         T/F average data before plotting?
    """
    # average serie(s)
    if avg:
        x = []
        x_err = []
        y = []
        y_err = []
        for array in data[0]:
            y.append(np.nanmean(array))
            y_err.append(np.nanstd(array))
        for array in data[1]:
            x.append(np.nanmean(array))
            x_err.append(np.nanstd(array))
        y = [np.array(y)]
        y_err = [np.array(y_err)]
        x = [np.array(x)]
        x_err = [np.array(x_err)]
    else:
        y = data[0]
        x = data[1]
    
    # setup colormap
    if len(x) > 7 :
        colors = 'tab20b'
    else:
        colors = 'Dark2'
    cmap = cm.get_cmap(colors, max(len(x), 2))
    
    # Add error bars
    for i,(xi,yi) in enumerate(zip(x,y)):
        label = labels[i] if labels else None
        c = cmap(i / len(x)) if color else None
        if avg:
            ax.errorbar(xi, yi, xerr=x_err[i], yerr=y_err[i], fmt='o', capsize=4, color=c)
        else:
            ax.scatter(xi,yi, color=c, label=label)

    # add linear fit
    if fit:
        for i,(xi,yi) in enumerate(zip(x,y)):

            mask = ~(np.isnan(xi) | np.isnan(yi))
            xi = xi[mask]
            yi = yi[mask]

            if len(xi) < 2:
                continue

            c = cmap(i / len(x)) if color else 'blue'
            coeffs = np.polyfit(xi,yi,1)
            fit_fn = np.poly1d(coeffs)

            x_line = np.linspace(min(xi), max(xi), 100)

            residuals = yi - fit_fn(xi)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((yi - np.mean(yi))**2)
            r2 = 1 - (ss_res / ss_tot)
        
            ax.plot(x_line, fit_fn(x_line), color=c)

            if annotate_fit:
                slope = fmt(coeffs[0])
                yint = fmt(coeffs[1])
                rsqr = fmt(r2)
                ax.annotate(f'y = ({slope})x + {yint}\nR2={rsqr}', 
                            xy=(0.05, 0.85 - i*0.05), xycoords='axes fraction', color=c)
    
    if series_labels:
        for idx,label in series_labels.items():
            xi,yi = x[idx], y[idx]
            valid = ~(np.isnan(xi) | np.isnan(yi))
            if not np.any(valid):
                continue
            ax.annotate(label, (xi[valid][-1],yi[valid][-1]),
                                fontsize=7, xytext=(4,0), textcoords='offset points')

    if legend and labels:
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.figure.tight_layout()
    apply_dark_theme(ax.figure, ax)

    def on_click(event):
        if event.inaxes != ax:
            return

        # initialize markers/annotations as list
        if not hasattr(ax, '_click_anns'):
            ax._click_anns = []
        if not hasattr(ax, '_click_markers'):
            ax._click_markers = []
        
        # double click to clear annotations
        if event.dblclick:
            for ann in ax._click_anns:
                ann.remove()
            for marker in ax._click_markers:
                marker.remove()
            ax._click_anns = []
            ax._click_markers = []
            ax.figure.canvas.draw_idle()
            return
        
        best_dist = float('inf')
        best_label = None
        best_x = None
        best_y = None
        
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

        for i, (xi,yi) in enumerate(zip(x,y)):
            valid = ~(np.isnan(xi) | np.isnan(yi))
            if not np.any(valid):
                continue
            x_iv = xi[valid]
            y_iv = yi[valid]
            
            distances = np.sqrt(((x_iv - event.xdata) / x_range)**2 +
                                ((y_iv - event.ydata) / y_range)**2)
            idx = np.argmin(distances)

            if distances[idx] < best_dist:
                best_dist = distances[idx]
                best_x = x_iv[idx]
                best_y = y_iv[idx]
                if len(x) == 1 and len(labels) > 1:
                    best_label = labels[idx]
                else:
                    best_label = labels[i] if labels else f"{best_x:.3f}, {best_y:.3f}"

        ann = ax.annotate(best_label, (best_x,best_y), xytext=(10,10),
                                    textcoords='offset points',
                                    bbox=dict(boxstyle='round', fc='black', alpha=0.7),
                                    color='white')
        marker, = ax.plot(best_x, best_y, 'o', markerfacecolor='none',
                                    markeredgecolor='red', markeredgewidth=2, markersize=12)
        
        ax._click_anns.append(ann)
        ax._click_markers.append(marker)

        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('button_press_event', on_click)

def plot_heatmap(ax: Axes, matrix, row_labels='', col_labels='', title='', cmap='RdYlGn'):
    """
    Plots a heatmap witha annotation and a colorbar

    Params
    ------
    ax                          ax to plot on
    matrix                      matrix of values to plot
    row/col_labels              labels for row/cols of the matrix
    title                       title
    cmap                        colormap to use
    """

    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    ax.figure.colorbar(im, ax=ax)

    if col_labels:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90)

    if row_labels:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)

    ax.set_title(title)
    ax.figure.tight_layout()
    apply_dark_theme(ax.figure, ax)

def plot_violin(ax: Axes, data, labels, ylabel='', title='', color=True):
    """
    Plots a set of violin plots

    Params
    ------
    ax                          ax to plot on
    data                        list of arrays to plot
    labels                      list of labels index matched to data
    ylabel/title                annotations
    color                       wether or not to color the plots
    """
    clean_data = [arr[~np.isnan(arr)] for arr in data]

    parts = ax.violinplot(clean_data, showmedians=True)

    if color:

        if len(data) > 7:
            cmap = cm.get_cmap('tab20b', len(data))
        else:
            cmap = cm.get_cmap('Dark2', max(len(data), 2))

        colors = [cmap(i/len(data)) for i in range(len(data))]
        for body,color in zip(parts['bodies'], colors):
            body.set_facecolor(color)
            body.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels)

    ax.set_ylabel(ylabel)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')
    ax.set_title(title)
    ax.figure.tight_layout()
    apply_dark_theme(ax.figure, ax)

def plot_histogram(ax: Axes, data, labels, xlabel='', title='', bins=20):
    """
    Plots overlaid histograms (or just one)

    Params
    ------
    ax                          ax to plot on
    data                        list of array distributinos to plot
    labels                      list of str labels index matched to data
    xlabel/title                annotations
    bins                        number of bins to break data into
    """

    for arr,label in zip(data,labels):
        ax.hist(arr, bins=bins, alpha=0.6, label=label)

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.figure.tight_layout()
    apply_dark_theme(ax.figure, ax)

def plot_bar(ax: Axes, data, labels, ylabel='', title='', cutoff=None, sorted_desc=True, type = 'avg'):
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
        avg = np.nanmean(arr)
        sd = np.nanstd(arr)
        if type == 'avg':
            values.append(avg)
            errs.append(sd)
        elif type == '%CV':
            cv = (sd/avg)*100 if avg != 0 else np.nan
            values.append(cv)
        elif type == 'count':
            values.append(np.nansum(arr))

    if sorted_desc:
        paris = sorted(zip(values,labels), reverse=True)
        values,labels = zip(*paris)

    if type == 'avg':
        ax.bar(range(len(values)), values, yerr=errs, capsize=4)
    else:
        ax.bar(range(len(values)), values)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)

    if cutoff is not None:
        for h in cutoff:
            ax.axhline(h, color='red', linestyle='--', label=f'{h:3f}')

    ax.set_ylabel(ylabel)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha='right')

    ax.set_title(title)
    ax.figure.tight_layout()
    apply_dark_theme(ax.figure, ax)

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
        apply_dark_theme(fig,ax)
        tables.append(fig)

    return tables

def plot_volcano():
    pass

def plot_corr_heatmap():
    pass

def plot_ranked_bar():
    pass

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
        cmap = cm.get_cmap('tab20b', len(group_names))
    else:
        cmap = cm.get_cmap('Dark2', max(len(group_names), 2))
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
        title = f"PC{pc_x+1} vs PC{pc_y+1} Plot"

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
    apply_dark_theme(ax.figure, ax)


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
                      ylabel: str = "Intensity",
                      ax: Axes = None):
    """
    Single ion chromatogram trace with optional peak bound overlays.
    Peak regions are shaded blue, bounds marked with points, apex with a red v at the top of the plot, 
    baseline with a red dotted horizontal line

    Params
    ------
    time_array          (n_timepoints,) retention time values
    intensity_array     (n_timepoints,) intensity values for one ion
    peaks               list of peak dicts with 'left_bound', 'right_bound', 'center' (scan indices)
                        and optionally 'molecule' for annotation labels
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(time_array, intensity_array, color='steelblue', linewidth=1.0)

    if peaks:
        for peak in peaks:
            lb = peak.get('left_bound')
            rb = peak.get('right_bound')
            center = peak.get('center')
            rt = peak.get('rt')
            bl_slope = peak.get('bl_slope')
            bl_yint = peak.get('bl_yint')

            if lb and rb :
                ax.fill_between(time_array[lb:rb+1], intensity_array[lb:rb+1], alpha=0.15, color='lightblue')
                ax.scatter(time_array[lb], intensity_array[lb], color='crimson')
                ax.scatter(time_array[rb], intensity_array[rb], color='crimson')

            if bl_slope and bl_yint:
                x = np.array(time_array[lb], time_array[rb])
                y = bl_slope * x + bl_yint
                ax.plot(x, y, color = 'crimson', linestyle='--')

            if center:
                ax.scatter(time_array[center], intensity_array[center], marker= '|', color='crimson')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()

    apply_dark_theme(fig,ax)

    return fig

def plot_peak(time_array: np.ndarray, intensity_array: np.ndarray, peak: dict,
              title: str = "", ax: Axes = None):
    """
    Zoomed view of a single peak

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

    pad = max(3, (rb - lb)//2)
    view_l = max(0, lb - pad)
    view_r = min(len(time_array) - 1, rb + pad)

    t = time_array[view_l:view_r + 1]
    y = intensity_array[view_l:view_r + 1]
    scan_indices = np.arange(view_l, view_r + 1)
    baseline = peak['bl_slope'] * (scan_indices-lb) + peak['bl_yint']

    rel_lb = lb - view_l
    rel_rb = rb - view_l

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # signal and baseline
    ax.plot(t, y, color='black', linewidth=1.2, label='Signal')
    ax.plot(t, baseline, color='crimson', linewidth=0.8, linestyle='--', label='Baseline')
    ax.fill_between(t[rel_lb:rel_rb + 1], baseline[rel_lb:rel_rb + 1],
                    y[rel_lb:rel_rb + 1], alpha=0.15, color='lightblue')
    
    # endpoints
    ax.scatter(time_array[lb], intensity_array[lb], color='crimson')
    ax.scatter(time_array[rb], intensity_array[rb], color='crimson')

    # center
    ax.scatter(time_array[center], intensity_array[center], color='crimson', marker='|')

    ax.set_xlabel(f"Retention Time (min)")
    ax.set_ylabel("Intensity")
    ax.set_title(title or peak.get('molecule', 'Peak'))
    ax.legend(fontsize=8)
    fig.tight_layout()

    apply_dark_theme(fig, ax)

    return fig

def plot_spectrum(mzs: np.ndarray, abundances: np.ndarray, ax: Axes = None):
    """
    Plots a spectra from a given mz and abundance array
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    if mzs is None or abundances is None:
        ax.text(0.5, 0.5, 'No Spectrum Available0', ha='center', va='center', 
                transform=ax.transAxes, color='white', fontsize=12)
    else:
        nonzero_mzs = mzs[abundances > 0]
        ax.bar(mzs, abundances, width=0.5, color='steelblue', edgecolor='none')
        ax.set_xlim(nonzero_mzs.min() - 5, nonzero_mzs.max() + 5)
        ax.set_ylim(0,110)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Realtive Abundance")

    fig.tight_layout()
    apply_dark_theme(fig,ax)

    return fig

# endregion

# region                 ---------- Utils ----------

def apply_dark_theme(fig, ax):
    """
    Applies dark theme that matches the app color scheme to figures
    """
    if not DARK_MODE:
        return
    
    fig.patch.set_facecolor('#31363b')
    ax.set_facecolor('#232629')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['bottom'].set_color('#4f5b62')
    ax.spines['left'].set_color('#4f5b62')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def fmt(val):
    """
    Formats a float value 
    """
    if isinstance(val,(float,np.floating)):
        if np.isnan(val):
            return 'N/A'
        if val == 0:
            return '0'
        elif abs(val) > 10000 or abs(val) < 0.0001:
            return f'{val:.4e}'
        return f'{val:.5g}'
    return str(val) if val is not None else 'N/A'

# endregion
