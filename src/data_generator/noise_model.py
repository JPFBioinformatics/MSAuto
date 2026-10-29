"""

Models noise to use for bootstrapping realistic chromatograms for validation/benchmarking
of analytical techniques and model training.

---------- Algorithm Overview ----------

SIGNAL PROCESSING

1) Use already implemented CWT + endpoint detection algorithm to differentiate between signal and noise sections
2) take each ion chromatogram's noise segments and fill in signal blank sections by connecting the nearest signal
point on each side with a line
4) for each segment of 13 scans:
    - linear fit for baseline storage
    - find the difference between fit and true signal at each point, add to a distribution for our noise model
        perhaps do abs(diff)?  don't know yet

BASELINE MODEL
1) stich together baseline segments, fill in gaps again by connecting nearset point on either side with a line
2) smooth baseline with 2 savgol passes: first is 5-poit lienar second is 7-point linear
3) save the full array to a npy file, index matched to the noise model it corresponds to

NOISE MODEL
1) take sampled differernces from baseline and store as residual list, store a censored bool mask as well
that contains a 1 in indices where the raw signal is <= abundance threshold for that segment of that matrix
(meaning it is truncated)
2) if censored.any() then submit the residuals and censored to a Regression Order Statistics (ROS) method to
impute the missing left-censored data and smooth the distribution
    Parameters:
    transform_in/transform_out      defaults to log/exp but since our data will be negative sometimes (diff
                                    from mean is theoretically centered at 0) we will make them both linear
                                    with lambda x: x
    min_uncensored = 2              as the minimum number of uncensored values before
                                    ROS can be used to impute results, falls back to simple substitution if
                                    ROS cannot be used
    max_fraction_censored = 0.8     if <= 80% of data is censored we can use the ROS to impute, if this is 
                                    exceeded then simple substitution is used
3) 

PEAK MODEL

"""

# region Imports
import subprocess, os, json, time
from pathlib import Path
from wqio.ros import ROS
import pandas as pd
import numpy as np
from collections import Counter


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.signal import savgol_filter
from scipy.stats import skew
from scipy.special import erfc, erfcx
from scipy.stats import probplot, skew, probplot, norm
from scipy.optimize import minimize
from scipy import sparse
from scipy.sparse.linalg import spsolve

from lmfit.models import GaussianModel
from lmfit import Model

from src.scripts.helpers import (plot_histogram, plot_table, plot_scatter, plot_hexbin)
from src.main_pipeline.utils import (get_noise_model_dir)
from src.data_generator.sample_cache import load_sample

from concurrent.futures import ProcessPoolExecutor, as_completed

# endregion

# region logging

import logging
logger = logging.getLogger(__name__)

# endregion

class NoiseModel:

    # default kwargs for missing peaks/bl noise analysis
    MISSED_PEAK_KWARGS = dict(k=2.0, min_run=3, hard_k=6.0, grow=2) 
    BL_NOISE_KWARGS = dict(seg_size=50, end_window=15, outlier_threshold=3.5, edge_trim=25, bl_lam=1e10, flat_len=100)


    def __init__(self,
                 cache_dirs,
                 model_name:str,
                 bic_margin: float = 2.0,
                 n_workers: int = None,
                 missed_kwargs: dict = None, bl_noise_kwargs: dict = None,
                 qc_limits: dict = None,
                 fit_peaks: bool = True, make_report: bool = True
                 ):

        # obj info
        self.model_name = model_name
        self.bic_margin = bic_margin
        self.missed_kwargs = {**self.MISSED_PEAK_KWARGS, **(missed_kwargs or {})}
        self.bl_noise_kwargs = {**self.BL_NOISE_KWARGS, **(bl_noise_kwargs or {})}
        self.fit_peaks = fit_peaks
        self.qc_limits = qc_limits
        cache_dirs = [Path(d) for d in cache_dirs]

        # per-sample info from cahce metadata
        self.samples = {}
        tasks = []
        for d in cache_dirs:
            with open (d / 'cache_meta.json') as f:
                meta = json.load(f)
            name = meta['sample']
            if name in self.samples:
                name = f"{name}_{d.name}"
            self.samples[name] = {
                'cache_dir': str(d),
                'scan_interval': meta['scan_interval'],
                'n_scans': meta['n_scans']
            }
            for ion, row in meta['ion_rows'].items():
                tasks.append({
                    'cache_dir': str(d),
                    'sample': name,
                    'ion': float(ion),
                    'row': row,
                    'n_peaks': meta['n_peaks_by_ion'][ion]
                })
        tasks.sort(key=lambda t: t['n_peaks'], reverse=True)  # process longest rows first

        # one BLAS thread per worker (st before workers start)
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

        # fit all rows of all samples in a single pool
        n_workers = n_workers or max(1, (os.cpu_count() or 2) -1)
        logger.info(f"Began Model Fitting: {len(self.samples)} samples, {len(tasks)} rows, {n_workers} workers")
        results = []
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                                 initargs=(self.bic_margin, self.missed_kwargs,
                                           self.bl_noise_kwargs, self.fit_peaks)) as ex:
            futures = [ex.submit(_run_ion, t) for t in tasks]
            for k,fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if k%50 == 0 or k == len(tasks):
                    logger.info(f"{k}/{len(tasks)} rows done")
        logger.info("Finished Model Fitting")

        # merge results
        results.sort(key=lambda r: (r['sample'], r['row']))
        self.exec_counter = Counter()
        fits = []
        self.bl_models, self.noise_models = {}, {}
        self.at_vals = {}
        self.missed_masks = {}
        max_scans = max(s['n_scans'] for s in self.samples.values())
        self.outlier_locations = np.zeros(max_scans,dtype=int)
        self.missed_locations = np.zeros(max_scans,dtype=int)
        for r in results:
            key = (r['sample'], r['ion'])
            self.bl_models[key] = r['baseline']
            self.noise_models[key] = r['residuals']
            self.at_vals[key] = r['at_vals']
            self.outlier_locations[:len(r['outliers'])] += r['outliers']
            self.missed_locations[:len(r['missed'])] += r['missed']
            self.exec_counter.update(r['errors'])
            self.missed_masks[key] = r['missed']
            fits.extend(r['fits'])
        logger.info(f"Fit Errors:\n{self.exec_counter.most_common(10)}")

        # tables
        diag_cols = ('skew', 'tail_ratio', 'acf1', 'frac_gt3', 'sigma', 'level')
        row_cols = ('sample', 'ion', 'row', 'n_outliers', 'kde_h', 'bl_count', 'signal_count', 'n_peaks', 'n_missed', 'bl_level', 'frac_censored',
                    *diag_cols, *(f'{c}_pre' for c in diag_cols), *(f'{c}_post' for c in diag_cols))
        self.row_df = pd.DataFrame([{k: r[k] for k in row_cols} for r in results])
        self.row_df['model_name'] = self.model_name
        self.row_df['frac_missed'] = self.row_df['n_missed'] / self.row_df['bl_count'].clip(lower=1)
        self.fits_df = pd.DataFrame(fits)
        self.fits_df['model_name'] = self.model_name

        # pooled arrays so plot_oc_and_h works unchanged
        self.outlier_counts = self.row_df['n_outliers'].to_numpy(dtype=float)
        self.kde_bandwiths = self.row_df['kde_h'].to_numpy(dtype=float)
        self.n_rows = len(self.row_df)

        # summary logs
        if self.fit_peaks:
            fd = self.fits_df
            logger.info(f"Fit status counts:\n{fd['status'].value_counts(dropna=False)}")
            logger.info(f"Model Counts:\n{pd.crosstab(fd['model'], fd['fronting'])}")
            logger.info(f"Fit Errors: {self.exec_counter.most_common(10)}")

        # QC + peak/noise joins (before the report so they're also saved by save_data)
        self._assign_qc(**(qc_limits or {}))
        if self.fit_peaks:
            self._add_peak_noise_stats()

        if make_report and self.fit_peaks:
            self.write_report()

    # region Report Writer

    def write_report(self):
        """
        builds nm_metrics.pdf, sections in the order you'd read/justify them
        """
        out_dir = get_noise_model_dir() / self.model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with PdfPages(out_dir / 'nm_metrics.pdf') as pdf:


            self._plot_summary(pdf)

            self._section_page(pdf, "1. Input data",
                               ["What the CWT picker gave the model: noise/signal points and peaks per row"])
            self._plot_input_section(pdf)

            self._section_page(pdf, "2. Noise cleaning & row QC",
                               ["Missed-peak filter and ROS effects, row QC labels",
                                "Generator draws only from qc == 'ok' rows"])
            self._plot_noise_section(pdf)
            self._plot_by_sample(pdf)

            self._section_page(pdf, "3. Noise model",
                               ["Shape of the residuals the generator resamples, example rows"])
            self._plot_pooled_residuals(pdf)
            self._plot_baseline_examples(pdf)

            self._section_page(pdf, "4. Peak models",
                               ["Fit outcomes, model choice (Gauss vs EMG), shape parameters, examples"])
            self._plot_peak_joint(pdf)
            self._plot_peak_fits(pdf)
            self._plot_peak_section(pdf)
            self._plot_fit_examples(pdf)

        logger.info(f"Wrote report to {out_dir / 'nm_metrics.pdf'}")

    @staticmethod
    def _section_page(pdf, title, lines=()):
        """
        divider page so the pdf reads in sections
        """
        fig = plt.figure(figsize=(8.5, 4))
        fig.text(0.05, 0.8, title, fontsize=18, weight='bold')
        for i, line in enumerate(lines):
            fig.text(0.05, 0.62 - 0.1 * i, line, fontsize=10)
        pdf.savefig(fig); plt.close(fig)
    
    def _plot_by_sample(self, pdf):
        rd = self.row_df
        # row QC mix per sample, one odd sample stands out
        ct = pd.crosstab(rd['sample'], rd['qc'], normalize='index')
        fig, ax = plt.subplots(figsize=(10, 4))
        ct.plot.bar(stacked=True, ax=ax, width=0.8)
        ax.set_ylabel('Fraction of rows'); ax.set_title('Row QC by sample'); ax.legend(fontsize=7)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # sigma vs level per sample, overlapping lines = noise can be pooled across samples
        ok = rd[(rd['qc'] == 'ok') & (rd['sigma'] > 0) & (rd['bl_level'] > 0)]
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, g in ok.groupby('sample'):
            if len(g) < 10:
                continue
            lx, ly = np.log10(g['bl_level']), np.log10(g['sigma'])
            b, a = np.polyfit(lx, ly, 1)
            xx = np.linspace(lx.min(), lx.max(), 50)
            ax.plot(10 ** xx, 10 ** (a + b * xx), lw=1.2, label=f"{name}  slope={b:.2f}")
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Baseline level'); ax.set_ylabel('Noise sigma')
        ax.set_title('Noise sigma vs level by sample (ok rows)\noverlapping lines = poolable')
        ax.legend(fontsize=7); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    def _plot_peak_joint(self, pdf, sn_min=20):
        d = self.fits_df
        d = d[d['status'].isin(['ok', 'no_errorbars']) & (d['model'] == 'emg') & (d['sn'] > sn_min)
              & (d['row_qc'] == 'ok')]
        if len(d) < 50:
            return
        plot_hexbin(pdf, f"EMG shape, joint (S/N>{sn_min}, ok rows, n={len(d)})",
                    "log10 sigma (scans)", "log10 tau/sigma",
                    np.log10(d['sigma']), np.log10(1 / (d['gamma'] * d['sigma'])))

    # endregion
    
    # region Plotting Helpers

    @staticmethod
    def _hist(pdf, title, xlabel, values, **kwargs):
        """
        plot_hisogram wrapper that skips bad values
        """
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if kwargs.get('log'):
            vals = vals[vals>0]
        if len(vals) == 0:
            logger.info(f"Skipped plot '{title.splitlines()[0]}' (no data)")
            return
        plot_histogram(pdf, title, xlabel, vals, **kwargs)

    @staticmethod
    def _plot_locations(pdf, title, counts):
        """
        counts per scan as a profile across the run (not a histogram of the counts)
        """

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.fill_between(np.arange(len(counts)), counts, step='mid', alpha=0.6)
        ax.set_xlabel('Scan idx'); ax.set_ylabel('Count (all rows)'); ax.set_title(title)
        pdf.savefig(fig); plt.close(fig)

    @staticmethod
    def _before_after(pdf, title, x, y, log=False):
        """
        step effect per row, points on the dashed line = unchanged
        """

        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if log:
            m &= (x > 0) & (y > 0)
        if not m.any():
            return
        x, y = x[m], y[m]
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=6, alpha=0.3, rasterized=True)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        if log:
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('before'); ax.set_ylabel('after'); ax.set_title(title)
        pdf.savefig(fig); plt.close(fig)

    # endregion

    # region QC Ploting

    def _plot_input_section(self, pdf):
        rd = self.row_df
        plot_histogram(pdf, "Noise (baseline) points per row", "N points", rd['bl_count'])
        plot_histogram(pdf, "Signal points per row", "N points", rd['signal_count'])
        plot_histogram(pdf, "Picked peaks per row", "N peaks", rd['n_peaks'])
        self._plot_oc_and_h(pdf)

    def _plot_peak_fits(self, pdf):
        df = self.fits_df
        is_emg = (df['model'] == 'emg').to_numpy()
        is_gauss = (df['model'] == 'gauss').to_numpy()
        is_front = (df['fronting'].astype(bool) & (df['model'] == 'emg')).to_numpy()
        is_tail = is_emg & ~is_front
        fit_ok = df['status'].isin(['ok', 'no_errorbars']).to_numpy()

        n_total = len(df)
        n_informed = int((df['guess_type'] == 0).sum())
        n_ok = int((df['status'] == 'ok').sum())
        n_noerr = int((df['status'] == 'no_errorbars').sum())
        tg_fails, sg_fails = int(df['tg_fail'].sum()), int(df['sg_fail'].sum())
        tau_over_sigma = 1.0 / (df['gamma'] * df['sigma'])

        def pct(k):
            return f"{100 * k / max(n_total, 1):.2f}%"

        # fit outcome summary table
        outcome = pd.crosstab(df['status'].fillna('none'), df['model'].fillna('none'), margins=True)
        plot_table(pdf, f"Peak Fit Outcomes (n={n_total}, informed={n_informed}, "
                        f"TG_fails={tg_fails}, SG_fails={sg_fails})",
                   [[idx] + list(row) for idx, row in outcome.iterrows()],
                   ['status'] + list(outcome.columns))

        # model choice
        self._hist(pdf, f"Delta BIC (gauss - emg), >{self.bic_margin} = EMG chosen\n"
                        f"EMG={is_emg.sum()} ({pct(is_emg.sum())}) "
                        f"Gauss={is_gauss.sum()} ({pct(is_gauss.sum())}) Fronting={is_front.sum()}",
                   "Delta BIC", df['delta_bic'], symlog=True, title_fontsize=10)

        # amplitudes
        self._hist(pdf, f"Amplitudes (area), all fitted\nok={n_ok} no_errbars={n_noerr}",
                   "Amplitude", df.loc[fit_ok, 'amplitude'], log=True)

        # sigmas by model
        for mask, name in [(is_gauss, "Gaussian"), (is_emg, "EMG")]:
            s = df.loc[mask, 'sigma']
            if len(s):
                self._hist(pdf, f"Sigma ({name}), n={len(s)}\n"
                                f"Med={np.nanmedian(s):.2f} Max={np.nanmax(s):.2f} scans",
                           "Sigma (scans)", s, log=True)

        # gammas by direction
        for mask, name in [(is_tail, "tailing"), (is_front, "fronting")]:
            g = df.loc[mask, 'gamma']
            if len(g):
                at_bound = int((g >= 9.99).sum() + (g <= 0.0101).sum())
                self._hist(pdf, f"Gamma (EMG {name}), n={len(g)}\n"
                                f"Med={np.nanmedian(g):.2f} at_bounds={at_bound}",
                           "Gamma (1/scans)", g, log=True)

        # tail shape, scale-free
        self._hist(pdf, "Tail length / width (tau/sigma), EMG only\n<0.3 ~ gaussian-like, >1 = strong tail",
                   "tau / sigma", tau_over_sigma[is_emg], log=True)

        # where do failures come from
        self._hist(pdf, f"Peak height: fitted (n={fit_ok.sum()})", "Height", df.loc[fit_ok, 'height'], log=True)
        self._hist(pdf, f"Peak height: failed/short (n={(~fit_ok).sum()})", "Height",
                   df.loc[~fit_ok, 'height'], log=True)
        self._hist(pdf, f"Window size: failed/short (n={(~fit_ok).sum()})", "N points",
                   df.loc[~fit_ok, 'n_points'], bin_size=1)

    def _plot_oc_and_h(self, pdf):
        """
        plots a histogram and table of outliers in KDE and Outleir count values
        """
        # outlier counts
        oc = self.outlier_counts
        oc_clean = oc[np.isfinite(oc)]
        oc_med = np.median(oc_clean)
        oc_mad = np.median(np.abs(oc_clean - oc_med))
        plot_histogram(pdf, f"Segment outliers per row\nMedian:{oc_med:.2f} | MAD:{oc_mad:.2f} | "
                            f"Min:{oc_clean.min():.0f} | Max:{oc_clean.max():.0f}",
                    "Outlier count", oc_clean, bin_size=1)
        
        # normalised by noise points so rows of different lengths compare
        bl = self.row_df['bl_count'].to_numpy(dtype=float)
        ol_pct = np.where(bl > 0, 100 * oc / np.where(bl > 0, bl, 1), np.nan)
        pct_med = np.nanmedian(ol_pct)
        pct_mad = np.nanmedian(np.abs(ol_pct - pct_med))
        self._hist(pdf, f"Segment outliers per row, % of noise points\n"
                        f"Median:{pct_med:.2f}% | MAD:{pct_mad:.2f}% | rows without noise pts: {int((bl == 0).sum())}",
                "Outliers (% of noise points)", ol_pct, n_bins=50)

        # kde h values
        h_data = self.kde_bandwiths
        h_clean = h_data[~np.isnan(h_data)]
        h_nans = np.sum(np.isnan(h_data))
        h_nums = np.sum(~np.isnan(h_data))
        h_median = np.nanmedian(h_data)
        h_mad = np.nanmedian(np.abs(h_data - h_median))
        h_title = f"KDE h values NaN's:{h_nans} | Successes:{h_nums}\nMedian:{h_median:.2f} | MAD:{h_mad:.2f} | Min:{np.nanmin(h_clean):.2f} | Max:{np.nanmax(h_clean):.2f}"
        plot_histogram(pdf, h_title, "KDE Bandwithd (h)", h_clean, log=True)

    def _plot_summary(self, pdf):
        rd, fd = self.row_df, self.fits_df
        qc = rd['qc'].value_counts()
        fit_ok = fd['status'].isin(['ok', 'no_errorbars'])
        rows = [
            ['samples', len(self.samples)],
            ['ion rows', len(rd)],
            ['rows ok / suspect / too_censored / no_data',
                f"{qc.get('ok', 0)} / {qc.get('suspect', 0)} / {qc.get('too_censored', 0)} / {qc.get('no_data', 0)}"],
            ['median noise points per row', int(rd['bl_count'].median())],
            ['median flagged fraction', f"{rd['frac_missed'].median():.2%}"],
            ['peaks fitted', f"{fit_ok.sum()} / {len(fd)} ({fit_ok.mean():.1%})"],
            ['EMG share of fitted', f"{(fd.loc[fit_ok, 'model'] == 'emg').mean():.1%}"],
            ['bic_margin', self.bic_margin],
            ['missed-peak filter', ', '.join(f"{k}={v}" for k, v in self.missed_kwargs.items())],
            ['baseline/noise', ', '.join(f"{k}={v}" for k, v in self.bl_noise_kwargs.items())],
            ['QC limits', ', '.join(f"{k}={v}" for k, v in self.qc_limits.items())],
        ]
        plot_table(pdf, f"Noise model '{self.model_name}'", rows, ['item', 'value'])

    def _plot_noise_section(self, pdf):
        rd = self.row_df

        # how much the filter removed, where in the run, and on which ions
        self._hist(pdf, f"Fraction of noise points flagged as missed peaks\nMed={rd['frac_missed'].median():.3f}",
                "Flagged fraction", rd['frac_missed'], n_bins=50)
        self._plot_locations(pdf, f"Flagged points by scan (total={int(self.missed_locations.sum())})",
                            self.missed_locations)
        self._plot_locations(pdf, f"Segment outliers by scan (total={int(self.outlier_locations.sum())})",
                            self.outlier_locations)
        plot_scatter(pdf, "Flagged fraction by ion", "m/z", "Flagged fraction", rd['ion'], rd['frac_missed'],
                    color_values=np.log1p(rd['n_peaks']), color_label='log(1 + picked peaks)', alpha=0.4)

        # each cleaning step, isolated
        for a, b, step in (('_pre', '_post', 'missed-peak filter'), ('_post', '', 'ROS imputation')):
            for col, log in (('tail_ratio', True), ('skew', False), ('sigma', True)):
                self._before_after(pdf, f"{col}: effect of {step}", rd[col + a], rd[col + b], log=log)

        # what's left
        self._hist(pdf, "Tail ratio, final (~1 = clean)", "Tail ratio", rd['tail_ratio'], n_bins=50)
        self._hist(pdf, "Lag-1 autocorrelation, final (high = drift left in)", "acf1", rd['acf1'], n_bins=50)
        self._hist(pdf, "Median residual / sigma (~0 = baseline centred in the noise)", "level / sigma",
                rd['level'] / rd['sigma'], n_bins=50)

        ok = (rd['sigma'] > 0) & (rd['bl_level'] > 0) & (rd['qc'] == 'ok')
        if ok.sum() >= 3:
            slope = np.polyfit(np.log10(rd.loc[ok, 'bl_level']), np.log10(rd.loc[ok, 'sigma']), 1)[0]
            plot_scatter(pdf, f"Noise sigma vs baseline level (ok rows)\nlog-log slope={slope:.2f} (0.5 = shot noise)",
                        "Baseline level", "Sigma", rd.loc[ok, 'bl_level'], rd.loc[ok, 'sigma'],
                        xscale='log', yscale='log', alpha=0.3)

        # QC
        qc_rows = [[name, int(m.sum()), f"{m.mean():.1%}"] for name, m in self.qc_checks.items()]
        qc_rows += [[f"qc = {k}", int(v), f"{v / len(rd):.1%}"] for k, v in rd['qc'].value_counts().items()]
        plot_table(pdf, f"Row QC (n={len(rd)})", qc_rows, ['check', 'rows', 'pct'])
        cols = ['sample', 'ion', 'n_peaks', 'frac_missed', 'tail_ratio', 'skew', 'acf1']
        worst = rd[rd['qc'] == 'suspect'].sort_values('frac_missed', ascending=False).head(15)
        plot_table(pdf, "Worst 15 suspect rows",
                [[r[c] if isinstance(r[c], str) else f"{r[c]:.3g}" for c in cols] for _, r in worst.iterrows()],
                cols)

    def _plot_pooled_residuals(self, pdf, max_points=200_000, seed=0):
        """all ok rows on one scale, shows the shape the generator will resample"""
        rd = self.row_df[self.row_df['qc'] == 'ok']
        zs = [(self.noise_models[(r['sample'], r['ion'])] - r['level']) / r['sigma'] for _, r in rd.iterrows()]
        if not zs:
            return
        z = np.concatenate(zs)
        if len(z) > max_points:
            z = np.random.default_rng(seed).choice(z, max_points, replace=False)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        xx = np.linspace(-8, 8, 400)
        a1.hist(np.clip(z, -8, 8), bins=np.linspace(-8, 8, 161), density=True, alpha=0.6, label='residuals')
        a1.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), 'k--', label='N(0,1)')
        a1.set_yscale('log'); a1.set_xlabel('z (per-row robust scale)'); a1.legend(fontsize=8)
        probplot(z, dist='norm', plot=a2)
        fig.suptitle(f"Pooled standardized residuals ({len(rd)} ok rows)\n"
                    f"frac |z|>3 = {(np.abs(z) > 3).mean():.4f} (normal = 0.0027)", fontsize=10)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    def _plot_baseline_examples(self, pdf, n_ok=4, n_worst=4, seed=0):
        rd = self.row_df
        picks = pd.concat([
            rd[rd['qc'] == 'ok'].sample(min(n_ok, (rd['qc'] == 'ok').sum()), random_state=seed),
            rd[rd['qc'] == 'suspect'].sort_values('frac_missed', ascending=False).head(n_worst),
        ])
        for _, r in picks.iterrows():
            key = (r['sample'], r['ion'])
            s = load_sample(self.samples[r['sample']]['cache_dir'])
            raw = np.asarray(s['intensity'][r['row']], dtype=float)
            sig = ~np.asarray(s['noise_mask'][r['row']], dtype=bool)
            missed = self.missed_masks[key]
            x = np.arange(len(raw))

            fig, ax = plt.subplots(figsize=(11, 4))
            edges = np.flatnonzero(np.diff(np.r_[0, sig.astype(np.int8), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                ax.axvspan(a, b, color='tab:blue', alpha=0.12, lw=0)
            ax.plot(x, raw, lw=0.6, color='k', label='raw')
            ax.plot(x, self.bl_models[key], lw=1.2, color='tab:green', label='baseline')
            ax.scatter(x[missed], raw[missed], s=8, color='tab:red', zorder=3, label='flagged')
            ax.set_yscale('symlog', linthresh=max(r['bl_level'], 1))
            ax.set_title(f"[{r['qc']}] {r['sample']} m/z {r['ion']:g} | flagged={r['frac_missed']:.1%} "
                        f"tail={r['tail_ratio']:.2f} acf1={r['acf1']:.2f} (blue = CWT signal)", fontsize=9)
            ax.legend(fontsize=7, loc='upper right')
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    def _plot_peak_section(self, pdf):
        fd = self.fits_df
        ok = fd['status'].isin(['ok', 'no_errorbars']) & (fd['sn'] > 0)
        d = fd[ok].copy()

        if len(d) == 0:
            logger.info("No fited peaks, skiping peak section")
            return

        # S/N range of real peaks -> range the generator should cover
        self._hist(pdf, f"Peak S/N (height / row noise sigma), n={len(d)}", "S/N", d['sn'], log=True)

        # noise-normalized fit quality (replaces raw redchi)
        self._hist(pdf, "Fit quality: redchi / noise sigma²\n~1 = fit within noise, >>1 = shape mismatch",
                "redchi_norm", d['redchi_norm'], log=True)

        # model choice depends on S/N
        if len(d) >= 50:
            d['sn_bin'] = pd.qcut(np.log10(d['sn']), 12, duplicates='drop')
            g = d.groupby('sn_bin', observed=True)
            x, p, n = g['sn'].median(), g['model'].apply(lambda s: (s == 'emg').mean()), g.size()
            fig, ax = plt.subplots()
            ax.errorbar(x, p, yerr=np.sqrt(p * (1 - p) / n), marker='o', capsize=3)
            ax.set_xscale('log'); ax.set_ylim(0, 1)
            ax.set_xlabel('Peak S/N (bin median)'); ax.set_ylabel('Fraction EMG')
            ax.set_title('EMG chosen vs S/N\nlow S/N cannot support the extra tail parameter')
            pdf.savefig(fig); plt.close(fig)

        # sensitivity of the EMG share to bic_margin
        db = fd['delta_bic'].dropna()
        margins = [(0, 'any'), (2, 'positive'), (6, 'strong'), (10, 'very strong')]
        plot_table(pdf, f"EMG share vs BIC margin (n={len(db)} peaks with both fits), current={self.bic_margin}",
                [[m, f"{(db > m).mean():.1%}", lab] for m, lab in margins],
                ['margin', 'EMG share', 'evidence (Kass & Raftery)'])

        # width vs retention, does the generator need RT-dependent widths?
        plot_hexbin(pdf, "Peak sigma vs retention (scan)", "Center scan", "log10 sigma (scans)",
                    d['center_scan'], np.log10(d['sigma']))
        emg = d[d['model'] == 'emg']
        if len(emg):
            plot_hexbin(pdf, "EMG tail (tau/sigma) vs S/N", "log10 S/N", "log10 tau/sigma",
                        np.log10(emg['sn']), np.log10(1 / (emg['gamma'] * emg['sigma'])))

    def _plot_fit_examples(self, pdf, n_per_page=4, seed=0):
        fd = self.fits_df
        ok = fd[fd['status'].isin(['ok', 'no_errorbars']) & np.isfinite(fd['redchi_norm'])]
        worker = NoiseModel._make_worker(self.bic_margin)
        for model in ('gauss', 'emg'):
            m = ok[ok['model'] == model]
            if len(m) < 3 * n_per_page:
                continue
            q10, q45, q55, q90 = m['redchi_norm'].quantile([0.1, 0.45, 0.55, 0.9])
            groups = {'best 10%': m[m['redchi_norm'] <= q10],
                    'median': m[m['redchi_norm'].between(q45, q55)],
                    'worst 10%': m[m['redchi_norm'] >= q90]}
            for label, sub in groups.items():
                sub = sub.sample(min(n_per_page, len(sub)), random_state=seed)
                fig, axes = plt.subplots(1, len(sub), figsize=(3.2 * len(sub), 3.2), squeeze=False)
                for ax, (_, p) in zip(axes[0], sub.iterrows()):
                    self._draw_fit(ax, worker, p)
                fig.suptitle(f"{model} fits, {label} by redchi_norm", fontsize=10)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    def _draw_fit(self, ax, worker, p):
        info = self.samples[p['sample']]
        s = load_sample(info['cache_dir'])
        row = {float(k): v for k, v in s['meta']['ion_rows'].items()}[float(p['ion'])]
        peak = next((pk for pk in s['peaks_by_ion'].get(p['ion'], []) if pk['center'] == p['center_scan']), None)
        if peak is None:
            ax.set_axis_off(); return
        worker.scan_interval = info['scan_interval']
        res = worker.gen_peak_model(peak, np.asarray(s['intensity'][row], dtype=float),
                                    bic_margin=self.bic_margin, return_curves=True)
        if not isinstance(res, tuple):              # 'short' / 'both_failed' return early
            ax.set_axis_off(); return
        out, x, y, g_fit, e_fit = res
        ax.plot(x, y, 'k.', ms=4, label='data')
        if g_fit is not None:
            ax.plot(x, g_fit, lw=1.2, label='gauss')
        if e_fit is not None:
            ax.plot(x, e_fit, lw=1.2, label='emg')
        ax.set_title(f"m/z {p['ion']:g} scan {p['center_scan']} S/N={p['sn']:.0f}\n"
                    f"chosen={out['model']} ΔBIC={out['delta_bic']:.1f} rχ²n={p['redchi_norm']:.1f}", fontsize=7)
        ax.legend(fontsize=6)

    # endregion

    # region QC Calculations

    @staticmethod
    def qc_labels(rd, tail_max=2.0, skew_max=0.5, acf_max=0.3, missed_max=0.10, suffix='_post'):
        """contamination checks on uncensored post-filter diagnostics by default"""
        tail, skw, acf = rd[f'tail_ratio{suffix}'], rd[f'skew{suffix}'], rd[f'acf1{suffix}']
        checks = {
            f'tail_ratio{suffix} > {tail_max}': tail > tail_max,
            f'skew{suffix} > {skew_max}': skw > skew_max,
            f'acf1{suffix} > {acf_max}': acf > acf_max,
            f'frac_missed > {missed_max:.0%}': rd['frac_missed'] > missed_max,
        }
        suspect = np.logical_or.reduce(list(checks.values()))
        no_data = rd['sigma'].isna() | rd[f'sigma{suffix}'].isna()    # unusable either before or after ROS

        # older row tables (before frac_censored existed) skip this check
        if 'frac_censored' in rd:
            too_censored = rd['frac_censored'] > 0.8
        else:
            too_censored = pd.Series(False, index=rd.index)

        labels = np.select([no_data, too_censored, suspect], ['no_data', 'too_censored', 'suspect'], default='ok')

        return labels, checks

    def _assign_qc(self, **limits):
        self.qc_limits = {**dict(tail_max=2.0, skew_max=0.5, acf_max=0.3, missed_max=0.10), **limits}
        self.row_df['qc'], self.qc_checks = self.qc_labels(self.row_df, **self.qc_limits)

    def _add_peak_noise_stats(self):
        """
        joins each peak to its row's noise sigma -> S/N and noise-normalized fit quality
        """

        noise = self.row_df[['sample', 'ion', 'sigma', 'qc']].rename(columns={'sigma': 'noise_sigma', 'qc': 'row_qc'})
        fd = self.fits_df.copy()
        fd['ion'] = fd['ion'].astype(float)
        fd = fd.merge(noise, on=['sample', 'ion'], how='left')
        fd['sn'] = fd['height'] / fd['noise_sigma']
        fd['redchi_norm'] = fd['redchi'] / fd['noise_sigma'] ** 2
        self.fits_df = fd

    @staticmethod
    def _flag_missed_peaks(residuals, valid, censored=None, run_mask = None, k=2.0, min_run=3, hard_k=6.0, grow=2):
        """
        Flags runs of positive residuals that look like unpicked peaks, sigma
        comes from lower half only which missed peaks won't inflate
        """
        # remove invalid values from censored
        est = valid if censored is None else valid & ~censored
        r = residuals[est]

        # return zeros if not enough points in residals
        if len(r) < 10:
            return np.zeros_like(valid)

        # get median, if no points are below median there is no noise and we return zeros
        med = np.median(r)
        lower = r[r<med]
        if len(lower) == 0:
            return np.zeros_like(valid)

        # get sigam of distribution and return zeros if sigma = 0
        sigma = np.median(med - lower) * 1.4826
        if sigma == 0:
            return np.zeros_like(valid)

        # flag consistent, high values as likely missed peaks
        search = valid if run_mask is None else run_mask
        high = search & (residuals - med > k * sigma)
        edges = np.flatnonzero(np.diff(np.r_[0, high.astype(np.int8), 0]))
        flag = np.zeros_like(valid)
        for s,e in zip(edges[::2], edges[1::2]):
            if e-s >= min_run or residuals[s:e].max() - med > hard_k * sigma:
                flag[max(0,s-grow):e + grow] = True
        return flag & valid

    @staticmethod
    def residual_diagnostics(r):
        nan_out = dict.fromkeys(('skew', 'tail_ratio', 'acf1', 'frac_gt3', 'sigma', 'level'), np.nan)
        r = np.asarray(r, dtype=float)
        if len(r) < 10:
            return nan_out
        med = np.median(r)
        sigma = 1.4826 * np.median(med - r[r <= med])     # lower-half scale
        if not sigma > 0:
            return{**nan_out, 'level':med}
        z = (r - med) / sigma
        n_up, n_down = int((z>3).sum()), int((z < -3).sum())
        return {
            'skew': skew(r),                                        # >~0.5 → positive contamination
            'tail_ratio': (n_up+1)/ (n_down+1),   # ~1 for symmetric noise
            'acf1': np.corrcoef(r[:-1], r[1:])[0, 1],               # high → slow structure/peaks left in
            'frac_gt3': (np.abs(z) > 3).mean(),                     # ~0.003 if normal, a few % is typical
            'sigma': sigma,
            'level': med,
        }

    @staticmethod
    def censored_normal_fit(y, limit, min_uncensored=30):
        """
        Tobit-style MLE of mean/sd for noise left-censored at a per-scan limit
        uncensored points use the normal density, censored ones the probability of falling below their limit
        also returns the upper-tail rate (points above mu + 3 sd over all points), which censoring
        can't bias since it only removes low values

        Returns
        -------
        dict cen_mu, cen_sigma, n_uncens, cen_upper_rate (NaN when too few uncensored points)
        """
        y, limit = np.asarray(y, float), np.asarray(limit, float)
        cen = y <= limit * (1 + 1e-6) + 1e-9
        yo, lc = y[~cen], limit[cen]
        out = {'cen_mu': np.nan, 'cen_sigma': np.nan, 'n_uncens': int(len(yo)), 'cen_upper_rate': np.nan}
        if len(yo) < min_uncensored:
            return out

        def nll(p):
            mu, log_s = p
            s = np.exp(log_s)
            return -(norm.logpdf(yo, mu, s).sum() + norm.logcdf(lc, mu, s).sum())

        mad = 1.4826 * np.median(np.abs(yo - np.median(yo)))
        x0 = [np.median(yo), np.log(max(mad, np.std(yo), 1e-6))]
        res = minimize(nll, x0, method='Nelder-Mead', options={'maxiter': 2000})
        if not res.success:
            return out

        mu, sigma = float(res.x[0]), float(np.exp(res.x[1]))
        out.update(cen_mu=mu, cen_sigma=sigma, cen_upper_rate=float((yo - mu > 3 * sigma).sum() / len(y)))
        return out


    # endregion

    # region Modeling
    def gen_bl_and_noise_models(self, raw_signal, noise_mask, at_start_idxs, at_vals, ion_idx,
                                seg_size=50, end_window=15, outlier_threshold=3.5, edge_trim=25,
                                bl_lam=1e10, flat_len=100):
        """
        Produces a noise and baseline model for a given row of an intensity matrix, noise first since the
        baseline is fit only through the noise points that survive it

        Params
        ------
        raw_signal                      raw signal array
        noise_mask                      bool mask 1=noise 0=signal for raw_signal
        at_start_idxs                   start idxs for abundance thresholds
        at_vals                         values array for each sgment for abundance thresholds
        seg_size                        size of segments for the local noise fits
        end_window                      how far to search for median when imputing baseline of signal sections
        outlier_threshold               threshold for median/mad based outlier-detection
        edge_trim                       scans at each end excluded from noise and baseline
        bl_lam                          whittaker stiffness, larger = smoother baseline
        flat_len                        slope penalty length in scans, smaller = flatter baseline, None = off

        Returns
        -------
        smoothed_baseline               whittaker baseline array for this row
        corrected_residuals             ROS corrected noise residuals for this row (signal - segment fit)
        n_outliers ... frac_censored    see gen_noise_model
        """
        noise = self.gen_noise_model(raw_signal, noise_mask, at_start_idxs, at_vals,
                                     seg_size=seg_size, end_window=end_window,
                                     outlier_threshold=outlier_threshold, edge_trim=edge_trim)

        smoothed_baseline = NoiseModel.gen_baseline_model(raw_signal, noise['valid'], lam=bl_lam, flat_len=flat_len)

        return (smoothed_baseline, noise['corrected_residuals'], noise['n_outliers'], noise['outliers'],
                noise['h'], noise['n_missed'], noise['missed'], noise['diag_before'], noise['diag_after'],
                noise['frac_censored'])

    def gen_noise_model(self, raw_signal, noise_mask, at_start_idxs, at_vals,
                        seg_size=50, end_window=15, outlier_threshold=3.5, edge_trim=25):
        """
        Noise residuals from local linear segment fits, then edge trim, outliers, censoring, missed peaks
        and ROS imputation of censored residuals

        Returns
        -------
        dict with valid (final noise mask, used for the baseline), corrected_residuals, n_outliers, outliers,
        h, n_missed, missed, diag_before, diag_after, frac_censored
        """
        # get the number of segments for this portion of signal
        n_segments = len(raw_signal) // seg_size
        if len(raw_signal) % seg_size != 0:
            n_segments += 1

        # remove peaks from signal and replace with linear imputed values so they don't skew segment fits
        signal = NoiseModel._impute_missing_signal(raw_signal, noise_mask, window=end_window)

        # process each segment for residuals, validity and outliers
        residuals = np.zeros_like(raw_signal, dtype=float)
        valid = np.zeros_like(raw_signal, dtype=bool)
        outliers = np.zeros_like(raw_signal, dtype=bool)
        for i in range(n_segments):

            # get start and end points
            start = seg_size * i
            end = start + seg_size

            # get segment info
            segment_noise_mask = noise_mask[start:end]
            segment = signal[start:end]

            # residuals against this segment's linear fit
            _, seg_res, seg_valid, seg_out_mask = NoiseModel._process_segment(segment, segment_noise_mask,
                                                                              threshold=outlier_threshold)
            # update stored values
            residuals[start:end] = seg_res
            valid[start:end] = seg_valid
            outliers[start:end] = seg_out_mask

        # trim edges
        if edge_trim:
            for m in (valid, outliers):
                m[:edge_trim] = False
                m[-edge_trim:] = False

        # count outliers only in noise points, not signal sections
        outliers &= noise_mask.astype(bool)
        n_outliers = int(outliers.sum())

        # detect row censoring
        censored = NoiseModel._detect_censored(raw_signal,
                                               abundance_thresholds=at_vals,
                                               start_idxs=at_start_idxs)

        # mask outlier scans
        uncens = valid & ~censored

        diag_before = NoiseModel.residual_diagnostics(residuals[uncens])

        missed = NoiseModel._flag_missed_peaks(residuals, valid, censored, run_mask=valid | outliers,
                                               **self.missed_kwargs)

        valid &= ~missed
        frac_censored = float(censored[valid].mean()) if valid.any() else np.nan
        n_missed = int(missed.sum())

        diag_after = NoiseModel.residual_diagnostics(residuals[valid & uncens])

        valid_censored = censored[valid]
        valid_residuals = residuals[valid]

        if censored.any() and len(valid_residuals) >= 3:
            df = pd.DataFrame({'residual': valid_residuals, 'censored': valid_censored})
            ros = ROS('residual', 'censored', df=df,
                      transform_in=lambda x: x, transform_out=lambda x: x,
                      min_uncensored=2, max_fraction_censored=0.8,
                      as_array=False)

            # wqio returns values sorted, put imputed values back on the censored scans by rank
            imputed = np.sort(ros.loc[ros['censored'].astype(bool), 'final'].to_numpy(dtype=float))
            corrected_residuals = valid_residuals.astype(float).copy()
            cen_pos = np.flatnonzero(valid_censored)
            cen_pos = cen_pos[np.argsort(valid_residuals[cen_pos], kind='stable')][:len(imputed)]
            corrected_residuals[cen_pos] = imputed
        else:
            corrected_residuals = valid_residuals

        # save outler info and kde bandwith info
        h = NoiseModel._silverman_rot(corrected_residuals) if len(corrected_residuals) >= 2 else np.nan

        return dict(valid=valid, corrected_residuals=corrected_residuals, n_outliers=n_outliers,
                    outliers=outliers, h=h, n_missed=n_missed, missed=missed,
                    diag_before=diag_before, diag_after=diag_after, frac_censored=frac_censored)

    @staticmethod
    def gen_baseline_model(raw_signal, valid, lam=1e10, flat_len=100):
        """
        smooth baseline through the final clean noise points (not peaks, outliers, missed peaks or trimmed
        edges), whittaker with an optional slope penalty for a flatter baseline
        """
        return NoiseModel._whittaker_baseline(raw_signal, valid, lam=lam, flat_len=flat_len)

    @staticmethod
    def _whittaker_baseline(y, use, lam=1e7, flat_len=100):
        """
        weighted whittaker smoother (Eilers 2003): only points in use (noise, not outliers) pull on the
        baseline, gaps are bridged by the smoothest curve, lam sets stiffness
        """
        n = len(y)
        if use.sum() < 3:
            return np.full(n, np.median(y[use]) if use.any() else np.median(y))
        w = use.astype(float)
        D2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
        A = sparse.diags(w) + lam * (D2.T @ D2)
        if flat_len:
            D1 = sparse.diags([-1.0, 1.0], [0, 1], shape=(n - 1, n))
            A = A + (lam / flat_len**2) * (D1.T @ D1)
        bl = spsolve(A.tocsc(), w * y)

        # no data beyond the first/last used point: hold flat rather than extend the end slope
        idx = np.flatnonzero(use)
        bl[:idx[0]] = bl[idx[0]]
        bl[idx[-1] + 1:] = bl[idx[-1]]
        return bl

    @staticmethod
    def _sg_baseline(y, window=201, order=2):
        """
        single savgol pass over the whole row, window forced odd and no longer than the row
        """
        n = len(y)
        window = min(window | 1, n if n % 2 else n - 1)     # | 1 makes it odd
        if window <= order:
            return y.copy()
        return savgol_filter(y, window_length=window, polyorder=order, mode='interp')

    def _estimate_emg_guess(self, peak, front_thresh=1.1):
        """
        Estimates satrting paramters for exponentially modified gaussian fit of a peak
        USP Tailing factor, 1 = symmetric, > 1 = tailing, < 1 = fronting
        """

        # peak params
        height = peak['height']
        fwhh = peak['fwhh']
        tailing_factor = peak['tailing_factor']

        # amplitude guess
        A_guess = height

        # center guess
        center_guess = peak['center'] - peak['left_bound']
        
        # counters and base gamma_guess value
        sg_fail = False
        tg_fail = False

        # gaussian estiamte of sigma
        scan_interval = self.scan_interval
        fwhh_to_sigma = 2 * np.sqrt(2*np.log(2))
        sigma_guess = (fwhh / scan_interval) / fwhh_to_sigma
        if sigma_guess == 0:
            sg_fail = True

        # estimate gamma from tailing factor 
        ratio = 2 * tailing_factor - 1
        if ratio <= 0:
            tg_fail = True
            return A_guess, center_guess, sigma_guess, np.nan, tg_fail, sg_fail, False

        fronting = ratio < 1 / front_thresh
        excess_tailing = (1 / ratio - 1) if fronting else (ratio - 1)

        if excess_tailing <= 0 or sigma_guess == 0:
            tg_fail = excess_tailing == 0
            gamma_guess = np.nan
        else:
            gamma_guess = 1 / (sigma_guess * excess_tailing)

        # adjust center guess if fronting
        if fronting:
            n = peak['right_bound'] - peak['left_bound']
            center_guess = (n - 1) - center_guess

        return A_guess, center_guess, sigma_guess, gamma_guess, tg_fail, sg_fail, fronting

    def _safe_fit(self, model, y, params, x):
        """
        Fits a model returns (result, None) on sccess or (None, reason) on a failure
        """

        try:
            result = model.fit(y, params, x=x)
        except (ValueError, RuntimeError, FloatingPointError, TypeError) as e:
            return None, f"{type(e).__name__}: {str(e)[:80]}"
        if not result.success:
            return None, f"no_converge {str(result.message)[:60]}"
        return result, None

    def _sigma_level_fit(self):
        """log10 sigma = a + b log10 level on ok rows, generator uses it to scale noise to any baseline"""
        rd = self.row_df
        ok = (rd['qc'] == 'ok') & (rd['sigma'] > 0) & (rd['bl_level'] > 0)
        if ok.sum() < 3:
            return None
        b, a = np.polyfit(np.log10(rd.loc[ok, 'bl_level']), np.log10(rd.loc[ok, 'sigma']), 1)
        return {'a': float(a), 'b': float(b), 'n_rows': int(ok.sum())}

    @staticmethod
    def emg_stable(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=1.0):
        """
        Fits an EMG like lmfit expgaussain but prevents exp overflow to improve performance
        """
        z = (center + gamma * sigma**2 - x) / (np.sqrt(2.0) * sigma)
        out = np.empty_like(z, dtype=float)
        pos = z >= 0

        out[pos] = np.exp(-(x[pos] - center)**2 / (2 * sigma**2)) * erfcx(z[pos])

        arg1 = gamma * (center - x[~pos] + gamma * sigma**2 / 2)
        out[~pos] = np.exp(arg1) * erfc(z[~pos])
        
        return amplitude * (gamma / 2) * out

    def gen_peak_model(self, peak: dict, raw_signal: np.ndarray, bic_margin=2.0,
                       return_curves=False):
        """
        Models a given peak as an exponentially modified gaussian or gaussian depending on which
        is a more accurate fit
        """

        # format outupt
        out = {
            # peak context
            'height': peak['height'],
            'ion': peak['ion'],
            'center_scan': peak['center'],
            'rt': peak['rt'],
            'n_points': int(peak['right_bound'] - peak['left_bound']) + 1,
            # fit results
            'model': None,
            'amplitude': np.nan,
            'sigma': np.nan,
            'gamma': np.nan,
            'redchi': np.nan,
            'delta_bic': np.nan,
            'fronting': False,
            # diagnostics
            'guess_type': 1,
            'tg_fail': False,
            'sg_fail': False,
            'status': None
        }

        # get peak segment
        left = int(peak['left_bound'])
        right = int(peak['right_bound'])
        signal = raw_signal[left:right+1]
        if len(signal) < 5:
            out['status'] = 'short'
            return out
        x = np.arange(len(signal), dtype=float)
        y = signal - peak['baseline']
        n = len(y)

        # metric guess usability checks
        fwhh_usable = np.isfinite(peak.get('fwhh', np.nan)) and peak['fwhh'] > 1e-9
        tf_usable = np.isfinite(peak.get('tailing_factor', np.nan))

        # make initial parameter guesses
        if fwhh_usable and tf_usable:
            A_guess, center_guess, sigma_guess, gamma_guess, tg_fail, sg_fail, fronting = self._estimate_emg_guess(peak)
            guess_type = 0
        else:
            center_guess = float(np.argmax(y))
            sigma_guess = n/6
            gamma_guess = 1.0
            A_guess = y.max()
            guess_type = 1
            tg_fail = False
            sg_fail = False
            fronting = False
        out.update(guess_type=guess_type, tg_fail=tg_fail, sg_fail=sg_fail, fronting=fronting)

        # clip guesses within bounds
        gamma_guess = np.clip(gamma_guess, 0.05, 8) if np.isfinite(gamma_guess) else 8
        sigma_guess = np.clip(sigma_guess, 0.4, 0.45*n)
        center_guess = np.clip(center_guess, 1, n-2)

        # convert a_guess from height to area
        A_guess = A_guess if np.isfinite(A_guess) else y.max()
        A_guess = max(A_guess, 1) * sigma_guess * np.sqrt(2*np.pi)

        # gaussian fit
        g_center = (n-1) - center_guess if fronting else center_guess
        g_params = self.gauss_model.make_params(amplitude=A_guess, center=g_center, sigma=sigma_guess)
        g_params['center'].set(min=0, max=n-1)
        g_params['sigma'].set(min=0.3, max=n/2)
        g_params['amplitude'].set(min=0)
        g_res, g_err = self._safe_fit(self.gauss_model, y, g_params, x)

        # EMG fit
        e_params = self.emg_model.make_params(amplitude=A_guess, center=center_guess, sigma=sigma_guess,
                                              gamma=gamma_guess)
        e_params['center'].set(min=0, max=n-1)
        e_params['sigma'].set(min=0.3, max=n/2)
        e_params['gamma'].set(min=0.01, max=10)
        e_params['amplitude'].set(min=0)
        y_fit = y[::-1] if fronting else y
        e_res, e_err = self._safe_fit(self.emg_model, y_fit, e_params, x)

        # tally features by model
        if g_err is not None:
            self.exec_counter[f"gauss | {g_err}"] += 1
        if e_err is not None:
            self.exec_counter[f"emg | {e_err}"] += 1

        # choose best model
        if g_res is None and e_res is None:
            out['status'] = 'both_failed'
            return out
        if g_res is not None and e_res is not None:
            out['delta_bic'] = g_res.bic - e_res.bic        # > 0 means EMG fits better
        if e_res is not None and (g_res is None or out['delta_bic'] > bic_margin):
            chosen = e_res
            out['model'] = 'emg'
            out['gamma'] = e_res.params['gamma'].value
        else:
            chosen = g_res
            out['model'] = 'gauss'
            out['fronting'] = False

        out['amplitude'] = chosen.params['amplitude'].value
        out['sigma'] = chosen.params['sigma'].value
        out['redchi'] = chosen.redchi
        out['status'] = 'ok' if chosen.errorbars else 'no_errorbars'

        # return curves for visualization
        if return_curves:
            g_fit = g_res.best_fit if g_res is not None else None
            e_fit = None
            if e_res is not None:
                e_fit = e_res.best_fit[::-1] if fronting else e_res.best_fit
            return out, x, y, g_fit, e_fit

        # return fit info
        return out
        
    @ staticmethod
    def _detect_censored(raw_signal, abundance_thresholds, start_idxs):
        """
        Looks through a raw signal for values that = abundance threshold for that segment and marks
        them as censored (bool True or 1) for ROC imputation
        
        Params
        ------
        raw_signal                      raw signal array
        abundance_thresholds            array of 10 possible abundance thresholds
        start_idxs                      start idxs of each 10 segments corresponding to the abundance thresholds

        Returns
        -------
        censored                        bool array len(raw_signal) where True = censored False = uncensored
        """
        seg_idx = np.searchsorted(start_idxs, np.arange(len(raw_signal)), side='right') - 1
        return np.isclose(raw_signal, np.asarray(abundance_thresholds)[seg_idx])

    @staticmethod
    def _impute_missing_signal(raw_signal:np.ndarray, noise_mask:np.ndarray, window: int=5):
        """
        Finds the medain in window points on eithe rside of a signal section of raw_signal and 
        fits a linear fit betweenthese two points, replacing the signal with this imputed value

        Params
        ------
        raw_signal                      raw signal array to process
        noise_mask                      0=signal 1=noise mask of raw_signal
        window                          how far to look from noise section when finding median

        Returns
        -------
        clean_signal                    signal with actual peaks removed and replaced with imputed baselines
        """

        clean_signal = np.copy(raw_signal)
        n = len(raw_signal)
        is_signal = ~noise_mask

        # find start/end ponits of signal sections
        diff = np.diff(is_signal.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if is_signal[0]:
            starts = np.concatenate([[0],starts])
        if is_signal[-1]:
            ends = np.concatenate([ends, [n]])

        # replace signal sections with lienarly imputed value
        for start,end in zip(starts,ends):
            left_slice = slice(max(0, start-window), start)
            right_slice = slice(end, min(n, end+window))

            left_vals = raw_signal[left_slice][noise_mask[left_slice]]
            right_vals = raw_signal[right_slice][noise_mask[right_slice]]

            if len(left_vals) == 0 and len(right_vals) == 0:
                continue
            elif len(left_vals) == 0:
                clean_signal[start:end] = np.median(right_vals)
            elif len(right_vals) == 0:
                clean_signal[start:end] = np.median(left_vals)
            else:
                left_anchor = np.median(left_vals)
                right_anchor = np.median(right_vals)
                clean_signal[start:end] = np.linspace(left_anchor, right_anchor, end-start)

        return clean_signal

    @staticmethod
    def _process_segment(segment_signal, segment_noise_mask, threshold=3.5):
        """
        takes a segment of signal and generates a local linear fit to it and calculates differences between
        true signal and the fit (residual)
        
        Params
        ------
        segment_signal                  section of signal for this segment

        Returns
        -------
        baseline                        array linear fit for this segment
        residuals                       array of residual difference between signal and fit
        valid_mask                      bool mask of poitns which are noise and not outliers for filtering 
                                        (true means include in noise model)
        n_outliers                      number of outliers detected in the fit
        """

        baseline, outlier_mask = NoiseModel._local_linear_fit(
            segment_signal, outlier_detection=True, threshold=threshold
            )
        residuals = segment_signal - baseline
        valid_mask = segment_noise_mask & ~outlier_mask

        return baseline, residuals, valid_mask, outlier_mask

    @ staticmethod
    def _smooth_baseline(baseline, start_idxs, filter_1=4, filter_2=5):
        """
        Smoothes a given baseline fist using a filter_1-scan local linear fit around each start_idx and then a global 
        filter_2-scan savgol filter and returns the smoothted baseline array

        Params
        ------
        baseline                        array of bseline values
        start_idxs                      start idex of segements in baseline
        filter_1/filter_2               size of filter for smoothing pass 1 and 2
                                        filter_1 should be even filter_2 should be odd

        Returns
        -------
        smoothed_baseline               smoothetd baseline array
        """
        # copy of basline for smoothing
        smoothed_baseline = np.copy(baseline)

        # make sure filter_1 is even
        if filter_1 % 2 != 0:
            filter_1 += 1

        # make sure filter_2 is odd
        if filter_2 % 2 == 0:
            filter_2 += 1

        for start_idx in start_idxs:

            # flags for starting and ending segments
            start = False
            end = False

            # get window on each side for the corner-smoothing
            window = filter_1 // 2
            segment = baseline[max(0,start_idx-window):start_idx+window]

            # pad segment if it is at the start or end
            if start_idx - window < 0:
                start_diff = abs(start_idx - window)
                pad = np.full(start_diff, baseline[0])
                segment = np.concatenate([pad, segment])
                start = True
            elif start_idx + window > len(baseline):
                end_diff = abs(start_idx + window - len(baseline))
                pad = np.full(end_diff, baseline[-1])
                segment = np.concatenate([segment,pad])
                end = True

            # generate local linear fit
            fit, _ = NoiseModel._local_linear_fit(segment, outlier_detection=False)

            # replace baseline fit with the adjusted fit
            if start:
                smoothed_baseline[:start_idx+window] = fit[start_diff:]
            elif end:
                smoothed_baseline[start_idx-window:start_idx+window-end_diff] = fit[:filter_1-end_diff]
            else:
                smoothed_baseline[start_idx-window:start_idx+window] = fit

        # final filter_2 savgol order=1 filter pass over entire baseline 
        smoothed_baseline = savgol_filter(smoothed_baseline, window_length=filter_2, polyorder=1)
        return smoothed_baseline

    @staticmethod
    def _local_linear_fit(y, outlier_detection=True, threshold=3.5):
        """
        Generates a local linear fit of a given y array using closed form least-squares linear regression on
        evenly spaced points, also rejects outlier points to cover in case of a missed parital peak at the start
        or end of the run
        """

        n = len(y)
        x = np.arange(n)

        if outlier_detection:
            outlier_mask = NoiseModel._detect_outliers(y, threshold=threshold)
            fit_y = y[~outlier_mask]
            fit_x = x[~outlier_mask]
            if len(fit_y) < 2:
                fit_x, fit_y = x,y
                outlier_mask = np.zeros(n, dtype=bool)
        else:
            fit_x, fit_y = x,y
            outlier_mask = np.zeros(n, dtype=bool)

        x_mean = fit_x.mean()
        y_mean = fit_y.mean()

        slope = np.sum((fit_x - x_mean) * (fit_y-y_mean)) / np.sum((fit_x-x_mean)**2)
        intercept = y_mean - slope * x_mean

        return slope * x + intercept, outlier_mask

    @staticmethod
    def _detect_outliers(y, threshold=3.5):
        """
        uses modz score to detect outliers in an array, returns a mask of outlier location (outlier = True)
        to exclude these points from linear fits etc...
        """
        median = np.nanmedian(y)
        mad = np.median(np.abs(y-median))
        if mad == 0:
            return np.zeros(len(y), dtype=bool)
        mod_z = 0.6745 * (y-median) / mad
        return np.abs(mod_z) > threshold

    @staticmethod
    def _silverman_rot(array, type:str = "MAD"):
        """
        Uses silverman's rule of thumb to estimate sigma for exponentially modified valley
        method
        """

        n = len(array)
        std = np.nanstd(array)

        if type == 'IQR':
            q75, q25 = np.nanpercentile(array,[75,25])
            iqr = (q75 - q25) / 1.349
            return 0.9 * min(std, iqr) * n**(-1/5)

        else:
            mad = np.nanmedian(np.abs(array - np.nanmedian(array))) * 1.4826
            return 0.9  * min(std, mad) * n**(-1/5)

    @classmethod
    def _make_worker(cls, bic_margin, missed_kwargs=None, bl_noise_kwargs=None, fit_peaks=True):
        w = cls.__new__(cls)
        w.scan_interval = None
        w.bic_margin = bic_margin
        w.missed_kwargs = missed_kwargs or dict(cls.MISSED_PEAK_KWARGS)
        w.bl_noise_kwargs = bl_noise_kwargs or dict(cls.BL_NOISE_KWARGS)
        w.fit_peaks = fit_peaks
        w.emg_model = Model(cls.emg_stable)
        w.gauss_model = GaussianModel()
        w.exec_counter = Counter()
        return w

    def process_ion(self, task):
        """
        fits baseline, noise, and peak models for one ion row, returns results for merging
        """
        self.scan_interval = task['scan_interval']
        self.exec_counter = Counter()
        bl, residuals, n_outliers, outliers, h, n_missed, missed, diag_before, diag_after, frac_censored = self.gen_bl_and_noise_models(
            task['raw_signal'], task['noise_mask'], task['at_start_idxs'], task['at_vals'],
            task['row'], **self.bl_noise_kwargs)
        fits = ([self.gen_peak_model(p, task['raw_signal'], bic_margin=self.bic_margin) for p in task['peaks']]
            if self.fit_peaks else [])

        for f in fits:
            f['sample'] = task['sample']
        return {
            'sample': task['sample'], 'ion': task['ion'], 'row': task['row'],
            'baseline': bl, 'residuals': np.asarray(residuals, dtype=float),
            'n_outliers': n_outliers, 'outliers': outliers, 'kde_h': h,
            'fits': fits, 'errors': self.exec_counter,
            'bl_count': int(task['noise_mask'].sum()),
            'signal_count': int((~task['noise_mask']).sum()),
            'n_peaks': len(task['peaks']),
            'at_vals': np.asarray(task['at_vals'], dtype=float),
            'n_missed': n_missed, 'missed': missed, 'bl_level': float(np.median(bl)), 'frac_censored': frac_censored,
            **NoiseModel.residual_diagnostics(np.asarray(residuals, dtype=float)),  # final, post ROS diagnostics on fully clean data
            **{f'{k}_pre':v for k,v in diag_before.items()},                        # diagnostics on valid + uncensored residuals
            **{f'{k}_post':v for k,v in diag_after.items()}                         # diagnostics after chekcing if there are probable missed peaks
        }

    @staticmethod
    def make_peak(x, height, center, model, sigma, gamma=np.nan, fronting=False):
        """peak with a given apex height, placed so its EMG/gauss 'center' parameter sits at `center`"""
        if model == 'gauss':
            return height * np.exp(-(x - center)**2 / (2 * sigma**2))

        # EMG: unit-area curve on a fine grid -> its maximum -> rescale to the requested height
        grid = np.linspace(-6 * sigma, 6 * sigma + 10 / gamma, 4001)
        unit_max = NoiseModel.emg_stable(grid, 1.0, 0.0, sigma, gamma).max()
        xx = (center - x) if fronting else (x - center)            # mirror for fronting peaks
        return (height / unit_max) * NoiseModel.emg_stable(xx, 1.0, 0.0, sigma, gamma)

    # endregion

    def save_data(self):
        """
        saves models to <app_dir>/noise_models/<model_name>/
                        peak_fits.csv, row_stats.csv, model_meta.json, <sample>/bl_noise.npz
        """
        out_dir = get_noise_model_dir() / self.model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # combined tables
        self.fits_df.to_csv(out_dir / 'peak_fits.csv', index=False)
        self.row_df.to_csv(out_dir / 'row_stats.csv', index=False)

        # per-sample baselines and residuals (scan counts can differ between samples)
        qc = self.row_df.set_index(['sample', 'ion'])['qc']
        for name in self.samples:
            keys = [k for k in self.bl_models if k[0] == name]
            ions = np.array([k[1] for k in keys])
            residuals = [self.noise_models[k] for k in keys]
            offsets = np.concatenate([[0], np.cumsum([len(r) for r in residuals])]).astype(np.int64)
            at_start_idxs = np.load(Path(self.samples[name]['cache_dir']) / 'at_start_idxs.npy')

            sample_dir = out_dir / name
            sample_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                sample_dir / 'bl_noise.npz',
                ions=ions,
                baselines=np.vstack([self.bl_models[k] for k in keys]),
                at_values=np.vstack([self.at_vals[k] for k in keys]),
                at_start_idxs=at_start_idxs,
                residuals_flat=np.concatenate(residuals) if residuals else np.array([]),
                residual_offsets=offsets,
                qc=np.array([qc[k] for k in keys]),
            )

        meta = {
            'model_name': self.model_name,
            'bic_margin': self.bic_margin,
            'missed_peak_kwargs': self.missed_kwargs,
            'bl_noise_kwargs': self.bl_noise_kwargs,
            'qc_limits': self.qc_limits,
            'samples': self.samples,                      # name -> scan_interval, n_scans
            'n_rows': int(len(self.row_df)),
            'n_peaks': int(len(self.fits_df)),
            'git_commit': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                         capture_output=True, text=True).stdout.strip(),
            'created': time.strftime('%Y-%m-%d %H:%M'),
            'cache_dirs': [s['cache_dir'] for s in self.samples.values()],   # name includes the config hash
            'sigma_vs_level': self._sigma_level_fit(),                        # noise scaling for the generator
        }

        with open(out_dir / 'model_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved noise model '{self.model_name}' ({len(self.samples)} samples) to {out_dir}")

# region parallel workers

_WORKER = None

def _init_worker(bic_margin, missed_kwargs, bl_noise_kwargs, fit_peaks):
    """runs once per worker process"""
    global _WORKER
    _WORKER = NoiseModel._make_worker(bic_margin, missed_kwargs, bl_noise_kwargs, fit_peaks)

def _run_ion(task):
    """loads one row from the sample cache and fits it"""
    s = load_sample(task['cache_dir'])
    row = task['row']
    full_task = {
        **task,
        'scan_interval': s['meta']['scan_interval'],
        'raw_signal': np.array(s['intensity'][row]),
        'noise_mask': np.array(s['noise_mask'][row]),
        'at_start_idxs': s['at_start_idxs'],
        'at_vals': np.array(s['at_values'][row]),
        'peaks': s['peaks_by_ion'].get(task['ion'], []),
    }
    return _WORKER.process_ion(full_task)

# endregion