from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from entities.wave import ExtremeRegime, MeanRegime, WaveRose
from interfaces.repositories import FigurePresenter

_STYLE = {
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
}

_SECTOR_LABELS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']


class MatplotlibPresenter(FigurePresenter):

    def wave_rose(self, rose: WaveRose, title: str, path: Path) -> Path:
        return self._plot_rose(
            rose, title, path,
            max_pct=40, step=5,
            cmap=cm.Spectral_r,
            legend_title='Altura de Ola[m]',
        )

    def extreme_wave_rose(self, rose: WaveRose, title: str, path: Path) -> Path:
        return self._plot_rose(
            rose, title, path,
            max_pct=50, step=10,
            cmap=cm.twilight,
            legend_title='Altura de Ola\nTemporal[m]',
        )

    def mean_regime(self, regime: MeanRegime, hs_data: np.ndarray, path: Path) -> Path:
        with plt.rc_context(_STYLE):
            hs = hs_data[hs_data > 0]

            # CDF empírica por histograma
            bins = np.arange(hs.min(), hs.max(), 0.02)
            N = np.histogram(hs, bins=bins)[0]
            centers = (bins[:-1] + bins[1:]) / 2
            bin_width = float(centers[1] - centers[0])
            area = float(np.sum(bin_width * N))
            n = N / area
            P1 = np.cumsum(bin_width * n)
            P11 = P1[:-1]
            y1 = centers[:-1]
            x1 = scipy_stats.norm.ppf(P11, loc=0, scale=1)

            PP = np.array([0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999, 0.9999])
            yticks = np.round(np.arange(0.5, np.max(y1) + 0.5, 0.5), 2)

            grid_kw = {'left': 0.12, 'bottom': 0.08, 'right': 0.98, 'top': 0.98}
            fig, ax = plt.subplots(figsize=(8, 8), gridspec_kw=grid_kw)

            ax.scatter(x1, np.log10(y1), s=20, c='red', alpha=0.5, zorder=3)

            # ajuste lognormal: log10(Hs) = (mu + sigma*z) / ln(10)
            z = np.linspace(x1.min(), x1.max(), 300)
            ax.plot(z, (regime.mu + regime.sigma * z) / np.log(10),
                    'b-', linewidth=1.5, label='Población 1')
            ax.plot(z, (regime.mu_2 + regime.sigma_2 * z) / np.log(10),
                    'g-', linewidth=1.5, label='Población 2')

            ax.grid(which='both', linestyle=':', linewidth=1)
            ax.set_xlim(-2, scipy_stats.norm.ppf(0.999999, loc=0, scale=1))
            ax.set_ylim(np.log10(0.5), np.log10(np.max(y1)) + 0.05 * np.log10(np.max(y1)))
            ax.set_xticks(scipy_stats.norm.ppf(PP, loc=0, scale=1))
            ax.set_xticklabels([f'{p * 100}%' for p in PP])
            ax.set_yticks(np.log10(yticks))
            ax.set_yticklabels(yticks)
            ax.set_xlabel('Probabilidad de no excedencia', fontweight='bold')
            ax.set_ylabel('Hs(m)', fontweight='bold')
            ax.legend()

            fig.savefig(path, dpi=300)
            plt.close(fig)

        return path

    def extreme_regime(
        self,
        regime_bm: ExtremeRegime,
        regime_pot: ExtremeRegime,
        path: Path,
    ) -> Path:
        with plt.rc_context(_STYLE):
            tr_bm  = [s[0] for s in regime_bm.estadisticos]
            hs_bm  = np.array([s[1] for s in regime_bm.estadisticos])
            lo_bm  = np.array([s[2] for s in regime_bm.estadisticos])
            hi_bm  = np.array([s[3] for s in regime_bm.estadisticos])

            tr_pot = [s[0] for s in regime_pot.estadisticos]
            hs_pot = np.array([s[1] for s in regime_pot.estadisticos])
            lo_pot = np.array([s[2] for s in regime_pot.estadisticos])
            hi_pot = np.array([s[3] for s in regime_pot.estadisticos])

            fig, ax = plt.subplots(figsize=(15, 8))

            ax.errorbar(tr_bm, hs_bm,
                        yerr=[hs_bm - lo_bm, hi_bm - hs_bm],
                        fmt='-ob', label='BM (GEV)')
            ax.fill_between(tr_bm, lo_bm, hi_bm, color='b', alpha=0.2)

            pot_label = f'POT (umbral = {regime_pot.pot:.2f} m)' if regime_pot.pot else 'POT'
            ax.errorbar(tr_pot, hs_pot,
                        yerr=[hs_pot - lo_pot, hi_pot - hs_pot],
                        fmt='-or', label=pot_label)
            ax.fill_between(tr_pot, lo_pot, hi_pot, color='r', alpha=0.2)

            ax.set_xscale('log')
            ax.grid(True, which='both', linestyle='--', linewidth=1)
            ax.set_xlabel('Periodo de retorno (años)', fontweight='bold')
            ax.set_ylabel('Hs(m)', fontweight='bold')
            ax.set_title('Hs: BM (GEV) vs POT — IC 95%')
            ax.legend()

            fig.savefig(path, dpi=300)
            plt.close(fig)

        return path

    @staticmethod
    def _plot_rose(
        rose: WaveRose,
        title: str,
        path: Path,
        max_pct: float,
        step: float,
        cmap,
        legend_title: str,
    ) -> Path:
        n_sectors = len(rose.sectors)
        n_bins = len(rose.hs_bins) - 1
        angles = np.linspace(0, 2 * np.pi, n_sectors, endpoint=False)
        width = 2 * np.pi / n_sectors
        colors = [cmap(i / n_bins) for i in range(n_bins)]

        with plt.rc_context(_STYLE):
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, polar=True)
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)

            bottoms = np.zeros(n_sectors)
            handles = []
            for b in range(n_bins):
                heights = np.array([rose.distribution[s][b] for s in range(n_sectors)])
                ax.bar(angles, heights, width=width * 0.9, bottom=bottoms,
                       color=colors[b], edgecolor='black', linewidth=0.5, zorder=3)
                lo = rose.hs_bins[b]
                hi = rose.hs_bins[b + 1]
                label = f'{lo}-{hi}' if hi < 90 else f'>{lo}'
                handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[b], label=label))
                bottoms += heights

            ax.set_xticks(angles)
            ax.set_xticklabels(_SECTOR_LABELS, fontsize=6)
            ax.set_ylim(0, max_pct)
            ax.set_yticks(np.arange(0, max_pct + step, step))
            ax.grid(linestyle='dashed', zorder=0)
            ax.set_title(title, pad=12)
            ax.legend(handles=handles, title=legend_title,
                      loc=(0.80, -0.05), fontsize=8, title_fontsize=10)
            ax.tick_params(axis='x', labelsize=6)

            fig.savefig(path, dpi=150)
            plt.close(fig)

        return path
