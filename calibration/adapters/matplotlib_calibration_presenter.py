from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import CubicSpline

from entities.calibration import CalibrationData, CalibrationResult
from interfaces.calibration import CalibrationPresenter


class MatplotlibCalibrationPresenter(CalibrationPresenter):
    """Figuras diagnósticas de calibración con matplotlib."""

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def cdf(
        self,
        data: CalibrationData,
        model: CalibrationResult,
        hs_calibrated: np.ndarray,
        path: Path,
    ) -> Path:
        n = len(data.hs_instrumental)
        probs = np.arange(1, n + 1) / (n + 1)
        x = self._gumbel(probs)

        hs_c_lo, hs_c_hi = self._ci_series(data, model)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.fill_between(
            x,
            np.sort(hs_c_lo),
            np.sort(hs_c_hi),
            color="salmon", alpha=0.5, label="CI$^{95\\%}$",
        )
        ax.plot(x, np.sort(data.hs_instrumental), color="blue",  lw=1.5, label="Instrumental")
        ax.plot(x, np.sort(data.hs_reanalysis),   color="green", lw=1.5, label="Reanalysis")
        ax.plot(x, np.sort(hs_calibrated),         color="red",   lw=1.5, label="Calibrated")

        tick_p  = [0.05, 0.50, 0.95, 0.995, 0.9995]
        tick_lbl = ["5%", "50%", "95%", "99.5%", "99.95%"]
        ax.set_xticks(self._gumbel(np.array(tick_p)))
        ax.set_xticklabels(tick_lbl, fontsize=11)
        ax.set_xlabel("Probability", fontsize=12)
        ax.set_ylabel("Hs [m]", fontsize=12)
        ax.set_title("Cumulative distribution function", fontsize=13)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle=":", color="gray", alpha=0.7)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def parameters_polar(self, model: CalibrationResult, path: Path) -> Path:
        theta_fine = np.linspace(0, 2 * np.pi, 361)
        deg_fine   = np.degrees(theta_fine)

        alpha_fine = model.alpha_spline(deg_fine % 360)
        beta_fine  = model.beta_spline(deg_fine % 360)

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        if model.confidence_intervals:
            ci = model.confidence_intervals
            nodes_ext = np.append(model.direction_nodes, 360.0)

            a_lo = CubicSpline(nodes_ext, np.append(ci.alpha_lower, ci.alpha_lower[0]), bc_type="periodic")
            a_hi = CubicSpline(nodes_ext, np.append(ci.alpha_upper, ci.alpha_upper[0]), bc_type="periodic")
            b_lo = CubicSpline(nodes_ext, np.append(ci.beta_lower,  ci.beta_lower[0]),  bc_type="periodic")
            b_hi = CubicSpline(nodes_ext, np.append(ci.beta_upper,  ci.beta_upper[0]),  bc_type="periodic")

            ax.fill_between(theta_fine, a_lo(deg_fine % 360), a_hi(deg_fine % 360),
                            color="gray", alpha=0.3)
            ax.fill_between(theta_fine, b_lo(deg_fine % 360), b_hi(deg_fine % 360),
                            color="gray", alpha=0.3)

        # referencia en r=1
        ax.plot(theta_fine, np.ones_like(theta_fine), color="black", lw=1.2, ls="-")

        ax.plot(theta_fine, alpha_fine, color="blue", lw=2, label=r"a($\theta$)")
        ax.plot(theta_fine, beta_fine,  color="red",  lw=2, label=r"b($\theta$)")

        r_max = max(alpha_fine.max(), beta_fine.max()) * 1.1
        ax.set_ylim(0, r_max)
        r_ticks = np.arange(0.25, r_max, 0.25)
        ax.set_yticks(r_ticks)
        ax.set_yticklabels(
            [f"{v:.2f}".rstrip("0").rstrip(".") if v not in (0.5, 0.75, 1.0, 1.25, 1.5)
             else f"{v}" for v in r_ticks],
            fontsize=8,
        )
        deg_ticks = np.arange(0, 360, 30)
        ax.set_xticks(np.radians(deg_ticks))
        ax.set_xticklabels([f"{d}°" for d in deg_ticks], fontsize=9)
        ax.set_title(r"Hs$^C$ = a($\theta$) Hs$^R$ $^{b(\theta)}$", fontsize=13, pad=20)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.10),
                  ncol=2, fontsize=11, frameon=True)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def parameters_table(self, model: CalibrationResult, path: Path) -> Path:
        nodes = model.direction_nodes
        alpha = model.alpha_nodes
        beta  = model.beta_nodes

        # Incluir nodo 360° (igual que 0°) como última fila
        nodes_ext = np.append(nodes, 360.0)
        alpha_ext = np.append(alpha, alpha[0])
        beta_ext  = np.append(beta,  beta[0])

        ci = model.confidence_intervals
        if ci:
            a_lo = np.append(ci.alpha_lower, ci.alpha_lower[0])
            a_hi = np.append(ci.alpha_upper, ci.alpha_upper[0])
            b_lo = np.append(ci.beta_lower,  ci.beta_lower[0])
            b_hi = np.append(ci.beta_upper,  ci.beta_upper[0])
            a_half = (a_hi - a_lo) / 2
            b_half = (b_hi - b_lo) / 2
        else:
            a_half = np.zeros_like(alpha_ext)
            b_half = np.zeros_like(beta_ext)

        col_labels = [r"$\theta$", r"a $\pm$ CI$^{95\%}$", r"b $\pm$ CI$^{95\%}$"]
        rows = [
            [f"{d:.1f}°", f"{a:.2f} ± {da:.3f}", f"{b:.2f} ± {db:.3f}"]
            for d, a, da, b, db in zip(nodes_ext, alpha_ext, a_half, beta_ext, b_half)
        ]

        n_rows = len(rows)
        fig_h  = 0.38 * (n_rows + 1) + 0.5
        fig, ax = plt.subplots(figsize=(8, fig_h))
        ax.axis("off")

        tbl = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 1.5)

        # cabecera en negrita con fondo gris claro
        for j in range(3):
            tbl[(0, j)].set_facecolor("#e8e8e8")
            tbl[(0, j)].set_text_props(fontweight="bold")

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _gumbel(p: np.ndarray) -> np.ndarray:
        return -np.log(-np.log(np.clip(p, 1e-9, 1 - 1e-9)))

    @staticmethod
    def _ci_series(
        data: CalibrationData,
        model: CalibrationResult,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Banda IC del Hs calibrado propagando la incertidumbre de alpha y beta."""
        if model.confidence_intervals is None:
            hs_c = model.alpha_spline(data.directions) * np.power(
                np.maximum(data.hs_reanalysis, 1e-10),
                model.beta_spline(data.directions),
            )
            return hs_c, hs_c

        ci = model.confidence_intervals
        nodes_ext = np.append(model.direction_nodes, 360.0)

        a_lo_sp = CubicSpline(
            nodes_ext, np.append(ci.alpha_lower, ci.alpha_lower[0]), bc_type="periodic"
        )
        a_hi_sp = CubicSpline(
            nodes_ext, np.append(ci.alpha_upper, ci.alpha_upper[0]), bc_type="periodic"
        )
        b_lo_sp = CubicSpline(
            nodes_ext, np.append(ci.beta_lower, ci.beta_lower[0]), bc_type="periodic"
        )
        b_hi_sp = CubicSpline(
            nodes_ext, np.append(ci.beta_upper, ci.beta_upper[0]), bc_type="periodic"
        )

        theta = data.directions % 360
        hs_r  = np.maximum(data.hs_reanalysis, 1e-10)

        hs_lo = a_lo_sp(theta) * np.power(hs_r, b_lo_sp(theta))
        hs_hi = a_hi_sp(theta) * np.power(hs_r, b_hi_sp(theta))
        return hs_lo, hs_hi