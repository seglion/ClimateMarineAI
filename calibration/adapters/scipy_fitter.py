from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from scipy.stats import t as t_dist

from entities.calibration import (
    CalibrationData,
    CalibrationDiagnostics,
    CalibrationResult,
    ConfidenceIntervals,
)
from interfaces.repositories import CalibrationFitter


class ScipyCalibrationFitter(CalibrationFitter):
    """Implementación del ajuste direccional de Mínguez et al. (2011).

    Cuantiles en escala Gumbel (Sec. 2a-2b) → L-BFGS-B (Sec. 2c) →
    Hessiano numérico para covarianza (Sec. 2d) → diagnósticos (Sec. 2f).
    """

    def __init__(self, n_quantiles: int = 20, confidence_level: float = 0.95):
        self._nq = n_quantiles
        self._conf = confidence_level

    # ── Interfaz pública ──────────────────────────────────────────────────────

    def fit(
        self,
        data: CalibrationData,
        n_nodes: int = 16,
        sector_width: float = 22.5,
    ) -> CalibrationResult:
        direction_nodes = np.linspace(0, 360, n_nodes, endpoint=False)

        dirs_flat, qi_flat, qr_flat = self._smooth_quantiles(
            data, sector_width
        )
        ndq = len(dirs_flat)

        alpha_opt, beta_opt, obj_val = self._optimize(
            dirs_flat, qi_flat, qr_flat, n_nodes, direction_nodes
        )
        alpha_spline, beta_spline = self._build_splines(
            direction_nodes, alpha_opt, beta_opt
        )

        residual_var = obj_val / max(ndq - n_nodes - 1, 1)
        params = np.concatenate([alpha_opt, beta_opt])

        cov = self._covariance(
            params, dirs_flat, qi_flat, qr_flat, residual_var, n_nodes
        )
        ci = self._confidence_intervals(
            alpha_opt, beta_opt, cov, ndq, n_nodes
        )

        hs_cal = self._apply(
            data.hs_reanalysis, data.directions, alpha_spline, beta_spline
        )
        diag = self._diagnostics(
            data.hs_instrumental, data.hs_reanalysis, hs_cal
        )

        return CalibrationResult(
            alpha_nodes=alpha_opt,
            beta_nodes=beta_opt,
            direction_nodes=direction_nodes,
            alpha_spline=alpha_spline,
            beta_spline=beta_spline,
            covariance_matrix=cov,
            residual_variance=residual_var,
            confidence_intervals=ci,
            diagnostics=diag,
        )

    # ── Pasos internos ────────────────────────────────────────────────────────

    def _quantile_probs(self, nd: int) -> np.ndarray:
        """Probabilidades equiespaciadas en escala Gumbel (Ecs. 11-14)."""
        q_lo = -np.log(-np.log(1.0 / nd))
        q_up = -np.log(-np.log(1.0 - 5.0 / nd))
        xq = q_lo + np.arange(self._nq) * (q_up - q_lo) / self._nq
        return np.exp(-np.exp(-xq))

    def _smooth_quantiles(
        self,
        data: CalibrationData,
        sector_width: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cuantiles empíricos suavizados por sector móvil (Sec. 2b).

        Devuelve tres arrays 1-D listos para la optimización:
        (direcciones, q_instrumental, q_reanalisis).
        """
        nd = len(data.hs_instrumental)
        q_probs = self._quantile_probs(nd)
        half = sector_width / 2.0
        min_pts = min(5 * self._nq, int(0.1 * nd))

        qi = np.full((360, self._nq), np.nan)
        qr = np.full((360, self._nq), np.nan)
        valid = np.zeros(360, dtype=bool)

        for i in range(360):
            diff = np.abs((data.directions - float(i) + 180) % 360 - 180)
            mask = diff <= half
            if mask.sum() < min_pts:
                continue
            valid[i] = True
            qi[i] = np.quantile(np.sort(data.hs_instrumental[mask]), q_probs)
            qr[i] = np.quantile(np.sort(data.hs_reanalysis[mask]), q_probs)

        # Rellenar sectores vacíos interpolando desde el válido más próximo
        if not np.all(valid):
            vi = np.where(valid)[0]
            if len(vi) >= 2:
                for idx in np.where(~valid)[0]:
                    diffs = (vi - idx) % 360
                    nearest = vi[np.argmin(np.where(diffs > 180, 360 - diffs, diffs))]
                    qi[idx] = qi[nearest]
                    qr[idx] = qr[nearest]
                    valid[idx] = True
            else:
                warnings.warn("Muy pocos sectores con datos para interpolar.")

        dirs_out, qi_out, qr_out = [], [], []
        for i in range(360):
            if not valid[i]:
                continue
            for j in range(self._nq):
                if not np.isnan(qi[i, j]) and qr[i, j] > 0:
                    dirs_out.append(float(i))
                    qi_out.append(qi[i, j])
                    qr_out.append(qr[i, j])

        return (
            np.array(dirs_out),
            np.array(qi_out),
            np.array(qr_out),
        )

    @staticmethod
    def _build_splines(
        direction_nodes: np.ndarray,
        alpha_nodes: np.ndarray,
        beta_nodes: np.ndarray,
    ) -> tuple[CubicSpline, CubicSpline]:
        """Splines cúbicos periódicos α(θ) y β(θ) (Ecs. 4-6)."""
        nodes_ext = np.append(direction_nodes, 360.0)
        alpha_ext = np.append(alpha_nodes, alpha_nodes[0])
        beta_ext = np.append(beta_nodes, beta_nodes[0])
        return (
            CubicSpline(nodes_ext, alpha_ext, bc_type="periodic"),
            CubicSpline(nodes_ext, beta_ext, bc_type="periodic"),
        )

    @staticmethod
    def _objective(
        params: np.ndarray,
        n_nodes: int,
        direction_nodes: np.ndarray,
        dirs: np.ndarray,
        qi: np.ndarray,
        qr: np.ndarray,
    ) -> float:
        """Suma de cuadrados Σ[Hs^I - α(θ)·(Hs^R)^β(θ)]² (Ec. 7)."""
        alpha_n = params[:n_nodes]
        beta_n = params[n_nodes:]
        if np.any(alpha_n <= 0):
            return 1e20
        a_sp, b_sp = ScipyCalibrationFitter._build_splines(
            direction_nodes, alpha_n, beta_n
        )
        hs_c = a_sp(dirs % 360) * np.power(np.maximum(qr, 1e-10), b_sp(dirs % 360))
        return float(np.sum((qi - hs_c) ** 2))

    def _optimize(
        self,
        dirs: np.ndarray,
        qi: np.ndarray,
        qr: np.ndarray,
        n_nodes: int,
        direction_nodes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        bounds = [(0.01, None)] * n_nodes + [(None, None)] * n_nodes
        res = minimize(
            self._objective,
            np.ones(2 * n_nodes),
            args=(n_nodes, direction_nodes, dirs, qi, qr),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-12},
        )
        if not res.success:
            warnings.warn(f"Optimización no convergió: {res.message}")
        return res.x[:n_nodes], res.x[n_nodes:], float(res.fun)

    def _covariance(
        self,
        params: np.ndarray,
        dirs: np.ndarray,
        qi: np.ndarray,
        qr: np.ndarray,
        residual_var: float,
        n_nodes: int,
    ) -> np.ndarray | None:
        """Hessiano numérico → S_β = (H_β/2σ²)⁻¹ (Ecs. 19-21)."""
        direction_nodes = np.linspace(0, 360, n_nodes, endpoint=False)
        n = len(params)
        eps = 1e-5
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                pp = params.copy(); pm = params.copy()
                mp = params.copy(); mm = params.copy()
                pp[i] += eps; pp[j] += eps
                pm[i] += eps; pm[j] -= eps
                mp[i] -= eps; mp[j] += eps
                mm[i] -= eps; mm[j] -= eps
                H[i, j] = (
                    self._objective(pp, n_nodes, direction_nodes, dirs, qi, qr)
                    - self._objective(pm, n_nodes, direction_nodes, dirs, qi, qr)
                    - self._objective(mp, n_nodes, direction_nodes, dirs, qi, qr)
                    + self._objective(mm, n_nodes, direction_nodes, dirs, qi, qr)
                ) / (4 * eps ** 2)
                H[j, i] = H[i, j]
        try:
            return np.linalg.inv(H / (2 * residual_var))
        except np.linalg.LinAlgError:
            warnings.warn("No se pudo invertir la matriz de Fisher.")
            return None

    def _confidence_intervals(
        self,
        alpha_opt: np.ndarray,
        beta_opt: np.ndarray,
        cov: np.ndarray | None,
        ndq: int,
        n_nodes: int,
    ) -> ConfidenceIntervals | None:
        """Intervalos de confianza para α y β (Ec. 22)."""
        if cov is None:
            return None
        dof = max(ndq - n_nodes - 1, 1)
        t = t_dist.ppf(1 - (1 - self._conf) / 2, dof)
        std = np.sqrt(np.diag(cov))
        return ConfidenceIntervals(
            alpha_lower=tuple(float(v) for v in alpha_opt - t * std[:n_nodes]),
            alpha_upper=tuple(float(v) for v in alpha_opt + t * std[:n_nodes]),
            beta_lower=tuple(float(v) for v in beta_opt - t * std[n_nodes:]),
            beta_upper=tuple(float(v) for v in beta_opt + t * std[n_nodes:]),
            level=self._conf,
        )

    @staticmethod
    def _apply(
        hs_rean: np.ndarray,
        directions: np.ndarray,
        alpha_spline: CubicSpline,
        beta_spline: CubicSpline,
    ) -> np.ndarray:
        """Ec. 23: Hs^C = α(θ)·(Hs^R)^β(θ)."""
        theta = directions % 360
        return alpha_spline(theta) * np.power(
            np.maximum(hs_rean, 1e-10), beta_spline(theta)
        )

    @staticmethod
    def _diagnostics(
        hs_inst: np.ndarray,
        hs_rean: np.ndarray,
        hs_cal: np.ndarray,
    ) -> CalibrationDiagnostics:
        """BIAS, RMS, RSI y r antes y después de calibrar (Ecs. 27-29)."""
        mu_i = np.mean(hs_inst)
        return CalibrationDiagnostics(
            bias_reanalysis=float(mu_i - np.mean(hs_rean)),
            bias_calibrated=float(mu_i - np.mean(hs_cal)),
            rms_reanalysis=float(np.sqrt(np.mean((hs_inst - hs_rean) ** 2))),
            rms_calibrated=float(np.sqrt(np.mean((hs_inst - hs_cal) ** 2))),
            rsi_reanalysis=float(np.sqrt(np.mean((hs_inst - hs_rean) ** 2)) / mu_i),
            rsi_calibrated=float(np.sqrt(np.mean((hs_inst - hs_cal) ** 2)) / mu_i),
            r_reanalysis=float(np.corrcoef(hs_inst, hs_rean)[0, 1]),
            r_calibrated=float(np.corrcoef(hs_inst, hs_cal)[0, 1]),
        )
