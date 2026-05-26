"""Calibración direccional de reanálisis de oleaje — Mínguez et al. (2011).

Modelo:  Hs^C = α(θ) · (Hs^R)^β(θ)

Uso típico (añadir wave/ y calibration/ a sys.path antes de importar):

    from use_cases.calibrate_waves import CalibrateWaves
    from adapters.scipy_fitter import ScipyCalibrationFitter
    from adapters.buoy_repository import PuertosEstadoBuoyRepository
    # La interfaz vive en interfaces.calibration (no interfaces.repositories)
    # para evitar colisión de namespace con wave/interfaces/repositories.py

    # 1. Cargar reanálisis (SIMAR, GOW, ERA5…)
    simar = simar_repo.load("3002002")

    # 2. Construir repositorio de boya con el reanálisis inyectado
    buoy_repo = PuertosEstadoBuoyRepository(simar, data_dir=Path("documents/"))

    # 3. Ejecutar calibración
    result = CalibrateWaves(buoy_repo, ScipyCalibrationFitter()).execute("Silleiro")

    # 4. result.hs_calibrated → alimentar a AnalyzeWaves
"""
