from pathlib import Path

class Paths:
    ROOT = Path(__file__).resolve().parents[2]
    DATA = ROOT / "data"
    DERIVATIVES = '/data/core-psy-archive/data/PRONIA/test_vbt_pipe/vbt_derivatives'
    RESOURCES = ROOT / "resources"
    FIGURES = ROOT / "figures"
    TYPE_OF_CONFOUNDS = "aCompCor"
    TYPE_OF_SWEEP = "sweep_simulations"

