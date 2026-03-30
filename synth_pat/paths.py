from pathlib import Path

class Paths:
    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    DERIVATIVES = DATA / "derivatives"
    RESOURCES = ROOT / "resources"
    RESULTS = ROOT / "results"
    FIGURES = ROOT / "figures"
    SNAKEMAKE = ROOT / "snakeproject2"

    TYPE_OF_SWEEP = "huifang_ppc_test"


