import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nuees_dynamiques_generales import NueesDynamiquesGenerales

X = np.vstack([np.random.default_rng(1).normal(-3, .3, (30, 2)),
               np.random.default_rng(2).normal(3, .3, (30, 2))])

def test_tous_les_modes():
    for mode in ("centroide", "points", "axes", "distribution"):
        m = NueesDynamiquesGenerales(2, representation=mode, n_init=3, random_state=4).fit(X)
        assert len(np.unique(m.labels_)) == 2
        assert np.isfinite(m.criterion_)

def test_kmeans_est_un_cas_particulier():
    m = NueesDynamiquesGenerales(2, representation="centroide", random_state=3).fit(X)
    assert all(np.asarray(r).shape == (2,) for r in m.representations_)
