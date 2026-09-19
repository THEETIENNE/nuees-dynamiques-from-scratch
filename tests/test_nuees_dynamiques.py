import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nuees_dynamiques import NueesDynamiques


def test_partition_simple_et_prediction():
    X = np.array([[0., 0.], [0., 1.], [10., 10.], [10., 11.]])
    modele = NueesDynamiques(2, n_init=5, random_state=7).fit(X)
    assert len(set(modele.labels_)) == 2
    assert modele.inertia_ == 1.0
    assert modele.predict([[0., .2], [10., 10.2]])[0] != modele.predict([[0., .2], [10., 10.2]])[1]


def test_reproductibilite():
    X = np.random.default_rng(4).normal(size=(30, 3))
    a = NueesDynamiques(3, random_state=9).fit(X)
    b = NueesDynamiques(3, random_state=9).fit(X)
    assert np.allclose(a.cluster_centers_, b.cluster_centers_)
    assert np.array_equal(a.labels_, b.labels_)
