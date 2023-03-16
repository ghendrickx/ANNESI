"""
Tests addressing the functioning of the normalisation, i.e. testing `utils.normalise.py`.

Author: Gijs G. Hendrickx
"""
import pytest

from utils import normalise

"""pytest.fixtures"""


@pytest.fixture
def data():
    return [
        [1, 2, 4],
        [4, 2, 1],
        [5, 0, 0],
    ]


@pytest.fixture
def norm():
    return [
        [0, 1, 1],
        [.75, 1, .25],
        [1, 0, 0],
    ]

@pytest.fixture
def sample():
    return [
        [1, 0, 100, 5, 500, .02, 2.5e-5, -1, 0, .02, 0, 0, 0]
    ]


"""TestClasses"""


class TestNormaliseDummyData:
    """Tests for the `utils.normalise.Normalise`-class: Dummy data."""

    def setup_method(self):
        """Initiate normalisation object."""
        self.norm = normalise.Normalise([
            [1, 2, 4],
            [4, 2, 1],
            [5, 0, 0],
        ], save=False)

    """Normalise and de-normalise"""

    def test_normalise(self, data, norm):
        out = self.norm(data)
        for v_range, c_range in zip(out, norm):
            for v, c in zip(v_range, c_range):
                assert v == c

    def test_normalise_exec(self, data, norm):
        out = self.norm.exec(data)
        for v_range, c_range in zip(out, norm):
            for v, c in zip(v_range, c_range):
                assert v == c

    def test_reverse_normalise(self, data, norm):
        out = self.norm.reverse(norm)
        for v_range, c_range in zip(out, data):
            for v, c, in zip(v_range, c_range):
                assert v == c


class TestNormaliseBuiltInScaler:

    def setup_method(self):
        self.norm = normalise.Normalise()

    """Use default scaler"""

    def test_scaler_type(self):
        from sklearn.preprocessing import MinMaxScaler

        scaler = self.norm.scaler
        assert isinstance(scaler, MinMaxScaler)

    def test_scaler_size(self, sample):
        out = self.norm(sample)
        assert out.shape == (1, 13)

    def test_scaler_size_small(self, sample):
        sample = [sample[0][:12]]
        with pytest.raises(ValueError):
            self.norm(sample)

    def test_scaler_size_big(self, sample):
        sample = [*sample[0], 1]
        with pytest.raises(ValueError):
            self.norm(sample)
