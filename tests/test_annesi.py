"""
Tests addressing the functioning of the neural network, i.e. testing `src.neural_network.py` and the supporting code in
`src._backend.py`.

Author: Gijs G. Hendrickx
"""
import logging

import pandas as pd
import pytest
import torch

from src import neural_network as nn

"""pytest.fixtures"""


@pytest.fixture
def nn_input_data():
    return dict(
        tidal_range=2.25,
        surge_level=0,
        river_discharge=7750,
        channel_depth=20,
        channel_width=1000,
        channel_friction=.023,
        convergence=1e-4,
        flat_depth_ratio=0,
        flat_width=500,
        flat_friction=.05,
        bottom_curvature=1e-5,
        meander_amplitude=1000,
        meander_length=20000,
    )


@pytest.fixture
def nn_input_data_range(nn_input_data):
    df = pd.DataFrame(data=nn_input_data, index=[0])
    for _ in range(9):
        df = df.append(nn_input_data, ignore_index=True)
    return df


"""TestClasses"""


class TestANNESI:
    """Tests for the NeuralNetwork-object."""

    def setup_method(self):
        """Initiate neural network."""
        self.annesi = nn.ANNESI()

    """Initiating ANNESI"""

    def test_type_model(self):
        assert isinstance(self.annesi.model, torch.nn.Module)

    def test_default_output_vars(self):
        assert all(out in ['L', 'V'] for out in self.annesi.output)

    def test_limited_output_vars(self):
        self.annesi.output = 'L'
        assert all(out in ['L'] for out in self.annesi.output)

    def test_limited_output_vars_fail(self, caplog):
        with caplog.at_level(logging.WARNING):
            self.annesi.output = 'non-existing output variable'
        assert 'unavailable output variable' in caplog.text.lower()

    """Model predictions"""

    def test_call(self, nn_input_data_range):
        out = self.annesi(nn_input_data_range, scan='full')
        assert len(out) == 10
        assert all(col in ['L', 'V'] for col in out.columns)

    def test_predict(self, nn_input_data_range):
        out = self.annesi.predict(nn_input_data_range, scan='full')
        assert len(out) == 10
        assert all(col in ['L', 'V'] for col in out.columns)

    def test_predict_mod_output(self, nn_input_data_range):
        self.annesi.output = 'L'
        out = self.annesi.predict(nn_input_data_range, scan='full')
        assert all(col in ['L'] for col in out.columns)

    def test_predict_error(self, nn_input_data_range):
        nn_input_data_range['channel_depth'] = 5
        nn_input_data_range['river_discharge'] = 16000
        with pytest.raises(ValueError):
            self.annesi.predict(nn_input_data_range, scan='full')

    def test_predict_skip(self, nn_input_data_range):
        nn_input_data_range.loc[0, 'channel_depth'] = 5
        nn_input_data_range['river_discharge'] = 16000
        out = self.annesi.predict(nn_input_data_range, scan='skip')
        assert len(out) == 9

    def test_predict_ignore(self, nn_input_data_range):
        nn_input_data_range.loc[0, 'channel_depth'] = 5
        nn_input_data_range['river_discharge'] = 16000
        out = self.annesi.predict(nn_input_data_range, scan='ignore')
        assert len(out) == 10

    def test_predict_ignore_warn(self, nn_input_data_range, caplog):
        nn_input_data_range.loc[0, 'channel_depth'] = 5
        nn_input_data_range['river_discharge'] = 16000
        with caplog.at_level(logging.WARNING):
            self.annesi.predict(nn_input_data_range, scan='ignore')
        assert 'use output with caution' in caplog.text.lower()

    """Single model predictions"""

    def test_single_predict(self, nn_input_data):
        out = self.annesi.single_predict(**nn_input_data)
        assert len(out) == 1
        assert all(col in ['L', 'V'] for col in out.columns)

    def test_single_predict_error(self, nn_input_data):
        nn_input_data.update({
            'channel_depth': 5,
            'river_discharge': 16000,
        })
        with pytest.raises(ValueError):
            self.annesi.single_predict(**nn_input_data)

    def test_single_predict_mod_output(self, nn_input_data):
        self.annesi.output = 'L'
        out = self.annesi.single_predict(**nn_input_data)
        assert isinstance(out, float)

    """Model predictions with uncertainty"""

    def test_estimate(self, nn_input_data):
        # noinspection PyTypeChecker
        nn_input_data['river_discharge'] = [7750, 20000]
        out = self.annesi.estimate(**nn_input_data, include_input=False)
        assert all(col in ['L', 'V'] for col in out.columns)

    def test_estimate_mod_output(self, nn_input_data):
        # noinspection PyTypeChecker
        nn_input_data['river_discharge'] = [7750, 20000]
        self.annesi.output = 'L'
        out = self.annesi.estimate(**nn_input_data, include_input=False)
        assert all(col in ['L'] for col in out.columns)

    def test_estimate_incl_input(self, nn_input_data):
        # noinspection PyTypeChecker
        nn_input_data['river_discharge'] = [7750, 20000]
        out = self.annesi.estimate(**nn_input_data, include_input=True)
        assert all(col in self.annesi.get_input_vars() + ['L', 'V'] for col in out.columns)
        assert not all(col in ['L', 'V'] for col in out.columns)