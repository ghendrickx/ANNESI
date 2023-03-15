"""
Neural network fitted to DFM-simulations: Frontend. The neural network's backend is in '._backend.py'.

Author: Gijs G. Hendrickx
"""
import logging

import torch

from src import _backend
from utils import check, filing, path

_LOG = logging.getLogger(__name__)


class ANNESI(_backend._NeuralNetwork):
    """ANNESI: Artificial Neural Network for Estuarine Salt Intrusion."""
    _f_scaler = 'annesi'

    input_vars = [
        'tidal_range', 'surge_level', 'river_discharge', 'channel_depth', 'channel_width', 'channel_friction',
        'convergence', 'flat_depth_ratio', 'flat_width', 'flat_friction', 'bottom_curvature', 'meander_amplitude',
        'meander_length',
    ]
    output_vars = ['L', 'V']

    def _import_model(self):
        """Import trained neural network: ANNESI.

        :return: neural network
        :rtype: torch.nn.Module
        """
        # TODO: Implement the use of the updated MLP-object (requires retraining the neural network...)
        # architecture of neural network
        model = _backend.OldMLP(len(self.input_vars), len(self.output_vars), hidden_dim=50)

        # import neural network
        wd = path.DirConfig(__file__)
        return filing.Import(wd.config_dir('_data')).from_pkl(model, file_name='annesi')

    def input_check(self, *args):
        """Execute input check.

        :param args: input parameters
        :type args: float

        :return: error messages
        :rtype: list[str]
        """
        return check.input_check(*args)
