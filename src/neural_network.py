"""
Neural network fitted to DFM-simulations: Frontend. The neural network's backend is in 'src._backend.py'.

Author: Gijs G. Hendrickx
"""
import logging

import pandas as pd
import torch

from src import _backend
from utils import check, filing, path, decorators

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

    def single_predict(
            self, tidal_range, surge_level, river_discharge, channel_depth, channel_width, channel_friction,
            convergence, flat_depth_ratio, flat_width, flat_friction, bottom_curvature, meander_amplitude,
            meander_length
    ):
        """Predict output of a single set of input parameters. This single predictor provides more guidance in the type
        of input; the `predict()`- and `predict_from_file()`-methods enable predictions based on multiple sets of input
        parameters, i.e. based on an input data set.

        :param tidal_range: tidal range
        :param surge_level: storm surge level
        :param river_discharge: river discharge
        :param channel_depth: channel depth
        :param channel_width: channel width
        :param channel_friction: channel friction
        :param convergence: convergence
        :param flat_depth_ratio: flat depth ratio
        :param flat_width: flat width
        :param flat_friction: flat friction
        :param bottom_curvature: bottom curvature
        :param meander_amplitude: meander amplitude
        :param meander_length: meander length

        :type tidal_range: float
        :type surge_level: float
        :type river_discharge: float
        :type channel_depth: float
        :type channel_width: float
        :type channel_friction: float
        :type convergence: float
        :type flat_depth_ratio: float
        :type flat_width: float
        :type flat_friction: float
        :type bottom_curvature: float
        :type meander_amplitude: float
        :type meander_length: float

        :return: model prediction
        :rtype: pandas.DataFrame, float
        """
        # convert `dict` to `pandas.DataFrame`
        data = pd.DataFrame({k: v for k, v in locals().items() if k in self.input_vars}, index=[0])

        # predict
        if len(self.output) == 1:
            return float(self.predict(data, scan='full').values)
        return self.predict(data, scan='full')

    def predict_from_file(self, file_name, scan='full', **kwargs):
        """Predict from *.csv-file containing all thirteen input parameters, listed in the first line of the *.csv-file.
        If the `export` argument is a `bool` or a `str`, the predictions are exported to a *.csv-file; when the argument
        is a `str`, this `str` is used as the file-name for this file, which defaults to 'output.csv' otherwise.

        :param file_name: file-name
        :param scan: method of scanning the input data, defaults to 'full'
            [see `src._backend._NeuralNetwork.predict()` for help]
        :param kwargs: optional arguments
            wd: working directory, defaults to None
            export: export predictions to *.csv-file, defaults to False
            optional arguments of `pandas.read_csv()`.

        :type file_name: str
        :type scan: str, optional
        :type kwargs: optional
            wd: utils.path.DirConfig, str, iterable[str]
            export: bool, str

        :return: model prediction(s)
        :return: pandas.DataFrame
        """
        # optional arguments
        wd = kwargs.pop('wd', None)
        export = kwargs.pop('export', False)

        # read input data
        data = filing.Import(wd).from_csv(file_name=file_name, **kwargs)

        # predict
        output = self.predict(data, scan=scan)

        # export data
        if export:
            filing.Export(wd).to_csv(output, file_name=export if isinstance(export, str) else None)

        # return predictions
        return output


@decorators.deprecated(msg='Use `src.neural_network.ANNESI` instead')
class NeuralNetwork(ANNESI):
    """The interface of a neural network, which defaults to a neural network trained on a large data set of hydrodynamic
    simulations using Delft3D Flexible Mesh (DFM). The DFM-simulations encompass idealised estuaries, and wide-ranging
    sets of parameters are evaluated.
    """

    @property
    @decorators.deprecated(msg='Use the `model`-attribute instead')
    def nn(self):
        """The default neural network (i.e. ANNESI) is loaded, despite any neural network provided with the initiation.
        This property is included for backward compatibility but the whole object will be deprecated in the next full
        version release: release of `ANNESI-v2.0`.

        :return: neural network
        :rtype: torch.nn.Module
        """
        return self.model

    @decorators.deprecated(msg='Its functionality will be removed')
    def save_as(self, f_type, file_name=None, directory=None):
        """Save neural network as one of the available export-formats:
         1. *.pkl   :   for usage within Python, using PyTorch.
         2. *.onnx  :   for integration of neural network in a website.

        :param f_type: file-type
        :param file_name: file-name, defaults to None
        :param directory: directory, defaults to None

        :type f_type: str
        :type file_name: str, optional
        :type directory: DirConfig, str, iterable[str], optional
        """
        # check available export-formats
        f_types = ('pkl', 'onnx')
        if f_type not in f_types:
            msg = f'NeuralNetwork can only be saved as {f_types}, {f_type} has been specified.'
            raise NotImplementedError(msg)

        # internally save neural network
        if directory == 'internal-save':
            msg = 'About to save neural network internally. This might overwrite an existing version. Continue? [y/n]'
            if input(msg) == 'y':
                directory = _backend._WD
            else:
                msg = 'Internally saving neural network aborted.'
                raise KeyboardInterrupt(msg)

        # export neural network
        export = filing.Export(directory)
        if f_type == 'pkl':
            export.to_pkl(self.nn, file_name=file_name)
        elif f_type == 'onnx':
            export.to_onnx(self.nn, file_name=file_name)
