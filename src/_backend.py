"""
Neural network fitted to DFM-simulations: Backend. The neural network's frontend is in 'neural_network.py'.

Author: Gijs G. Hendrickx
"""
import abc
import logging

import numpy as np
import pandas as pd
import torch

from utils import path, normalise

LOG = logging.getLogger(__name__)

"""Configuration parameters"""
DEVICE = 'cpu'
_WD = path.DirConfig(__file__).config_dir('_data')
_FILE_BASE = 'annesi'


# TODO: Deprecate this object
class OldMLP(torch.nn.Module):
    """Multilayer Perceptron: Default neural network."""

    def __init__(self, input_dim, output_dim, hidden_dim=50):
        """
        :param input_dim: dimension of input data
        :param output_dim: dimension of output data
        :param hidden_dim: hidden dimensions

        :type input_dim: int
        :type output_dim: int
        :type hidden_dim: int
        """
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def __repr__(self):
        """Object-representation."""
        return f'MLP(input_dim={self.features[0].in_features}, output_dim={self.features[-1].out_features})'

    def __str__(self):
        """String-representation."""
        return f'MLP: Multilayer Perceptron'

    def forward(self, x):
        """Forward passing of neural network.

        :param x: data
        :type x: torch.tensor

        :return: forward passed data
        """
        return self.features(x)


class MLP(torch.nn.Module):
    """Neural network architecture: Multilayer Perceptron."""

    def __init__(self, input_dim, output_dim, **kwargs):
        """
        :param input_dim: dimension of input vector
        :param output_dim: dimension of output vector
        :param kwargs: optional arguments
            hidden_dim: dimension of hidden layer(s), defaults to 50
            hidden_layers: number of hidden layers, defaults to 1

        :type input_dim: int
        :type output_dim: int
        :type kwargs: optional
            hidden_dim: int
            hidden_layers: int
        """
        super().__init__()

        # optional arguments
        hidden_dim = kwargs.get('hidden_dim', 50)
        hidden_layers = kwargs.get('hidden_layers', 1)
        assert hidden_layers > 0

        # hidden layers
        hidden = hidden_layers * [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(inplace=True)]

        # neural network architecture
        self.features = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            *hidden,
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """Forward passing of neural network.

        :param x: data
        :type x: torch.tensor

        :return: forward passed data
        :rtype: torch.tensor
        """
        return self.features(x)


class _NeuralNetwork(abc.ABC):
    """The interface of a neural network, which defaults to a neural network trained on a large data set of hydrodynamic
    simulations using Delft3D Flexible Mesh (DFM). The DFM-simulations encompass idealised estuaries, and wide-ranging
    sets of parameters are evaluated.

    This is the abstract-class containing the functions of the interface. Sub-classes further detail the specifics of
    the exact input parameters used with which configuration of neural network.
    """
    _req_attr = '_f_scaler', 'input_vars', 'output_vars'

    _reduced_output_vars = None
    _model = None

    _norm = None
    _f_scaler = None

    input_vars = []
    output_vars = []

    def __init__(self, **kwargs):
        """Loads trained neural network, which is stored inside the package.

        :param kwargs: optional arguments
            device: running device for neural network, defaults to DEVICE

        :type kwargs: optional
            device: str
        """
        self._device = kwargs.get('device', DEVICE)

    def __call__(self, data, **kwargs):
        """Call neural network, i.e. predict output with neural network.

        :param data: input data
        :param kwargs: optional arguments of `.predict()`

        :type data: pandas.DataFrame

        :return: model prediction
        :rtype: pandas.DataFrame
        """
        return self.predict(data, **kwargs)

    def __init_subclass__(cls):
        """Verify if all required attributes are defined in the subclass."""
        if not all(cls.__dict__.get(a) for a in cls._req_attr):
            attr = {a: getattr(cls, a) or 'UNDEFINED' for a in cls._req_attr}
            msg = f'Required attributes remain undefined in {cls.__name__}: {attr}'
            raise AttributeError(msg)

    @property
    def model(self):
        """Neural network. When none is specified during the initiation, the default neural network is loaded and used.

        :return: (trained) neural network
        :rtype: torch.nn.Module
        """
        if self._model is None:
            self._model = self._import_model()
        return self._model

    @abc.abstractmethod
    def _import_model(self):
        """Import trained neural network.

        :return: neural network
        :rtype: torch.nn.Module
        """

    @abc.abstractmethod
    def input_check(self, *args, **kwargs):
        """Execute input check.

        :param args: input parameters
        :param kwargs: optional arguments

        :type args: float

        :return: error messages
        :rtype: list[str]
        """

    @property
    def output(self):
        """Output variables, which may be a selection of the available output parameters. The available output
        parameters are defined by:

        >>> _NeuralNetwork.get_output_vars()

        :return: output variables
        :rtype: list
        """
        return self._reduced_output_vars or self.output_vars

    @output.setter
    def output(self, output_vars):
        """Set output variables of interest, which must be a selection of the available output parameters. The available
        output parameters are defined by:

        >>> _NeuralNetwork.get_output_vars()

        When the output variables are set to None, this is considered as resetting the output definition, which returns
        all available output variables.

        :param output_vars: selection of output parameters
        :type output_vars: iterable[str], None
        """
        # reset to default, i.e. all output variables
        if output_vars is None:
            self._reduced_output_vars = None

        # set to (selection of) available output variables

        def _warning(key):
            if key not in self.output_vars:
                LOG.warning(f'Unavailable output variable: \"{key}\" [skipped]')
            return key in self.output_vars

        if isinstance(output_vars, str):
            output_vars = [output_vars]

        self._reduced_output_vars = [k for k in output_vars if _warning(k)]

    def scan_input(self, data, scan):
        """Scan the input space for validity of samples. Three different scanning methods are included, which determine
        the handling of invalid samples:
         1. 'full'      :   apply the input check and raise an error if there is an invalid model configuration.
         2. 'skip'      :   apply the input check and remove invalid model configurations from the data.
         3. 'ignore'    :   ignore the input check and predict for all model configurations, invalid or valid.

        :param data: input data
        :param scan: scanning method

        :type data: pandas.DataFrame
        :type scan: str

        :return: scanned input data
        :rtype: pandas.DataFrame
        """
        # scanning data: raise error if any sample is invalid
        if scan == 'full':
            msg = data.apply(lambda r: self.input_check(*r[self.input_vars]), axis=1)
            warnings = msg[msg.astype(bool)]
            if len(warnings):
                raise ValueError(
                    f'Input is considered (partly) physical invalid:'
                    f'\n\t{len(data) - len(warnings):,} invalid samples '
                    f'({(len(data) - len(warnings)) / len(data) * 100:.1f}%)'
                    f'\n{warnings}'
                    f'\n\nSee documentation for scanning options.'
                )

        # scanning data set: skip invalid samples
        elif scan == 'skip':

            def check(*args):
                """Perform input check and return a warning when invalid samples are encountered."""
                msg_ = self.input_check(*args)
                # input check: failed
                if msg_:
                    LOG.info(msg_)
                    return None
                # input check: passed
                return args

            size = len(data)
            data = data.apply(lambda r: check(*[r[p] for p in self.input_vars]), axis=1, result_type='broadcast')
            data.dropna(inplace=True)
            if not len(data) == size:
                LOG.warning(
                    f'{size - len(data):,} samples have been skipped ({(size - len(data)) / size * 100:.1f}%).'
                )

        # scanning data set: ignore input check
        elif scan == 'ignore':
            msg = data.apply(lambda r: self.input_check(*r[self.input_vars]), axis=1)
            warnings = msg[msg.astype(bool)]
            if len(warnings):
                LOG.warning(
                    f'Input is considered (partly) physical invalid: Use output with caution!'
                    f'\n\t{len(warnings):,} invalid samples '
                    f'({len(warnings) / len(data) * 100:.1f}%)'
                    f'\n{warnings}'
                )

        # scanning data set: invalid scanning option
        else:
            msg = f'Scanning option \"{scan}\" not included; see documentation for help.'
            raise NotImplementedError(msg)

        # return checked data set
        return data

    @property
    def norm(self):
        """
        :return: normalise-object
        :rtype: normalise.Normalise
        """
        if self._norm is None:
            self._norm = normalise.Normalise(file_scaler=self._f_scaler)

        return self._norm

    @property
    def scaler(self):
        """
        :return: implemented scaler
        :rtype: sklearn.preprocessing.BaseEstimator
        """
        return self.norm.scaler

    def predict(self, data, scan='full'):
        """Predict output.

        The scanning method is based on the `scan`-argument, which can have one of three values:
         1. 'full'      :   apply the input check and raise an error if there is an invalid model configuration.
         2. 'skip'      :   apply the input check and remove invalid model configurations from the data.
         3. 'ignore'    :   ignore the input check and predict for all model configurations, invalid or valid.

        :param data: input data
        :param scan: method of scanning the input data, defaults to 'full'

        :type data: pandas.DataFrame
        :type scan: str, optional

        :return: model prediction
        :rtype: pandas.DataFrame
        """
        # scan input data
        data = self.scan_input(data[self.input_vars], scan=scan)

        # normalise input data
        norm_data = self.norm(data)

        # use neural network
        x = torch.tensor(norm_data).float().to(self._device)
        y = self.model(x)

        # store as pandas.DataFrame
        output = pd.DataFrame(data=y.detach().cpu(), columns=self.output_vars, dtype=float, index=data.index)

        # return selected output
        return output[self.output]

    def estimate(self, **kwargs):
        """Provides an estimate for incomplete input data. The rough estimate is based on assessing a range values of
        the undefined parameter(s). Based on the resulting spreading, a rough estimate is provided. A parameter can also
        be given as a list (or tuple) containing the minimum and maximum values of a range of interest. Between these
        two provided extremes, a number of samples are evaluated (equal to `parameter_samples`).

        The size of the range assessed per undefined parameter is defined by `parameter_samples`, which iterates through
        the minimum and maximum of the undefined parameter. Multiple values are assessed due to the expected non-linear
        response. However, the number of iterations may rise quickly when more parameters remain undefined: The number
        of iterations required equals `parameter_samples` to the power of the number of undefined parameters. As an
        example, with `parameter_samples` equal to 3 (default), and three undefined parameters, the total number of
        iterations equals 3 ** 3, or 27.

        :param kwargs: optional arguments
            parameter_samples: range-size of undefined input parameters, defaults to 3
            grid_limits: include grid-limits checks, defaults to None
            scan: method of scanning the input data, defaults to 'skip'
            reset_index: reset index of pandas.DataFrame, defaults to True
            include_input: include input data, defaults to True
            statistics: return statistics of estimates, defaults to False
            file_scaler: file of scaler to extract data ranges from, defaults to None
            kwargs: definitions of input parameters

        :type kwargs: optional
            parameter_samples: int
            grid_limits: bool
            scan: str
            reset_index; bool
            include_input: bool
            file_scaler: str
            statistics: bool
        """
        # optional arguments
        size = kwargs.get('parameter_samples', 3)
        grid_limits = kwargs.get('grid_limits')
        f_scaler = kwargs.get('file_scaler')

        # extract input data
        input_data = {k: kwargs.get(k) for k in self.input_vars}

        # determine estimate when all input parameters are provided (float)
        if all(isinstance(v, (float, int)) for v in input_data.values()):
            sample = pd.DataFrame(**input_data, index=[0])
            return self.predict(sample, scan='full', grid_limits=grid_limits)

        # define input space
        scaler = self.norm.scaler if f_scaler is None else normalise.Normalise(file_scaler=f_scaler).scaler
        arrays = dict()
        for i, v in enumerate(self.input_vars):
            # no range defined
            if input_data.get(v) is None:
                arrays[v] = np.linspace(scaler.data_min_[i], scaler.data_max_[i], size)
            # single value definition
            elif isinstance(input_data[v], (float, int)):
                arrays[v] = input_data[v]
            # range defined
            elif len(input_data[v]) == 2:
                v_min = max(min(input_data[v]), scaler.data_min_[i])
                v_max = min(max(input_data[v]), scaler.data_max_[i])
                arrays[v] = np.linspace(v_min, v_max, size)
                # warning message: range outside training data
                if min(input_data[v]) < scaler.data_min_[i] or max(input_data[v]) > scaler.data_max_[i]:
                    LOG.warning(f'Defined range exceeds training data; \"{v}\" range used: {v_min, v_max}')
            # array defined
            else:
                arrays[v] = input_data[v]

        # create input space
        data = pd.DataFrame(
            data=np.array(np.meshgrid(*[v for v in arrays.values()])).T.reshape(-1, len(arrays)),
            columns=self.input_vars
        )

        # predict output space
        data[self.output] = self.predict(data, scan=kwargs.get('scan', 'skip'), grid_limits=grid_limits)
        data.dropna(inplace=True)

        # reset index
        if kwargs.get('reset_index', True):
            data.reset_index(drop=True, inplace=True)

        # in-/exclude input space
        if not kwargs.get('include_input', True):
            data = data[self.output]

        # return statistics
        if kwargs.get('statistics', False):
            return data.describe()
        # return estimates
        return data

    @classmethod
    def get_input_vars(cls):
        """Get class-level defined input variables.

        :return: input variables
        :rtype: list
        """
        return cls.input_vars

    @classmethod
    def get_output_vars(cls):
        """get class-level defined output variables.

        :return: output variables
        :rtype: list
        """
        return cls.output_vars
