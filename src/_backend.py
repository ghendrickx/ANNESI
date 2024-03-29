"""
Neural network fitted to DFM-simulations: Backend. The neural network's frontend is in 'neural_network.py'.

Author: Gijs G. Hendrickx
"""
import abc

import numpy as np
import pandas as pd
import logging

import torch as torch
from sklearn.model_selection import train_test_split

from utils.files_dirs import DirConfig
from utils.data_conv import Export, Import

DEVICE = 'cpu'
LOG = logging.getLogger(__name__)

"""Configuration parameters"""
_WD = DirConfig(__file__).config_dir('_data')
_FILE_BASE = 'annesi'
_INPUT_VARS = (
    'tidal_range', 'surge_level', 'river_discharge', 'channel_depth', 'channel_width', 'channel_friction',
    'convergence', 'flat_depth_ratio', 'flat_width', 'flat_friction', 'bottom_curvature', 'meander_amplitude',
    'meander_length',
)
_OUTPUT_VARS = ('L', 'V')


class MLP(torch.nn.Module):
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


class _NNData(abc.ABC):
    """Parent-class that provides all basic data for all neural network related objects."""
    _wd = DirConfig(_WD)
    _input_vars = list(_INPUT_VARS)
    _output_vars = list(_OUTPUT_VARS)

    @property
    def wd(self):
        """
        :return: internal working directory
        :rtype: DirConfig
        """
        return self._wd

    @wd.setter
    def wd(self, wd_):
        """
        :param wd_: working directory
        :type wd_: DirConfig, str, iterable
        """
        self._wd = DirConfig(wd_)

    @staticmethod
    def _set_vars(variables):
        """Set variables.

        :param variables: input/output variables
        :type variables: iterable, str

        :return: list of variables
        :rtype: list
        """
        return [variables] if isinstance(variables, str) else list(variables)

    @property
    def input_vars(self):
        """
        :return: input variables
        :rtype: list
        """
        return self._input_vars

    @input_vars.setter
    def input_vars(self, variables):
        """
        :param variables: input variables
        :type variables: iterable, str
        """
        self.set_input_vars(variables)

    @classmethod
    def set_input_vars(cls, variables):
        """Set class-level input variables

        :param variables: input variables
        :type variables: iterable, str
        """
        cls._input_vars = cls._set_vars(variables)

    @property
    def output_vars(self):
        """
        :return: output variables
        :rtype: list
        """
        return self._output_vars

    @output_vars.setter
    def output_vars(self, variables):
        """
        :param variables: output variables
        :type variables: iterable, str
        """
        self.set_output_vars(variables)

    @classmethod
    def set_output_vars(cls, variables):
        """Set class-level output variables.

        :param variables: output variables
        :type variables: iterable, str
        """
        cls._output_vars = cls._set_vars(variables)


class Training(_NNData):
    """Training and testing environment for a neural network, provided with (1) a neural network; (2) an optimiser; and
    (3) a loss-function. When no data is provided for the training, the internal train data set is used; a data set of
    thousands of simulations of idealised estuaries using the modelling system Delft3D Flexible Mesh (DFM).
    """

    def __init__(self, model, optimiser, loss_function):
        """
        :param model: neural network
        :param optimiser: optimiser
        :param loss_function: loss-function

        :type model: torch.nn.Module
        :type optimiser: torch.optim.Optimizer
        :type loss_function: torch.loss._Loss
        """
        self._model = model.to(DEVICE)
        self._optimiser = optimiser
        self._loss_function = loss_function

        self._x_train, self._y_train, self._x_test, self._y_test = (None,) * 4

    @property
    def model(self):
        """
        :return: neural network model
        :rtype: torch.nn.Module
        """
        return self._model

    @property
    def optimiser(self):
        """
        :return: neural network optimiser
        :rtype: torch.optim.Optimizer
        """
        return self._optimiser

    @property
    def loss_function(self):
        """
        :return: neural network loss-function
        :rtype: torch.loss._Loss
        """
        return self._loss_function

    def load_data(self, file_name=None, x_cols=None, y_cols=None, directory=None, test_size=.2):
        """Load data from file that is split in a training and testing data set based on the provided test-size.

        :param file_name: file name, defaults to None
        :param x_cols: input variables, defaults to None
        :param y_cols: output variables, defaults to None
        :param directory: directory, defaults to None
        :param test_size: fraction of data used for testing, defaults to 0.2

        :type file_name: str, optional
        :type x_cols: str, list[str], optional
        :type y_cols: str, list[str], optional
        :type directory: DirConfig, str, list[str], tuple[str], optional
        :type test_size: float, optional
        """
        # load data from file
        file_name = 'nn_data.csv' if file_name is None else file_name
        file = DirConfig(self.wd if directory is None else directory).config_dir(file_name)
        df = pd.read_csv(file)

        # normalise input data
        x = InputData.normalise(df[self.input_vars if x_cols is None else x_cols])
        y = df[self.output_vars if y_cols is None else y_cols].to_numpy()

        # split data in training and testing data
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(x, y, test_size=test_size)

    def fit(self, epochs, x_train=None, y_train=None, random_sample_size=None):
        """Fit/train neural network to training data. When no training input and output data are provided, the
        internally stored data set is used.

        :param epochs: number of epochs, i.e. iterations
        :param x_train: training input data, defaults to None
        :param y_train: training output data, defaults to None
        :param random_sample_size: random sample size in training, defaults to None

        :type epochs: int
        :type x_train: iterable, optional
        :type y_train: iterable, optional
        :type random_sample_size: int, optional

        :return: losses for every iteration
        :rtype: list
        """
        # set training data
        x_train = self._x_train if x_train is None else x_train
        y_train = self._y_train if y_train is None else y_train

        # set/determine sample sizes
        n_train = len(x_train)
        random_sample_size = int(.1 * n_train) if random_sample_size is None else random_sample_size

        # initiate list of losses
        loss_list = []

        # train neural network
        for epoch in range(epochs):
            # reset optimiser: reset all gradients to zero
            self.optimiser.zero_grad()

            # randomly select samples from training data
            sel = np.random.choice(range(n_train), random_sample_size)
            x = torch.tensor(x_train[sel]).float().to(DEVICE)
            y_true = torch.tensor(y_train[sel]).float().to(DEVICE)

            # determine model's prediction
            y = self.model(x)

            # calculate losses/errors
            loss = self.loss_function(y, y_true)

            # back-propagate information to learn
            loss.backward()
            self.optimiser.step()

            # append performance
            loss_list.append(float(loss.detach().cpu()))
            LOG.info(f'Training\t:\tepoch {epoch}; loss = {loss:.4f}')

        # return list of losses: model's performance
        return loss_list

    def test(self, x_test=None, y_test=None):
        """Test neural network performance to test data. When to test input and output data are provided, the internally
        stored data set is used.

        :param x_test: test input data, defaults to None
        :param y_test: test output data, defaults to None

        :type x_test: iterable, optional
        :type y_test: iterable, optional

        :return: losses of test data
        :rtype: float
        """
        # set testing data
        x_test = self._x_test if x_test is None else x_test
        y_test = self._y_test if y_test is None else y_test

        # translate testing data to `torch.tensor`-objects
        x = torch.tensor(x_test).float().to(DEVICE)
        y_true = torch.tensor(y_test).float().to(DEVICE)

        # determine model's prediction
        y = self.model(x)

        # determine model's performance
        loss = self.loss_function(y, y_true)
        loss = float(loss.detach().cpu())

        # return model's performance
        LOG.info(f'Testing result (mean)\t:\t{loss:.4f}')
        return loss

    def predict(self, x):
        """Predict output data based on provided input data; i.e. use the neural network.

        :param x: input data
        :type x: float, iterable

        :return: neural network prediction
        :rtype: float, iterable
        """
        x = torch.tensor(x).float().to(DEVICE)
        y = self.model(x)
        return y.detach().cpu()

    def save(self, file_name=None, directory=None, to_pkl=True, to_onnx=True):
        """Save trained neural network.

        :param file_name: file name, defaults to None
        :param directory: directory, defaults to None
        :param to_pkl: save as *.pkl-file, defaults to True
        :param to_onnx: save as *.onnx-file, defaults to True

        :type file_name: str, optional
        :type directory: DirConfig, str, list[str], tuple[str], optional
        :type to_pkl: bool, optional
        :type to_onnx: bool, optional
        """
        export = Export(self.wd if directory is None else directory)
        meta_data = {
            'train': len(self._x_train),
            'test': len(self._x_test)
        }

        if to_pkl:
            export.to_pkl(self.model, _FILE_BASE if file_name is None else file_name, **meta_data)

        if to_onnx:
            export.to_onnx(self.model, _FILE_BASE if file_name is None else file_name, **meta_data)


class InputData(_NNData):
    """Input data object to ensure consistent scaling."""
    _scaler = None
    _scaler_is_fitted = False

    def __init__(self, file_name, directory=None, **kwargs):
        """
        :param file_name: file name of data set
        :param directory: directory, defaults to None
        :param kwargs: pandas.read_csv key-worded arguments

        :type file_name: str
        :type directory: DirConfig, str, list[str], tuple[str], optional
        """
        self._file = DirConfig(self.wd if directory is None else directory, create_dir=False).config_dir(file_name)

        df = pd.read_csv(self._file, **kwargs)
        self._raw = df[self.input_vars]
        self._norm = pd.DataFrame(data=self.normalise(df[self.input_vars]), columns=self.input_vars)

    @property
    def raw_data(self):
        """Raw data; i.e. not-normalised data.

        :return: raw data
        :rtype: pandas.DataFrame
        """
        return self._raw

    @property
    def norm_data(self):
        """Normalised data.

        :return: normalised data
        :rtype: pandas.DataFrame
        """
        return self._norm

    @property
    def scaler(self):
        """Scaler.

        :return: scaler-object
        :rtype: BaseEstimator
        """
        self._verify_scaler()
        return self._scaler

    @classmethod
    def get_scaler(cls):
        """Get scaler.

        :return: scaler-object
        :rtype: BaseEstimator
        """
        cls._verify_scaler()
        return cls._scaler

    @classmethod
    def _set_scaler(cls, scaler):
        """Set scaler, or reset scaler.

        :param scaler: scaler
        :type scaler: BaseEstimator, object
        """
        LOG.critical(f'Resetting/setting of scaler may influence performance of the neural network.')
        if input('Continue? [y/n] ') == 'y':
            file = input('Provide directory and file name of data to fit the scaler to: ')
            if not DirConfig().existence_file(file):
                raise FileNotFoundError(file)
            fit_data = pd.read_csv(file)

            cls._scaler = scaler
            cls._scaler.fit(fit_data[cls._input_vars])
            cls._scaler_is_fitted = True

            if input('Normalise raw data? [y/n] ') == 'y':
                LOG.warning('Not yet implemented.')

            if input('Save scaler? [y/n] ') == 'y':
                if input('Overwrite internal scaler? [y/n] ') == 'y':
                    Export(_WD).to_gz(cls._scaler, file_name=_FILE_BASE)
                else:
                    directory = input('Provide directory: ')
                    file_name = input('Provide file name: ')
                    Export(directory).to_gz(cls._scaler, file_name=file_name)

    @classmethod
    def normalise(cls, data):
        """Normalise data.

        :param data: data to be normalised
        :type data: iterable

        :return: normalised data
        :rtype: numpy.array
        """
        if not cls._scaler_is_fitted:
            try:
                cls._load()
            except FileNotFoundError:
                LOG.critical(f'Scaler file not found; normalisation not possible.')
                cls._scaler_is_fitted = False
                return None

        return cls._scaler.transform(data)

    @classmethod
    def _verify_scaler(cls):
        """Verify if scaler is loaded and/or fitted."""
        if not cls._scaler_is_fitted:
            try:
                cls._load()
            except FileNotFoundError:
                LOG.critical(f'Scaler file not found; no scaler defined.')
                cls._scaler_is_fitted = False

    @classmethod
    def _load(cls):
        """Load scaler data."""
        cls._scaler = Import(_WD).from_gz(file_name=_FILE_BASE)
        cls._scaler_is_fitted = True
