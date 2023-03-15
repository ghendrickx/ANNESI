"""
Export and import data/files.

Author: Gijs G. Hendrickx
"""
import logging

import pandas as pd
import joblib
import torch
import time

from utils import path

LOG = logging.getLogger(__name__)


def _file_name(default, extension=None):
    """Decorator to define default file-name and do the logging of file-handling.

    :param default: default file-name
    :type default: str
    """
    def decorator(func):
        """Decorator function."""

        def wrapper(self, *args, **kwargs):
            """Wrapper function."""
            file_name = default_file_name(kwargs.pop('file_name', None), default=default, extension=extension)
            out = func(self, *args, file_name=file_name, **kwargs)
            self._log(file_name)
            return out

        return wrapper

    return decorator


def default_file_name(file_name, default, extension=None):
    """Use default file-name if none is defined. The extension is based on the default file-name provided if not
    stated explicitly.

    :param file_name: file-name
    :param default: default file-name
    :param extension: file extension, defaults to None

    :type file_name: str, None
    :type default: str
    :type extension: str, optional

    :return: file-name
    :rtype: str
    """
    if file_name is None:
        LOG.info(f'Default file-name used: {default}')
        return default

    if extension is None:
        extension = f'.{default.split(".")[-1]}'

    if not file_name.endswith(extension):
        return f'{file_name.split(".")[0]}{extension}'

    return file_name


class _DataConversion:
    """Parent-class for exporting and importing data/files."""
    _wd = None

    def __init__(self, wd=None):
        """
        :param wd: working directory, defaults to None
        :type wd: DirConfig, str, list, tuple, optional
        """
        self._wd = path.DirConfig(wd)

    def _log(self, file_name):
        """Log file exported/imported.

        :param file_name: file name
        :type file_name: str
        """
        LOG.info(f'File {self.__class__.__name__.lower()}ed\t:\t{self._wd.config_dir(file_name)}')

    @property
    def working_dir(self):
        """
        :return: working directory
        :rtype: DirConfig
        """
        return self._wd


class Export(_DataConversion):
    """Exporting data/files."""
    _wd = None

    @_file_name(default='output.csv')
    def to_csv(self, data, *, file_name=None, **kwargs):
        """Export data to *.csv-file

        :param data: output data
        :param file_name: file-name, defaults to None
        :param kwargs: key-worded arguments for exporting pandas.DataFrame to *.csv

        :type data: pandas.DataFrame, dict, iterable
        :type file_name: str, optional
        :type kwargs: dict, optional

        :return: *.csv-file
        """
        # optional arguments
        index = kwargs.pop('index', False)

        # convert to pandas.DataFrame
        if not hasattr(data, 'to_csv'):
            data = pd.DataFrame(data=data)

        # export to *.csv
        data.to_csv(self._wd.config_dir(file_name), index=index, **kwargs)

    @_file_name(default='scaler.gz')
    def to_gz(self, scaler, *, file_name=None):
        """Dump fitted scaler for later re-use to a *.gz-file.

        :param scaler: scaler
        :param file_name: file name, defaults to None

        :type scaler: BaseEstimator
        :type file_name: str, optional

        :return: *.gz-file
        """
        # export scaler
        joblib.dump(scaler, self._wd.config_dir(file_name))

    @_file_name(default='annesi.pkl')
    def to_pkl(self, neural_network, *, file_name=None, **kwargs):
        """Save neural network as `*.pkl`-file.

        :param neural_network: neural network
        :param file_name: file name, defaults to None
        :param kwargs: meta data of neural network (see documentation of `Export._export_meta_data()`)

        :type neural_network: torch.nn.Module
        :type file_name: str, optional

        :return: *.pkl-file
        """
        # export neural network as *.pkl
        torch.save(neural_network.state_dict(), self.working_dir.config_dir(file_name))

        # export meta-data
        self._export_meta_data(file_name=f'{file_name.split(".")[0]}-pkl.txt', model=neural_network, **kwargs)

    @_file_name(default='annesi.onnx')
    def to_onnx(self, neural_network, *, file_name=None, **kwargs):
        """Save neural network as `*.onnx`-file for use on a (static) web-page.

        :param neural_network: neural network
        :param file_name: file-name, defaults to None
        :param kwargs: meta data of neural network (see documentation of `Export._export_meta_data()`)

        :type neural_network: torch.nn.Module
        :type file_name: str, optional

        :return: *.onnx-file
        """
        # ensure neural network is in inference-mode
        neural_network.eval()

        # define dummy input to trace the graph
        dummy_input = torch.randn(1, neural_network.features[0].in_features)

        # export neural network as *.onnx
        torch.onnx.export(neural_network, dummy_input, self.working_dir.config_dir(file_name))

        # export meta-data
        self._export_meta_data(file_name=f'{file_name.split(".")[0]}-onnx.txt', model=neural_network, **kwargs)

    @_file_name(default='meta-data.txt')
    def _export_meta_data(self, *, file_name=None, **kwargs):
        """Export a `*.log`-file with information about the exported model, and when it is (last) saved. Depending on
        the information provided in the `kwargs`, the content of the file is created. The `*.log`-file includes (or can
        include) the following information:
         -  (default)       :   date and time of exporting
         -  model           :   model type, features, and (internal) design (torch.nn.Module, optional)
         -  train           :   size of training data set (int, optional)
         -  test            :   size of testing data set (int, optional)
         -  train & test    :   size of overall data set (int, optional)
        The last is included if both the `train` and `test` key-word are provided.

        :param file_name: file-name, defaults to None
        :param kwargs: meta data

        :type file_name: str, optional
        """
        with open(self.working_dir.config_dir(file_name), mode='w') as f:
            f.write(f'Model exported on {time.ctime(time.time())}\n')

            if kwargs.get('model'):
                model = kwargs.get('model')
                f.write(f'Model type: {model}\n')
                f.write(f'\tinput features: {model.features[0].in_features}\n')
                f.write(f'\toutput features: {model.features[-1].out_features}\n')

                f.write(f'Model design: {model.features}\n')

            if kwargs.get('train'):
                f.write(f'Size of training data set: {kwargs.get("train")}\n')

            if kwargs.get('test'):
                f.write(f'Size of testing data set: {kwargs.get("test")}\n')

            if kwargs.get('train') and kwargs.get('test'):
                f.write(f'Size of overall data set: {kwargs.get("train") + kwargs.get("test")}\n')


class Import(_DataConversion):
    """Importing data/files."""
    _wd = None

    @_file_name(default='data.csv')
    def from_csv(self, *, file_name=None, **kwargs):
        """Load data from a *.csv-file.

        :param file_name: file name, defaults to None
        :param kwargs: optional arguments for `pandas.read_csv()`

        :type file_name: str, optional

        :return: data
        :rtype: pandas.DataFrame
        """
        # read data
        return pd.read_csv(self._wd.config_dir(file_name), **kwargs)

    @_file_name(default='annesi.gz')
    def from_gz(self, *, file_name=None):
        """Load fitted scaler from previous fitting to re-use from a *.gz-file.

        :param file_name: file name, defaults to None
        :type file_name: str, optional

        :return: scaler
        :rtype: BaseEstimator
        """
        # import scaler
        return joblib.load(self._wd.config_dir(file_name))

    @_file_name(default='annesi.pkl')
    def from_pkl(self, model, *, file_name=None):
        """Load previously trained neural network to (re-)use from a *.pkl-file.

        :param model: neural network design
        :param file_name: file name

        :type model: torch.nn.Module
        :type file_name: str, optional

        :return: trained neural network
        :rtype: torch.nn.Module
        """
        # import neural network
        model.load_state_dict(torch.load(self._wd.config_dir(file_name)))

        # return neural network
        return model
