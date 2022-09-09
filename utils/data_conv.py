"""
Export and import data/files.

Author: Gijs G. Hendrickx
"""
import logging

import joblib
import torch
import time

from utils.files_dirs import DirConfig

LOG = logging.getLogger(__name__)


def default_file_name(file_name, default, extension=None):
    """Use default file name if none is defined. The extension is based on the default file name provided if not
    stated explicitly.

    :param file_name: file name
    :param default: default file name
    :param extension: file extension, defaults to None

    :type file_name: str, None
    :type default: str
    :type extension: str, optional

    :return: file name
    :rtype: str
    """
    # return default file name
    if file_name is None:
        LOG.info(f'Default file name used: {default}')
        return default

    # determine file extension
    if extension is None:
        extension = f'.{default.split(".")[-1]}'

    # return file name
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
        self._wd = DirConfig(wd)

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

    def to_csv(self, data, file_name=None, **kwargs):
        """Export data to *.csv-file

        :param data: output data
        :param file_name: file name, defaults to None
        :param kwargs: key-worded arguments for exporting pandas.DataFrame to *.csv

        :type data: pandas.DataFrame
        :type file_name: str, optional
        :type kwargs: dict, optional

        :return: *.csv-file
        """
        # default file-name
        file_name = default_file_name(file_name, default='output.csv')

        # export data set
        data.to_csv(self._wd.config_dir(file_name), **kwargs)

        # log export
        self._log(file_name)

    def to_gz(self, scaler, file_name=None):
        """Dump fitted scaler for later re-use to a *.gz-file.

        :param scaler: scaler
        :param file_name: file name, defaults to None

        :type scaler: BaseEstimator
        :type file_name: str, optional

        :return: *.gz-file
        """
        # default file-name
        file_name = default_file_name(file_name, default='scaler.gz')

        # export scaler
        joblib.dump(scaler, self._wd.config_dir(file_name))

        # log export
        self._log(file_name)

    def to_pkl(self, neural_network, file_name=None, **kwargs):
        """Save neural network as `*.pkl`-file.

        :param neural_network: neural network
        :param file_name: file name, defaults to None
        :param kwargs: meta data of neural network (see documentation of `Export._export_meta_data()`)

        :type neural_network: torch.nn.Module
        :type file_name: str, optional

        :return: *.pkl-file
        """
        import torch

        file_name = default_file_name(file_name, default='annesi.pkl')

        torch.save(neural_network.state_dict(), self.working_dir.config_dir(file_name))
        self._export_meta_data('annesi-pkl.log', model=neural_network, **kwargs)

        self._log(file_name)

    def to_onnx(self, neural_network, file_name=None, **kwargs):
        """Save neural network as `*.onnx`-file for use on a (static) web-page.

        :param neural_network: neural network
        :param file_name: file-name, defaults to None
        :param kwargs: meta data of neural network (see documentation of `Export._export_meta_data()`)

        :type neural_network: torch.nn.Module
        :type file_name: str, optional

        :return: *.onnx-file
        """
        import torch

        file_name = default_file_name(file_name, default='annesi.onnx')

        # ensure neural network is in inference-mode
        neural_network.eval()
        # define dummy input to trace the graph
        dummy_input = torch.randn(1, neural_network.features[0].in_features)

        # export neural network to ONNX
        torch.onnx.export(neural_network, dummy_input, self.working_dir.config_dir(file_name))
        self._export_meta_data('annesi-onnx.log', model=neural_network, **kwargs)

        self._log(file_name)

    def _export_meta_data(self, file_name, **kwargs):
        """Export a `*.log`-file with information about the exported model, and when it is (last) saved. Depending on
        the information provided in the `kwargs`, the content of the file is created. The `*.log`-file includes (or can
        include) the following information:
         -  (default)       :   date and time of exporting
         -  model           :   model type, features, and (internal) design (torch.nn.Module, optional)
         -  train           :   size of training data set (int, optional)
         -  test            :   size of testing data set (int, optional)
         -  train & test    :   size of overall data set (int, optional)
        The last is included if both the `train` and `test` key-word are provided.

        :param file_name: file-name
        :param kwargs: meta data

        :type file_name: str
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

        self._log(file_name)


class Import(_DataConversion):
    """Importing data/files."""
    _wd = None

    def from_gz(self, file_name=None):
        """Load fitted scaler from previous fitting to re-use from a *.gz-file.

        :param file_name: file name, defaults to None
        :type file_name: str, optional

        :return: scaler
        :rtype: BaseEstimator
        """
        # default file-name
        file_name = default_file_name(file_name, default='scaler.gz')

        # import scaler
        scaler = joblib.load(self._wd.config_dir(file_name))

        # log import
        self._log(file_name)

        # return scaler
        return scaler

    def from_pkl(self, model, file_name=None):
        """Load previously trained neural network to (re-)use from a *.pkl-file.

        :param model: neural network design
        :param file_name: file name

        :type model: torch.nn.Module
        :type file_name: str, optional

        :return: trained neural network
        :rtype: torch.nn.Module
        """
        # default file-name
        file_name = default_file_name(file_name, default='annesi.pkl')

        # import neural network
        model.load_state_dict(torch.load(self._wd.config_dir(file_name)))

        # log import
        self._log(file_name)

        # return neural network
        return model
