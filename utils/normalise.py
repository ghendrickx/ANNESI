"""
Normalisation object.

Author: Gijs G. Hendrickx
"""
from sklearn.preprocessing import StandardScaler

from utils import filing, path


class Normalise:
    """Normalise data based on the zero-mean-unit-variance scaling."""
    _scaler = None
    _scaler_is_fitted = False

    _file_name = 'scaler'
    _wd = path.DirConfig(__file__).config_dir('_data')

    def __init__(self, data=None, **kwargs):
        """When a data set is provided, the scaler is automatically fitted to this data set and internally saved; unless
        stated otherwise by the key-worded arguments (i.e. `kwargs`).

        :param data: data, defaults to None
        :param kwargs: optional arguments for `fit_scaler()`-method

        :type data: iterable[float]
        """
        if data is not None:
            self.fit_scaler(data, **kwargs)

        if kwargs.get('scaler'):
            self.scaler = kwargs.get('scaler')
        elif kwargs.get('file_scaler'):
            self.load_scaler(file_name=kwargs.get('file_scaler'), wd=kwargs.get('wd'))

    def __call__(self, data):
        """"Execute normalisation.

        :param data: data
        :type data: iterable[float]

        :return: normalised data
        :rtype: numpy.array
        """
        return self.exec(data)

    """Scaler-settings"""

    @property
    def scaler(self):
        """
        :return: scaler
        :rtype: BaseEstimator
        """
        if not self._scaler_is_fitted:
            try:
                self.load_scaler()
            except FileNotFoundError:
                msg = 'No scaler defined.'
                raise FileNotFoundError(msg)

        return self._scaler

    @scaler.setter
    def scaler(self, scaler_):
        if hasattr(scaler_, 'fit') and hasattr(scaler_, 'transform'):
            self._scaler = scaler_
        else:
            msg = f'Provided scaler-object does not comply with the requirements: ' \
                f'`fit()`-method = {hasattr(scaler_, "fit")}; `transform()`-method = {hasattr(scaler_, "transform")}'
            raise TypeError(msg)

    def load_scaler(self, file_name=None, wd=None):
        """Load scaler.

        :param file_name: file-name, defaults to None
        :param wd: working directory, defaults to None

        :type file_name: str, optional
        :type wd: DirConfig, str, iterable, optional
        """
        wd_ = wd or self._wd
        file_name_ = file_name or self._file_name

        self._scaler = filing.Import(wd_).from_gz(file_name=file_name_)
        self._scaler_is_fitted = True

    def save_scaler(self, file_name=None, wd=None):
        """Save scaler.

        :param file_name: file-name, defaults to None
        :param wd: working directory, defaults to None

        :type file_name: str, optional
        :type wd: DirConfig, str, iterable[str], defaults to None
        """
        wd_ = wd or self._wd
        file_name_ = file_name or self._file_name

        filing.Export(wd_).to_gz(self.scaler, file_name=file_name_)

    def fit_scaler(self, data, save=True, **kwargs):
        """Fit the scaler: zero-mean-unit-variance scaling.

        :param data: data
        :param save: save scaler, defaults to True
        :param kwargs: optional arguments for saving

        :type data: iterable[float]
        :type save: bool, optional
        """
        # set scaler-object
        if self._scaler is None:
            self.scaler = kwargs.get('scaler', StandardScaler())

        # fit scaler
        self._scaler.fit(data)
        self._scaler_is_fitted = True

        # save scaler
        if save:
            self.save_scaler(file_name=kwargs.get('file_name'), wd=kwargs.get('wd'))

    """Execute normalisation"""

    def exec(self, data):
        """Execute normalisation.

        :param data: data
        :type data: iterable[float]

        :return: normalised data
        :rtype: numpy.array
        """
        return self.scaler.transform(data)

    """Reverse normalisation"""

    def reverse(self, data):
        """Reverse normalisation.

        :param data: normalised data
        :type data: iterable[float]

        :return: de-normalised data
        :rtype: numpy.ndarray
        """
        return self.scaler.inverse_transform(data)


class NormalisedData:
    """Normalised data set based on pre-fitted scaler."""

    def __init__(self, data, **kwargs):
        """
        :param data: data
        :param kwargs: optional arguments to `.Normalise()`

        :type data: iterable[float]
        """
        self._norm = Normalise(**kwargs)
        self._data = self._norm(data)

    @property
    def data(self):
        """
        :return: normalised data
        :rtype: numpy.ndarray
        """
        return self._data

    def de_normalise(self, data):
        """De-normalise data using the inverse normalisation method.

        :param data: normalised data
        :type data: iterable[float]

        :return: de-normalised data
        :rtype: numpy.ndarray
        """
        return self._norm.reverse(data)
