"""
Tests addressing the functioning of code for importing and exporting of data, i.e. testing `utils.filing.py`.

Author: Gijs G. Hendrickx
"""
import os
import time

import pytest

from utils import filing, path

"""pytest.fixtures"""


@pytest.fixture
def default_file_name_():
    def auto_file_name(default, extension=None):
        return filing.default_file_name(file_name='file', default=default, extension=extension)
    return auto_file_name


"""TestClasses"""


class TestDefaultFileName:
    """Tests for the `utils.filing.default_file_name()`-method."""

    def test_default(self):
        file_name = filing.default_file_name(file_name=None, default='default.txt')
        assert file_name == 'default.txt'

    def test_auto_ext(self, default_file_name_):
        file_name = default_file_name_(default='default.txt')
        assert file_name == 'file.txt'

    def test_manual_ext(self, default_file_name_):
        file_name = default_file_name_(default='default_ext.txt', extension='_ext.txt')
        assert file_name == 'file_ext.txt'

    def test_manual_ext_wrong_usage(self, default_file_name_):
        file_name = default_file_name_(default='default_ext.txt')
        assert file_name == 'file.txt'

    def test_auto_ext_double(self):
        file_name = filing.default_file_name(file_name='file.txt', default='default.txt')
        assert file_name == 'file.txt'

    def test_manual_ext_double(self):
        file_name = filing.default_file_name(file_name='file_ext.txt', default='default_ext.txt', extension='_ext.txt')
        assert file_name == 'file_ext.txt'

    def test_auto_ext_xyz(self, default_file_name_):
        file_name = default_file_name_(default='default.csv')
        assert file_name == 'file.csv'

    def test_auto_double_ext(self):
        file_name = filing.default_file_name(file_name='file.txt', default='default.csv')
        assert file_name == 'file.csv'


class TestImport:
    """Tests for the `utils.filing.Import`-class."""

    def setup_method(self):
        self._import = filing.Import(__file__.split(os.sep)[:-2])

    """Import data methods"""

    def test_from_csv(self):
        with pytest.raises(FileNotFoundError):
            self._import.from_csv()

    def test_from_gz(self):
        from sklearn.preprocessing import MinMaxScaler

        scaler = self._import.from_gz(file_name='utils/_data/annesi.gz')
        assert isinstance(scaler, MinMaxScaler)

    def test_from_pkl(self):
        from torch.nn import Module
        from src._backend import OldMLP

        model = self._import.from_pkl(OldMLP(13, 2), file_name='src/_data/annesi.pkl')
        assert isinstance(model, Module)


class TestExport:
    """Tests for the `utils.filing.Export`-class."""

    def setup_method(self):
        self.export = filing.Export(__file__)
        self.path = path.DirConfig(__file__)

    """Export data methods"""

    def test_to_csv(self):
        data = dict(a=[1, 2], b=[4, 0])
        self.export.to_csv(data)
        self.path.delete_file('output.csv')

    def test_to_csv_custom(self):
        data = dict(a=[1, 2], b=[4, 0])
        self.export.to_csv(data, file_name='data.csv')
        self.path.delete_file('data.csv')
