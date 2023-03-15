"""
Tests addressing the functioning of code for importing and exporting of data, i.e. testing `utils.filing.py`.

Author: Gijs G. Hendrickx
"""
import pytest

from utils import filing


"""pytest.fixtures"""


@pytest.fixture
def default_file_name_():
    def auto_file_name(default, extension=None):
        return filing.default_file_name(file_name='file', default=default, extension=extension)
    return auto_file_name


"""TestClasses"""


class TestDefaultFileName:
    """Tests for the _default_file_name-method."""

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


# TODO: Include tests for utils.filing.Import- and .Export-classes
