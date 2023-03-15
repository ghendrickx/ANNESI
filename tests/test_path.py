"""
Tests addressing the functioning of the code for defining files and directories, i.e. testing `utils.path.py`.

Author: Gijs G. Hendrickx
"""
import os

import pytest

from utils import path

"""pytest.fixtures"""


@pytest.fixture
def save_init_dir_config():
    def dir_config(*home_dir):
        return path.DirConfig(*home_dir, create_dir=False)
    return dir_config


"""TestClasses"""


class TestDirConfig:
    """Tests for the DirConfig-object."""

    def setup_method(self):
        """Initiate standard absolute and relative directories, which are considered the true result."""
        self.abs_dir = os.sep.join(['C:', 'folder1', 'folder2'])
        self.rel_dir = os.sep.join([os.getcwd(), 'folder1', 'folder2'])

    """Setting home-directory at object-level"""

    def test_default_init(self, save_init_dir_config):
        wd = save_init_dir_config()
        assert str(wd) == os.getcwd()

    def test_init_relative(self, save_init_dir_config):
        wd = save_init_dir_config('folder1', 'folder2')
        assert str(wd) == self.rel_dir

    def test_init_absolute(self, save_init_dir_config):
        wd = save_init_dir_config('C:', 'folder1', 'folder2')
        assert str(wd) == self.abs_dir

    def test_init_self(self, save_init_dir_config):
        wd = save_init_dir_config(save_init_dir_config('C:', 'folder1', 'folder2'))
        assert str(wd) == self.abs_dir

    """Configuring directory relative to home-directory"""

    def test_config_dir_relative_init(self, save_init_dir_config):
        wd = save_init_dir_config().config_dir('folder1', 'folder2')
        assert str(wd) == self.rel_dir

    def test_config_dir_absolute_init(self, save_init_dir_config):
        wd = save_init_dir_config('C:').config_dir('folder1', 'folder2')
        assert str(wd) == self.abs_dir

    def test_config_dir_list_abs_init(self, save_init_dir_config):
        wd = save_init_dir_config('C:').config_dir(['folder1', 'folder2'])
        assert str(wd) == self.abs_dir

    def test_config_dir_tuple_abs_init(self, save_init_dir_config):
        wd = save_init_dir_config('C:').config_dir(('folder1', 'folder2'))
        assert str(wd) == self.abs_dir

    def test_config_dir_str_abs_init(self, save_init_dir_config):
        wd = save_init_dir_config('C:').config_dir('folder1/folder2')
        assert str(wd) == self.abs_dir

    def test_config_dir_self_abs_init(self, save_init_dir_config):
        wd = save_init_dir_config(save_init_dir_config('C:')).config_dir('folder1', 'folder2')
        assert str(wd) == self.abs_dir

    def test_config_dir_skip_empty_init(self, save_init_dir_config):
        wd = save_init_dir_config().config_dir('C:', 'folder1', 'folder2')
        assert str(wd) == self.abs_dir

    def test_config_dir_skip_rel_init(self, save_init_dir_config):
        wd = save_init_dir_config('another', 'folder').config_dir('C:', 'folder1', 'folder2')
        assert str(wd) == self.abs_dir

    def test_config_dir_skip_abs_init(self, save_init_dir_config):
        wd = save_init_dir_config('D:', 'another', 'folder').config_dir('C:', 'folder1', 'folder2')
        assert str(wd) == self.abs_dir

    """Complex input combinations: __init__()"""

    def test_abs_init_combined_list_str(self, save_init_dir_config):
        wd = save_init_dir_config('C:', ['folder1', 'folder2'])
        assert str(wd) == self.abs_dir

    def test_abs_init_combined_tuple_str(self, save_init_dir_config):
        wd = save_init_dir_config('C:', ('folder1', 'folder2'))
        assert str(wd) == self.abs_dir

    def test_abs_init_combined_list_tuple(self, save_init_dir_config):
        wd = save_init_dir_config(['C:'], ('folder1', 'folder2'))
        assert str(wd) == self.abs_dir

    def test_abs_init_combined_self_str(self, save_init_dir_config):
        wd = save_init_dir_config(save_init_dir_config('C:'), 'folder1', 'folder2')
        assert str(wd) == self.abs_dir

    def test_abs_init_combined_self_list(self, save_init_dir_config):
        wd = save_init_dir_config(save_init_dir_config('C:'), ['folder1', 'folder2'])
        assert str(wd) == self.abs_dir

    """Combined input combinations: config_dir()"""

    def test_config_dir_combined_list_str(self, save_init_dir_config):
        wd = save_init_dir_config('C:').config_dir(['folder1'], 'folder2')
        assert str(wd) == self.abs_dir

    def test_config_dir_combined_tuple_str(self, save_init_dir_config):
        wd = save_init_dir_config('C:').config_dir(('folder1',), 'folder2')
        assert str(wd) == self.abs_dir

    def test_config_dir_combined_list_tuple(self, save_init_dir_config):
        wd = save_init_dir_config().config_dir(['C:'], ('folder1', 'folder2'))
        assert str(wd) == self.abs_dir

    def test_config_dir_combined_self_str(self, save_init_dir_config):
        wd = save_init_dir_config().config_dir(save_init_dir_config('C:'), 'folder1', 'folder2')
        assert str(wd) == self.abs_dir
