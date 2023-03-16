"""
Tests addressing the validity of the input check, i.e. testing `utils.check.py`.

Author: Gijs G. Hendrickx
"""
import pytest

from utils import check

"""supporting functions"""


def check_input(data, key):
    """Search if `key` is in collection of error messages resulting from the input check of `data`.

    :param data: sample data
    :param key: key-word of a specific error message

    :type data: dict
    :type key: str

    :return: key-word in error messages
    :rtype: bool
    """
    # perform input check
    msg = check.input_check(**data)
    # search for key-word in error messages
    return any(key in item for item in msg)


"""pytest.fixtures"""


@pytest.fixture
def sample():
    return dict(
        tidal_range=3,
        surge_level=0,
        river_discharge=1000,
        channel_depth=15,
        channel_width=1000,
        channel_friction=.023,
        convergence=1e-4,
        flat_depth_ratio=0,
        flat_width=0,
        flat_friction=.023,
        bottom_curvature=0,
        meander_amplitude=0,
        meander_length=0
    )


"""TestClasses"""


class TestInputCheck:
    """Tests for the `utils.check.input_check()`-method."""

    def test_valid_input(self, sample):
        msg = check.input_check(**sample)
        assert msg == []

    def test_channel_depth(self, sample):
        sample.update({
            'river_discharge': 16e3,
            'channel_depth': 5,
        })
        assert check_input(sample, 'channel_depth')

    def test_channel_width(self, sample):
        sample.update({
            'river_discharge': 16e3,
            'channel_width': 500,
        })
        assert check_input(sample, 'channel_width')

    def test_flat_depth_ratio(self, sample):
        sample.update({
            'flat_depth_ratio': -2,
        })
        assert check_input(sample, 'flat_depth_ratio')

    def test_flat_depth(self, sample):
        sample.update({
            'tidal_range': 12,
            'channel_depth': 5,
            'flat_depth_ratio': 1,
        })
        assert check_input(sample, 'flat_depth')

    def test_flat_width(self, sample):
        sample.update({
            'channel_width': 500,
            'flat_width': 1000,
        })
        assert check_input(sample, 'flat_width_ratio')

    def test_bottom_curvature(self, sample):
        sample.update({
            'channel_width': 3000,
            'channel_depth': 5,
            'bottom_curvature': 1e-6,
        })
        assert check_input(sample, 'bottom_curvature')

    def test_meander_amplitude(self, sample):
        sample.update({
            'channel_width': 500,
            'flat_width': 0,
            'meander_amplitude': 2500,
        })
        check_input(sample, 'meander_amplitude')

    def test_meander_length_lower(self, sample):
        sample.update({
            'meander_amplitude': 500,
            'meander_length': 3500,
        })
        check_input(sample, 'meander_length')

    def test_meander_length_upper(self, sample):
        sample.update({
            'meander_amplitude': 500,
            'meander_length': 12700,
        })
        check_input(sample, 'meander_length')

    def test_flow_velocity(self, sample):
        sample.update({
            'river_discharge': 16e3,
            'channel_depth': 5,
        })
        assert check_input(sample, 'channel_depth')
