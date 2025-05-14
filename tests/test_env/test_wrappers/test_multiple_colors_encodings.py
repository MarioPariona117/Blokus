import pytest
import torch
import numpy as np

from gymnasium_env import BlokusEnv, BlokusAction
from gymnasium_env.wrappers import MultipleColorsEncoding

BOARD_SIZE = 6

@pytest.fixture
def mock_env():
    """Create a mock BlokusEnv for testing."""
    return BlokusEnv(
        board_size=BOARD_SIZE,
        num_players=2,
    )

@pytest.fixture
def wrapper(mock_env):
    """Create a MultipleColorsEncoding wrapper for testing."""
    return MultipleColorsEncoding(mock_env)

def test_reset(wrapper):
    """Test the reset method of the wrapper."""
    obs, info = wrapper.reset()
    assert "encoding" in obs
    assert isinstance(obs["encoding"], torch.Tensor)
    assert obs["encoding"].shape == (wrapper.total_channels, wrapper.env.board_size, wrapper.env.board_size)
    assert torch.all(obs["encoding"] == 0)

def test_step(wrapper):
    """Test the step method of the wrapper."""
    wrapper.reset()
    action_id = 0  # Mock action ID
    obs, reward, term, trunc, info = wrapper.step(action_id)
    assert "encoding" in obs
    # Ensure encoding is updated for the current player
    assert isinstance(obs["encoding"], torch.Tensor)
    assert obs["encoding"].shape == (wrapper.total_channels, wrapper.env.board_size, wrapper.env.board_size)
    non_zero_indices = torch.nonzero(obs["encoding"])
    assert len(non_zero_indices) == 1
    assert tuple(non_zero_indices[0].tolist()) == (0, 0, 0)

def test_attribute_forwarding(wrapper):
    """Test that missing attributes are forwarded to the original environment."""
    assert wrapper.board_size == wrapper.env.board_size
    assert wrapper.num_players == wrapper.env.num_players