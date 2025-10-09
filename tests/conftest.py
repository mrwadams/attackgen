"""Shared pytest fixtures for AttackGen tests."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing.

    Yields:
        Path: Temporary directory path that will be cleaned up after test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Path:
    """Create a temporary file for testing.

    Args:
        temp_dir: Temporary directory fixture.

    Returns:
        Path: Temporary file path.
    """
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("test content")
    return temp_file


@pytest.fixture
def mock_env_vars(monkeypatch) -> Dict[str, str]:
    """Create mock environment variables for testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Dict[str, str]: Dictionary of mock environment variables.
    """
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GOOGLE_API_KEY": "test-google-key",
        "MISTRAL_API_KEY": "test-mistral-key",
        "GROQ_API_KEY": "test-groq-key",
        "LANGCHAIN_API_KEY": "test-langchain-key",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def mock_api_key(monkeypatch) -> str:
    """Create a mock API key for testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        str: Mock API key.
    """
    api_key = "test-api-key-12345"
    monkeypatch.setenv("TEST_API_KEY", api_key)
    return api_key


@pytest.fixture
def sample_mitre_technique() -> Dict[str, str]:
    """Create a sample MITRE ATT&CK technique for testing.

    Returns:
        Dict[str, str]: Sample technique data.
    """
    return {
        "id": "T1566",
        "name": "Phishing",
        "description": "Adversaries may send phishing messages to gain access to victim systems.",
        "tactic": "Initial Access",
    }


@pytest.fixture
def sample_threat_group() -> Dict[str, str]:
    """Create a sample threat actor group for testing.

    Returns:
        Dict[str, str]: Sample threat group data.
    """
    return {
        "id": "G0016",
        "name": "APT29",
        "description": "APT29 is a threat group that has been attributed to the Russian government.",
        "aliases": ["Cozy Bear", "The Dukes"],
    }


@pytest.fixture
def sample_organization_config() -> Dict[str, str]:
    """Create sample organization configuration for testing.

    Returns:
        Dict[str, str]: Sample organization configuration.
    """
    return {
        "industry": "Financial Services",
        "size": "Large (1000+ employees)",
        "attack_matrix": "Enterprise",
    }


@pytest.fixture
def mock_streamlit_session_state(mocker):
    """Mock Streamlit session state for testing.

    Args:
        mocker: Pytest-mock mocker fixture.

    Returns:
        Mock: Mocked session state object.
    """
    mock_state = {}
    return mocker.patch("streamlit.session_state", mock_state)


@pytest.fixture
def disable_streamlit_caching(mocker):
    """Disable Streamlit caching decorators for testing.

    Args:
        mocker: Pytest-mock mocker fixture.
    """
    # Mock cache decorators to act as pass-through
    def identity_decorator(func):
        return func

    mocker.patch("streamlit.cache_data", identity_decorator)
    mocker.patch("streamlit.cache_resource", identity_decorator)
