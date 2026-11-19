import pytest

from audit_env.generator.episode import generate_episode


@pytest.fixture(scope="session")
def episodes():
    """A pool of generated episodes reused across tests (generation is deterministic)."""
    return {seed: generate_episode(seed) for seed in range(40)}


@pytest.fixture(scope="session")
def errored(episodes):
    return [ep for ep in episodes.values() if ep.truth["issues"]]


@pytest.fixture(scope="session")
def clean(episodes):
    return [ep for ep in episodes.values() if not ep.truth["issues"]]
