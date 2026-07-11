import os
import sys
import typing

# Patch Python 3.12 ForwardRef._evaluate compatibility with older Pydantic v1 / Langgraph
if sys.version_info >= (3, 12):
    original_evaluate = typing.ForwardRef._evaluate
    def patched_evaluate(self, globalns, localns, *args, **kwargs):
        if 'recursive_guard' in kwargs:
            if kwargs['recursive_guard'] is None:
                kwargs['recursive_guard'] = set()
            return original_evaluate(self, globalns, localns, *args, **kwargs)
            
        if len(args) == 1:
            rec_guard = args[0]
            kwargs['recursive_guard'] = rec_guard
            args = ((),)
            
        if 'recursive_guard' not in kwargs:
            kwargs['recursive_guard'] = set()
            
        return original_evaluate(self, globalns, localns, *args, **kwargs)
    typing.ForwardRef._evaluate = patched_evaluate

# Patch httpx Client/AsyncClient init compatibility with older Starlette TestClient
import httpx
import inspect
for client_cls in (httpx.Client, httpx.AsyncClient):
    original_init = client_cls.__init__
    def make_patched_init(orig_init):
        def patched_init(self, *args, **kwargs):
            if 'app' in kwargs:
                sig = inspect.signature(orig_init)
                if 'app' not in sig.parameters:
                    kwargs.pop('app')
            return orig_init(self, *args, **kwargs)
        return patched_init
    client_cls.__init__ = make_patched_init(original_init)

import asyncio
import inspect
import pytest
from unittest.mock import MagicMock, patch

# Ensure testing env vars are set
os.environ["TESTING"] = "true"
os.environ["STORAGE_MODE"] = "local"
os.environ["REDIS_URL"] = "memory://"
os.environ["API_KEY"] = "dev-secret-key-123" # Fixes test_api.py key mismatch
os.environ["INTERNAL_API_KEY"] = "internal-test-key"


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: run async tests with asyncio")


def pytest_pyfunc_call(pyfuncitem):
    test_function = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_function):
        return None

    fixture_names = pyfuncitem._fixtureinfo.argnames
    kwargs = {name: pyfuncitem.funcargs[name] for name in fixture_names}
    asyncio.run(test_function(**kwargs))
    return True

@pytest.fixture
def mock_redis():
    """Mock redis client to avoid connecting to real redis service."""
    mock_client = MagicMock()
    mock_client.ping.return_value = True
    mock_client.eval.return_value = '{"total_cases": 1, "agreements": 1, "disagreements": 0, "agreement_rate": 1.0}'
    mock_client.get.return_value = None
    
    with patch("redis.Redis") as mock_redis_class:
        mock_redis_class.from_url.return_value = mock_client
        mock_redis_class.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_minio():
    """Mock Minio client for S3 storage tests."""
    mock_client = MagicMock()
    mock_client.bucket_exists.return_value = True
    
    with patch("minio.Minio") as mock_minio_class:
        mock_minio_class.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_inference_api():
    """Mock calls to the external inference API service."""
    with patch("requests.post") as mock_post:
        # Default mock response for /encode/text, /estimate, etc.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [0.1] * 768,
            "prediction": [0],
            "mean_confidence": [0.85],
            "std_deviation": [0.03],
            "all_probs": [[0.85, 0.05, 0.05, 0.03, 0.02]]
        }
        mock_post.return_value = mock_response
        yield mock_post

@pytest.fixture(autouse=True)
def mock_open_clip_tokenizer():
    """Mock open_clip.get_tokenizer to raise an exception, avoiding network calls during tests."""
    with patch("open_clip.get_tokenizer", side_effect=RuntimeError("Mock network block for offline testing")):
        yield
