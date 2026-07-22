from squadpitch_ai.api.app import create_app
from squadpitch_ai.worker.app import Worker


def test_api_and_worker_import() -> None:
    assert create_app is not None
    assert Worker is not None
