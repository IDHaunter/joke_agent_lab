from fastapi.testclient import TestClient

from backend.app.api import routes
from backend.app.main import app

client = TestClient(app)
VALID_SESSION_ID = "a1f88a92-8bd7-4a22-9a9e-e079d1eeb84d"


def test_health_does_not_initialize_ollama_runtime() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_agent_answer(monkeypatch) -> None:
    def fake_agent(session_id, message):
        assert str(session_id) == VALID_SESSION_ID
        assert message == "Tell me a joke"
        return "A test joke."

    monkeypatch.setattr(routes, "run_agent", fake_agent)

    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": VALID_SESSION_ID,
            "message": "  Tell me a joke  ",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": VALID_SESSION_ID,
        "answer": "A test joke.",
    }


def test_chat_rejects_whitespace_only_message() -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": VALID_SESSION_ID,
            "message": "   ",
        },
    )

    assert response.status_code == 422


def test_chat_rejects_invalid_session_id() -> None:
    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": "not-a-uuid",
            "message": "Hello",
        },
    )

    assert response.status_code == 422


def test_reset_endpoint_calls_agent_reset(monkeypatch) -> None:
    received = {}

    def fake_reset(session_id):
        received["session_id"] = str(session_id)

    monkeypatch.setattr(routes, "reset_agent_chat", fake_reset)

    response = client.post(
        "/api/v1/reset",
        json={"session_id": VALID_SESSION_ID},
    )

    assert response.status_code == 200
    assert response.json() == {
        "session_id": VALID_SESSION_ID,
        "status": "reset",
    }
    assert received["session_id"] == VALID_SESSION_ID