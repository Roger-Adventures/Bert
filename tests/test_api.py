from fastapi.testclient import TestClient

from serve_api import create_app


def test_predict_endpoint_returns_decision_fields() -> None:
    client = TestClient(create_app(model_dir=None))
    response = client.post("/predict", json={"text": "加微信领取优惠券，今天最后一天。"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "ad"
    assert "risk_band" in payload
    assert "threshold_reason" in payload
    assert "reasons" in payload


def test_index_route_serves_demo_page() -> None:
    client = TestClient(create_app(model_dir=None))
    response = client.get("/")
    assert response.status_code == 200
    html = response.content.decode("utf-8")
    assert "中文内容审核决策原型" in html
    assert "系统概览" in html
    assert "讲解模式" not in html


def test_docs_and_openapi_routes_are_available() -> None:
    client = TestClient(create_app(model_dir=None))
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_health_includes_environment_fields() -> None:
    client = TestClient(create_app(model_dir=None))
    payload = client.get("/health").json()
    assert "python_executable" in payload
    assert "python_version" in payload
    assert "torch_available" in payload
    assert "transformers_available" in payload
    assert payload["docs_url"] == "/docs"
