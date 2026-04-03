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
    assert "中文内容审核策略原型" in html
    assert "moderation strategy prototype" in html
    assert "审核输入" in html
    assert "审核结果" in html
    assert "验证结果" in html
    assert "处置结论" in html
    assert "模型参考概率" in html
    assert "/static/styles.css" in html
    assert "/static/app.js" in html
    assert "讲解模式" not in html
    assert "系统状态与调试入口" not in html


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


def test_demo_content_json_is_served_with_expected_categories() -> None:
    client = TestClient(create_app(model_dir=None))
    response = client.get("/static/demo_content.json")
    assert response.status_code == 200

    payload = response.json()
    assert payload["default_sample_id"] == "ad_review_1"
    assert sorted(group["id"] for group in payload["sample_groups"]) == ["abuse", "ad", "normal", "sexual"]
    assert len(payload["sample_groups"]) == 4
    assert {group["id"]: len(group["items"]) for group in payload["sample_groups"]} == {
        "normal": 2,
        "abuse": 3,
        "sexual": 3,
        "ad": 3,
    }
    assert len(payload["cases"]) == 7
    assert all("group_label" in item for item in payload["cases"])
    assert all("text" in item for item in payload["cases"])


def test_static_assets_for_demo_page_are_served() -> None:
    client = TestClient(create_app(model_dir=None))
    assert client.get("/static/styles.css").status_code == 200

    app_js = client.get("/static/app.js")
    assert app_js.status_code == 200
    assert "/static/demo_content.json" in app_js.text
    assert "中文内容审核策略原型" not in app_js.text
