from pipeline import ModerationPipeline


def test_rules_only_pipeline_flags_ad_text() -> None:
    pipeline = ModerationPipeline(model_dir=None)
    result = pipeline.predict("wx 联系，优惠价只限今天，想做代理私聊我")
    assert result["label"] == "ad"
    assert result["action"] in {"review", "block"}
    assert result["risk_band"] in {"medium", "high"}
    assert result["threshold_reason"]


def test_rules_only_pipeline_allows_normal_text() -> None:
    pipeline = ModerationPipeline(model_dir=None)
    result = pipeline.predict("今天把周报写完了，准备下班回家。")
    assert result["label"] == "normal"
    assert result["action"] == "allow"
    assert result["risk_band"] == "low"


def test_rules_only_pipeline_reviews_soft_sexual_text() -> None:
    pipeline = ModerationPipeline(model_dir=None)
    result = pipeline.predict("主页里总在发擦边写真，最好让人工再看一下。")
    assert result["label"] == "sexual"
    assert result["action"] in {"review", "block"}
    assert result["source"] == "rules"
