from pipeline import ModerationPipeline


def test_rules_only_pipeline_flags_ad_text() -> None:
    pipeline = ModerationPipeline(model_dir=None)
    result = pipeline.predict("wx 联系，优惠价只限今天，想做代理私聊我")
    assert result["label"] == "ad"
    assert result["action"] == "block"
    assert result["risk_band"] == "high"
    assert result["threshold_reason"]
    assert result["source"] == "rules"


def test_rules_only_pipeline_allows_normal_text() -> None:
    pipeline = ModerationPipeline(model_dir=None)
    result = pipeline.predict("今天把周报写完了，准备下班回家。")
    assert result["label"] == "normal"
    assert result["action"] == "allow"
    assert result["risk_band"] == "low"


def test_rules_only_pipeline_reviews_soft_sexual_text() -> None:
    pipeline = ModerationPipeline(model_dir=None)
    result = pipeline.predict("运营复盘里专门提到“擦边”判定边界，例子其实是普通穿搭视频。")
    assert result["label"] == "sexual"
    assert result["action"] == "review"
    assert result["source"] == "rules"


def test_pipeline_aggregates_multiple_rule_hits_into_one_reason() -> None:
    pipeline = ModerationPipeline(model_dir=None)
    result = pipeline.predict("wx 联系，优惠价只限今天，想做代理私聊我。")

    grouped_ad_reasons = [reason for reason in result["reasons"] if reason.startswith("广告引流：")]

    assert len(grouped_ad_reasons) == 1
    assert "联系方式引流模式" in grouped_ad_reasons[0]
    assert "广告营销关键词" in grouped_ad_reasons[0]
    assert "wx 联系" in grouped_ad_reasons[0]
    assert "优惠价" in grouped_ad_reasons[0]
    assert len(result["rule_hits"]) >= 2


def test_pipeline_has_a_stable_review_example_for_demo() -> None:
    pipeline = ModerationPipeline(model_dir=None)
    result = pipeline.predict("最近有人复盘“代理”话术被滥用的案例，主题更像治理讨论。")

    assert result["action"] == "review"
    assert result["label"] == "ad"


def test_probabilities_remain_model_reference_scores() -> None:
    pipeline = ModerationPipeline(model_dir=None)
    result = pipeline.predict("最近有人复盘“代理”话术被滥用的案例，主题更像治理讨论。")

    assert result["probabilities"]["normal"] == 0.78
    assert result["probabilities"]["ad"] == 0.08
    assert result["label"] == "ad"
    assert result["action"] == "review"
