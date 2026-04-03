from rules import find_rule_hits


def test_ad_rule_detects_contact_and_promotion() -> None:
    hits = find_rule_hits("加微信 abc8899 领取优惠券，手机号 13912345678")
    labels = {hit.label for hit in hits}
    assert "ad" in labels
    assert any(hit.rule_name == "phone_number" for hit in hits)


def test_safe_text_has_no_rule_hits() -> None:
    hits = find_rule_hits("这篇教程写得很细，适合刚入门的新手。")
    assert hits == []
