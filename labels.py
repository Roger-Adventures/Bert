from __future__ import annotations

LABELS = ["normal", "abuse", "sexual", "ad"]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

LABEL_DISPLAY_NAME = {
    "normal": "正常内容",
    "abuse": "辱骂攻击",
    "sexual": "低俗色情",
    "ad": "广告引流",
}

ACTION_DISPLAY_NAME = {
    "allow": "放行",
    "review": "人审",
    "block": "拦截",
}

RISK_BAND_DISPLAY_NAME = {
    "low": "低风险",
    "medium": "中风险",
    "high": "高风险",
}
