from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class RuleDefinition:
    label: str
    rule_name: str
    pattern: re.Pattern[str]
    severity: float
    score_boost: float
    reason: str


@dataclass(frozen=True)
class RuleHit:
    label: str
    rule_name: str
    matched_text: str
    severity: float
    score_boost: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


RULES: list[RuleDefinition] = [
    RuleDefinition(
        label="ad",
        rule_name="contact_wechat_or_qq",
        pattern=re.compile(
            r"(?:微\s*信|v\s*x|v\s*信|wx|加\s*v|加\s*微|q\s*q|扣\s*扣)"
            r"(?:\s*[:：]?\s*[a-zA-Z0-9_-]{3,}|\s*(?:联系|咨询|私聊|进群|留一下))",
            re.IGNORECASE,
        ),
        severity=0.92,
        score_boost=0.28,
        reason="命中联系方式引流模式",
    ),
    RuleDefinition(
        label="ad",
        rule_name="phone_number",
        pattern=re.compile(r"1[3-9]\d{9}"),
        severity=0.93,
        score_boost=0.30,
        reason="命中手机号联系方式",
    ),
    RuleDefinition(
        label="ad",
        rule_name="promotion_keywords",
        pattern=re.compile(
            r"(优惠券|优惠价|返利|刷单|兼职|代理|加盟|引流|批发价|接急单|代写|拉人|涨赞|买粉|清仓|回本|接单赚钱|代发|推文|日结|日撸|挂机|小红书|快手认证|视频号)",
            re.IGNORECASE,
        ),
        severity=0.78,
        score_boost=0.20,
        reason="命中广告营销关键词",
    ),
    RuleDefinition(
        label="sexual",
        rule_name="sexual_invitation",
        pattern=re.compile(
            r"(约会|约吗|约炮|裸聊|私密照|福利视频|成人资源|成人视频|擦边|暧昧聊天|果照|原味内衣|黄游|瑟瑟|色视频|禁漫|91)",
            re.IGNORECASE,
        ),
        severity=0.88,
        score_boost=0.25,
        reason="命中低俗色情邀约表达",
    ),
    RuleDefinition(
        label="sexual",
        rule_name="private_chat_induction",
        pattern=re.compile(
            r"(私聊看图|私信看图|加我看图|单独约|懂的来|晚上更精彩|私密照片|通过一下|加我一下|自己扣)",
            re.IGNORECASE,
        ),
        severity=0.80,
        score_boost=0.18,
        reason="命中私聊诱导表达",
    ),
    RuleDefinition(
        label="abuse",
        rule_name="abuse_keywords",
        pattern=re.compile(
            r"(闭嘴|脑子进水|恶心人|滚远点|垃圾观点|丢人|蠢货|蠢猪|蠢|没人想理你|野种|有病|没教养|傻雕|傻子|丑东西|老不死)",
            re.IGNORECASE,
        ),
        severity=0.76,
        score_boost=0.18,
        reason="命中辱骂攻击词",
    ),
]


def find_rule_hits(text: str) -> list[RuleHit]:
    hits: list[RuleHit] = []
    seen: set[tuple[str, str, str]] = set()

    for rule in RULES:
        for match in rule.pattern.finditer(text):
            matched_text = match.group(0)
            key = (rule.label, rule.rule_name, matched_text)
            if key in seen:
                continue
            seen.add(key)
            hits.append(
                RuleHit(
                    label=rule.label,
                    rule_name=rule.rule_name,
                    matched_text=matched_text,
                    severity=rule.severity,
                    score_boost=rule.score_boost,
                    reason=rule.reason,
                )
            )

    return hits


def max_rule_severity(hits: Iterable[RuleHit]) -> float:
    return max((hit.severity for hit in hits), default=0.0)
