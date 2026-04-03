from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from labels import (
    ACTION_DISPLAY_NAME,
    ID_TO_LABEL,
    LABEL_DISPLAY_NAME,
    LABELS,
    RISK_BAND_DISPLAY_NAME,
)
from rules import RuleHit, find_rule_hits, max_rule_severity

try:
    import torch
except ImportError:  # pragma: no cover - runtime fallback
    torch = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover - runtime fallback
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


@dataclass
class Decision:
    action: str
    action_name: str
    risk_band: str
    risk_band_name: str
    threshold_reason: str


@dataclass
class ModerationResult:
    text: str
    label: str
    label_name: str
    risk_score: float
    risk_band: str
    risk_band_name: str
    action: str
    action_name: str
    threshold_reason: str
    source: str
    model_loaded: bool
    model_confidence: float
    probabilities: dict[str, float]
    rule_hits: list[dict[str, object]]
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["risk_score"] = round(self.risk_score, 4)
        payload["model_confidence"] = round(self.model_confidence, 4)
        payload["probabilities"] = {
            label: round(score, 4) for label, score in self.probabilities.items()
        }
        return payload


class ModerationPipeline:
    def __init__(self, model_dir: str | Path | None = "artifacts/moderation_macbert") -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.load_error: str | None = None
        self.policy_version = "v1.1"

        if torch is not None and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self._try_load_model()

    def _try_load_model(self) -> None:
        if self.model_dir is None or not self.model_dir.exists():
            self.load_error = "model directory not found, using rules fallback"
            return

        if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
            self.load_error = "transformers or torch not installed, using rules fallback"
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.load_error = f"failed to load model: {exc}"
            self.model_loaded = False
            self.model = None
            self.tokenizer = None

    def predict(self, text: str) -> dict[str, object]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("text must not be empty")

        rule_hits = find_rule_hits(clean_text)
        model_probs, model_confidence = self._predict_with_model(clean_text)
        combined_scores = self._combine_scores(model_probs, rule_hits)
        label = max(combined_scores, key=combined_scores.get)
        risk_score = 1.0 - combined_scores["normal"]
        decision = self._make_decision(label, risk_score, model_confidence, rule_hits)
        source = self._build_source(rule_hits)
        reasons = self._build_reasons(label, model_confidence, rule_hits, decision)

        result = ModerationResult(
            text=clean_text,
            label=label,
            label_name=LABEL_DISPLAY_NAME[label],
            risk_score=risk_score,
            risk_band=decision.risk_band,
            risk_band_name=decision.risk_band_name,
            action=decision.action,
            action_name=decision.action_name,
            threshold_reason=decision.threshold_reason,
            source=source,
            model_loaded=self.model_loaded,
            model_confidence=model_confidence,
            probabilities=combined_scores,
            rule_hits=[hit.to_dict() for hit in rule_hits],
            reasons=reasons,
        )
        return result.to_dict()

    def batch_predict(self, texts: list[str]) -> list[dict[str, object]]:
        return [self.predict(text) for text in texts]

    def _predict_with_model(self, text: str) -> tuple[dict[str, float], float]:
        if not self.model_loaded or self.model is None or self.tokenizer is None or torch is None:
            base_probs = {"normal": 0.78, "abuse": 0.08, "sexual": 0.06, "ad": 0.08}
            return base_probs, 0.0

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).tolist()

        probabilities = {ID_TO_LABEL[idx]: float(score) for idx, score in enumerate(probs)}
        return probabilities, float(max(probs))

    def _combine_scores(
        self,
        model_probs: dict[str, float],
        rule_hits: list[RuleHit],
    ) -> dict[str, float]:
        combined = {label: float(model_probs.get(label, 0.0)) for label in LABELS}
        max_rule_by_label = {label: 0.0 for label in LABELS}
        boost_by_label = {label: 0.0 for label in LABELS}

        for hit in rule_hits:
            max_rule_by_label[hit.label] = max(max_rule_by_label[hit.label], hit.severity)
            boost_by_label[hit.label] += hit.score_boost

        for label in LABELS:
            boosted = min(0.99, combined[label] + boost_by_label[label])
            combined[label] = max(combined[label], boosted, max_rule_by_label[label])

        if rule_hits and combined["normal"] > 0.70:
            combined["normal"] = max(0.05, combined["normal"] - 0.20)

        total = sum(combined.values())
        if total <= 0:
            return {label: 0.0 for label in LABELS}

        return {label: score / total for label, score in combined.items()}

    def _make_decision(
        self,
        label: str,
        risk_score: float,
        model_confidence: float,
        rule_hits: list[RuleHit],
    ) -> Decision:
        highest_rule = max_rule_severity(rule_hits)
        low_confidence = self.model_loaded and model_confidence < 0.55

        if label == "normal" and highest_rule < 0.88 and risk_score < 0.35:
            return Decision(
                action="allow",
                action_name=ACTION_DISPLAY_NAME["allow"],
                risk_band="low",
                risk_band_name=RISK_BAND_DISPLAY_NAME["low"],
                threshold_reason="低风险样本：无强规则命中且综合风险分较低，允许放行。",
            )

        if label in {"sexual", "ad"} and (risk_score >= 0.72 or highest_rule >= 0.90):
            return Decision(
                action="block",
                action_name=ACTION_DISPLAY_NAME["block"],
                risk_band="high",
                risk_band_name=RISK_BAND_DISPLAY_NAME["high"],
                threshold_reason="高风险样本：命中高危类别且风险分或规则强度达到拦截阈值，直接拦截。",
            )

        if label == "abuse" and (risk_score >= 0.68 or highest_rule >= 0.85):
            return Decision(
                action="block",
                action_name=ACTION_DISPLAY_NAME["block"],
                risk_band="high",
                risk_band_name=RISK_BAND_DISPLAY_NAME["high"],
                threshold_reason="高风险样本：辱骂攻击强度较高，命中拦截阈值，直接拦截。",
            )

        if risk_score >= 0.45 or highest_rule >= 0.75 or low_confidence:
            reason = "中风险样本："
            if low_confidence:
                reason += "模型置信度偏低，进入人工审核。"
            elif highest_rule >= 0.75:
                reason += "命中规则但未达到直接拦截阈值，进入人工审核。"
            else:
                reason += "综合风险分处于审核区间，进入人工审核。"
            return Decision(
                action="review",
                action_name=ACTION_DISPLAY_NAME["review"],
                risk_band="medium",
                risk_band_name=RISK_BAND_DISPLAY_NAME["medium"],
                threshold_reason=reason,
            )

        return Decision(
            action="allow",
            action_name=ACTION_DISPLAY_NAME["allow"],
            risk_band="low",
            risk_band_name=RISK_BAND_DISPLAY_NAME["low"],
            threshold_reason="低风险样本：未命中强规则，模型风险分较低，允许放行。",
        )

    def _build_source(self, rule_hits: list[RuleHit]) -> str:
        if self.model_loaded and rule_hits:
            return "model+rules"
        if self.model_loaded:
            return "model"
        return "rules"

    def _build_reasons(
        self,
        label: str,
        model_confidence: float,
        rule_hits: list[RuleHit],
        decision: Decision,
    ) -> list[str]:
        reasons = [f"主标签为 {LABEL_DISPLAY_NAME[label]}。"]
        reasons.append(f"最终动作：{decision.action_name}，风险等级为 {decision.risk_band_name}。")

        if self.model_loaded:
            reasons.append(f"模型置信度为 {model_confidence:.2f}。")
        else:
            reasons.append("当前未加载微调模型，使用规则兜底策略。")

        for hit in rule_hits[:3]:
            reasons.append(f"{LABEL_DISPLAY_NAME[hit.label]}：{hit.reason} -> {hit.matched_text}")

        return reasons


if __name__ == "__main__":
    pipeline = ModerationPipeline()
    sample_texts = [
        "加微信领取优惠券，今天最后一天",
        "这次更新后加载速度提升明显",
        "你这人说话真欠，没人想理你",
    ]

    for text in sample_texts:
        print(json.dumps(pipeline.predict(text), ensure_ascii=False, indent=2))
