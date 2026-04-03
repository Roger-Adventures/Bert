from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from labels import (
    ACTION_DISPLAY_NAME,
    ID_TO_LABEL,
    LABEL_DISPLAY_NAME,
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


DEFAULT_MODEL_DIR = "artifacts/moderation_macbert"
DEFAULT_MODEL_PROBABILITIES = {
    "normal": 0.78,
    "abuse": 0.08,
    "sexual": 0.06,
    "ad": 0.08,
}
POLICY_VERSION = "v1.1"


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
    def __init__(self, model_dir: str | Path | None = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.load_error: str | None = None
        self.policy_version = POLICY_VERSION

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
        model_label = max(model_probs, key=model_probs.get)
        label = self._select_label(model_label, rule_hits)
        base_risk_score = 1.0 - model_probs["normal"]
        risk_score = self._combine_scores(base_risk_score, rule_hits)
        decision = self._make_decision(label, base_risk_score, model_confidence, rule_hits)
        source = self._build_source(rule_hits)
        reasons = self._build_reasons(model_label, label, model_confidence, rule_hits, decision)

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
            probabilities=model_probs,
            rule_hits=[hit.to_dict() for hit in rule_hits],
            reasons=reasons,
        )
        return result.to_dict()

    def _predict_with_model(self, text: str) -> tuple[dict[str, float], float]:
        if not self.model_loaded or self.model is None or self.tokenizer is None or torch is None:
            return DEFAULT_MODEL_PROBABILITIES, 0.0

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

    def _select_label(self, model_label: str, rule_hits: list[RuleHit]) -> str:
        strongest_rule_hit = self._strongest_rule_hit(rule_hits)
        if strongest_rule_hit is None:
            return model_label

        if strongest_rule_hit.severity >= 0.90:
            return strongest_rule_hit.label

        if model_label == "normal" and strongest_rule_hit.severity >= 0.75:
            return strongest_rule_hit.label

        return model_label

    def _combine_scores(self, base_risk_score: float, rule_hits: list[RuleHit]) -> float:
        highest_rule = max_rule_severity(rule_hits)
        if highest_rule >= 0.90:
            return max(base_risk_score, highest_rule)

        if highest_rule >= 0.75:
            return max(base_risk_score, 0.55)

        return base_risk_score

    def _build_decision(self, action: str, threshold_reason: str) -> Decision:
        risk_band = "low"
        if action == "review":
            risk_band = "medium"
        elif action == "block":
            risk_band = "high"

        return Decision(
            action=action,
            action_name=ACTION_DISPLAY_NAME[action],
            risk_band=risk_band,
            risk_band_name=RISK_BAND_DISPLAY_NAME[risk_band],
            threshold_reason=threshold_reason,
        )

    def _make_decision(
        self,
        label: str,
        base_risk_score: float,
        model_confidence: float,
        rule_hits: list[RuleHit],
    ) -> Decision:
        highest_rule = max_rule_severity(rule_hits)
        model_decision = self._make_model_decision(label, base_risk_score, model_confidence)

        if highest_rule >= 0.90:
            return self._build_decision(
                action="block",
                threshold_reason="强规则命中：内容命中高确定性风险模式，建议直接拦截。",
            )

        if model_decision.action == "block":
            return model_decision

        if highest_rule >= 0.75:
            return self._build_decision(
                action="review",
                threshold_reason="规则提示风险：内容命中风险模式但未达到强规则拦截标准，建议进入人工审核。",
            )

        if model_decision.action == "review":
            return model_decision

        return self._build_decision(
            action="allow",
            threshold_reason="模型判断为低风险内容，且未命中需要升级处理的规则，建议放行。",
        )

    def _make_model_decision(
        self,
        label: str,
        base_risk_score: float,
        model_confidence: float,
    ) -> Decision:
        low_confidence = self.model_loaded and model_confidence < 0.55

        if label == "normal" and base_risk_score < 0.35:
            return self._build_decision(
                action="allow",
                threshold_reason="模型判断为低风险内容，综合风险分较低，建议放行。",
            )

        if label in {"sexual", "ad"} and base_risk_score >= 0.72:
            return self._build_decision(
                action="block",
                threshold_reason=f"模型判断为高风险{LABEL_DISPLAY_NAME[label]}内容，且风险分达到拦截阈值，建议直接拦截。",
            )

        if label == "abuse" and base_risk_score >= 0.68:
            return self._build_decision(
                action="block",
                threshold_reason="模型判断为高风险辱骂攻击内容，且风险分达到拦截阈值，建议直接拦截。",
            )

        if low_confidence:
            return self._build_decision(
                action="review",
                threshold_reason="模型判断存在风险，但当前置信度偏低，建议进入人工审核。",
            )

        if base_risk_score >= 0.45:
            return self._build_decision(
                action="review",
                threshold_reason="模型判断内容存在风险，风险分进入人工审核区间，建议进入人工审核。",
            )

        return self._build_decision(
            action="allow",
            threshold_reason="模型判断为低风险内容，综合风险分较低，建议放行。",
        )

    def _build_source(self, rule_hits: list[RuleHit]) -> str:
        if self.model_loaded and rule_hits:
            return "model+rules"
        if self.model_loaded:
            return "model"
        return "rules"

    def _strongest_rule_hit(self, rule_hits: list[RuleHit]) -> RuleHit | None:
        if not rule_hits:
            return None
        return max(rule_hits, key=lambda hit: hit.severity)

    def _build_reasons(
        self,
        model_label: str,
        label: str,
        model_confidence: float,
        rule_hits: list[RuleHit],
        decision: Decision,
    ) -> list[str]:
        reasons = [f"模型主判断：{LABEL_DISPLAY_NAME[model_label]}。"]
        reasons.append(
            f"最终标签：{LABEL_DISPLAY_NAME[label]}；审核动作：{decision.action_name}，风险等级为 {decision.risk_band_name}。"
        )

        if self.model_loaded:
            reasons.append(f"模型置信度：{model_confidence:.2f}。")
        else:
            reasons.append("当前未加载微调模型，使用规则兜底和默认模型概率。")

        reasons.extend(self._build_grouped_rule_reasons(rule_hits))

        return reasons

    def _build_grouped_rule_reasons(self, rule_hits: list[RuleHit]) -> list[str]:
        if not rule_hits:
            return []

        grouped: dict[str, dict[str, list[str]]] = {}
        for hit in rule_hits:
            if hit.label not in grouped:
                grouped[hit.label] = {"reasons": [], "matched_texts": []}

            normalized_reason = hit.reason.removeprefix("命中")
            if normalized_reason not in grouped[hit.label]["reasons"]:
                grouped[hit.label]["reasons"].append(normalized_reason)
            if hit.matched_text not in grouped[hit.label]["matched_texts"]:
                grouped[hit.label]["matched_texts"].append(hit.matched_text)

        summaries: list[str] = []
        for grouped_label, payload in grouped.items():
            reason_text = "、".join(payload["reasons"])
            matched_text = "、".join(payload["matched_texts"])
            summaries.append(
                f"{LABEL_DISPLAY_NAME[grouped_label]}：命中{reason_text}；命中内容：{matched_text}。"
            )

        return summaries
