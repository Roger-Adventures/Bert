const DEMO_CONTENT_URL = "/static/demo_content.json";

const LABEL_NAMES = {
  normal: "正常内容",
  abuse: "辱骂攻击",
  sexual: "低俗色情",
  ad: "广告引流"
};

const EMPTY_DEMO_CONTENT = {
  default_sample_id: null,
  sample_groups: [],
  cases: []
};

let demoContent = { ...EMPTY_DEMO_CONTENT };

const elements = {
  sampleRow: document.getElementById("sampleRow"),
  inputText: document.getElementById("inputText"),
  predictButton: document.getElementById("predictButton"),
  decisionHero: document.getElementById("decisionHero"),
  decisionHeroPill: document.getElementById("decisionHeroPill"),
  decisionHeroTitle: document.getElementById("decisionHeroTitle"),
  decisionHeroSummary: document.getElementById("decisionHeroSummary"),
  labelValue: document.getElementById("labelValue"),
  actionValue: document.getElementById("actionValue"),
  bandValue: document.getElementById("bandValue"),
  scoreValue: document.getElementById("scoreValue"),
  sourceValue: document.getElementById("sourceValue"),
  thresholdValue: document.getElementById("thresholdValue"),
  ruleValue: document.getElementById("ruleValue"),
  statusLine: document.getElementById("statusLine"),
  probabilityList: document.getElementById("probabilityList"),
  casesTableBody: document.getElementById("casesTableBody"),
  summaryHeadline: document.getElementById("summaryHeadline"),
  summaryChip: document.getElementById("summaryChip"),
  summaryDetail: document.getElementById("summaryDetail"),
  fallbackWarning: document.getElementById("fallbackWarning")
};

function actionPillClass(action) {
  if (action === "allow") return "pill pill-allow";
  if (action === "review") return "pill pill-review";
  return "pill pill-block";
}

function sampleChipClass(action) {
  if (action === "allow") return "sample-chip sample-chip--allow";
  if (action === "review") return "sample-chip sample-chip--review";
  return "sample-chip sample-chip--block";
}

function groupRuleHits(ruleHits) {
  if (!ruleHits.length) {
    return "未命中规则";
  }

  const grouped = new Map();
  for (const hit of ruleHits) {
    if (!grouped.has(hit.label)) {
      grouped.set(hit.label, {
        labelName: LABEL_NAMES[hit.label] ?? hit.label,
        reasons: [],
        matchedTexts: []
      });
    }

    const entry = grouped.get(hit.label);
    const normalizedReason = hit.reason.startsWith("命中") ? hit.reason.slice(2) : hit.reason;

    if (!entry.reasons.includes(normalizedReason)) {
      entry.reasons.push(normalizedReason);
    }

    if (!entry.matchedTexts.includes(hit.matched_text)) {
      entry.matchedTexts.push(hit.matched_text);
    }
  }

  return Array.from(grouped.values())
    .map((entry) => {
      const reasonText = entry.reasons.join("、");
      const matchedText = entry.matchedTexts.join("、");
      return `${entry.labelName}：命中${reasonText}；命中内容：${matchedText}。`;
    })
    .join(" ");
}

function renderEmptySamples(message) {
  elements.sampleRow.innerHTML = "";
  const note = document.createElement("div");
  note.className = "sample-note";
  note.textContent = message;
  elements.sampleRow.appendChild(note);
}

function renderEmptyCases(message) {
  elements.casesTableBody.innerHTML = "";
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 5;
  cell.className = "muted";
  cell.textContent = message;
  row.appendChild(cell);
  elements.casesTableBody.appendChild(row);
}

function renderSampleGroups(groups) {
  elements.sampleRow.innerHTML = "";

  if (!groups.length) {
    renderEmptySamples("示例配置未加载，可直接输入文本进行审核。");
    return;
  }

  for (const group of groups) {
    const groupCard = document.createElement("article");
    groupCard.className = "sample-group";

    const groupHead = document.createElement("div");
    groupHead.className = "sample-group-head";

    const title = document.createElement("strong");
    title.textContent = group.label;

    const description = document.createElement("span");
    description.textContent = group.description;

    groupHead.append(title, description);

    const items = document.createElement("div");
    items.className = "sample-group-items";

    for (const item of group.items) {
      const button = document.createElement("button");
      button.className = sampleChipClass(item.action);
      button.type = "button";
      button.title = item.text;

      const actionTag = document.createElement("span");
      actionTag.className = "sample-chip-tag";
      actionTag.textContent = item.action_label;

      const label = document.createElement("span");
      label.className = "sample-chip-label";
      label.textContent = item.title;

      button.append(actionTag, label);
      button.addEventListener("click", async () => {
        elements.inputText.value = item.text;
        await runPrediction();
      });

      items.appendChild(button);
    }

    groupCard.append(groupHead, items);
    elements.sampleRow.appendChild(groupCard);
  }
}

async function fetchJson(url, errorMessage, options = undefined) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(errorMessage);
  }
  return response.json();
}

function renderHealth(health) {
  const modelLoaded = Boolean(health.model_loaded);

  elements.summaryHeadline.textContent = modelLoaded
    ? "模型已加载，可输出完整审核结果"
    : "模型未加载，当前使用规则兜底";
  elements.summaryChip.className = `status-chip ${modelLoaded ? "status-ok" : "status-warn"}`;
  elements.summaryChip.textContent = modelLoaded ? "model ready" : "rules fallback";
  elements.summaryDetail.textContent = modelLoaded
    ? "当前服务可以输出标签、风险分、审核动作和命中原因。"
    : "建议用 python run_demo.py --reload 启动服务，并通过 /health 检查当前解释器。";
  elements.fallbackWarning.hidden = modelLoaded;
}

function renderProbabilities(probabilities) {
  elements.probabilityList.innerHTML = "";

  for (const [label, score] of Object.entries(probabilities)) {
    const item = document.createElement("li");
    const name = document.createElement("span");
    const bar = document.createElement("div");
    const value = document.createElement("strong");

    name.textContent = LABEL_NAMES[label] ?? label;
    bar.className = "bar";
    bar.innerHTML = `<span style="width: ${Math.max(score * 100, 2)}%"></span>`;
    value.textContent = score.toFixed(4);

    item.append(name, bar, value);
    elements.probabilityList.appendChild(item);
  }
}

function resetResultView(summary, statusLine) {
  elements.decisionHero.className = "decision-hero decision-hero--pending";
  elements.decisionHeroPill.textContent = "等待执行";
  elements.decisionHeroTitle.textContent = "等待输入文本";
  elements.decisionHeroSummary.textContent = summary;

  elements.labelValue.textContent = "-";
  elements.actionValue.textContent = "-";
  elements.bandValue.textContent = "-";
  elements.scoreValue.textContent = "-";
  elements.sourceValue.textContent = "-";
  elements.thresholdValue.textContent = "-";
  elements.ruleValue.textContent = "-";
  elements.statusLine.textContent = statusLine;
  elements.probabilityList.innerHTML = "";
}

function renderDecisionHero(payload) {
  elements.decisionHero.className = `decision-hero decision-hero--${payload.action}`;
  elements.decisionHeroPill.textContent = `${payload.action_name} | ${payload.risk_band_name}`;

  if (payload.action === "allow") {
    elements.decisionHeroTitle.textContent = "建议直接放行";
    elements.decisionHeroSummary.textContent = `${payload.threshold_reason} 当前样本属于低风险内容。`;
    return;
  }

  if (payload.action === "review") {
    elements.decisionHeroTitle.textContent = "建议进入人工复核";
    elements.decisionHeroSummary.textContent = `${payload.threshold_reason} 当前样本更适合通过人审来降低误杀。`;
    return;
  }

  elements.decisionHeroTitle.textContent = "建议直接拦截";
  elements.decisionHeroSummary.textContent = `${payload.threshold_reason} 当前样本已经达到高风险处理阈值。`;
}

function renderMainResult(payload) {
  renderDecisionHero(payload);
  elements.labelValue.textContent = `${payload.label_name} (${payload.label})`;
  elements.actionValue.innerHTML = `<span class="${actionPillClass(payload.action)}">${payload.action_name}</span>`;
  elements.bandValue.textContent = payload.risk_band_name;
  elements.scoreValue.textContent = payload.risk_score.toFixed(4);
  elements.sourceValue.textContent = payload.source;
  elements.thresholdValue.textContent = payload.threshold_reason;
  elements.ruleValue.textContent = groupRuleHits(payload.rule_hits);
  elements.statusLine.textContent = payload.model_loaded
    ? `由 ${payload.source} 给出处置，模型参考置信度 ${payload.model_confidence.toFixed(4)}。`
    : `当前为规则兜底模式，由 ${payload.source} 给出处置。`;

  renderProbabilities(payload.probabilities);
}

async function predict(text) {
  return fetchJson("/predict", "审核接口调用失败", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
}

async function runPrediction() {
  const text = elements.inputText.value.trim();
  if (!text) {
    resetResultView("执行审核后，这里会优先展示审核动作、风险等级和处置依据。", "请输入待审核文本。");
    return;
  }

  elements.predictButton.disabled = true;
  elements.predictButton.textContent = "审核中...";

  try {
    const payload = await predict(text);
    renderMainResult(payload);
  } catch (error) {
    resetResultView("审核结果未返回。", error.message);
    elements.decisionHeroPill.textContent = "调用失败";
    elements.decisionHeroTitle.textContent = "审核结果未返回";
  } finally {
    elements.predictButton.disabled = false;
    elements.predictButton.textContent = "运行审核决策";
  }
}

async function renderCaseTable() {
  elements.casesTableBody.innerHTML = "";

  if (!demoContent.cases.length) {
    renderEmptyCases("案例配置未加载，可手动输入文本验证结果。");
    return;
  }

  for (const item of demoContent.cases) {
    try {
      const payload = await predict(item.text);

      const row = document.createElement("tr");
      const groupCell = document.createElement("td");
      const textCell = document.createElement("td");
      const labelCell = document.createElement("td");
      const actionCell = document.createElement("td");
      const thresholdCell = document.createElement("td");

      groupCell.textContent = item.group_label;
      textCell.textContent = item.text;
      labelCell.textContent = payload.label_name;
      actionCell.innerHTML = `<span class="${actionPillClass(payload.action)}">${payload.action_name}</span>`;
      thresholdCell.className = "muted";
      thresholdCell.textContent = payload.threshold_reason;

      row.append(groupCell, textCell, labelCell, actionCell, thresholdCell);
      elements.casesTableBody.appendChild(row);
    } catch (error) {
      renderEmptyCases(`案例加载失败：${error.message}`);
      break;
    }
  }
}

function findDefaultSample(groups, defaultSampleId) {
  for (const group of groups) {
    for (const item of group.items) {
      if (item.id === defaultSampleId) {
        return item;
      }
    }
  }

  return groups[0]?.items?.[0] ?? null;
}

async function bootstrap() {
  elements.predictButton.addEventListener("click", runPrediction);

  try {
    demoContent = await fetchJson(DEMO_CONTENT_URL, "示例配置加载失败");
    renderSampleGroups(demoContent.sample_groups ?? []);

    const defaultSample = findDefaultSample(
      demoContent.sample_groups ?? [],
      demoContent.default_sample_id
    );
    if (defaultSample) {
      elements.inputText.value = defaultSample.text;
    }
  } catch (_error) {
    demoContent = { ...EMPTY_DEMO_CONTENT };
    renderEmptySamples("示例配置加载失败，可直接输入文本进行审核。");
    renderEmptyCases("案例配置加载失败。");
  }

  try {
    const health = await fetchJson("/health", "健康检查失败");
    renderHealth(health);
  } catch (error) {
    elements.summaryHeadline.textContent = "健康检查失败";
    elements.summaryChip.className = "status-chip status-warn";
    elements.summaryChip.textContent = "health error";
    elements.summaryDetail.textContent = `${error.message}。请检查服务是否已正常启动。`;
  }

  if (elements.inputText.value.trim()) {
    await runPrediction();
  }

  await renderCaseTable();
}

bootstrap();
