# 中文内容审核策略原型

这是一个面向内容风控/策略岗面试准备的轻量项目。  
项目重点不是做复杂模型堆叠，而是展示一条完整的审核决策链路：

`风险识别 -> 风险分层 -> 放行 / 人审 / 拦截`

## 项目特点

- 任务标签固定为 `normal / abuse / sexual / ad`
- 决策动作固定为 `allow / review / block`
- 模型负责识别风险类型，规则主要负责升级审核动作
- 保留“规则 + 模型 + 阈值”三层逻辑，但不做严格概率融合
- 支持网页演示、接口调用和健康检查

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
python prepare_dataset.py
```

会生成：

- `data/train.csv`
- `data/dev.csv`
- `data/test.csv`
- `data/dataset_summary.json`

这些文件属于可再生成数据，默认不纳入版本控制。

### 3. 训练模型

```bash
python train.py
```

默认训练配置：

- 模型：`hfl/chinese-macbert-base`
- `max_length=128`
- `batch_size=8`
- `num_train_epochs=3`
- `learning_rate=2e-5`

训练完成后会生成：

- `artifacts/moderation_macbert/`
- `reports/confusion_matrix.png`
- `reports/model_metrics.json`

其中 `artifacts/`、`reports/` 下的生成结果默认只保留本地，不提交到仓库。

### 4. 启动演示

```bash
python run_demo.py --reload
```

不要直接使用系统里的：

```bash
uvicorn serve_api:app --reload
```

这样可以避免服务落到错误的 Python 解释器里。

## 演示入口

- `http://127.0.0.1:8000/`：展示页
- `http://127.0.0.1:8000/docs`：接口文档
- `http://127.0.0.1:8000/health`：健康检查

`/predict` 的核心输出字段：

- `label`
- `risk_score`
- `risk_band`
- `action`
- `source`
- `model_confidence`
- `rule_hits`
- `reasons`
- `threshold_reason`

字段语义：

- `probabilities`：模型原始概率，用于展示模型参考判断
- `risk_score`：用于审核动作决策的风险分，不等同于严格概率

## 数据说明

- 主数据集使用 `ChineseHarm-Bench`
- 标签映射为：
  - `不违规 -> normal`
  - `谩骂引战 -> abuse`
  - `低俗色情 -> sexual`
  - `黑产广告 -> ad`
- `博彩` 和 `欺诈` 暂不纳入当前四分类标签体系
- `abuse` 在主数据不足时可回退补充 `COLDataset` offensive 样本
- 当前数据集仍然用于演示审核链路，不代表真实生产覆盖度

## 决策逻辑

- 模型先给出风险类型基线判断
- 强规则命中时直接拦截
- 中强规则命中时至少进入人审
- 其余情况由模型风险分和置信度决定放行、人审或拦截

## 目录说明

- [run_demo.py](./run_demo.py)：统一启动入口
- [serve_api.py](./serve_api.py)：FastAPI 服务
- [pipeline.py](./pipeline.py)：审核决策主逻辑
- [rules.py](./rules.py)：规则命中逻辑
- [prepare_dataset.py](./prepare_dataset.py)：数据构造与切分
- [train.py](./train.py)：模型训练脚本
- `data/`：数据脚本生成的训练、验证、测试集目录
- `reports/`：训练脚本生成的评估结果目录
- [static/index.html](./static/index.html)：展示页结构
- [static/app.js](./static/app.js)：展示页交互逻辑
- [static/styles.css](./static/styles.css)：展示页样式
- [static/demo_content.json](./static/demo_content.json)：示例与案例配置

## 补充材料

- [docs/interview_brief.md](./docs/interview_brief.md)
