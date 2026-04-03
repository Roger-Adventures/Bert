# 中文内容审核决策原型

这是一个为内容风控/策略岗面试准备的轻量项目。  
项目目标不是证明“模型有多复杂”，而是展示一条更接近真实业务的审核链路：

**识别内容风险类型 -> 计算综合风险 -> 决定放行 / 人审 / 拦截**

## 适合面试展示的点

- 标签体系清晰：`normal / abuse / sexual / ad`
- 决策动作明确：`allow / review / block`
- 输出可解释：有 `risk_score`、`risk_band`、`threshold_reason`、`rule_hits`
- 规则和模型协同：规则兜底高精度模式，模型处理语义表达
- 有可直接现场演示的网页原型

## 模型与数据

- 默认模型：`hfl/chinese-macbert-base`
- 默认训练目标：本机 `RTX 4060 8GB` 上半小时内完成一轮轻量微调
- 数据策略：
  - `abuse` 使用 `COLDataset` offensive 样本补强
  - `normal / sexual / ad` 采用面试展示级定向构造样本

## 推荐启动方式

请使用下面这个命令：

```bash
python run_demo.py --reload
```

不要直接使用系统里的：

```bash
uvicorn serve_api:app --reload
```

原因是它可能落到错误的 Python 解释器，导致：

- `http://127.0.0.1:8000/docs` 打不开
- `/health` 显示 `model_loaded=false`
- 页面进入 `rules fallback`

## 如何判断是不是起错环境

打开：

- `http://127.0.0.1:8000/health`

重点看这些字段：

- `python_executable`
- `python_version`
- `torch_available`
- `transformers_available`
- `model_loaded`

如果 `model_loaded` 是 `false`，优先检查：

1. 你是不是用 `python run_demo.py` 启动的
2. IDE 当前选中的解释器是不是项目实际使用的 Python

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

### 3. 训练模型

```bash
python train.py
```

默认训练配置：

- `model_name = hfl/chinese-macbert-base`
- `max_length = 128`
- `batch_size = 8`
- `num_train_epochs = 3`
- `learning_rate = 2e-5`

训练完成后会生成：

- `artifacts/moderation_macbert/`
- `reports/confusion_matrix.png`
- `reports/model_metrics.json`

### 4. 启动服务

```bash
python run_demo.py --reload
```

访问地址：

- `http://127.0.0.1:8000/`：现场演示页
- `http://127.0.0.1:8000/docs`：API 文档
- `http://127.0.0.1:8000/health`：健康检查

## API 输出重点

`/predict` 返回的关键字段：

- `label`：内容类型
- `risk_score`：综合风险分
- `risk_band`：低风险 / 中风险 / 高风险
- `action`：放行 / 人审 / 拦截
- `source`：`model` / `rules` / `model+rules`
- `model_confidence`：模型置信度
- `rule_hits`：命中的规则
- `reasons`：解释性原因列表
- `threshold_reason`：动作阈值原因

## 现场演示页包含什么

- 顶部系统状态：模型是否加载、解释器路径、docs 链接
- 中间审核输入：示例文本、运行审核按钮
- 右侧决策输出：标签、动作、风险分、规则命中、阈值原因
- 下方结果支撑：混淆矩阵和典型案例表

## VS Code 调试

项目已经提供：

- `.vscode/launch.json`

直接选择：

- `Run Moderation Demo`

即可用当前 IDE 解释器启动服务。

## 面试时推荐怎么讲

1. 这是一个审核决策原型，不只是文本分类器。
2. 模型负责语义识别，规则负责高确定性模式拦截。
3. 内容风控需要 `allow / review / block` 三档，而不是只有“过/不过”。
4. 我没有盲目追求更大模型，而是选择更适合本地环境和演示节奏的 `MacBERT Base`。

## 讲解材料

- [docs/interview_brief.md](docs/interview_brief.md)
