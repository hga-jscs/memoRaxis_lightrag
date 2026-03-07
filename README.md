# memoRaxis_lightrag

`memoRaxis_lightrag` 是从 `memoRaxis` 中拆分出的 LightRAG 单后端实验仓库。该仓库延续了 memoRaxis 中“推理适配器（R-axis）”与“记忆系统（M-axis）”解耦”的设计思路，并以 LightRAG 作为唯一的记忆后端实现，用于在 MemoryAgentBench 上完成 ingest、infer、evaluate 的完整实验流程。

当前仓库的核心目标有两点：

1. 以 LightRAG 为唯一 M 轴后端，形成独立、清晰、可复现的实验仓库；
2. 在统一的 R1 / R2 / R3 推理范式下，考察 LightRAG 的检索支持能力与整体问答表现。

---

## 1. 项目结构

```text
memoRaxis_lightrag/
├─ config/
│  ├─ config.yaml
│  └─ prompts.yaml
├─ docs/
│  └─ bluePrint.md
├─ external/
├─ MemoryAgentBench/
├─ scripts/
│  ├─ LightRAG_MAB/
│  │  ├─ data/
│  │  ├─ ingest/
│  │  ├─ infer/
│  │  ├─ evaluate/
│  │  ├─ analyze/
│  │  └─ debug/
├─ src/
│  ├─ __init__.py
│  ├─ adaptors.py
│  ├─ benchmark_utils.py
│  ├─ config.py
│  ├─ lightrag_memory.py
│  ├─ llm_interface.py
│  ├─ logger.py
│  └─ memory_interface.py
├─ main.py
├─ run_all_tasks.py
└─ requirements.txt
```

其中：

- `src/adaptors.py`：三种推理适配器实现  
  - `SingleTurnAdaptor`（R1）  
  - `IterativeAdaptor`（R2）  
  - `PlanAndActAdaptor`（R3）

- `src/lightrag_memory.py`：LightRAG 记忆后端实现

- `scripts/LightRAG_MAB/`：围绕 MemoryAgentBench 的实验脚本  
  - `data/`：数据预处理与 preview JSON 生成  
  - `ingest/`：索引构建  
  - `infer/`：问题推理与作答  
  - `evaluate/`：机械评测与其他评测脚本  
  - `analyze/`：结果分析

- `run_all_tasks.py`：四个任务的一键总控脚本

---

## 2. 仓库定位

该仓库的定位不是面向生产环境的通用型 RAG 产品，而是一个研究型实验仓库。它的主要价值在于：

- 将 LightRAG 从原始多后端实验框架中拆分出来；
- 把实验变量收敛到单一 M 轴；
- 保留统一的 R1 / R2 / R3 推理适配器；
- 支持在 MemoryAgentBench 上完成完整链路复现：数据读取 → 建索引 → 推理 → 评测。

当前版本已经具备“单后端独立运行与复现”的条件，但整体仍保持研究型项目的轻量结构。

---

## 3. 运行环境

建议环境如下：

- Python 3.11
- Windows / Linux / macOS 均可
- 已安装可用的 LightRAG 包
- 已准备可访问的 LLM 与 Embedding 接口
- 建议在 Anaconda Prompt 中运行

推荐的使用方式是统一在同一个 Conda 环境中完成所有操作，避免因终端切换导致路径、环境变量或依赖状态不一致。

---

## 4. 安装步骤

### 4.1 创建并激活环境（Anaconda）

```bash
conda create -n memoraxis_lightrag python=3.11 -y
conda activate memoraxis_lightrag
```

### 4.2 进入项目根目录

Windows 下：

```bash
cd /d D:\memoRaxis_lightrag
```

按实际路径替换即可。

### 4.3 安装基础依赖

```bash
pip install -r requirements.txt
```

### 4.4 安装 LightRAG

若仓库中包含 `third_party/LightRAG`，则执行：

```bash
pip install -e third_party/LightRAG
```

若仓库中没有该目录，而是使用外部安装方式，则执行：

```bash
pip install lightrag
```

---

## 5. 配置说明

配置文件路径为：

```text
config/config.yaml
```

配置通常包含两部分：LLM 配置与 Embedding 配置。

示例：

```yaml
llm:
  provider: openai_compat
  model: gpt-4o-mini
  base_url: "https://your-llm-base-url/v1"
  api_key: "YOUR_LLM_API_KEY"
  timeout: 120

embedding:
  provider: openai_compat
  model: text-embedding-3-small
  base_url: "https://your-embedding-base-url/v1"
  api_key: "YOUR_EMBEDDING_API_KEY"
  dim: 1536

database: {}
```

其中：

- `llm` 用于 LightRAG 建图、摘要、关系抽取，以及 R1 / R2 / R3 适配器问答阶段；
- `embedding` 用于向量化与检索；
- `database` 对于本仓库不是必需项，可保留为空。

若仓库对外公开，`config.yaml` 更适合作为占位配置文件使用，真实 key 与真实地址由本地环境单独填写。

---

## 6. 数据准备

当前脚本会读取 `MemoryAgentBench/data/` 下的数据文件。四个主要任务对应的数据文件如下：

```text
MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet
MemoryAgentBench/data/Conflict_Resolution-00000-of-00001.parquet
MemoryAgentBench/data/Long_Range_Understanding-00000-of-00001.parquet
MemoryAgentBench/data/Test_Time_Learning-00000-of-00001.parquet
```

若评测脚本需要 preview JSON，可先执行：

```bash
python scripts/LightRAG_MAB/data/convert_all_data.py
```

或者单独执行：

```bash
python scripts/LightRAG_MAB/data/convert_parquet_to_json.py
```

生成后的 preview 文件通常位于：

```text
MemoryAgentBench/preview_samples/
```

---

## 7. 两种使用方式

当前仓库同时保留两种运行方式：

### 7.1 一键运行

一键运行由 `run_all_tasks.py` 提供，用于将四个任务串成一条完整链路：(详见docs路径下的说明)

- 预处理
- ingest
- infer
- evaluate

它适合以下场景：

- 初次验证仓库是否可运行
- 按 README 快速复现
- 小规模 smoke test
- 一次性完成四个任务的基线测试

最简命令为：

```bash
python run_all_tasks.py --reset
```

这条命令默认会跑：

- Accurate_Retrieval
- Conflict_Resolution
- Long_Range_Understanding
- Test_Time_Learning

并执行完整的：

```text
预处理 → ingest → infer → evaluate
```

### 7.2 CLI 分步运行

CLI 分步运行是直接调用 `scripts/LightRAG_MAB/...` 下的脚本，分别完成：

- data
- ingest
- infer
- evaluate
- analyze

它适合以下场景：

- 单独调试某个任务
- 某个参数修改后只重跑一段流程
- 定位错误
- 批量实验与对比

两种方式并不冲突。前者用于“跑通全链路”，后者用于“精细控制与局部重跑”。

---

## 8. 一键运行：`run_all_tasks.py`

### 8.1 最简命令

四个任务全跑：

```bash
python run_all_tasks.py --reset
```

### 8.2 常用示例

只跑 `Accurate_Retrieval` 和 `Conflict_Resolution`：

```bash
python run_all_tasks.py --tasks acc conflict --reset
```

只跑 R1 和 R2：

```bash
python run_all_tasks.py --adaptors R1 R2 --reset
```

复用已有索引，跳过 ingest：

```bash
python run_all_tasks.py --skip_ingest
```

只做推理与评测，跳过预处理和 ingest：

```bash
python run_all_tasks.py --skip_preprocess --skip_ingest
```

小规模 smoke test：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 3 --conflict_limit 3 --long_limit 1 --ttl_limit 3 --reset
```

### 8.3 一键脚本参数总览

#### 通用参数

`--tasks`  
选择要执行的任务集合。默认值为：

```bash
--tasks acc conflict long ttl
```

可选值：

- `acc`：Accurate_Retrieval
- `conflict`：Conflict_Resolution
- `long`：Long_Range_Understanding
- `ttl`：Test_Time_Learning

---

`--adaptors`  
选择推理适配器。默认值为：

```bash
--adaptors R1 R2 R3
```

可选值：

- `R1`
- `R2`
- `R3`
- `all`

---

`--mode`  
指定 LightRAG 的检索模式。默认值：

```bash
--mode naive
```

可选值：

- `naive`
- `mix`
- `local`
- `global`
- `hybrid`

通常以 `naive` 作为默认起点更稳妥。

---

`--storage_dir`  
指定 LightRAG 索引存储目录。默认值：

```bash
--storage_dir out/lightrag_storage
```

不同实验使用不同目录，可以避免索引覆盖。

---

`--output_suffix`  
给推理结果文件名增加后缀。默认值为空字符串：

```bash
--output_suffix ""
```

该参数适合区分不同实验结果。  
但 `Conflict_Resolution` 的官方评测脚本对文件名较敏感，因此默认建议留空。

---

`--reset`  
在 ingest 前重置对应 workspace。  
第一次运行或希望彻底重建索引时使用。

---

`--skip_preprocess`  
跳过 preview JSON 生成。  
当 `MemoryAgentBench/preview_samples/` 已经存在且可用时可使用。

---

`--skip_ingest`  
跳过索引构建，直接复用已有索引。  
适合调整 adaptor、limit、评测逻辑时使用。

---

`--skip_infer`  
跳过推理阶段。  
适合只做 ingest 或只检查索引构建。

---

`--skip_eval`  
跳过评测阶段。  
适合先只看模型输出结果。

---

### 8.4 四个任务的专属参数

#### Accurate_Retrieval

`--acc_instance_idx`  
指定实例编号。默认值：

```bash
0
```

支持格式：

- `0`
- `0-3`
- `0,2,5`
- `0-2,5,7`

---

`--acc_limit`  
限制每个实例回答多少个问题。默认值：

```bash
5
```

smoke test 常用较小值，例如 `3` 或 `5`。

---

`--acc_chunk_size`  
ingest 时的 chunk 大小。默认值：

```bash
850
```

值小则切分更细，值大则保留更多上下文。

---

#### Conflict_Resolution

`--conflict_instance_idx`  
默认值：

```bash
0-7
```

通常该任务实例数不多，默认全跑即可。

---

`--conflict_limit`  
默认值：

```bash
-1
```

`-1` 表示不限制，全部问题都跑。

---

`--conflict_min_chars`  
默认值：

```bash
800
```

控制 ingest 时的最小文本块长度。

---

#### Long_Range_Understanding

`--long_instance_idx`  
默认值：

```bash
0-39
```

默认覆盖较多实例，完整运行时间较长。

---

`--long_limit`  
默认值：

```bash
-1
```

`-1` 表示不限制，全部问题都跑。

---

`--long_chunk_size`  
默认值：

```bash
1200
```

---

`--long_overlap`  
默认值：

```bash
100
```

该任务对长上下文切分较敏感。若需要更多上下文连续性，可适当增大 `chunk_size` 与 `overlap`。

---

#### Test_Time_Learning

`--ttl_instance_idx`  
默认值：

```bash
0-5
```

---

`--ttl_limit`  
默认值：

```bash
-1
```

`-1` 表示不限制，全部问题都跑。

---

## 9. CLI 分步运行

当需要局部控制、调试或分步实验时，可直接调用 `scripts/LightRAG_MAB/...` 中的脚本。

### 9.1 数据预处理

生成 preview samples：

```bash
python scripts/LightRAG_MAB/data/convert_all_data.py
```

或：

```bash
python scripts/LightRAG_MAB/data/convert_parquet_to_json.py
```

### 9.2 Ingest

#### Accurate_Retrieval

```bash
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 850 --save_dir out/lightrag_storage --mode naive --reset
```

#### Conflict_Resolution

```bash
python scripts/LightRAG_MAB/ingest/ingest_conflict_resolution.py --instance_idx 0-7 --min_chars 800 --save_dir out/lightrag_storage --mode naive --reset
```

#### Long_Range_Understanding

```bash
python scripts/LightRAG_MAB/ingest/ingest_long_range.py --instance_idx 0-39 --chunk_size 1200 --overlap 100 --save_dir out/lightrag_storage --mode naive --reset
```

#### Test_Time_Learning

```bash
python scripts/LightRAG_MAB/ingest/ingest_test_time.py --instance_idx 0-5 --save_dir out/lightrag_storage --mode naive --reset
```

### 9.3 Infer

#### Accurate_Retrieval

```bash
python scripts/LightRAG_MAB/infer/infer_accurate_retrieval.py --instance_idx 0 --adaptor all --limit 5 --storage_dir out/lightrag_storage --mode naive --output_suffix ""
```

#### Conflict_Resolution

```bash
python scripts/LightRAG_MAB/infer/infer_conflict_resolution.py --instance_idx 0-7 --adaptor all --limit -1 --storage_dir out/lightrag_storage --mode naive --output_suffix ""
```

#### Long_Range_Understanding

```bash
python scripts/LightRAG_MAB/infer/infer_long_range.py --instance_idx 0-39 --adaptor all --limit -1 --storage_dir out/lightrag_storage --mode naive --output_suffix ""
```

#### Test_Time_Learning

```bash
python scripts/LightRAG_MAB/infer/infer_test_time.py --instance_idx 0-5 --adaptor all --limit -1 --storage_dir out/lightrag_storage --mode naive --output_suffix ""
```

### 9.4 Evaluate

#### Accurate_Retrieval

```bash
python scripts/LightRAG_MAB/evaluate/evaluate_mechanical.py --results out/acc_ret_results_0.json --instance MemoryAgentBench/preview_samples/Accurate_Retrieval/instance_0.json
```

#### Conflict_Resolution

```bash
python scripts/LightRAG_MAB/evaluate/evaluate_conflict_official.py
```

#### Long_Range_Understanding

```bash
python scripts/LightRAG_MAB/evaluate/evaluate_long_range_A.py --results out/long_range_results_0.json --instance_folder MemoryAgentBench/preview_samples/Long_Range_Understanding
```

#### Test_Time_Learning

```bash
python scripts/LightRAG_MAB/evaluate/evaluate_ttl_mechanical.py --results_pattern out/ttl_results_*.json
```

---

## 10. 推荐使用流程

### 第一步：冒烟测试

先用一键脚本做最小规模验证，确认环境、接口与路径都正常：

```bash
python run_all_tasks.py --acc_instance_idx 0 --conflict_instance_idx 0 --long_instance_idx 0 --ttl_instance_idx 0 --acc_limit 3 --conflict_limit 3 --long_limit 1 --ttl_limit 3 --reset
```

### 第二步：标准运行

四个任务完整跑一轮：

```bash
python run_all_tasks.py --reset
```

### 第三步：进入科研调参

当需要局部修改参数、只重跑某一段流程时，改用 CLI 分步运行。例如：

- 修改 chunk size 后只重跑 ingest
- 修改 adaptor 后只重跑 infer
- 修改评测逻辑后只重跑 evaluate

这种方式更适合实验迭代与结果对比。

---

## 11. 输出结果说明

### 索引输出目录

```text
out/lightrag_storage/
```

不同任务、不同实例的 workspace 会在该目录下生成对应子目录。

### 推理结果输出

通常输出在：

```text
out/
```

典型文件名包括：

```text
out/acc_ret_results_0.json
out/conflict_res_results_0.json
out/long_range_results_0.json
out/ttl_results_0.json
```

若指定了 `--output_suffix`，则文件名会附带后缀。

### 评测输出

评测结果通常直接打印在终端中。  
不同任务对应不同的评测方式：

- Accurate_Retrieval：mechanical evaluation
- Conflict_Resolution：official evaluation
- Long_Range_Understanding：LLM-based evaluation
- Test_Time_Learning：mechanical evaluation

其中 `Long_Range_Understanding` 的评测通常更慢，也更消耗 token，因为其评测脚本本身依赖 LLM Judge。

---

## 12. 当前版本的边界

当前版本已经具备“单后端独立运行与复现”的条件，但仍然属于研究代码风格，主要体现在：

- 默认入口仍以脚本方式组织，尚未完全统一为单一 CLI 框架；
- 评测设计仍以研究型实验流程为主；
- 配置与外部模型接口紧密相关，依赖本地实验环境。

因此，该仓库的重点不是作为通用产品交付，而是作为一个清晰、可复现、适合后续重构的 LightRAG 独立实验仓库。

---

## 13. 项目目的

`memoRaxis_lightrag` 的主要意义不在于展示一个封装完整的成品系统，而在于：

- 将 LightRAG 从原始多后端实验框架中拆分出来；
- 把实验变量收敛到单一 M 轴；
- 保留统一的 R1 / R2 / R3 推理适配器；
- 为后续的独立重构、结果分析与文档沉淀提供清晰起点。