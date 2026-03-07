# memoRaxis_lightrag
这个仓库致力于将lightrag的代码整合进入memoRaxis并进行测试

# memoRaxis_lightrag

`memoRaxis_lightrag` 是从 `memoRaxis` 中拆分出的 LightRAG 单后端实验仓库。  
该仓库保留了 memoRaxis 中“推理适配器（R-axis）”与“记忆系统（M-axis）”解耦的设计思路，并以 LightRAG 作为唯一的记忆后端实现，用于在 MemoryAgentBench 上完成 ingest、infer、evaluate 的完整实验流程。

当前仓库关注的核心目标有两点：

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
└─ requirements.txt

## 2. 运行方法
建议环境：

Python 3.11

Windows / Linux / macOS 均可

已安装可用的 LightRAG 包

已准备可访问的 LLM 与 Embedding 接口

pip install -r requirements.txt
pip install -e third_party/LightRAG

配置文件路径：config/config.yaml

当前脚本默认读取：MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet
若机械评测需要使用 preview JSON，可执行：python scripts/LightRAG_MAB/data/convert_parquet_to_json.py
