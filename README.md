# memoRaxis-LightRAG 操作手册（简化版）

本仓库的目标是以 LightRAG 作为唯一记忆后端，完成 MemoryAgentBench 四个任务的 ingest、infer 与 token 统计实验。

## 1. 环境

```bash
conda create -n memoraxis_lightrag python=3.10 -y
conda activate memoraxis_lightrag
pip install -r requirements.txt
```

配置 API：

```text
config/config.yaml
```

## 2. 数据准备

```bash
python scripts/LightRAG_MAB/data/convert_parquet_to_json.py
```

## 3. 建索引（ingest）

```bash
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py   --instance_idx 0   --chunk_size 12000   --reset
```

统一支持的参数：

- `--max_chunks`：限制 ingest 的 chunk 数量，用于 token 曲线实验。
- `--output_suffix`：为工作区与输出文件追加后缀，避免覆盖旧结果。
- ingest 完成后会自动输出 probe JSON 到 `out/*_ingest_probe_*.json`。

## 4. 推理（infer）

```bash
python scripts/LightRAG_MAB/infer/infer_accurate_retrieval.py   --instance_idx 0   --adaptor all   --limit 10
```

统一输出：

- 每题结果会写入 memory token 统计字段 `memory_token_summary`。
- adaptor 级别会在控制台打印汇总 memory token。
- infer 结果文件统一输出到 `out/*_results_*.json`。

## 5. Token 统计实验（核心）

以下命令用于生成 ingest token 曲线数据：

```bash
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 12000 --max_chunks 1   --output_suffix probe_1   --reset
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 12000 --max_chunks 2   --output_suffix probe_2   --reset
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 12000 --max_chunks 4   --output_suffix probe_4   --reset
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 12000 --max_chunks 8   --output_suffix probe_8   --reset
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 12000 --max_chunks 16  --output_suffix probe_16  --reset
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 12000 --max_chunks 32  --output_suffix probe_32  --reset
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 12000 --max_chunks 64  --output_suffix probe_64  --reset
python scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py --instance_idx 0 --chunk_size 12000 --max_chunks 128 --output_suffix probe_128 --reset
```

以下命令用于生成 infer token 数据：

```bash
python scripts/LightRAG_MAB/infer/infer_accurate_retrieval.py --instance_idx 0 --adaptor all --limit 10 --output_suffix token_test
```

可视化建议：

- x 轴：chunk 数
- y 轴：token 数 / API calls
- 对比 ingest 与 infer 的增长趋势

## 6. 输出结构

ingest probe 示例：

```json
{
  "final_chunk_count": 32,
  "ingest_token_summary": {
    "total": {
      "total_tokens": 12345
    }
  }
}
```

infer result 示例：

```json
{
  "memory_token_summary": {
    "total": {
      "total_tokens": 456
    }
  }
}
```

## 7. 项目结构

```text
src/
  lightrag_memory.py

scripts/LightRAG_MAB/
  ingest/
  infer/
  evaluate/
```
