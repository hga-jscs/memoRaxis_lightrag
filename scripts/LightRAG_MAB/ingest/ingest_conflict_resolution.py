import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices
from src.lightrag_memory import LightRAGMemory

logger = get_logger()


def chunk_facts(context: str, min_chars: int = 800) -> List[str]:
    """
    Conflict Resolution 专用切分策略：
    按行读取 Fact，累积直到缓冲区字符数 > min_chars，然后作为一个 Chunk。
    """
    lines = [line.strip() for line in context.split("\n") if line.strip()]

    chunks = []
    current_chunk_lines = []
    current_length = 0

    for line in lines:
        current_chunk_lines.append(line)
        current_length += len(line)

        if current_length > min_chars:
            chunk_text = "\n".join(current_chunk_lines)
            chunks.append(chunk_text)
            current_chunk_lines = []
            current_length = 0

    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines)
        chunks.append(chunk_text)

    return chunks


def _probe_path(prefix: str, instance_idx: int, output_suffix: str) -> Path:
    suffix_part = f"_{output_suffix}" if output_suffix else ""
    return Path("out") / f"{prefix}_ingest_probe_{instance_idx}{suffix_part}.json"


def ingest_one_instance(
    instance_idx: int,
    min_chars: int,
    save_dir: str,
    mode: str,
    reset: bool,
    max_chunks: int | None,
    output_suffix: str,
):
    logger.info("=== Processing Conflict_Resolution Instance %s (LightRAG) ===", instance_idx)
    data_path = f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json"

    if not Path(data_path).exists():
        logger.error("Data file not found: %s", data_path)
        return

    try:
        data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Error loading instance %s: %s", instance_idx, e)
        return

    chunks = chunk_facts(data["context"], min_chars=min_chars)
    original_chunk_count = len(chunks)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    final_chunk_count = len(chunks)

    workspace_name = f"lightrag_conflict_{instance_idx}"
    if output_suffix:
        workspace_name += f"_{output_suffix}"
    workspace = Path(save_dir) / workspace_name
    workspace.mkdir(parents=True, exist_ok=True)

    memory = LightRAGMemory(working_dir=str(workspace), mode=mode)
    if reset:
        memory.reset()

    logger.info(
        "[ingest-debug] dataset=Conflict_Resolution instance=%s original_chunks=%s final_chunks=%s min_chars=%s workspace=%s",
        instance_idx,
        original_chunk_count,
        final_chunk_count,
        min_chars,
        workspace,
    )
    print(f"Starting ingestion for Instance {instance_idx} ({final_chunk_count} chunks) into {workspace} ...")
    for i, chunk in enumerate(chunks, start=1):
        memory.add_memory(chunk, metadata={"chunk_id": i - 1, "instance_idx": instance_idx})
        print(
            f"[ingest-progress] instance={instance_idx} queued={i}/{final_chunk_count} chunk_chars={len(chunk)}",
            end="\r",
            flush=True,
        )

    memory.begin_token_session()
    memory.build_index(doc_id=f"conflict_{instance_idx}")
    ingest_token_summary = memory.end_token_session()

    probe = {
        "dataset": "Conflict_Resolution",
        "instance_idx": instance_idx,
        "original_chunk_count": original_chunk_count,
        "final_chunk_count": final_chunk_count,
        "min_chars": min_chars,
        "char_count": len(data.get("context", "")),
        "mode": mode,
        "workspace": str(workspace),
        "output_suffix": output_suffix,
        "ingest_token_summary": ingest_token_summary,
    }
    probe_path = _probe_path("conflict_res", instance_idx, output_suffix)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_text(json.dumps(probe, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "[ingest-probe] saved=%s total_tokens=%s",
        probe_path,
        probe.get("ingest_token_summary", {}).get("total", {}).get("total_tokens", 0),
    )
    print(f"\nInstance {instance_idx} complete. LightRAG workspace saved -> {workspace}")
    print(f"[ingest-probe] {probe_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest Conflict_Resolution data (LightRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-7", help="Index range (e.g., '0-7')")
    parser.add_argument("--min_chars", type=int, default=800, help="Minimum chars per chunk")
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--output_suffix", type=str, default="")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out/lightrag_storage",
        help="Where to save LightRAG workspaces",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="naive",
        choices=["naive", "mix", "local", "global", "hybrid"],
        help="LightRAG query mode (start with naive)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete workspace dir before ingest (for clean reruns)",
    )
    args = parser.parse_args()

    indices = parse_instance_indices(args.instance_idx)
    logger.info("Target instances: %s", indices)

    for idx in indices:
        ingest_one_instance(
            instance_idx=idx,
            min_chars=args.min_chars,
            save_dir=args.save_dir,
            mode=args.mode,
            reset=args.reset,
            max_chunks=args.max_chunks,
            output_suffix=args.output_suffix,
        )


if __name__ == "__main__":
    main()
