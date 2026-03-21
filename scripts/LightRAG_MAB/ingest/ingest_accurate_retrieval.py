import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import load_benchmark_data, chunk_context, parse_instance_indices
from src.lightrag_memory import LightRAGMemory

logger = get_logger()


def _probe_path(prefix: str, instance_idx: int, output_suffix: str) -> Path:
    suffix_part = f"_{output_suffix}" if output_suffix else ""
    return Path("out") / f"{prefix}_ingest_probe_{instance_idx}{suffix_part}.json"


def ingest_one_instance(
    instance_idx: int,
    chunk_size: int,
    save_dir: str,
    mode: str,
    reset: bool,
    max_chunks: int | None,
    output_suffix: str,
):
    logger.info("=== Processing Instance %s (LightRAG) ===", instance_idx)
    data_path = "MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet"

    try:
        data = load_benchmark_data(data_path, instance_idx)
    except Exception as e:
        logger.error("Error loading instance %s: %s", instance_idx, e)
        return

    chunks = chunk_context(data["context"], chunk_size=chunk_size)
    original_chunk_count = len(chunks)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    final_chunk_count = len(chunks)

    workspace_name = f"lightrag_acc_ret_{instance_idx}"
    if output_suffix:
        workspace_name += f"_{output_suffix}"
    out_dir = Path(save_dir) / workspace_name
    out_dir.mkdir(parents=True, exist_ok=True)

    memory = LightRAGMemory(working_dir=str(out_dir), mode=mode)
    if reset:
        memory.reset()

    logger.info(
        "[ingest-debug] dataset=Accurate_Retrieval instance=%s original_chunks=%s final_chunks=%s chunk_size=%s workspace=%s",
        instance_idx,
        original_chunk_count,
        final_chunk_count,
        chunk_size,
        out_dir,
    )
    print(f"Starting ingestion of {final_chunk_count} chunks into {out_dir} ...")
    for i, chunk in enumerate(chunks, start=1):
        memory.add_memory(chunk, metadata={"doc_id": i - 1, "instance_idx": instance_idx})
        print(
            f"[ingest-progress] instance={instance_idx} queued={i}/{final_chunk_count} chunk_chars={len(chunk)}",
            end="\r",
            flush=True,
        )

    memory.begin_token_session()
    memory.build_index(doc_id=f"acc_ret_{instance_idx}")
    ingest_token_summary = memory.end_token_session()

    probe = {
        "dataset": "Accurate_Retrieval",
        "instance_idx": instance_idx,
        "original_chunk_count": original_chunk_count,
        "final_chunk_count": final_chunk_count,
        "chunk_size": chunk_size,
        "char_count": len(data.get("context", "")),
        "mode": mode,
        "workspace": str(out_dir),
        "output_suffix": output_suffix,
        "ingest_token_summary": ingest_token_summary,
    }
    probe_path = _probe_path("acc_ret", instance_idx, output_suffix)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path.write_text(json.dumps(probe, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "[ingest-probe] saved=%s total_tokens=%s",
        probe_path,
        probe.get("ingest_token_summary", {}).get("total", {}).get("total_tokens", 0),
    )
    print(f"\nIngestion complete. LightRAG workspace saved at: {out_dir}")
    print(f"[ingest-probe] {probe_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest MemoryAgentBench data (LightRAG)")
    parser.add_argument(
        "--instance_idx",
        type=str,
        default="0",
        help="Index range (e.g., '0', '0-5', '1,3')",
    )
    parser.add_argument("--chunk_size", type=int, default=850, help="Fallback chunk size")
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
            chunk_size=args.chunk_size,
            save_dir=args.save_dir,
            mode=args.mode,
            reset=args.reset,
            max_chunks=args.max_chunks,
            output_suffix=args.output_suffix,
        )


if __name__ == "__main__":
    main()
