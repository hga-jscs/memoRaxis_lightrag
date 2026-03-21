import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.logger import get_logger
from src.benchmark_utils import parse_instance_indices, chunk_context
from src.lightrag_memory import LightRAGMemory

logger = get_logger()


def _probe_path(prefix: str, instance_idx: int, output_suffix: str) -> Path:
    suffix_part = f"_{output_suffix}" if output_suffix else ""
    return Path("out") / f"{prefix}_ingest_probe_{instance_idx}{suffix_part}.json"


def ingest_one_instance(
    instance_idx: int,
    chunk_size: int,
    overlap: int,
    save_dir: str,
    mode: str,
    reset: bool,
    max_chunks: int | None,
    output_suffix: str,
):
    logger.info("=== Processing Long_Range_Understanding Instance %s (LightRAG) ===", instance_idx)
    data_path = f"MemoryAgentBench/preview_samples/Long_Range_Understanding/instance_{instance_idx}.json"

    if not Path(data_path).exists():
        logger.error("Data file not found: %s", data_path)
        return

    try:
        data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Error loading instance %s: %s", instance_idx, e)
        return

    chunks = chunk_context(data["context"], chunk_size=chunk_size, overlap=overlap)
    original_chunk_count = len(chunks)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    final_chunk_count = len(chunks)

    logger.info(
        "[ingest-debug] dataset=Long_Range_Understanding instance=%s original_chunks=%s final_chunks=%s chunk_size=%s overlap=%s",
        instance_idx,
        original_chunk_count,
        final_chunk_count,
        chunk_size,
        overlap,
    )

    workspace_name = f"lightrag_long_range_{instance_idx}"
    if output_suffix:
        workspace_name += f"_{output_suffix}"
    workspace = Path(save_dir) / workspace_name
    workspace.mkdir(parents=True, exist_ok=True)

    memory = LightRAGMemory(working_dir=str(workspace), mode=mode)
    if reset:
        memory.reset()

    print(f"Starting ingestion for Instance {instance_idx} ({final_chunk_count} chunks) into {workspace} ...")
    for i, chunk in enumerate(chunks, start=1):
        memory.add_memory(chunk, metadata={"chunk_id": i - 1, "instance_idx": instance_idx})
        print(
            f"[ingest-progress] instance={instance_idx} queued={i}/{final_chunk_count} chunk_chars={len(chunk)}",
            end="\r",
            flush=True,
        )

    memory.begin_token_session()
    memory.build_index(doc_id=f"long_range_{instance_idx}")
    ingest_token_summary = memory.end_token_session()

    probe = {
        "dataset": "Long_Range_Understanding",
        "instance_idx": instance_idx,
        "original_chunk_count": original_chunk_count,
        "final_chunk_count": final_chunk_count,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "char_count": len(data.get("context", "")),
        "mode": mode,
        "workspace": str(workspace),
        "output_suffix": output_suffix,
        "ingest_token_summary": ingest_token_summary,
    }
    probe_path = _probe_path("long_range", instance_idx, output_suffix)
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
    parser = argparse.ArgumentParser(description="Ingest Long_Range_Understanding data (LightRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-39", help="Index range (e.g., '0-39')")
    parser.add_argument("--chunk_size", type=int, default=1200, help="Chunk size for sliding window")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap for sliding window")
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
    logger.info("Config: Chunk Size=%s, Overlap=%s, Mode=%s, Max Chunks=%s", args.chunk_size, args.overlap, args.mode, args.max_chunks)

    for idx in indices:
        ingest_one_instance(
            instance_idx=idx,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            save_dir=args.save_dir,
            mode=args.mode,
            reset=args.reset,
            max_chunks=args.max_chunks,
            output_suffix=args.output_suffix,
        )


if __name__ == "__main__":
    main()
