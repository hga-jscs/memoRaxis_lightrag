import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.benchmark_utils import parse_instance_indices
from src.lightrag_memory import LightRAGMemory
from src.logger import get_logger
from src.runner_utils import run_one_question

logger = get_logger()


def evaluate_instance(instance_idx: int, adaptors: list, limit: int = -1, output_suffix: str = "", storage_dir: str = "out/lightrag_storage", mode: str = "naive"):
    logger.info("=== Evaluating Conflict_Resolution Instance %s (LightRAG) ===", instance_idx)

    data_path = Path(f"MemoryAgentBench/preview_samples/Conflict_Resolution/instance_{instance_idx}.json")
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return

    data = json.loads(data_path.read_text(encoding="utf-8"))
    workspace = Path(storage_dir) / f"lightrag_conflict_{instance_idx}"
    if not workspace.exists():
        logger.error("LightRAG workspace not found: %s (run ingest first)", workspace)
        return

    memory = LightRAGMemory(working_dir=str(workspace), mode=mode)
    questions, answers = data["questions"], data["answers"]
    if limit > 0:
        questions, answers = questions[:limit], answers[:limit]

    results = {"dataset": "Conflict_Resolution", "instance_idx": instance_idx, "results": {}}

    for adaptor_name in adaptors:
        adaptor_results = []
        for i, (q, a) in enumerate(zip(questions, answers)):
            logger.info("[%s] Q%s/%s", adaptor_name, i + 1, len(questions))
            try:
                res, report = run_one_question(
                    adaptor_name,
                    q,
                    memory,
                    dataset="Conflict_Resolution",
                    instance_idx=instance_idx,
                    question_idx=i,
                )
                logger.info("[%s] token_debug=%s", adaptor_name, json.dumps(report.get("by_stage", {}), ensure_ascii=False))
                adaptor_results.append({
                    "question": q,
                    "answer": res.answer,
                    "ground_truth": a,
                    "steps": res.steps_taken,
                    "tokens": report["grand_total_tokens"],
                    "replan": res.replan_count,
                    "token_report": report,
                    "token_breakdown": res.token_breakdown,
                    "memory_token_breakdown": res.memory_token_breakdown,
                })
            except Exception as e:
                logger.error("Error on Q%s: %s", i, e)
                adaptor_results.append({"question": q, "error": str(e)})

        results["results"][adaptor_name] = adaptor_results

    output_dir = Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"conflict_res_results_{instance_idx}" + (f"_{output_suffix}" if output_suffix else "") + ".json"
    out_file = output_dir / filename
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results saved to %s", out_file)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Conflict_Resolution (LightRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-7")
    parser.add_argument("--adaptor", nargs="+", default=["R1", "R2"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--storage_dir", type=str, default="out/lightrag_storage")
    parser.add_argument("--mode", type=str, default="naive", choices=["naive", "mix", "local", "global", "hybrid"])
    parser.add_argument("--output_suffix", type=str, default="")
    args = parser.parse_args()

    for idx in parse_instance_indices(args.instance_idx):
        evaluate_instance(idx, args.adaptor, args.limit, args.output_suffix, args.storage_dir, args.mode)


if __name__ == "__main__":
    main()
