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
    data_path = Path(f"MemoryAgentBench/preview_samples/Test_Time_Learning/instance_{instance_idx}.json")
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return
    data = json.loads(data_path.read_text(encoding="utf-8"))

    output_dir = Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / (f"ttl_results_{instance_idx}" + (f"_{output_suffix}" if output_suffix else "") + ".json")
    results = {"dataset": "Test_Time_Learning", "instance_idx": instance_idx, "results": {}}
    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text(encoding="utf-8"))
            if existing.get("instance_idx") == instance_idx:
                results = existing
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)

    workspace = Path(storage_dir) / f"lightrag_ttl_{instance_idx}"
    if not workspace.exists():
        logger.error("LightRAG workspace not found: %s", workspace)
        return

    memory = LightRAGMemory(working_dir=str(workspace), mode=mode)
    questions, answers = data["questions"], data["answers"]
    if limit > 0:
        questions, answers = questions[:limit], answers[:limit]

    for adaptor_name in adaptors:
        adaptor_results = results["results"].setdefault(adaptor_name, [])
        for i in range(len(adaptor_results), len(questions)):
            q, a = questions[i], answers[i]
            task_query = q + ("\n\nInstruction: Based on the dialogue history, recommend 5 movies. Output their titles or IDs." if instance_idx == 0 else "\n\nInstruction: Classify the intent of the user query based on the labeled examples in memory. Output ONLY the numeric Label ID.")
            try:
                res, report = run_one_question(adaptor_name, task_query, memory, dataset="Test_Time_Learning", instance_idx=instance_idx, question_idx=i)
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
                adaptor_results.append({"question": q, "error": str(e)})
            out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Instance %s finished. Results saved to %s", instance_idx, out_file)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Test_Time_Learning (LightRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-5")
    parser.add_argument("--adaptor", nargs="+", default=["R1", "R2"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--storage_dir", type=str, default="out/lightrag_storage")
    parser.add_argument("--mode", type=str, default="naive", choices=["naive", "mix", "local", "global", "hybrid"])
    parser.add_argument("--output_suffix", type=str, default="lightrag")
    args = parser.parse_args()

    for idx in parse_instance_indices(args.instance_idx):
        evaluate_instance(idx, args.adaptor, args.limit, args.output_suffix, args.storage_dir, args.mode)


if __name__ == "__main__":
    main()
