import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.benchmark_utils import load_benchmark_data, parse_instance_indices
from src.lightrag_memory import LightRAGMemory
from src.logger import get_logger
from src.runner_utils import run_one_question

logger = get_logger()


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def is_correct_mechanical(prediction: str, references: List[str]) -> bool:
    negative_patterns = [
        "does not contain any information", "insufficient information", "not mentioned in the context",
        "no information related to", "上下文没有提到", "没有找到相关信息", "信息不足",
    ]
    pred_norm = (prediction or "").lower()
    if any(pattern in pred_norm for pattern in negative_patterns):
        return False
    pred_n = normalize_text(prediction)
    return any((normalize_text(ref) in pred_n) for ref in references or [] if normalize_text(ref))


def load_ground_truth_acc_ret(instance_idx: int) -> Dict[str, List[str]]:
    gt_path = Path(f"MemoryAgentBench/preview_samples/Accurate_Retrieval/instance_{instance_idx}.json")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth not found: {gt_path}")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    qa_map: Dict[str, List[str]] = {}
    for q, a in zip(gt.get("questions", []), gt.get("answers", [])):
        qa_map[str(q)] = a if isinstance(a, list) else [str(a)]
    return qa_map


def score_and_annotate(results: Dict[str, List[dict]], qa_map: Dict[str, List[str]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for adaptor_name, preds in results.items():
        correct = 0
        for item in preds:
            ok = is_correct_mechanical(item.get("answer", ""), qa_map.get(item.get("question", ""), []))
            item["is_correct"] = bool(ok)
            correct += int(ok)
        total = len(preds)
        summary[adaptor_name] = {"accuracy": (correct / total) if total else 0.0, "correct_count": correct, "total_questions": total}
    return summary


def evaluate_adaptor(name: str, questions: list, limit: int, memory: LightRAGMemory, instance_idx: int) -> list:
    results = []
    target_questions = questions if limit == -1 else questions[:limit]
    for i, q in enumerate(target_questions):
        logger.info("[%s] Running Q%s/%s", name, i + 1, len(target_questions))
        try:
            res, report = run_one_question(name, q, memory, dataset="Accurate_Retrieval", instance_idx=instance_idx, question_idx=i)
            logger.info("[%s] token_debug=%s", name, json.dumps(report.get("by_stage", {}), ensure_ascii=False))
            results.append({
                "question": q,
                "answer": res.answer,
                "steps": res.steps_taken,
                "tokens": report["grand_total_tokens"],
                "replan": res.replan_count,
                "token_report": report,
                "token_breakdown": res.token_breakdown,
                "memory_token_breakdown": res.memory_token_breakdown,
            })
        except Exception as e:
            logger.error("[%s] Failed on Q%s: %s", name, i + 1, e)
            results.append({"question": q, "error": str(e)})
    return results


def evaluate_one_instance(instance_idx: int, adaptors_to_run: List[str], limit: int, storage_dir: str, mode: str, output_suffix: str = "", print_scores: bool = True):
    data = load_benchmark_data("MemoryAgentBench/data/Accurate_Retrieval-00000-of-00001.parquet", instance_idx)
    questions = list(data["questions"])
    workspace = Path(storage_dir) / f"lightrag_acc_ret_{instance_idx}"
    if not workspace.exists():
        logger.error("LightRAG workspace not found: %s", workspace)
        return

    memory = LightRAGMemory(working_dir=str(workspace), mode=mode)
    results: Dict[str, list] = {}
    for adaptor_name in ["R1", "R2", "R3"]:
        if "all" in adaptors_to_run or adaptor_name in adaptors_to_run:
            results[adaptor_name] = evaluate_adaptor(adaptor_name, questions, limit, memory, instance_idx)

    final_report = {"dataset": "Accurate_Retrieval", "instance_idx": instance_idx, "results": results}
    try:
        score_summary = score_and_annotate(final_report["results"], load_ground_truth_acc_ret(instance_idx))
        final_report["scores"] = {"mechanical_accuracy": score_summary}
        if print_scores:
            for adaptor_name, m in score_summary.items():
                logger.info("[%s] Accuracy=%.4f (%s/%s)", adaptor_name, m["accuracy"], m["correct_count"], m["total_questions"])
    except Exception as e:
        logger.warning("Scoring failed: %s", e)

    output_dir = Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"acc_ret_results_{instance_idx}" + (f"_{output_suffix}" if output_suffix else "") + ".json"
    out_file = output_dir / filename
    out_file.write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved results to %s", out_file)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Accurate_Retrieval (LightRAG)")
    parser.add_argument("--instance_idx", type=str, default="0-14")
    parser.add_argument("--adaptor", nargs="+", default=["all"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--storage_dir", type=str, default="out/lightrag_storage")
    parser.add_argument("--mode", type=str, default="naive", choices=["naive", "mix", "local", "global", "hybrid"])
    parser.add_argument("--output_suffix", type=str, default="lightrag")
    parser.add_argument("--no_print_scores", action="store_true")
    args = parser.parse_args()

    for idx in parse_instance_indices(args.instance_idx):
        evaluate_one_instance(idx, args.adaptor, args.limit, args.storage_dir, args.mode, args.output_suffix, not args.no_print_scores)


if __name__ == "__main__":
    main()
