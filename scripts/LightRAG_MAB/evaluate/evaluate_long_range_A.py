import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.config import get_config
from src.llm_interface import OpenAIClient
from src.logger import get_logger
from src.token_ledger import TokenLedger

logger = get_logger()

FLUENCY_PROMPT = """请评价以下摘要的流畅性，返回 JSON: {\"fluency\": 0|1|2|3}\n摘要:\n{summary}"""
RECALL_PROMPT = """请根据关键点判断摘要覆盖了多少项，返回 JSON: {\"recall\": <int>}\n关键点:\n{keypoints}\n摘要:\n{summary}"""
PRECISION_PROMPT = """请判断摘要中有多少句与专家摘要一致，返回 JSON: {\"precision\": <int>, \"sentence_count\": <int>}\n专家摘要:\n{expert_summary}\n摘要:\n{summary}"""


class SummarizationJudge:
    def __init__(self, ledger: TokenLedger):
        conf = get_config()
        self.ledger = ledger
        self.llm = OpenAIClient(api_key=conf.llm["api_key"], base_url=conf.llm["base_url"], model=conf.llm["model"], ledger=ledger)

    def judge_fluency(self, summary: str) -> int:
        prompt = FLUENCY_PROMPT.format(summary=summary)
        res = self.llm.generate_json(prompt, stage="evaluate.lrua", substage="fluency")
        return int(res.get("fluency", 0))

    def judge_recall(self, summary: str, keypoints: list) -> int:
        prompt = RECALL_PROMPT.format(keypoints="\n".join(f"- {k}" for k in keypoints), summary=summary)
        res = self.llm.generate_json(prompt, stage="evaluate.lrua", substage="recall")
        return int(res.get("recall", 0))

    def judge_precision(self, summary: str, expert_summary: str) -> tuple[int, int]:
        prompt = PRECISION_PROMPT.format(expert_summary=expert_summary, summary=summary)
        res = self.llm.generate_json(prompt, stage="evaluate.lrua", substage="precision")
        return int(res.get("precision", 0)), int(res.get("sentence_count", 1))


def main():
    parser = argparse.ArgumentParser(description="LRU-A Summarization Evaluator")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--instance_folder", type=str, default="MemoryAgentBench/preview_samples/Long_Range_Understanding")
    args = parser.parse_args()

    results_data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    instance_idx = results_data["instance_idx"]
    ground_truth = json.loads((Path(args.instance_folder) / f"instance_{instance_idx}.json").read_text(encoding="utf-8"))

    keypoints = ground_truth["metadata"].get("keypoints", [])
    expert_summary = ground_truth["answers"][0] if ground_truth.get("answers") else ""

    ledger = TokenLedger()
    judge = SummarizationJudge(ledger)

    final_eval = {"dataset": "LRU-A", "instance_idx": instance_idx, "metrics": {}}
    for adaptor_name, predictions in results_data["results"].items():
        if not predictions:
            continue
        prediction = predictions[0].get("answer", "")
        with ledger.scope(dataset="Long_Range_Understanding", instance_idx=instance_idx, adaptor=adaptor_name, question_idx=0):
            f_score = judge.judge_fluency(prediction)
            r_found = judge.judge_recall(prediction, keypoints)
            p_found, p_total = judge.judge_precision(prediction, expert_summary)

        recall = r_found / len(keypoints) if keypoints else 0
        precision = p_found / p_total if p_total > 0 else 0
        f1 = f_score * 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
        final_eval["metrics"][adaptor_name] = {
            "fluency": f_score,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "raw": {"recall_found": r_found, "recall_total": len(keypoints), "precision_found": p_found, "precision_total": p_total},
        }

    final_eval["token_report"] = ledger.summarize()
    logger.info("evaluate token_debug=%s", json.dumps(final_eval["token_report"].get("by_stage", {}), ensure_ascii=False))

    output_dir = Path("out/eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"eval_lru_a_{instance_idx}.json"
    output_file.write_text(json.dumps(final_eval, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"LRU-A Evaluation Report for Instance {instance_idx} saved to {output_file}")


if __name__ == "__main__":
    main()
