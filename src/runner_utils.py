from __future__ import annotations

from typing import Dict, Tuple

from .adaptors import AdaptorResult, IterativeAdaptor, PlanAndActAdaptor, SingleTurnAdaptor
from .config import get_config
from .llm_interface import OpenAIClient
from .lightrag_memory import LightRAGMemory
from .token_ledger import TokenLedger

ADAPTOR_MAP = {
    "R1": SingleTurnAdaptor,
    "R2": IterativeAdaptor,
    "R3": PlanAndActAdaptor,
}


def run_one_question(
    adaptor_name: str,
    task: str,
    memory: LightRAGMemory,
    *,
    dataset: str,
    instance_idx: int,
    question_idx: int,
) -> Tuple[AdaptorResult, Dict]:
    conf = get_config()
    ledger = TokenLedger()

    with ledger.scope(
        dataset=dataset,
        instance_idx=instance_idx,
        adaptor=adaptor_name,
        question_idx=question_idx,
    ):
        llm = OpenAIClient(
            api_key=conf.llm["api_key"],
            base_url=conf.llm["base_url"],
            model=conf.llm["model"],
            ledger=ledger,
        )
        adaptor_cls = ADAPTOR_MAP[adaptor_name]
        adaptor = adaptor_cls(llm, memory)
        result = adaptor.run(task)

    report = ledger.summarize()
    report["memory"] = result.memory_token_breakdown
    report["grand_total_tokens"] = (
        report["total"]["total_tokens"] + report.get("memory", {}).get("total", {}).get("total_tokens", 0)
    )
    return result, report
