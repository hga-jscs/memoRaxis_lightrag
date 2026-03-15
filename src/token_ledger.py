from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class TokenEvent:
    run_id: str
    stage: str
    substage: str = ""
    dataset: Optional[str] = None
    instance_idx: Optional[int] = None
    adaptor: Optional[str] = None
    question_idx: Optional[int] = None
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


class TokenLedger:
    def __init__(self, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._events: List[TokenEvent] = []
        self._ctx: Dict[str, Any] = {}

    def bind(self, **ctx: Any) -> None:
        self._ctx.update({k: v for k, v in ctx.items() if v is not None})

    @contextmanager
    def scope(self, **ctx: Any) -> Iterator["TokenLedger"]:
        old = dict(self._ctx)
        self.bind(**ctx)
        try:
            yield self
        finally:
            self._ctx = old

    def add(
        self,
        *,
        stage: str,
        substage: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
        api_calls: int = 1,
        model: str = "",
        provider: str = "",
        **meta: Any,
    ) -> None:
        if total_tokens is None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)

        event = TokenEvent(
            run_id=self.run_id,
            stage=stage,
            substage=substage,
            dataset=self._ctx.get("dataset"),
            instance_idx=self._ctx.get("instance_idx"),
            adaptor=self._ctx.get("adaptor"),
            question_idx=self._ctx.get("question_idx"),
            model=model,
            provider=provider,
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            total_tokens=int(total_tokens or 0),
            api_calls=int(api_calls or 0),
            meta=meta,
        )
        self._events.append(event)

    @property
    def events(self) -> List[Dict[str, Any]]:
        return [asdict(x) for x in self._events]

    def summarize(self) -> Dict[str, Any]:
        by_stage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_calls": 0,
            }
        )

        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}

        for ev in self._events:
            key = ev.stage if not ev.substage else f"{ev.stage}.{ev.substage}"
            by_stage[key]["prompt_tokens"] += ev.prompt_tokens
            by_stage[key]["completion_tokens"] += ev.completion_tokens
            by_stage[key]["total_tokens"] += ev.total_tokens
            by_stage[key]["api_calls"] += ev.api_calls

            total["prompt_tokens"] += ev.prompt_tokens
            total["completion_tokens"] += ev.completion_tokens
            total["total_tokens"] += ev.total_tokens
            total["api_calls"] += ev.api_calls

        return {
            "run_id": self.run_id,
            "total": total,
            "by_stage": dict(by_stage),
            "events": self.events,
        }

    def dump_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summarize(), ensure_ascii=False, indent=2), encoding="utf-8")
