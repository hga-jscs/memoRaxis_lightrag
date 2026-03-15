import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("out")
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def collect_reports():
    rows = []
    # sidecar reports
    for p in (ROOT / "token_reports").glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        tr = data.get("token_report", {})
        for ev in tr.get("events", []):
            rows.append({
                "dataset": data.get("dataset", ev.get("dataset", "")),
                "instance_idx": data.get("instance_idx", ev.get("instance_idx", "")),
                "adaptor": ev.get("adaptor", ""),
                "question_idx": ev.get("question_idx", ""),
                "stage": ev.get("stage", ""),
                "substage": ev.get("substage", ""),
                "prompt_tokens": ev.get("prompt_tokens", 0),
                "completion_tokens": ev.get("completion_tokens", 0),
                "total_tokens": ev.get("total_tokens", 0),
                "api_calls": ev.get("api_calls", 0),
            })

    for p in ROOT.glob("*results*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        dataset = data.get("dataset", "")
        instance_idx = data.get("instance_idx", "")
        for adaptor, items in data.get("results", {}).items():
            for i, item in enumerate(items):
                report = item.get("token_report", {})
                for ev in report.get("events", []):
                    rows.append({
                        "dataset": ev.get("dataset", dataset),
                        "instance_idx": ev.get("instance_idx", instance_idx),
                        "adaptor": ev.get("adaptor", adaptor),
                        "question_idx": ev.get("question_idx", i),
                        "stage": ev.get("stage", ""),
                        "substage": ev.get("substage", ""),
                        "prompt_tokens": ev.get("prompt_tokens", 0),
                        "completion_tokens": ev.get("completion_tokens", 0),
                        "total_tokens": ev.get("total_tokens", 0),
                        "api_calls": ev.get("api_calls", 0),
                    })
                for mev in report.get("memory", {}).get("events", []):
                    rows.append({
                        "dataset": dataset,
                        "instance_idx": instance_idx,
                        "adaptor": adaptor,
                        "question_idx": i,
                        "stage": mev.get("stage", ""),
                        "substage": "",
                        "prompt_tokens": mev.get("prompt_tokens", 0),
                        "completion_tokens": mev.get("completion_tokens", 0),
                        "total_tokens": mev.get("total_tokens", 0),
                        "api_calls": mev.get("api_calls", 0),
                    })
    return rows


def summarize(rows):
    by_dataset = defaultdict(int)
    by_adaptor = defaultdict(int)
    by_stage = defaultdict(int)
    for r in rows:
        by_dataset[str(r["dataset"])] += int(r["total_tokens"])
        by_adaptor[str(r["adaptor"])] += int(r["total_tokens"])
        key = f"{r['stage']}.{r['substage']}" if r["substage"] else str(r["stage"])
        by_stage[key] += int(r["total_tokens"])
    return by_dataset, by_adaptor, by_stage


def main():
    rows = collect_reports()
    csv_path = REPORT_DIR / "token_usage_events.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "instance_idx", "adaptor", "question_idx", "stage", "substage", "prompt_tokens", "completion_tokens", "total_tokens", "api_calls"])
        writer.writeheader()
        writer.writerows(rows)

    by_dataset, by_adaptor, by_stage = summarize(rows)
    md = ["# Token Usage Summary", "", "## By Dataset", "|dataset|total_tokens|", "|---|---:|"]
    for k, v in sorted(by_dataset.items()):
        md.append(f"|{k}|{v}|")
    md += ["", "## By Adaptor", "|adaptor|total_tokens|", "|---|---:|"]
    for k, v in sorted(by_adaptor.items()):
        md.append(f"|{k}|{v}|")
    md += ["", "## By Stage/Substage", "|stage|total_tokens|", "|---|---:|"]
    for k, v in sorted(by_stage.items(), key=lambda x: x[1], reverse=True):
        md.append(f"|{k}|{v}|")

    md_path = REPORT_DIR / "token_usage_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
