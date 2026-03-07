# -*- coding: utf-8 -*-
"""
run_all_tasks.py

用于串联 LightRAG 子项目在 MemoryAgentBench 上的四个任务：
1. Accurate_Retrieval
2. Conflict_Resolution
3. Long_Range_Understanding
4. Test_Time_Learning

流程：
- convert preview json
- ingest
- infer
- evaluate

示例：
python run_all_tasks.py --reset
python run_all_tasks.py --tasks acc conflict
python run_all_tasks.py --adaptors R1 R2
python run_all_tasks.py --skip_ingest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd: List[str]) -> None:
    print("\n" + "=" * 100)
    print("RUN:", " ".join([repr(x) if x == "" else x for x in cmd]))
    print("=" * 100)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} 不存在: {path}")


def parse_indices(spec: str) -> List[int]:
    """
    支持:
    0
    0-3
    0,2,5
    0-2,5,7-8
    """
    spec = spec.strip()
    if not spec:
        return []
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            for x in range(start, end + 1):
                result.add(x)
        else:
            result.add(int(part))
    return sorted(result)


def build_common_infer_args(adaptors: List[str], limit: int, storage_dir: str, mode: str, output_suffix: str) -> List[str]:
    args = [
        "--adaptor",
        *adaptors,
        "--limit",
        str(limit),
        "--storage_dir",
        storage_dir,
        "--mode",
        mode,
        "--output_suffix",
        output_suffix,
    ]
    return args


def preprocess_preview_samples(python_exe: str) -> None:
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/data/convert_all_data.py",
    ]
    run_cmd(cmd)


def ingest_acc(python_exe: str, instance_idx: str, chunk_size: int, storage_dir: str, mode: str, reset: bool) -> None:
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/ingest/ingest_accurate_retrieval.py",
        "--instance_idx",
        instance_idx,
        "--chunk_size",
        str(chunk_size),
        "--save_dir",
        storage_dir,
        "--mode",
        mode,
    ]
    if reset:
        cmd.append("--reset")
    run_cmd(cmd)


def infer_acc(python_exe: str, instance_idx: str, adaptors: List[str], limit: int, storage_dir: str, mode: str, output_suffix: str) -> None:
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/infer/infer_accurate_retrieval.py",
        "--instance_idx",
        instance_idx,
        *build_common_infer_args(adaptors, limit, storage_dir, mode, output_suffix),
    ]
    run_cmd(cmd)


def eval_acc(python_exe: str, instance_idx: str, output_suffix: str) -> None:
    for idx in parse_indices(instance_idx):
        result_file = PROJECT_ROOT / "out" / f"acc_ret_results_{idx}{'_' + output_suffix if output_suffix else ''}.json"
        gt_file = PROJECT_ROOT / "MemoryAgentBench" / "preview_samples" / "Accurate_Retrieval" / f"instance_{idx}.json"

        ensure_exists(result_file, "Accurate_Retrieval 推理结果")
        ensure_exists(gt_file, "Accurate_Retrieval ground truth")

        cmd = [
            python_exe,
            "scripts/LightRAG_MAB/evaluate/evaluate_mechanical.py",
            "--results",
            str(result_file),
            "--instance",
            str(gt_file),
        ]
        run_cmd(cmd)


def ingest_conflict(python_exe: str, instance_idx: str, min_chars: int, storage_dir: str, mode: str, reset: bool) -> None:
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/ingest/ingest_conflict_resolution.py",
        "--instance_idx",
        instance_idx,
        "--min_chars",
        str(min_chars),
        "--save_dir",
        storage_dir,
        "--mode",
        mode,
    ]
    if reset:
        cmd.append("--reset")
    run_cmd(cmd)


def infer_conflict(python_exe: str, instance_idx: str, adaptors: List[str], limit: int, storage_dir: str, mode: str) -> None:
    # 注意：
    # evaluate_conflict_official.py 默认匹配 out/conflict_res_results_*.json，
    # 为避免 suffix 导致匹配不到，这里强制 output_suffix=""
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/infer/infer_conflict_resolution.py",
        "--instance_idx",
        instance_idx,
        "--adaptor",
        *adaptors,
        "--limit",
        str(limit),
        "--storage_dir",
        storage_dir,
        "--mode",
        mode,
        "--output_suffix",
        "",
    ]
    run_cmd(cmd)


def eval_conflict(python_exe: str) -> None:
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/evaluate/evaluate_conflict_official.py",
    ]
    run_cmd(cmd)


def ingest_long(python_exe: str, instance_idx: str, chunk_size: int, overlap: int, storage_dir: str, mode: str, reset: bool) -> None:
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/ingest/ingest_long_range.py",
        "--instance_idx",
        instance_idx,
        "--chunk_size",
        str(chunk_size),
        "--overlap",
        str(overlap),
        "--save_dir",
        storage_dir,
        "--mode",
        mode,
    ]
    if reset:
        cmd.append("--reset")
    run_cmd(cmd)


def infer_long(python_exe: str, instance_idx: str, adaptors: List[str], limit: int, storage_dir: str, mode: str, output_suffix: str) -> None:
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/infer/infer_long_range.py",
        "--instance_idx",
        instance_idx,
        "--adaptor",
        *adaptors,
        "--limit",
        str(limit),
        "--storage_dir",
        storage_dir,
        "--mode",
        mode,
        "--output_suffix",
        output_suffix,
    ]
    run_cmd(cmd)


def eval_long(python_exe: str, instance_idx: str, output_suffix: str) -> None:
    for idx in parse_indices(instance_idx):
        result_file = PROJECT_ROOT / "out" / f"long_range_results_{idx}{'_' + output_suffix if output_suffix else ''}.json"
        ensure_exists(result_file, "Long_Range_Understanding 推理结果")

        cmd = [
            python_exe,
            "scripts/LightRAG_MAB/evaluate/evaluate_long_range_A.py",
            "--results",
            str(result_file),
            "--instance_folder",
            "MemoryAgentBench/preview_samples/Long_Range_Understanding",
        ]
        run_cmd(cmd)


def ingest_ttl(python_exe: str, instance_idx: str, storage_dir: str, mode: str, reset: bool) -> None:
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/ingest/ingest_test_time.py",
        "--instance_idx",
        instance_idx,
        "--save_dir",
        storage_dir,
        "--mode",
        mode,
    ]
    if reset:
        cmd.append("--reset")
    run_cmd(cmd)


def infer_ttl(python_exe: str, instance_idx: str, adaptors: List[str], limit: int, storage_dir: str, mode: str, output_suffix: str) -> None:
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/infer/infer_test_time.py",
        "--instance_idx",
        instance_idx,
        "--adaptor",
        *adaptors,
        "--limit",
        str(limit),
        "--storage_dir",
        storage_dir,
        "--mode",
        mode,
        "--output_suffix",
        output_suffix,
    ]
    run_cmd(cmd)


def eval_ttl(python_exe: str, output_suffix: str) -> None:
    pattern = f"out/ttl_results_*{'_' + output_suffix if output_suffix else ''}.json"
    cmd = [
        python_exe,
        "scripts/LightRAG_MAB/evaluate/evaluate_ttl_mechanical.py",
        "--results_pattern",
        pattern,
    ]
    run_cmd(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all 4 LightRAG tasks end-to-end")

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["acc", "conflict", "long", "ttl"],
        choices=["acc", "conflict", "long", "ttl"],
        help="要执行的任务集合，默认四个全跑",
    )
    parser.add_argument(
        "--adaptors",
        nargs="+",
        default=["R1", "R2", "R3"],
        choices=["R1", "R2", "R3", "all"],
        help="推理适配器，默认 R1 R2 R3",
    )
    parser.add_argument("--mode", type=str, default="naive", choices=["naive", "mix", "local", "global", "hybrid"])
    parser.add_argument("--storage_dir", type=str, default="out/lightrag_storage")
    parser.add_argument("--output_suffix", type=str, default="", help="建议默认留空，便于评测脚本直接匹配")
    parser.add_argument("--reset", action="store_true", help="ingest 前重置对应 workspace")

    parser.add_argument("--acc_instance_idx", type=str, default="0")
    parser.add_argument("--acc_limit", type=int, default=5)
    parser.add_argument("--acc_chunk_size", type=int, default=850)

    parser.add_argument("--conflict_instance_idx", type=str, default="0-7")
    parser.add_argument("--conflict_limit", type=int, default=-1)
    parser.add_argument("--conflict_min_chars", type=int, default=800)

    parser.add_argument("--long_instance_idx", type=str, default="0-39")
    parser.add_argument("--long_limit", type=int, default=-1)
    parser.add_argument("--long_chunk_size", type=int, default=1200)
    parser.add_argument("--long_overlap", type=int, default=100)

    parser.add_argument("--ttl_instance_idx", type=str, default="0-5")
    parser.add_argument("--ttl_limit", type=int, default=-1)

    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_ingest", action="store_true")
    parser.add_argument("--skip_infer", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")

    args = parser.parse_args()

    python_exe = sys.executable

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    ensure_exists(config_path, "配置文件")

    if not args.skip_preprocess:
        preprocess_preview_samples(python_exe)

    if "acc" in args.tasks:
        if not args.skip_ingest:
            ingest_acc(
                python_exe=python_exe,
                instance_idx=args.acc_instance_idx,
                chunk_size=args.acc_chunk_size,
                storage_dir=args.storage_dir,
                mode=args.mode,
                reset=args.reset,
            )
        if not args.skip_infer:
            infer_acc(
                python_exe=python_exe,
                instance_idx=args.acc_instance_idx,
                adaptors=args.adaptors,
                limit=args.acc_limit,
                storage_dir=args.storage_dir,
                mode=args.mode,
                output_suffix=args.output_suffix,
            )
        if not args.skip_eval:
            eval_acc(
                python_exe=python_exe,
                instance_idx=args.acc_instance_idx,
                output_suffix=args.output_suffix,
            )

    if "conflict" in args.tasks:
        if not args.skip_ingest:
            ingest_conflict(
                python_exe=python_exe,
                instance_idx=args.conflict_instance_idx,
                min_chars=args.conflict_min_chars,
                storage_dir=args.storage_dir,
                mode=args.mode,
                reset=args.reset,
            )
        if not args.skip_infer:
            infer_conflict(
                python_exe=python_exe,
                instance_idx=args.conflict_instance_idx,
                adaptors=args.adaptors,
                limit=args.conflict_limit,
                storage_dir=args.storage_dir,
                mode=args.mode,
            )
        if not args.skip_eval:
            eval_conflict(python_exe=python_exe)

    if "long" in args.tasks:
        if not args.skip_ingest:
            ingest_long(
                python_exe=python_exe,
                instance_idx=args.long_instance_idx,
                chunk_size=args.long_chunk_size,
                overlap=args.long_overlap,
                storage_dir=args.storage_dir,
                mode=args.mode,
                reset=args.reset,
            )
        if not args.skip_infer:
            infer_long(
                python_exe=python_exe,
                instance_idx=args.long_instance_idx,
                adaptors=args.adaptors,
                limit=args.long_limit,
                storage_dir=args.storage_dir,
                mode=args.mode,
                output_suffix=args.output_suffix,
            )
        if not args.skip_eval:
            eval_long(
                python_exe=python_exe,
                instance_idx=args.long_instance_idx,
                output_suffix=args.output_suffix,
            )

    if "ttl" in args.tasks:
        if not args.skip_ingest:
            ingest_ttl(
                python_exe=python_exe,
                instance_idx=args.ttl_instance_idx,
                storage_dir=args.storage_dir,
                mode=args.mode,
                reset=args.reset,
            )
        if not args.skip_infer:
            infer_ttl(
                python_exe=python_exe,
                instance_idx=args.ttl_instance_idx,
                adaptors=args.adaptors,
                limit=args.ttl_limit,
                storage_dir=args.storage_dir,
                mode=args.mode,
                output_suffix=args.output_suffix,
            )
        if not args.skip_eval:
            eval_ttl(
                python_exe=python_exe,
                output_suffix=args.output_suffix,
            )

    print("\n全部完成。")


if __name__ == "__main__":
    main()