"""GraphForge 命令行入口。

    python -m graphforge.cli info
    python -m graphforge.cli run       --seed 0 --graph karate
    python -m graphforge.cli benchmark --seeds 0 1 2 --graph lfr --output benchmark.json
    python -m graphforge.cli operators
    python -m graphforge.cli heuristics

作者: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .core.capabilities import capability_report
from .core.config import load_config
from .core.errors import GraphForgeError
from .graph.edges import available_operators
from .graph.heuristics import available_heuristics
from .pipeline.benchmark import MAIN_METHOD, run_benchmark, to_jsonable
from .pipeline.link_prediction import LinkPredictionPipeline


def _configure_stdout() -> None:
    """Windows 控制台编码兜底。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - 非标准 stdout 时忽略
        pass


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2))


# --------------------------------------------------------------------- 子命令
def cmd_info(_args: argparse.Namespace) -> int:
    report = capability_report()
    print("GraphForge capability report")
    for key in sorted(report):
        print(f"  {key:<12}: {report[key]}")
    print(f"  {'operators':<12}: {', '.join(available_operators())}")
    print(f"  {'heuristics':<12}: {', '.join(available_heuristics())}")
    return 0


def cmd_operators(_args: argparse.Namespace) -> int:
    for name in available_operators():
        print(name)
    return 0


def cmd_heuristics(_args: argparse.Namespace) -> int:
    for name in available_heuristics():
        print(name)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(
        seed=args.seed,
        graph=args.graph,
        n_nodes=args.nodes,
        n_communities=args.communities,
        homophily=args.homophily,
        avg_degree=args.avg_degree,
        p=args.p,
        q=args.q,
        dim=args.dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window=args.window,
        negative_samples=args.negative,
        epochs=args.epochs,
        edge_operator=args.operator,
        normalize_embeddings=args.normalize_embeddings,
        use_heuristic_features=args.heuristic_features,
        classifier=args.classifier,
        classifier_c=args.classifier_c,
        hpo_enabled=args.hpo,
        hpo_trials=args.trials,
        tier1_only=args.tier1_only,
    )
    pipeline = LinkPredictionPipeline(config, seed=args.seed, force_tier1=args.tier1_only)
    output = pipeline.run()
    print(f"✅ seed={output.seed} graph={output.graph.name} "
          f"nodes={output.graph.n_nodes} edges={output.graph.n_edges}")
    print(f"✅ AUC={output.result.auc:.4f} AP={output.result.ap:.4f} "
          f"ACC={output.result.accuracy:.4f} ({output.result.seconds:.2f}s)")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(to_jsonable(output.as_dict()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"✅ 结果已写入 {path}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = load_config(
        seed=args.seed,
        graph=args.graph,
        n_nodes=args.nodes,
        n_communities=args.communities,
        homophily=args.homophily,
        avg_degree=args.avg_degree,
        p=args.p,
        q=args.q,
        dim=args.dim,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        window=args.window,
        negative_samples=args.negative,
        epochs=args.epochs,
        edge_operator=args.operator,
        normalize_embeddings=args.normalize_embeddings,
        use_heuristic_features=args.heuristic_features,
        classifier=args.classifier,
        classifier_c=args.classifier_c,
        hpo_enabled=args.hpo,
        hpo_trials=args.trials,
        tier1_only=args.tier1_only,
    )
    seeds = tuple(int(s) for s in args.seeds)
    payload = run_benchmark(
        config,
        seeds=seeds,
        hpo_trials=args.trials,
        force_tier1=args.tier1_only,
        avg_degree=args.avg_degree,
        verbose=True,
    )
    payload["main_method"] = MAIN_METHOD
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✅ benchmark 已写入 {path}")
    if args.json_stdout:
        _print_json(payload)
    else:
        print(payload["table"])
    return 0


# ------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphforge",
        description="GraphForge —— node2vec 图表示学习与链接预测（作者: 晨星）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="打印后端能力探测结果")
    sub.add_parser("operators", help="列出可用边特征算子")
    sub.add_parser("heuristics", help="列出可用启发式基线")

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--seed", type=int, default=42, help="基础随机种子")
        p.add_argument("--graph", choices=("sbm", "lfr", "karate"), default="sbm")
        p.add_argument("--nodes", type=int, default=700, help="合成图节点数")
        p.add_argument("--communities", type=int, default=4, help="SBM 社群数")
        p.add_argument("--homophily", type=float, default=10.0,
                       help="SBM 同配比 p_in / p_out")
        p.add_argument("--avg-degree", type=float, default=10.0, dest="avg_degree")
        p.add_argument("--p", type=float, default=1.0, help="node2vec 回退参数 p")
        p.add_argument("--q", type=float, default=1.0, help="node2vec 外扩参数 q")
        p.add_argument("--dim", type=int, default=64, help="嵌入维度")
        p.add_argument("--walk-length", type=int, default=32)
        p.add_argument("--num-walks", type=int, default=6)
        p.add_argument("--window", type=int, default=5)
        p.add_argument("--negative", type=int, default=5, dest="negative")
        p.add_argument("--epochs", type=int, default=1)
        p.add_argument("--operator", choices=available_operators(), default="hadamard")
        p.add_argument("--normalize-embeddings", action="store_true",
                       dest="normalize_embeddings",
                       help="嵌入按行 L2 归一化后再做边算子")
        p.add_argument("--heuristic-features", action="store_true", dest="heuristic_features")
        p.add_argument("--classifier", choices=("auto", "sklearn", "numpy"), default="auto")
        p.add_argument("--classifier-c", type=float, default=1.0, dest="classifier_c",
                       help="逻辑回归正则强度 C")
        p.add_argument("--hpo", action="store_true", help="开启超参搜索（仅用 train/valid）")
        p.add_argument("--trials", type=int, default=8, help="超参搜索试验数")
        p.add_argument("--tier1-only", action="store_true", dest="tier1_only",
                       help="强制纯 numpy 离线兜底路径")

    run_parser = sub.add_parser("run", help="单 seed 端到端运行")
    _common(run_parser)
    run_parser.add_argument("--output", type=str, default="", help="结果 JSON 输出路径")

    bench_parser = sub.add_parser("benchmark", help="多 seed 基准对比")
    _common(bench_parser)
    bench_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    bench_parser.add_argument("--output", type=str, default="", help="benchmark JSON 输出路径")
    bench_parser.add_argument("--json-stdout", action="store_true", dest="json_stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "info": cmd_info,
        "operators": cmd_operators,
        "heuristics": cmd_heuristics,
        "run": cmd_run,
        "benchmark": cmd_benchmark,
    }
    try:
        return handlers[args.command](args)
    except GraphForgeError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
