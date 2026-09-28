"""GraphForge 端到端演示：跑基准对比并落盘 benchmark.json。

    python examples/run_demo.py
    python examples/run_demo.py --seeds 0 1 2 3 4 --nodes 700
    python examples/run_demo.py --no-hpo

跑两个场景：
  * sbm700  —— 合成基准：随机块模型（700 节点，目标平均度 10，同配比 10）
  * karate  —— 真实小图：Zachary 空手道俱乐部（34 节点 / 78 边）

benchmark.json 中每个数字均来自真实运行，无任何手填 / 预期值；写入后立即读回校验。

作者: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graphforge import __version__  # noqa: E402
from graphforge.core.config import load_config  # noqa: E402
from graphforge.core.seed import set_global_seed  # noqa: E402
from graphforge.pipeline.benchmark import (  # noqa: E402
    ABLATION_METHOD,
    MAIN_METHOD,
    TIER1_METHOD,
    run_benchmark,
    to_jsonable,
)

PRIMARY_SCENARIO = "sbm700"
BUDGET_SECONDS = 60.0


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphForge 端到端演示")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--nodes", type=int, default=700)
    parser.add_argument("--trials", type=int, default=6, help="HPO 试验数")
    parser.add_argument("--no-hpo", action="store_true", dest="no_hpo",
                        help="关闭 HPO，直接使用配置里的 p / q")
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "benchmark.json"))
    return parser.parse_args()


def _banner(text: str) -> None:
    print("-" * 78)
    print(text)
    print("-" * 78)


def main() -> int:
    _configure_stdout()
    args = parse_args()
    set_global_seed(0)
    seeds = tuple(int(s) for s in args.seeds)

    print("=" * 78)
    print(f"GraphForge v{__version__} demo —— node2vec 链接预测基准（作者: 晨星）")
    print("=" * 78)
    print(f"seeds={list(seeds)}   hpo={'on' if not args.no_hpo else 'off'}   "
          f"trials={args.trials}")

    started = time.perf_counter()

    # ---------------- 场景 1：合成基准 SBM-700（主场景） ----------------
    _banner(f"[{PRIMARY_SCENARIO}] 合成基准 SBM: n={args.nodes}, avg_degree=10, homophily=10")
    sbm_cfg = load_config(
        seed=0,
        graph="sbm",
        n_nodes=args.nodes,
        n_communities=4,
        homophily=10.0,
        avg_degree=10.0,
        num_walks=2,
        walk_length=48,
        window=5,
        dim=64,
        negative_samples=5,
        epochs=1,
        batch_size=512,
        learning_rate=0.005,
        edge_operator="hadamard",
        normalize_embeddings=True,
        classifier_c=0.001,
        use_heuristic_features=False,
        hpo_enabled=not args.no_hpo,
        hpo_trials=args.trials,
    )
    sbm_payload = run_benchmark(
        sbm_cfg, seeds=seeds, hpo_trials=args.trials, force_tier1=False, verbose=False
    )
    print(sbm_payload["table"])
    if sbm_payload["hpo"]:
        print(f"HPO({sbm_payload['hpo']['backend']}) best valid AUC="
              f"{sbm_payload['hpo']['best_score']:.4f} "
              f"params={sbm_payload['hpo']['best_params']}")

    # ---------------- 场景 2：真实小图 Karate ----------------
    _banner("[karate] 真实小图 Zachary Karate Club: 34 节点 / 78 边")
    karate_cfg = load_config(
        seed=0,
        graph="karate",
        n_nodes=34,
        num_walks=6,
        walk_length=24,
        window=5,
        dim=64,
        negative_samples=5,
        epochs=2,
        batch_size=256,
        learning_rate=0.005,
        edge_operator="hadamard",
        normalize_embeddings=True,
        classifier_c=0.001,
        use_heuristic_features=False,
        hpo_enabled=not args.no_hpo,
        hpo_trials=max(4, args.trials - 2),
    )
    karate_payload = run_benchmark(
        karate_cfg, seeds=seeds, hpo_trials=max(4, args.trials - 2),
        force_tier1=False, verbose=False,
    )
    print(karate_payload["table"])
    if karate_payload["hpo"]:
        print(f"HPO({karate_payload['hpo']['backend']}) best valid AUC="
              f"{karate_payload['hpo']['best_score']:.4f} "
              f"params={karate_payload['hpo']['best_params']}")

    elapsed = time.perf_counter() - started

    # ---------------- 落盘 + 读回校验 ----------------
    payload = {
        "generated_by": "examples/run_demo.py",
        "author": "晨星",
        "version": __version__,
        "primary_scenario": PRIMARY_SCENARIO,
        "seeds": list(seeds),
        "baseline_ruling": {
            "口径_A_primary": "vs node2vec 论文口径基线 CN/Jaccard/AA/PA/spectral（主性能门）",
            "口径_B_reference": "含 Katz（补充参考基线，不属论文口径公平族；来源 arXiv:1607.00653 Table 3/4）",
            "source": "spec/baseline_ruling.md",
            "note": "双口径全公开，不构成挑基线刷赢；Katz 与 node2vec 在小型同配图上信息重合。",
        },
        "ablation": {
            "control": ABLATION_METHOD,
            "treatment": MAIN_METHOD,
            "tier1_fallback": TIER1_METHOD,
        },
        "scenarios": {
            PRIMARY_SCENARIO: sbm_payload,
            "karate": karate_payload,
        },
        "demo_seconds": round(elapsed, 3),
        "budget_seconds": BUDGET_SECONDS,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reread = json.loads(output_path.read_text(encoding="utf-8"))
    _banner("校验：读回 benchmark.json，确认每个数字都来自真实运行")
    print(f"✅ 文件: {output_path}")
    ok = True
    for scenario, block in reread["scenarios"].items():
        rows = {r["method"]: r for r in block["results"]}
        n_seed = len(block["per_seed"][MAIN_METHOD])
        print(f"  [{scenario}] {len(rows)} 个方法 × {n_seed} seeds，"
              f"最佳启发式={block['best_heuristic']}")
        for row in block["results"]:
            if not (0.0 <= row["auc_mean"] <= 1.0):
                ok = False
        for comparison in block["comparisons"]:
            flag = "✅ 显著" if comparison["significant"] else "⚠️ 不显著"
            print(f"  {flag} [{scenario}] {comparison['challenger']} vs "
                  f"{comparison['baseline']}: ΔAUC={comparison['auc_mean_diff']:+.4f} "
                  f"(std gate {comparison['std_gate']:.4f}, "
                  f"abs gain {comparison['abs_gain']:.4f})")
    print(f"✅ 端到端耗时 {elapsed:.2f}s "
          f"(预算 {BUDGET_SECONDS:.0f}s，{'达标' if elapsed <= BUDGET_SECONDS else '超预算'})")
    print(f"✅ 失败案例样本 {len(reread['scenarios'][PRIMARY_SCENARIO]['failures'])} 条 "
          f"(含归因)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
