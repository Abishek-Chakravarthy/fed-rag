import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time


DEFAULT_ALPHA_VALUES = [0.0, 0.3, 0.7, 1.0]
DEFAULT_MECHANISM_SEEDS = [42]
DEFAULT_STRESS_SEEDS = [42, 52, 62]
DEFAULT_BENCHMARK_MODE = "mechanism"
DEFAULT_STRESS_NOISE_MODES = ["hard_negative"]
DEFAULT_STRESS_NOISE_RATIOS = [0.5, 0.7]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUTPUT_DIR, "output_csv_files")
LOG_DIR = os.path.join(OUTPUT_DIR, "output_log_files")
SCRIPT = os.path.join(OUTPUT_DIR, "federated_noisy_qa.py")


def build_run_slug(
    *,
    alpha: float,
    seed: int,
    benchmark_mode: str,
    noise_mode: str,
    noise_ratio: float,
    num_rounds: int,
    local_epochs: int,
) -> str:
    track = benchmark_mode.replace("-", "_")
    mode = noise_mode.replace("-", "_")
    ratio = f"{noise_ratio:.2f}".replace(".", "p")
    return (
        f"alpha_{alpha:.1f}_track_{track}_seed_{seed}_mode_{mode}_"
        f"ratio_{ratio}_r{num_rounds}_e{local_epochs}"
    )


def mean_and_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def ratio_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def load_csv_rows(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_progress(progress_path: str, record: dict) -> None:
    file_exists = os.path.exists(progress_path)
    fieldnames = [
        "benchmark_mode",
        "alpha",
        "seed",
        "noise_mode",
        "noise_ratio",
        "rounds",
        "local_epochs",
        "run_slug",
        "csv_path",
        "manifest_path",
        "acceptance_path",
        "stdout_log",
        "elapsed_seconds",
        "returncode",
        "status",
    ]
    with open(progress_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: record.get(key, "") for key in fieldnames})


def expected_effective_settings(*, benchmark_mode: str, noise_mode: str, noise_ratio: float) -> tuple[str, float]:
    if benchmark_mode == "mechanism":
        return "shuffle", 0.0
    return noise_mode, noise_ratio


def run_one(
    alpha,
    seed,
    benchmark_mode,
    noise_mode,
    noise_ratio,
    rounds,
    local_epochs,
    *,
    force=False,
) -> dict:
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    effective_noise_mode, effective_noise_ratio = expected_effective_settings(
        benchmark_mode=benchmark_mode,
        noise_mode=noise_mode,
        noise_ratio=noise_ratio,
    )
    run_slug = build_run_slug(
        alpha=alpha,
        seed=seed,
        benchmark_mode=benchmark_mode,
        noise_mode=effective_noise_mode,
        noise_ratio=effective_noise_ratio,
        num_rounds=rounds,
        local_epochs=local_epochs,
    )
    stdout_log = os.path.join(LOG_DIR, f"sweep_{run_slug}.log")
    csv_path = os.path.join(CSV_DIR, f"results_{run_slug}.csv")
    manifest_path = os.path.join(CSV_DIR, f"manifest_{run_slug}.json")
    acceptance_path = os.path.join(CSV_DIR, f"acceptance_{run_slug}.json")

    if (
        not force
        and os.path.exists(csv_path)
        and os.path.exists(manifest_path)
        and os.path.exists(acceptance_path)
    ):
        print(
            f"SKIP  | track={benchmark_mode} | α={alpha:.1f} | seed={seed} | "
            f"mode={effective_noise_mode} | ratio={effective_noise_ratio:.2f}"
        )
        return {
            "benchmark_mode": benchmark_mode,
            "alpha": alpha,
            "seed": seed,
            "noise_mode": effective_noise_mode,
            "noise_ratio": effective_noise_ratio,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "run_slug": run_slug,
            "stdout_log": stdout_log,
            "csv_path": csv_path,
            "manifest_path": manifest_path,
            "acceptance_path": acceptance_path,
            "elapsed_seconds": 0.0,
            "returncode": 0,
            "status": "skipped_existing",
        }

    print(
        f"RUN   | track={benchmark_mode} | α={alpha:.1f} | seed={seed} | "
        f"mode={effective_noise_mode} | ratio={effective_noise_ratio:.2f}"
    )

    start = time.time()
    with open(stdout_log, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            [
                sys.executable,
                SCRIPT,
                "--alpha",
                str(alpha),
                "--seed",
                str(seed),
                "--benchmark-mode",
                benchmark_mode,
                "--noise-mode",
                noise_mode,
                "--noise-ratio",
                str(noise_ratio),
                "--rounds",
                str(rounds),
                "--local-epochs",
                str(local_epochs),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=OUTPUT_DIR,
        )
    elapsed = time.time() - start

    status = "completed" if proc.returncode == 0 else "failed"
    print(
        f"{status.upper():<5} | track={benchmark_mode} | α={alpha:.1f} | seed={seed} | "
        f"mode={effective_noise_mode} | ratio={effective_noise_ratio:.2f} | "
        f"time={elapsed / 60:.1f}m"
    )

    return {
        "benchmark_mode": benchmark_mode,
        "alpha": alpha,
        "seed": seed,
        "noise_mode": effective_noise_mode,
        "noise_ratio": effective_noise_ratio,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "run_slug": run_slug,
        "stdout_log": stdout_log,
        "csv_path": csv_path,
        "manifest_path": manifest_path,
        "acceptance_path": acceptance_path,
        "elapsed_seconds": elapsed,
        "returncode": proc.returncode,
        "status": status,
    }


def summarize_group(run_records, *, benchmark_mode, noise_mode, noise_ratio, rounds, local_epochs):
    summary_rows = []
    acceptance_summary = {
        "benchmark_mode": benchmark_mode,
        "noise_mode": noise_mode,
        "noise_ratio": noise_ratio,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "criteria": {},
    }

    alpha_to_records = {}
    for record in run_records:
        alpha_to_records.setdefault(record["alpha"], []).append(record)

    for alpha in sorted(alpha_to_records):
        records = alpha_to_records[alpha]
        final_rows = []
        acceptance_reports = []
        all_artifacts_present = True

        for record in records:
            if record["returncode"] != 0:
                all_artifacts_present = False
                continue
            if not (
                os.path.exists(record["csv_path"])
                and os.path.exists(record["manifest_path"])
                and os.path.exists(record["acceptance_path"])
            ):
                all_artifacts_present = False
                continue

            rows = load_csv_rows(record["csv_path"])
            if not rows:
                all_artifacts_present = False
                continue
            final_rows.append(rows[-1])
            acceptance_reports.append(load_json(record["acceptance_path"]))

        final_losses = [float(row["avg_loss"]) for row in final_rows]
        final_train_losses = [float(row.get("avg_train_loss", 0.0) or 0.0) for row in final_rows]
        final_test_mrr = [float(row.get("final_test_mrr") or row.get("post_mrr", 0.0)) for row in final_rows]
        final_test_ndcg = [float(row.get("final_test_ndcg_at_k") or row.get("post_ndcg_at_k", 0.0)) for row in final_rows]
        final_test_recall = [
            float(row.get("final_test_recall_at_k") or row.get("post_recall_at_k", 0.0))
            for row in final_rows
        ]
        server_val_mrr = [float(row.get("server_val_mrr") or row.get("post_mrr", 0.0)) for row in final_rows]
        best_rounds = [float(row.get("selected_best_round", 0.0) or 0.0) for row in final_rows]
        deltas = [float(row.get("aggregated_delta_norm", 0.0) or 0.0) for row in final_rows]

        focus_weights = []
        lower_quality_weights = []
        clean_weights = []
        retriever_models = set()

        for row in final_rows:
            retriever_models.add(row.get("retriever_model", ""))
            noisy_client_id = row.get("noisy_client_id", "4")
            focus_weights.append(float(row.get(f"client_{noisy_client_id}_weight", 0.0)))

            if benchmark_mode == "mechanism":
                for cid in ["3", "4"]:
                    lower_quality_weights.append(float(row.get(f"client_{cid}_weight", 0.0) or 0.0))
                for cid in ["0", "1", "2"]:
                    clean_weights.append(float(row.get(f"client_{cid}_weight", 0.0) or 0.0))

        loss_mean, loss_std = mean_and_std(final_losses)
        train_loss_mean, train_loss_std = mean_and_std(final_train_losses)
        mrr_mean, mrr_std = mean_and_std(final_test_mrr)
        ndcg_mean, ndcg_std = mean_and_std(final_test_ndcg)
        recall_mean, recall_std = mean_and_std(final_test_recall)
        server_val_mean, server_val_std = mean_and_std(server_val_mrr)
        focus_weight_mean, focus_weight_std = mean_and_std(focus_weights)
        best_round_mean, best_round_std = mean_and_std(best_rounds)
        delta_mean, delta_std = mean_and_std(deltas)
        lower_quality_weight_mean, lower_quality_weight_std = mean_and_std(lower_quality_weights)
        clean_weight_mean, clean_weight_std = mean_and_std(clean_weights)

        summary_rows.append(
            {
                "benchmark_mode": benchmark_mode,
                "alpha": f"{alpha:.1f}",
                "noise_mode": noise_mode,
                "noise_ratio": f"{noise_ratio:.2f}",
                "retriever_model": ";".join(sorted(model for model in retriever_models if model)),
                "seeds_run": len(records),
                "successful_runs": len(final_rows),
                "final_quality_loss_mean": f"{loss_mean:.6f}",
                "final_quality_loss_std": f"{loss_std:.6f}",
                "final_train_loss_mean": f"{train_loss_mean:.6f}",
                "final_train_loss_std": f"{train_loss_std:.6f}",
                "final_test_mrr_mean": f"{mrr_mean:.6f}",
                "final_test_mrr_std": f"{mrr_std:.6f}",
                "final_test_ndcg_mean": f"{ndcg_mean:.6f}",
                "final_test_ndcg_std": f"{ndcg_std:.6f}",
                "final_test_recall_mean": f"{recall_mean:.6f}",
                "final_test_recall_std": f"{recall_std:.6f}",
                "server_val_mrr_mean": f"{server_val_mean:.6f}",
                "server_val_mrr_std": f"{server_val_std:.6f}",
                "focus_client_weight_mean": f"{focus_weight_mean:.6f}",
                "focus_client_weight_std": f"{focus_weight_std:.6f}",
                "lower_quality_client_weight_mean": f"{lower_quality_weight_mean:.6f}",
                "lower_quality_client_weight_std": f"{lower_quality_weight_std:.6f}",
                "clean_client_weight_mean": f"{clean_weight_mean:.6f}",
                "clean_client_weight_std": f"{clean_weight_std:.6f}",
                "selected_best_round_mean": f"{best_round_mean:.6f}",
                "selected_best_round_std": f"{best_round_std:.6f}",
                "final_delta_mean": f"{delta_mean:.6f}",
                "final_delta_std": f"{delta_std:.6f}",
                "artifacts_present": str(all_artifacts_present),
            }
        )

        if alpha == 0.0:
            applicable = [
                report["alpha_zero_matches_fedavg"]["passed"]
                for report in acceptance_reports
                if report["alpha_zero_matches_fedavg"]["applicable"]
            ]
            acceptance_summary["criteria"]["alpha_zero_matches_fedavg"] = all(applicable) if applicable else False
        else:
            applicable = [
                report["higher_loss_clients_downweighted"]["passed"]
                for report in acceptance_reports
                if report["higher_loss_clients_downweighted"]["applicable"]
            ]
            acceptance_summary["criteria"][f"alpha_{alpha:.1f}_higher_loss_downweighted"] = (
                all(applicable) if applicable else False
            )
            if benchmark_mode == "mechanism":
                mechanism_applicable = [
                    report["lower_quality_clients_downweighted"]["passed"]
                    for report in acceptance_reports
                    if report["lower_quality_clients_downweighted"]["applicable"]
                ]
                acceptance_summary["criteria"][f"alpha_{alpha:.1f}_lower_quality_downweighted"] = (
                    all(mechanism_applicable) if mechanism_applicable else False
                )

    if summary_rows:
        mrr_means = {float(row["alpha"]): float(row["final_test_mrr_mean"]) for row in summary_rows}
        ndcg_means = {float(row["alpha"]): float(row["final_test_ndcg_mean"]) for row in summary_rows}
        focus_weight_means = {float(row["alpha"]): float(row["focus_client_weight_mean"]) for row in summary_rows}
        acceptance_summary["criteria"]["distinct_alphas_distinct_weights"] = (
            max(focus_weight_means.values()) - min(focus_weight_means.values())
        ) > 1e-3
        acceptance_summary["criteria"]["reproducible_artifacts_present"] = all(
            row["artifacts_present"] == "True" for row in summary_rows
        )

        nonzero_alphas = [alpha for alpha in mrr_means if alpha > 0.0]
        alpha_zero_mrr = mrr_means.get(0.0, 0.0)
        alpha_zero_ndcg = ndcg_means.get(0.0, 0.0)
        acceptance_summary["criteria"]["nonzero_alpha_beats_fedavg_on_mrr"] = any(
            mrr_means[alpha] > alpha_zero_mrr for alpha in nonzero_alphas
        )
        acceptance_summary["criteria"]["nonzero_alpha_beats_fedavg_on_ndcg"] = any(
            ndcg_means[alpha] > alpha_zero_ndcg for alpha in nonzero_alphas
        )

        best_alpha_by_mrr = max(mrr_means, key=mrr_means.get)
        acceptance_summary["criteria"]["mid_alpha_best_on_mrr"] = best_alpha_by_mrr in {0.3, 0.7}
        acceptance_summary["criteria"]["alpha_one_not_best_on_mrr"] = best_alpha_by_mrr != 1.0
        acceptance_summary["criteria"]["thesis_ready_mechanism_signal"] = (
            acceptance_summary["criteria"]["distinct_alphas_distinct_weights"]
            and acceptance_summary["criteria"]["nonzero_alpha_beats_fedavg_on_mrr"]
        )
        acceptance_summary["evidence"] = {
            "best_alpha_by_final_test_mrr_mean": best_alpha_by_mrr,
            "mrr_means": mrr_means,
            "ndcg_means": ndcg_means,
            "focus_weight_means": focus_weight_means,
        }

    summary_path = os.path.join(
        CSV_DIR,
        f"summary_track_{benchmark_mode}_mode_{noise_mode}_ratio_{ratio_slug(noise_ratio)}_"
        f"r{rounds}_e{local_epochs}.csv",
    )
    fieldnames = list(summary_rows[0].keys()) if summary_rows else [
        "benchmark_mode",
        "alpha",
        "noise_mode",
        "noise_ratio",
        "retriever_model",
        "seeds_run",
        "successful_runs",
        "final_quality_loss_mean",
        "final_quality_loss_std",
        "final_train_loss_mean",
        "final_train_loss_std",
        "final_test_mrr_mean",
        "final_test_mrr_std",
        "final_test_ndcg_mean",
        "final_test_ndcg_std",
        "final_test_recall_mean",
        "final_test_recall_std",
        "server_val_mrr_mean",
        "server_val_mrr_std",
        "focus_client_weight_mean",
        "focus_client_weight_std",
        "lower_quality_client_weight_mean",
        "lower_quality_client_weight_std",
        "clean_client_weight_mean",
        "clean_client_weight_std",
        "selected_best_round_mean",
        "selected_best_round_std",
        "final_delta_mean",
        "final_delta_std",
        "artifacts_present",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    acceptance_path = os.path.join(
        CSV_DIR,
        f"acceptance_summary_track_{benchmark_mode}_mode_{noise_mode}_"
        f"ratio_{ratio_slug(noise_ratio)}_r{rounds}_e{local_epochs}.json",
    )
    with open(acceptance_path, "w", encoding="utf-8") as f:
        json.dump(acceptance_summary, f, indent=2, sort_keys=True)

    return summary_path, acceptance_path, summary_rows, acceptance_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QA-FedAvg mechanism or stress sweeps across alpha values and seeds."
    )
    parser.add_argument("--benchmark-mode", choices=["mechanism", "stress"], default=DEFAULT_BENCHMARK_MODE)
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHA_VALUES)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--noise-modes", nargs="+", type=str)
    parser.add_argument("--noise-ratios", nargs="+", type=float)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute runs even if their artifacts already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seeds is None:
        args.seeds = (
            DEFAULT_MECHANISM_SEEDS if args.benchmark_mode == "mechanism" else DEFAULT_STRESS_SEEDS
        )
    if args.noise_modes is None:
        args.noise_modes = ["shuffle"] if args.benchmark_mode == "mechanism" else DEFAULT_STRESS_NOISE_MODES
    if args.noise_ratios is None:
        args.noise_ratios = [0.0] if args.benchmark_mode == "mechanism" else DEFAULT_STRESS_NOISE_RATIOS

    print("=" * 72)
    print("QA-FEDAVG BENCHMARK SWEEP")
    print(f"track        : {args.benchmark_mode}")
    print(f"alphas       : {args.alphas}")
    print(f"seeds        : {args.seeds}")
    print(f"noise modes  : {args.noise_modes}")
    print(f"noise ratios : {args.noise_ratios}")
    print(f"rounds       : {args.rounds}")
    print(f"local epochs : {args.local_epochs}")
    print("=" * 72)

    total_start = time.time()
    grouped_records = {}

    for noise_mode in args.noise_modes:
        for noise_ratio in args.noise_ratios:
            effective_mode, effective_ratio = expected_effective_settings(
                benchmark_mode=args.benchmark_mode,
                noise_mode=noise_mode,
                noise_ratio=noise_ratio,
            )
            group_key = (args.benchmark_mode, effective_mode, effective_ratio)
            grouped_records[group_key] = []
            progress_path = os.path.join(
                CSV_DIR,
                f"progress_track_{args.benchmark_mode}_mode_{effective_mode}_"
                f"ratio_{ratio_slug(effective_ratio)}_r{args.rounds}_e{args.local_epochs}.csv",
            )
            for alpha in args.alphas:
                for seed in args.seeds:
                    record = run_one(
                        alpha=alpha,
                        seed=seed,
                        benchmark_mode=args.benchmark_mode,
                        noise_mode=noise_mode,
                        noise_ratio=noise_ratio,
                        rounds=args.rounds,
                        local_epochs=args.local_epochs,
                        force=args.force,
                    )
                    grouped_records[group_key].append(record)
                    append_progress(progress_path, record)

    for (benchmark_mode, noise_mode, noise_ratio), records in grouped_records.items():
        summary_path, acceptance_path, summary_rows, acceptance_summary = summarize_group(
            records,
            benchmark_mode=benchmark_mode,
            noise_mode=noise_mode,
            noise_ratio=noise_ratio,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
        )
        print(f"\nSummary CSV     : {summary_path}")
        print(f"Acceptance JSON : {acceptance_path}")
        evidence = acceptance_summary.get("evidence") or {}
        if evidence.get("best_alpha_by_final_test_mrr_mean") is not None:
            print(
                f"Best alpha by mean final test MRR "
                f"({benchmark_mode}, mode={noise_mode}, ratio={noise_ratio:.2f}): "
                f"{evidence['best_alpha_by_final_test_mrr_mean']}"
            )
        for row in summary_rows:
            print(
                f"  α={row['alpha']}: "
                f"test_mrr={row['final_test_mrr_mean']}±{row['final_test_mrr_std']}, "
                f"test_ndcg={row['final_test_ndcg_mean']}±{row['final_test_ndcg_std']}, "
                f"focus_w={row['focus_client_weight_mean']}±{row['focus_client_weight_std']}"
            )

    total_elapsed = time.time() - total_start
    print(f"\nTotal sweep time: {total_elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
