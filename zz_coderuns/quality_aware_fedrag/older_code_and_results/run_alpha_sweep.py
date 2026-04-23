import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time


DEFAULT_ALPHA_VALUES = [0.0, 0.3, 0.7, 1.0]
DEFAULT_SEEDS = [42, 52, 62]
DEFAULT_NOISE_MODES = [
    "shuffle",
    "hard_negative",
]
DEFAULT_NOISE_RATIOS = [0.8]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUTPUT_DIR, "output_csv_files")
LOG_DIR = os.path.join(OUTPUT_DIR, "output_log_files")
SCRIPT = os.path.join(OUTPUT_DIR, "federated_noisy_qa.py")


def build_run_slug(
    *,
    alpha: float,
    seed: int,
    noise_mode: str,
    noise_ratio: float,
    num_rounds: int,
    local_epochs: int,
) -> str:
    mode = noise_mode.replace("-", "_")
    ratio = f"{noise_ratio:.2f}".replace(".", "p")
    return (
        f"alpha_{alpha:.1f}_seed_{seed}_mode_{mode}_"
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


def run_one(alpha, seed, noise_mode, noise_ratio, rounds, local_epochs, *, force=False) -> dict:
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    client_split_mode = "equal" if noise_mode == "shuffle" else "unequal"
    noisy_fraction = 0.4

    run_slug = build_run_slug(
        alpha=alpha,
        seed=seed,
        noise_mode=noise_mode,
        noise_ratio=noise_ratio,
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
            f"SKIP  | α={alpha:.1f} | seed={seed} | "
            f"mode={noise_mode} | ratio={noise_ratio:.2f}"
        )
        return {
            "alpha": alpha,
            "seed": seed,
            "noise_mode": noise_mode,
            "noise_ratio": noise_ratio,
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
        f"RUN   | α={alpha:.1f} | seed={seed} | "
        f"mode={noise_mode} | ratio={noise_ratio:.2f}"
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
                "--noise-mode",
                noise_mode,
                "--noise-ratio",
                str(noise_ratio),
                "--rounds",
                str(rounds),
                "--local-epochs",
                str(local_epochs),
                "--client-split-mode",
                client_split_mode,
                "--noisy-client-fraction",
                str(noisy_fraction),
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=OUTPUT_DIR,
        )
    elapsed = time.time() - start

    status = "completed" if proc.returncode == 0 else "failed"
    print(
        f"{status.upper():<5} | α={alpha:.1f} | seed={seed} | "
        f"mode={noise_mode} | ratio={noise_ratio:.2f} | "
        f"time={elapsed / 60:.1f}m"
    )

    return {
        "alpha": alpha,
        "seed": seed,
        "noise_mode": noise_mode,
        "noise_ratio": noise_ratio,
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


def summarize_group(run_records, *, noise_mode, noise_ratio, rounds, local_epochs):
    summary_rows = []
    acceptance_summary = {
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
        final_post_mrr = [
            float(row.get("final_test_mrr") or row.get("post_mrr", 0.0))
            for row in final_rows
        ]
        final_post_recall = [
            float(row.get("final_test_recall_at_k") or row.get("post_recall_at_k", 0.0))
            for row in final_rows
        ]
        final_post_ndcg = [
            float(row.get("final_test_ndcg_at_k") or row.get("post_ndcg_at_k", 0.0))
            for row in final_rows
        ]
        final_noisy_weight = []
        for row in final_rows:
            noisy_client_id = row.get("noisy_client_id", "2")
            final_noisy_weight.append(
                float(row.get(f"client_{noisy_client_id}_weight", 0.0))
            )
        final_delta = [float(row["aggregated_delta_norm"]) for row in final_rows]
        final_best_round = [
            float(row.get("selected_best_round", 0.0) or 0.0)
            for row in final_rows
        ]

        loss_mean, loss_std = mean_and_std(final_losses)
        mrr_mean, mrr_std = mean_and_std(final_post_mrr)
        recall_mean, recall_std = mean_and_std(final_post_recall)
        ndcg_mean, ndcg_std = mean_and_std(final_post_ndcg)
        noisy_weight_mean, noisy_weight_std = mean_and_std(final_noisy_weight)
        delta_mean, delta_std = mean_and_std(final_delta)
        best_round_mean, best_round_std = mean_and_std(final_best_round)

        summary_rows.append(
            {
                "alpha": f"{alpha:.1f}",
                "noise_mode": noise_mode,
                "noise_ratio": f"{noise_ratio:.2f}",
                "seeds_run": len(records),
                "successful_runs": len(final_rows),
                "final_loss_mean": f"{loss_mean:.6f}",
                "final_loss_std": f"{loss_std:.6f}",
                "final_post_mrr_mean": f"{mrr_mean:.6f}",
                "final_post_mrr_std": f"{mrr_std:.6f}",
                "final_post_recall_mean": f"{recall_mean:.6f}",
                "final_post_recall_std": f"{recall_std:.6f}",
                "final_post_ndcg_mean": f"{ndcg_mean:.6f}",
                "final_post_ndcg_std": f"{ndcg_std:.6f}",
                "final_noisy_weight_mean": f"{noisy_weight_mean:.6f}",
                "final_noisy_weight_std": f"{noisy_weight_std:.6f}",
                "selected_best_round_mean": f"{best_round_mean:.6f}",
                "selected_best_round_std": f"{best_round_std:.6f}",
                "final_delta_mean": f"{delta_mean:.6f}",
                "final_delta_std": f"{delta_std:.6f}",
                "artifacts_present": str(all_artifacts_present),
            }
        )

        if alpha == 0.0:
            acceptance_summary["criteria"]["alpha_zero_matches_fedavg"] = all(
                report["alpha_zero_matches_fedavg"]["passed"]
                for report in acceptance_reports
                if report["alpha_zero_matches_fedavg"]["applicable"]
            ) if acceptance_reports else False
        else:
            key = f"alpha_{alpha:.1f}_higher_loss_downweighted"
            acceptance_summary["criteria"][key] = all(
                report["higher_loss_clients_downweighted"]["passed"]
                for report in acceptance_reports
                if report["higher_loss_clients_downweighted"]["applicable"]
            ) if acceptance_reports else False

    if summary_rows:
        metric_means = {
            float(row["alpha"]): float(row["final_post_mrr_mean"])
            for row in summary_rows
        }
        noisy_weight_means = {
            float(row["alpha"]): float(row["final_noisy_weight_mean"])
            for row in summary_rows
        }
        acceptance_summary["criteria"]["distinct_alphas_distinct_weights"] = (
            max(noisy_weight_means.values()) - min(noisy_weight_means.values())
        ) > 1e-3
        acceptance_summary["criteria"]["reproducible_artifacts_present"] = all(
            row["artifacts_present"] == "True" for row in summary_rows
        )

        best_alpha = max(metric_means, key=metric_means.get)
        best_is_midpoint = best_alpha not in {
            min(metric_means.keys()),
            max(metric_means.keys()),
        }
        acceptance_summary["criteria"]["inverted_u_reproduced"] = best_is_midpoint
        acceptance_summary["criteria"]["thesis_revision_needed"] = not best_is_midpoint
        acceptance_summary["evidence"] = {
            "best_alpha_by_final_post_mrr_mean": best_alpha,
            "metric_means": metric_means,
            "noisy_weight_means": noisy_weight_means,
        }

    summary_path = os.path.join(
        CSV_DIR,
        f"summary_mode_{noise_mode}_ratio_{ratio_slug(noise_ratio)}_"
        f"r{rounds}_e{local_epochs}.csv",
    )
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [
            "alpha",
            "noise_mode",
            "noise_ratio",
            "seeds_run",
            "successful_runs",
            "final_loss_mean",
            "final_loss_std",
            "final_post_mrr_mean",
            "final_post_mrr_std",
            "final_post_recall_mean",
            "final_post_recall_std",
            "final_post_ndcg_mean",
            "final_post_ndcg_std",
            "final_noisy_weight_mean",
            "final_noisy_weight_std",
            "selected_best_round_mean",
            "selected_best_round_std",
            "final_delta_mean",
            "final_delta_std",
            "artifacts_present",
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    acceptance_path = os.path.join(
        CSV_DIR,
        f"acceptance_summary_mode_{noise_mode}_ratio_{ratio_slug(noise_ratio)}_"
        f"r{rounds}_e{local_epochs}.json",
    )
    with open(acceptance_path, "w", encoding="utf-8") as f:
        json.dump(acceptance_summary, f, indent=2, sort_keys=True)

    return summary_path, acceptance_path, summary_rows, acceptance_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run QA-FedAvg alpha sweeps across seeds and noise modes."
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHA_VALUES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--noise-modes",
        nargs="+",
        type=str,
        default=DEFAULT_NOISE_MODES,
    )
    parser.add_argument(
        "--noise-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_NOISE_RATIOS,
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute runs even if their artifacts already exist",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("QA-FEDAVG BENCHMARK SWEEP")
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
            group_key = (noise_mode, noise_ratio)
            grouped_records[group_key] = []
            progress_path = os.path.join(
                CSV_DIR,
                f"progress_mode_{noise_mode}_ratio_{ratio_slug(noise_ratio)}_"
                f"r{args.rounds}_e{args.local_epochs}.csv",
            )
            for alpha in args.alphas:
                for seed in args.seeds:
                    record = run_one(
                        alpha=alpha,
                        seed=seed,
                        noise_mode=noise_mode,
                        noise_ratio=noise_ratio,
                        rounds=args.rounds,
                        local_epochs=args.local_epochs,
                        force=args.force,
                    )
                    grouped_records[group_key].append(record)
                    append_progress(progress_path, record)

    for (noise_mode, noise_ratio), records in grouped_records.items():
        summary_path, acceptance_path, summary_rows, acceptance_summary = summarize_group(
            records,
            noise_mode=noise_mode,
            noise_ratio=noise_ratio,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
        )
        print(f"\nSummary CSV     : {summary_path}")
        print(f"Acceptance JSON : {acceptance_path}")
        if acceptance_summary.get("evidence"):
            print(
                f"Best alpha by mean final test MRR "
                f"({noise_mode}, ratio={noise_ratio:.2f}): "
                f"{acceptance_summary['evidence']['best_alpha_by_final_post_mrr_mean']}"
            )
        for row in summary_rows:
            print(
                f"  α={row['alpha']}: "
                f"loss={row['final_loss_mean']}±{row['final_loss_std']}, "
                f"post_mrr={row['final_post_mrr_mean']}±{row['final_post_mrr_std']}, "
                f"noisy_w={row['final_noisy_weight_mean']}±{row['final_noisy_weight_std']}"
            )

    total_elapsed = time.time() - total_start
    print(f"\nTotal sweep time: {total_elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
