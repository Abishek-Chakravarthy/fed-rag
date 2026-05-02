import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time


DEFAULT_TAU_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8]
DEFAULT_SEEDS = [42, 52, 62]

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(OUTPUT_DIR, "output_csv_files")
LOG_DIR = os.path.join(OUTPUT_DIR, "output_log_files")
SCRIPT = os.path.join(OUTPUT_DIR, "federated_das.py")


def build_run_slug(
    *, tau: float, seed: int, num_rounds: int, local_epochs: int, target: str
) -> str:
    target_slug = target.replace("-", "_")
    return (
        f"tau_{tau:.2f}_seed_{seed}_target_{target_slug}_"
        f"r{num_rounds}_e{local_epochs}"
    )


def mean_and_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def load_csv_rows(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_progress(progress_path: str, record: dict) -> None:
    file_exists = os.path.exists(progress_path)
    fieldnames = [
        "tau",
        "seed",
        "target",
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


def run_one(
    tau: float,
    seed: int,
    target: str,
    rounds: int,
    local_epochs: int,
    *,
    force: bool = False,
) -> dict:
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    run_slug = build_run_slug(
        tau=tau,
        seed=seed,
        num_rounds=rounds,
        local_epochs=local_epochs,
        target=target,
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
        print(f"SKIP  | τ={tau:.2f} | seed={seed} | target={target}")
        return {
            "tau": tau,
            "seed": seed,
            "target": target,
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

    print(f"RUN   | τ={tau:.2f} | seed={seed} | target={target}")
    start = time.time()
    with open(stdout_log, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            [
                sys.executable,
                SCRIPT,
                "--tau",
                str(tau),
                "--seed",
                str(seed),
                "--rounds",
                str(rounds),
                "--local-epochs",
                str(local_epochs),
                "--target",
                target,
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=OUTPUT_DIR,
        )
    elapsed = time.time() - start

    status = "completed" if proc.returncode == 0 else "failed"
    print(
        f"{status.upper():<9}| τ={tau:.2f} | seed={seed} | "
        f"target={target} | time={elapsed / 60:.1f}m"
    )
    return {
        "tau": tau,
        "seed": seed,
        "target": target,
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


def summarize_group(run_records, *, target: str, rounds: int, local_epochs: int):
    summary_rows = []
    acceptance_summary = {
        "target": target,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "criteria": {},
    }

    tau_to_records = {}
    for record in run_records:
        tau_to_records.setdefault(record["tau"], []).append(record)

    for tau in sorted(tau_to_records):
        records = tau_to_records[tau]
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
        final_server_val_mrr = [float(row.get("server_val_mrr", 0.0)) for row in final_rows]
        final_server_val_ndcg = [float(row.get("server_val_ndcg_at_k", 0.0)) for row in final_rows]
        final_test_mrr = [float(row.get("final_test_mrr", 0.0)) for row in final_rows]
        final_test_recall = [float(row.get("final_test_recall_at_k", 0.0)) for row in final_rows]
        final_test_ndcg = [float(row.get("final_test_ndcg_at_k", 0.0)) for row in final_rows]
        final_selected = [float(row.get("num_selected", 0.0)) for row in final_rows]
        final_best_round = [float(row.get("selected_best_round", 0.0) or 0.0) for row in final_rows]

        loss_mean, loss_std = mean_and_std(final_losses)
        server_val_mrr_mean, server_val_mrr_std = mean_and_std(final_server_val_mrr)
        server_val_ndcg_mean, server_val_ndcg_std = mean_and_std(final_server_val_ndcg)
        test_mrr_mean, test_mrr_std = mean_and_std(final_test_mrr)
        test_recall_mean, test_recall_std = mean_and_std(final_test_recall)
        test_ndcg_mean, test_ndcg_std = mean_and_std(final_test_ndcg)
        selected_mean, selected_std = mean_and_std(final_selected)
        best_round_mean, best_round_std = mean_and_std(final_best_round)

        summary_rows.append(
            {
                "tau": f"{tau:.2f}",
                "target": target,
                "seeds_run": len(records),
                "successful_runs": len(final_rows),
                "final_loss_mean": f"{loss_mean:.6f}",
                "final_loss_std": f"{loss_std:.6f}",
                "server_val_mrr_mean": f"{server_val_mrr_mean:.6f}",
                "server_val_mrr_std": f"{server_val_mrr_std:.6f}",
                "server_val_ndcg_mean": f"{server_val_ndcg_mean:.6f}",
                "server_val_ndcg_std": f"{server_val_ndcg_std:.6f}",
                "final_test_mrr_mean": f"{test_mrr_mean:.6f}",
                "final_test_mrr_std": f"{test_mrr_std:.6f}",
                "final_test_recall_mean": f"{test_recall_mean:.6f}",
                "final_test_recall_std": f"{test_recall_std:.6f}",
                "final_test_ndcg_mean": f"{test_ndcg_mean:.6f}",
                "final_test_ndcg_std": f"{test_ndcg_std:.6f}",
                "avg_num_selected_mean": f"{selected_mean:.6f}",
                "avg_num_selected_std": f"{selected_std:.6f}",
                "selected_best_round_mean": f"{best_round_mean:.6f}",
                "selected_best_round_std": f"{best_round_std:.6f}",
                "artifacts_present": str(all_artifacts_present),
            }
        )

        if tau == 0.0:
            acceptance_summary["criteria"]["tau_zero_selects_all"] = all(
                report["tau_zero_selects_all"]["passed"]
                for report in acceptance_reports
                if report["tau_zero_selects_all"]["applicable"]
            ) if acceptance_reports else False
        else:
            key = f"tau_{tau:.2f}_selective_thresholding"
            acceptance_summary["criteria"][key] = all(
                report["selection_matches_threshold_policy"]["passed"]
                for report in acceptance_reports
                if report["selection_matches_threshold_policy"]["applicable"]
            ) if acceptance_reports else False

    if summary_rows:
        metric_means = {
            float(row["tau"]): float(row["final_test_mrr_mean"]) for row in summary_rows
        }
        acceptance_summary["criteria"]["target_client_always_selected"] = all(
            load_json(record["acceptance_path"])["target_client_always_selected"]["passed"]
            for record in run_records
            if record["returncode"] == 0 and os.path.exists(record["acceptance_path"])
        )
        acceptance_summary["criteria"]["distinct_tau_trajectories"] = len(
            {row["tau"] for row in summary_rows}
        ) == len(summary_rows)
        acceptance_summary["criteria"]["reproducible_artifacts_present"] = all(
            row["artifacts_present"] == "True" for row in summary_rows
        )
        best_tau = max(metric_means, key=metric_means.get)
        acceptance_summary["evidence"] = {
            "best_tau_by_final_test_mrr_mean": best_tau,
            "metric_means": metric_means,
        }

    summary_path = os.path.join(
        CSV_DIR,
        f"summary_target_{target.replace('-', '_')}_r{rounds}_e{local_epochs}.csv",
    )
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys())
            if summary_rows
            else [
                "tau",
                "target",
                "seeds_run",
                "successful_runs",
                "final_loss_mean",
                "final_loss_std",
                "server_val_mrr_mean",
                "server_val_mrr_std",
                "server_val_ndcg_mean",
                "server_val_ndcg_std",
                "final_test_mrr_mean",
                "final_test_mrr_std",
                "final_test_recall_mean",
                "final_test_recall_std",
                "final_test_ndcg_mean",
                "final_test_ndcg_std",
                "avg_num_selected_mean",
                "avg_num_selected_std",
                "selected_best_round_mean",
                "selected_best_round_std",
                "artifacts_present",
            ],
        )
        writer.writeheader()
        if summary_rows:
            writer.writerows(summary_rows)

    acceptance_path = os.path.join(
        CSV_DIR,
        f"acceptance_summary_target_{target.replace('-', '_')}_r{rounds}_e{local_epochs}.json",
    )
    with open(acceptance_path, "w", encoding="utf-8") as f:
        json.dump(acceptance_summary, f, indent=2, sort_keys=True)

    return summary_path, acceptance_path


def main():
    parser = argparse.ArgumentParser(description="Resumable DAS-FedAvg tau sweep")
    parser.add_argument("--taus", nargs="+", type=float, default=DEFAULT_TAU_VALUES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--target", type=str, default="nfcorpus")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    progress_path = os.path.join(
        CSV_DIR,
        f"progress_target_{args.target.replace('-', '_')}_r{args.rounds}_e{args.local_epochs}.csv",
    )

    print("=" * 72)
    print("DAS-FEDAVG TAU SWEEP")
    print(f"taus          : {args.taus}")
    print(f"seeds         : {args.seeds}")
    print(f"target        : {args.target}")
    print(f"rounds        : {args.rounds}")
    print(f"local_epochs  : {args.local_epochs}")
    print("=" * 72)

    total_start = time.time()
    run_records = []
    for tau in args.taus:
        for seed in args.seeds:
            record = run_one(
                tau,
                seed,
                args.target,
                args.rounds,
                args.local_epochs,
                force=args.force,
            )
            append_progress(progress_path, record)
            run_records.append(record)

    summary_path, acceptance_path = summarize_group(
        run_records,
        target=args.target,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
    )

    total_elapsed = time.time() - total_start
    print(f"\nTotal sweep time: {total_elapsed / 60:.1f} minutes")
    print(f"Progress ledger : {progress_path}")
    print(f"Summary CSV     : {summary_path}")
    print(f"Acceptance JSON : {acceptance_path}")


if __name__ == "__main__":
    main()
