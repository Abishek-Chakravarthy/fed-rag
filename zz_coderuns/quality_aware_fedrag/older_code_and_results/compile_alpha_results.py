import argparse
import csv
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def find_result_csvs(inputs: list[str]) -> list[Path]:
    result_paths: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file() and path.name.startswith("results_") and path.suffix == ".csv":
            result_paths.append(path)
        elif path.is_dir():
            result_paths.extend(sorted(path.rglob("results_*.csv")))
    unique_paths = sorted(set(result_paths))
    if not unique_paths:
        raise FileNotFoundError("No results_*.csv files were found in the provided inputs.")
    return unique_paths


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def aggregate_runs(csv_paths: list[Path]) -> tuple[list[dict], list[dict]]:
    final_rows: list[dict] = []
    round_rows: list[dict] = []

    for csv_path in csv_paths:
        rows = load_rows(csv_path)
        if not rows:
            continue
        final_row = rows[-1].copy()
        final_row["source_csv"] = str(csv_path)
        final_rows.append(final_row)

        for row in rows:
            round_copy = row.copy()
            round_copy["source_csv"] = str(csv_path)
            round_rows.append(round_copy)

    if not final_rows:
        raise ValueError("Found result CSVs, but all of them were empty.")

    summary_by_alpha: dict[float, dict[str, list[float] | str | int]] = {}
    for row in final_rows:
        alpha = float(row["alpha"])
        bucket = summary_by_alpha.setdefault(
            alpha,
            {
                "noise_mode": row.get("noise_mode", ""),
                "noise_ratio": row.get("noise_ratio", ""),
                "seed_count": 0,
                "avg_loss": [],
                "avg_train_loss": [],
                "server_val_mrr": [],
                "server_val_recall_at_k": [],
                "server_val_ndcg_at_k": [],
                "final_test_mrr": [],
                "final_test_recall_at_k": [],
                "final_test_ndcg_at_k": [],
                "selected_best_round": [],
            },
        )
        bucket["seed_count"] = int(bucket["seed_count"]) + 1
        bucket["avg_loss"].append(float(row["avg_loss"]))
        if row.get("avg_train_loss"):
            bucket["avg_train_loss"].append(float(row["avg_train_loss"]))
        bucket["server_val_mrr"].append(float(row.get("server_val_mrr") or row.get("post_mrr", 0.0)))
        bucket["server_val_recall_at_k"].append(float(row.get("server_val_recall_at_k") or row.get("post_recall_at_k", 0.0)))
        bucket["server_val_ndcg_at_k"].append(float(row.get("server_val_ndcg_at_k") or row.get("post_ndcg_at_k", 0.0)))
        bucket["final_test_mrr"].append(float(row.get("final_test_mrr") or row.get("post_mrr", 0.0)))
        bucket["final_test_recall_at_k"].append(float(row.get("final_test_recall_at_k") or row.get("post_recall_at_k", 0.0)))
        bucket["final_test_ndcg_at_k"].append(float(row.get("final_test_ndcg_at_k") or row.get("post_ndcg_at_k", 0.0)))
        if row.get("selected_best_round"):
            bucket["selected_best_round"].append(float(row["selected_best_round"]))

    summary_rows: list[dict] = []
    for alpha in sorted(summary_by_alpha):
        bucket = summary_by_alpha[alpha]
        loss_mean, loss_std = mean_std(bucket["avg_loss"])  # type: ignore[arg-type]
        train_loss_mean, train_loss_std = mean_std(bucket["avg_train_loss"])  # type: ignore[arg-type]
        server_val_mrr_mean, server_val_mrr_std = mean_std(bucket["server_val_mrr"])  # type: ignore[arg-type]
        server_val_recall_mean, server_val_recall_std = mean_std(bucket["server_val_recall_at_k"])  # type: ignore[arg-type]
        server_val_ndcg_mean, server_val_ndcg_std = mean_std(bucket["server_val_ndcg_at_k"])  # type: ignore[arg-type]
        final_test_mrr_mean, final_test_mrr_std = mean_std(bucket["final_test_mrr"])  # type: ignore[arg-type]
        final_test_recall_mean, final_test_recall_std = mean_std(bucket["final_test_recall_at_k"])  # type: ignore[arg-type]
        final_test_ndcg_mean, final_test_ndcg_std = mean_std(bucket["final_test_ndcg_at_k"])  # type: ignore[arg-type]
        best_round_mean, best_round_std = mean_std(bucket["selected_best_round"])  # type: ignore[arg-type]
        summary_rows.append(
            {
                "alpha": f"{alpha:.1f}",
                "seed_count": str(bucket["seed_count"]),
                "noise_mode": str(bucket["noise_mode"]),
                "noise_ratio": str(bucket["noise_ratio"]),
                "final_loss_mean": f"{loss_mean:.6f}",
                "final_loss_std": f"{loss_std:.6f}",
                "final_train_loss_mean": f"{train_loss_mean:.6f}",
                "final_train_loss_std": f"{train_loss_std:.6f}",
                "server_val_mrr_mean": f"{server_val_mrr_mean:.6f}",
                "server_val_mrr_std": f"{server_val_mrr_std:.6f}",
                "server_val_recall_mean": f"{server_val_recall_mean:.6f}",
                "server_val_recall_std": f"{server_val_recall_std:.6f}",
                "server_val_ndcg_mean": f"{server_val_ndcg_mean:.6f}",
                "server_val_ndcg_std": f"{server_val_ndcg_std:.6f}",
                "final_test_mrr_mean": f"{final_test_mrr_mean:.6f}",
                "final_test_mrr_std": f"{final_test_mrr_std:.6f}",
                "final_test_recall_mean": f"{final_test_recall_mean:.6f}",
                "final_test_recall_std": f"{final_test_recall_std:.6f}",
                "final_test_ndcg_mean": f"{final_test_ndcg_mean:.6f}",
                "final_test_ndcg_std": f"{final_test_ndcg_std:.6f}",
                "selected_best_round_mean": f"{best_round_mean:.6f}",
                "selected_best_round_std": f"{best_round_std:.6f}",
            }
        )

    return summary_rows, round_rows


def write_summary(summary_rows: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "compiled_alpha_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def plot_final_loss(summary_rows: list[dict], output_dir: Path) -> Path:
    alphas = [float(row["alpha"]) for row in summary_rows]
    losses = [float(row["final_loss_mean"]) for row in summary_rows]
    loss_err = [float(row["final_loss_std"]) for row in summary_rows]

    plt.figure(figsize=(8, 5))
    plt.errorbar(alphas, losses, yerr=loss_err, marker="o", linewidth=2, capsize=4)
    plt.xlabel("Alpha")
    plt.ylabel("Final Quality Loss")
    plt.title("QA-FedAvg: Alpha vs Final Quality Loss")
    plt.grid(True, linestyle="--", alpha=0.4)

    best_idx = min(range(len(losses)), key=lambda idx: losses[idx])
    plt.scatter([alphas[best_idx]], [losses[best_idx]], s=80)
    plt.annotate(
        f"best α={alphas[best_idx]:.1f}",
        (alphas[best_idx], losses[best_idx]),
        textcoords="offset points",
        xytext=(8, -14),
    )

    plot_path = output_dir / "alpha_vs_final_loss.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def plot_post_metrics(summary_rows: list[dict], output_dir: Path) -> list[Path]:
    metric_specs = [
        ("final_test_mrr_mean", "final_test_mrr_std", "Final Test MRR", "alpha_vs_final_test_mrr.png"),
        (
            "final_test_ndcg_mean",
            "final_test_ndcg_std",
            "Final Test NDCG@k",
            "alpha_vs_final_test_ndcg.png",
        ),
    ]

    alphas = [float(row["alpha"]) for row in summary_rows]
    output_paths: list[Path] = []
    for mean_key, std_key, label, filename in metric_specs:
        means = [float(row[mean_key]) for row in summary_rows]
        errs = [float(row[std_key]) for row in summary_rows]

        plt.figure(figsize=(8, 5))
        plt.errorbar(alphas, means, yerr=errs, marker="o", linewidth=2, capsize=4)
        plt.xlabel("Alpha")
        plt.ylabel(label)
        plt.title(f"QA-FedAvg: Alpha vs Final {label}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plot_path = output_dir / filename
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        output_paths.append(plot_path)
    return output_paths


def plot_round_curves(round_rows: list[dict], output_dir: Path) -> Path:
    grouped: dict[float, list[tuple[int, float]]] = {}
    for row in round_rows:
        alpha = float(row["alpha"])
        round_no = int(row["round"])
        server_val_mrr = float(row.get("server_val_mrr") or row.get("post_mrr", 0.0))
        grouped.setdefault(alpha, []).append((round_no, server_val_mrr))

    plt.figure(figsize=(8, 5))
    for alpha in sorted(grouped):
        pairs = sorted(grouped[alpha], key=lambda pair: pair[0])
        rounds = [pair[0] for pair in pairs]
        values = [pair[1] for pair in pairs]
        plt.plot(rounds, values, marker="o", linewidth=2, label=f"α={alpha:.1f}")

    plt.xlabel("Round")
    plt.ylabel("Server Validation MRR")
    plt.title("Per-Round Server-Validation MRR by Alpha")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plot_path = output_dir / "server_val_mrr_over_rounds_by_alpha.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile quality_aware_fedrag results_*.csv files and plot alpha curves."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more result CSV files and/or directories containing results_*.csv files",
    )
    parser.add_argument(
        "--output-dir",
        default="compiled_alpha_plots",
        help="Directory where the combined summary CSV and plots will be written",
    )
    args = parser.parse_args()

    csv_paths = find_result_csvs(args.inputs)
    summary_rows, round_rows = aggregate_runs(csv_paths)

    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = write_summary(summary_rows, output_dir)
    loss_plot = plot_final_loss(summary_rows, output_dir)
    metric_plots = plot_post_metrics(summary_rows, output_dir)
    round_plot = plot_round_curves(round_rows, output_dir)

    print(f"Loaded {len(csv_paths)} result CSV files")
    print(f"Summary CSV: {summary_path}")
    print(f"Loss plot : {loss_plot}")
    for plot_path in metric_plots:
        print(f"Metric plot: {plot_path}")
    print(f"Round plot: {round_plot}")


if __name__ == "__main__":
    main()
