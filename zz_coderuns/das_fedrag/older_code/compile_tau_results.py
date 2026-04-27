import argparse
import csv
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
        raise FileNotFoundError("No results_*.csv files found.")
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
        raise ValueError("All result CSVs were empty.")

    summary_by_tau: dict[float, dict] = {}
    for row in final_rows:
        tau = float(row["tau"])
        bucket = summary_by_tau.setdefault(
            tau,
            {
                "target_dataset": row.get("target_dataset", ""),
                "seed_count": 0,
                "avg_loss": [],
                "server_val_mrr": [],
                "server_val_ndcg_at_k": [],
                "final_test_mrr": [],
                "final_test_recall_at_k": [],
                "final_test_ndcg_at_k": [],
                "num_selected": [],
                "selected_best_round": [],
            },
        )
        bucket["seed_count"] = int(bucket["seed_count"]) + 1
        bucket["avg_loss"].append(float(row["avg_loss"]))
        bucket["server_val_mrr"].append(float(row.get("server_val_mrr", 0.0)))
        bucket["server_val_ndcg_at_k"].append(float(row.get("server_val_ndcg_at_k", 0.0)))
        bucket["final_test_mrr"].append(float(row.get("final_test_mrr", 0.0)))
        bucket["final_test_recall_at_k"].append(float(row.get("final_test_recall_at_k", 0.0)))
        bucket["final_test_ndcg_at_k"].append(float(row.get("final_test_ndcg_at_k", 0.0)))
        bucket["num_selected"].append(float(row.get("num_selected", 0.0)))
        bucket["selected_best_round"].append(float(row.get("selected_best_round", 0.0)))

    summary_rows: list[dict] = []
    for tau in sorted(summary_by_tau):
        bucket = summary_by_tau[tau]
        loss_mean, loss_std = mean_std(bucket["avg_loss"])
        server_val_mrr_mean, server_val_mrr_std = mean_std(bucket["server_val_mrr"])
        server_val_ndcg_mean, server_val_ndcg_std = mean_std(bucket["server_val_ndcg_at_k"])
        test_mrr_mean, test_mrr_std = mean_std(bucket["final_test_mrr"])
        test_recall_mean, test_recall_std = mean_std(bucket["final_test_recall_at_k"])
        test_ndcg_mean, test_ndcg_std = mean_std(bucket["final_test_ndcg_at_k"])
        selected_mean, selected_std = mean_std(bucket["num_selected"])
        best_round_mean, best_round_std = mean_std(bucket["selected_best_round"])
        summary_rows.append(
            {
                "tau": f"{tau:.2f}",
                "target_dataset": str(bucket["target_dataset"]),
                "seed_count": str(bucket["seed_count"]),
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
            }
        )

    return summary_rows, round_rows


def write_summary(summary_rows: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "compiled_tau_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def plot_metric(summary_rows: list[dict], output_dir: Path, *, mean_key: str, std_key: str, ylabel: str, title: str, filename: str) -> Path:
    taus = [float(row["tau"]) for row in summary_rows]
    means = [float(row[mean_key]) for row in summary_rows]
    errs = [float(row[std_key]) for row in summary_rows]

    plt.figure(figsize=(8, 5))
    plt.errorbar(taus, means, yerr=errs, marker="o", linewidth=2, capsize=4)
    plt.xlabel("Tau")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plot_path = output_dir / filename
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def plot_round_curves(round_rows: list[dict], output_dir: Path) -> Path:
    grouped: dict[float, list[tuple[int, float]]] = {}
    for row in round_rows:
        tau = float(row["tau"])
        round_no = int(row["round"])
        server_val_mrr = float(row.get("server_val_mrr", 0.0))
        grouped.setdefault(tau, []).append((round_no, server_val_mrr))

    plt.figure(figsize=(8, 5))
    for tau in sorted(grouped):
        pairs = sorted(grouped[tau], key=lambda p: p[0])
        rounds = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        plt.plot(rounds, values, marker="o", linewidth=2, label=f"τ={tau:.2f}")

    plt.xlabel("Round")
    plt.ylabel("Server Validation MRR")
    plt.title("DAS-FedAvg: Server Validation MRR by Tau")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plot_path = output_dir / "server_val_mrr_over_rounds_by_tau.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def plot_client_selection(round_rows: list[dict], output_dir: Path) -> Path:
    grouped: dict[float, dict[str, list]] = {}
    for row in round_rows:
        tau = float(row["tau"])
        round_no = int(row["round"])
        for key, value in row.items():
            if key.endswith("_selected") and key.startswith("client_"):
                cid = key.replace("client_", "").replace("_selected", "")
                domain_key = f"client_{cid}_domain"
                domain = row.get(domain_key, cid)
                label = f"{cid} ({domain})"
                grouped.setdefault(tau, {}).setdefault(label, []).append(
                    (round_no, int(value) if value else 0)
                )

    if not grouped:
        return output_dir / "client_selection_heatmap.png"

    fig, axes = plt.subplots(1, len(grouped), figsize=(4 * len(grouped), 4), squeeze=False)
    for col, tau in enumerate(sorted(grouped)):
        ax = axes[0][col]
        clients = sorted(grouped[tau].keys())
        data_matrix = []
        for client in clients:
            entries = sorted(grouped[tau][client], key=lambda x: x[0])
            data_matrix.append([v for _, v in entries])

        rounds = sorted(set(r for entries in grouped[tau].values() for r, _ in entries))
        ax.imshow(data_matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax.set_yticks(range(len(clients)))
        ax.set_yticklabels(clients, fontsize=8)
        ax.set_xticks(range(len(rounds)))
        ax.set_xticklabels(rounds)
        ax.set_xlabel("Round")
        ax.set_title(f"τ={tau:.2f}")

    fig.suptitle("Client Selection Heatmap")
    plot_path = output_dir / "client_selection_heatmap.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile DAS-FedAvg results_*.csv files and plot tau curves."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Result CSV files and/or directories containing results_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="compiled_tau_plots",
        help="Directory for summary CSV and plots",
    )
    args = parser.parse_args()

    csv_paths = find_result_csvs(args.inputs)
    summary_rows, round_rows = aggregate_runs(csv_paths)

    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = write_summary(summary_rows, output_dir)
    loss_plot = plot_metric(
        summary_rows,
        output_dir,
        mean_key="final_loss_mean",
        std_key="final_loss_std",
        ylabel="Final Aggregated Loss",
        title="DAS-FedAvg: Tau vs Final Loss",
        filename="tau_vs_final_loss.png",
    )
    mrr_plot = plot_metric(
        summary_rows,
        output_dir,
        mean_key="final_test_mrr_mean",
        std_key="final_test_mrr_std",
        ylabel="Final Test MRR",
        title="DAS-FedAvg: Tau vs Final Test MRR",
        filename="tau_vs_final_test_mrr.png",
    )
    ndcg_plot = plot_metric(
        summary_rows,
        output_dir,
        mean_key="final_test_ndcg_mean",
        std_key="final_test_ndcg_std",
        ylabel="Final Test NDCG@k",
        title="DAS-FedAvg: Tau vs Final Test NDCG@k",
        filename="tau_vs_final_test_ndcg.png",
    )
    round_plot = plot_round_curves(round_rows, output_dir)
    selection_plot = plot_client_selection(round_rows, output_dir)

    print(f"Loaded {len(csv_paths)} result CSV files")
    print(f"Summary CSV   : {summary_path}")
    print(f"Loss plot     : {loss_plot}")
    print(f"MRR plot      : {mrr_plot}")
    print(f"NDCG plot     : {ndcg_plot}")
    print(f"Round plot    : {round_plot}")
    print(f"Selection plot: {selection_plot}")


if __name__ == "__main__":
    main()
