#!/usr/bin/env python3
# pyright: ignore
"""Statistical analysis for transformer pretraining experiment."""

import json
import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "paper" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load experiment results from JSONL."""
    results = []
    results_file = RESULTS_DIR / "all_results.jsonl"
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return pd.DataFrame(results)


def compute_anova(df) -> float:
    """Compute two-way ANOVA across architecture/pretraining groups."""
    grouped = df.groupby(["arch", "pretrained"])["accuracy"].apply(list)
    groups = [vals for vals in grouped.values if len(vals) > 1]
    if len(groups) < 2:
        return float("nan")
    _, pvalue = stats.f_oneway(*groups)
    return float(pvalue)


def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    group1 = [float(value) for value in group1]
    group2 = [float(value) for value in group2]
    if not group1 or not group2:
        return float("nan")
    mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
    std1 = statistics.stdev(group1) if len(group1) > 1 else 0.0
    std2 = statistics.stdev(group2) if len(group2) > 1 else 0.0
    pooled_std = math.sqrt((std1**2 + std2**2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((mean1 - mean2) / pooled_std)


def _save_figure(fig, filename):
    path = FIGURES_DIR / filename
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_mean_std(values):
    data = [float(value) for value in values]
    if not data:
        return "nan"
    mean = statistics.mean(data)
    std = statistics.stdev(data) if len(data) > 1 else float("nan")
    if math.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} \\pm {std:.3f}"


def generate_figures(df):
    """Generate all 6 figures."""
    plt.switch_backend("Agg")
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 300

    # 1. Interaction plot (2x2 bar chart)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    sns.barplot(
        data=df,
        x="arch",
        y="accuracy",
        hue="pretrained",
        order=sorted(df["arch"].unique()),
        ax=ax,
        errorbar="sd",
    )
    ax.set_title("Architecture x Pretraining Interaction")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Architecture")
    ax.legend(title="Pretrained")
    _save_figure(fig, "interaction_plot.pdf")

    # 2. Dataset comparison (grouped bars)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(
        data=df,
        x="dataset",
        y="accuracy",
        hue="arch",
        order=sorted(df["dataset"].unique()),
        ax=ax,
        errorbar="sd",
    )
    ax.set_title("Dataset Comparison by Architecture")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=30)
    _save_figure(fig, "dataset_comparison.pdf")

    # 3. t-SNE visualization (4 panels)
    datasets = sorted(df["dataset"].unique())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5))
    for ax, dataset in zip(axes.flatten(), datasets, strict=False):
        subset = df[df["dataset"] == dataset].copy()
        if subset.empty:
            ax.set_axis_off()
            continue
        features = subset[["arch", "pretrained", "eval_method", "seed", "accuracy"]].copy()
        features = pd.get_dummies(features, columns=["arch", "pretrained", "eval_method"])
        if len(features) < 3:
            ax.text(0.5, 0.5, "Not enough samples", ha="center", va="center")
            ax.set_axis_off()
            continue
        perplexity = min(30, max(1, len(features) - 1))
        embedding = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            init="pca",
        ).fit_transform(features.values)
        subset = subset.assign(tsne_x=embedding[:, 0], tsne_y=embedding[:, 1])
        sns.scatterplot(
            data=subset,
            x="tsne_x",
            y="tsne_y",
            hue="arch",
            style="pretrained",
            ax=ax,
            legend=False,
        )
        ax.set_title(dataset)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    fig.suptitle("t-SNE Projection of Experiment Conditions", y=1.02)
    _save_figure(fig, "tsne_overview.pdf")

    # 4. Accuracy heatmap
    heatmap_data = (
        df.assign(group=lambda x: x["arch"] + "_" + x["pretrained"].astype(str))
        .pivot_table(index="dataset", columns="group", values="accuracy", aggfunc="mean")
        .reindex(index=sorted(df["dataset"].unique()))
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis", ax=ax)
    ax.set_title("Mean Accuracy Heatmap")
    ax.set_xlabel("Architecture / Pretraining")
    ax.set_ylabel("Dataset")
    _save_figure(fig, "accuracy_heatmap.pdf")

    # 5. Pre-training benefit (delta) by dataset
    deltas = []
    for dataset in sorted(df["dataset"].unique()):
        subset = df[df["dataset"] == dataset]
        mean_pre = subset[subset["pretrained"]]["accuracy"].mean()
        mean_scratch = subset[~subset["pretrained"]]["accuracy"].mean()
        deltas.append({"dataset": dataset, "delta": mean_pre - mean_scratch})
    delta_df = pd.DataFrame(deltas)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    sns.barplot(data=delta_df, x="dataset", y="delta", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Pretraining Benefit by Dataset")
    ax.set_ylabel("Accuracy Delta (pretrained - scratch)")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="x", rotation=30)
    _save_figure(fig, "pretraining_delta.pdf")

    # 6. Training curves (if available)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    curve_column = None
    for candidate in ["train_curve", "val_curve", "curve"]:
        if candidate in df.columns:
            curve_column = candidate
            break
    if curve_column is None:
        ax.text(0.5, 0.5, "Training curves not available", ha="center", va="center")
        ax.set_axis_off()
    else:
        for arch in sorted(df["arch"].unique()):
            curves = [
                [float(value) for value in c]
                for c in df[df["arch"] == arch][curve_column]
                if isinstance(c, list)
            ]
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            mean_curve = [statistics.mean(curve[i] for curve in curves) for i in range(min_len)]
            ax.plot(mean_curve, label=arch)
        ax.set_title("Training Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
    _save_figure(fig, "training_curves.pdf")


def generate_latex_tables(df, stats_output):
    """Generate LaTeX tables with booktabs."""

    def build_results_table(df_subset, caption):
        datasets = sorted(df_subset["dataset"].unique())
        newline = "\\\\"
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            f"Dataset & DeiT-Small (PT) & DeiT-Small (Scratch) & ResNet34 (PT) & ResNet34 (Scratch) {newline}",
            "\\midrule",
        ]
        for dataset in datasets:
            row = [dataset]
            for arch in ["deit_small", "resnet34"]:
                for pretrained in [True, False]:
                    values = df_subset[
                        (df_subset["dataset"] == dataset)
                        & (df_subset["arch"] == arch)
                        & (df_subset["pretrained"] == pretrained)
                    ]["accuracy"]
                    row.append(_format_mean_std(values))
            lines.append(" & ".join(row) + f" {newline}")
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                f"\\caption{{{caption}}}",
                "\\end{table}",
            ]
        )
        return "\n".join(lines) + "\n"

    linear_df = df[df["eval_method"] == "linear_probe"]
    knn_df = df[df["eval_method"] == "knn"]

    main_table = build_results_table(linear_df, "Linear probe accuracy (mean \\pm std).")
    knn_table = build_results_table(knn_df, "k-NN accuracy (mean \\pm std).")

    newline = "\\\\"
    interaction_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lc}",
        "\\toprule",
        f"Metric & Value {newline}",
        "\\midrule",
        f"ANOVA p-value & {stats_output['anova_interaction_pvalue']:.6f} {newline}",
        f"Cohen's d (DeiT-Small) & {stats_output['cohens_d_deit']:.3f} {newline}",
        f"Cohen's d (ResNet34) & {stats_output['cohens_d_resnet']:.3f} {newline}",
        f"Interaction delta & {stats_output['interaction_delta']:.3f} {newline}",
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Interaction statistics and effect sizes.}",
        "\\end{table}",
    ]
    interaction_table = "\n".join(interaction_lines) + "\n"

    (TABLES_DIR / "main_results.tex").write_text(main_table)
    (TABLES_DIR / "knn_results.tex").write_text(knn_table)
    (TABLES_DIR / "interaction.tex").write_text(interaction_table)


def main():
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} experiments")

    print("Computing statistics...")
    anova_p = compute_anova(df)

    stats_by_arch = {}
    for arch in sorted(df["arch"].unique()):
        subset = df[df["arch"] == arch]
        pretrained_acc = list(subset[subset["pretrained"]]["accuracy"])
        scratch_acc = list(subset[~subset["pretrained"]]["accuracy"])
        mean_pre = statistics.mean(pretrained_acc) if pretrained_acc else float("nan")
        mean_scratch = statistics.mean(scratch_acc) if scratch_acc else float("nan")
        stats_by_arch[arch] = {
            "delta": float(mean_pre - mean_scratch),
            "cohens_d": compute_cohens_d(pretrained_acc, scratch_acc),
        }

    interaction_delta = stats_by_arch["deit_small"]["delta"] - stats_by_arch["resnet34"]["delta"]

    stats_output = {
        "anova_interaction_pvalue": anova_p,
        "cohens_d_deit": stats_by_arch["deit_small"]["cohens_d"],
        "cohens_d_resnet": stats_by_arch["resnet34"]["cohens_d"],
        "interaction_delta": float(interaction_delta),
    }

    with open(RESULTS_DIR / "statistical_analysis.json", "w") as f:
        json.dump(stats_output, f, indent=2)

    print("Generating figures...")
    generate_figures(df)

    print("Generating LaTeX tables...")
    generate_latex_tables(df, stats_output)

    print("Analysis complete!")


if __name__ == "__main__":
    main()
