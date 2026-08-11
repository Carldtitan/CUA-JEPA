from __future__ import annotations

import csv
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_charts"

MODEL_NAMES = {
    "model2": "Model 2\nSFT only",
    "model3": "Model 3\nNo-action JEPA + SFT",
    "model4": "Model 4\nAction JEPA + SFT",
}
SHORT_NAMES = {"model2": "Model 2", "model3": "Model 3", "model4": "Model 4"}
COLORS = {"model2": "#0072B2", "model3": "#E69F00", "model4": "#D55E00"}
ACTION_ORDER = [
    "click",
    "double_click",
    "right_click",
    "moveTo",
    "write",
    "press",
    "hotkey",
    "scroll",
]

SFT_DIRS = {
    "model2": ROOT / "artifacts" / "sft-model2-full-seed20260810-20260810T220702Z",
    "model3": ROOT / "artifacts" / "sft-model3-full-seed20260810-20260810T235313Z",
    "model4": ROOT / "artifacts" / "sft-model4-full-seed20260810-20260810T220704Z",
}
JEPA_DIRS = {
    "model3": ROOT / "artifacts" / "model3-model3_full-seed20260809-20260810T214351Z",
    "model4": ROOT / "artifacts" / "model4-model4_full-seed20260809-20260810T052900Z",
}
DATA_AUDIT_PATH = ROOT / "tmp" / "sft-final-audit" / "dataset_audit.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def minutes(seconds: float) -> float:
    return seconds / 60


def annotate_bars(ax, bars, fmt=lambda value: f"{value:.1f}", pad=3) -> None:
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            fmt(height),
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def moving_average(values: list[float], window: int = 5) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < window:
        return array
    return np.convolve(array, np.ones(window) / window, mode="same")


def add_source_note(fig, text: str) -> None:
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=7, color="#555555")


def style_axis(ax, *, grid: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, alpha=0.22, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_bars(
    ax,
    categories: list[str],
    values: dict[str, list[float]],
    *,
    ylabel: str,
    percent_axis: bool = False,
    legend: bool = True,
) -> None:
    x = np.arange(len(categories))
    width = 0.24
    offsets = [-width, 0, width]
    for offset, model in zip(offsets, ["model2", "model3", "model4"], strict=True):
        bars = ax.bar(
            x + offset,
            values[model],
            width,
            label=SHORT_NAMES[model],
            color=COLORS[model],
        )
        if len(categories) <= 6:
            annotate_bars(ax, bars, (lambda v: f"{v:.0f}%") if percent_axis else None)
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel)
    if percent_axis:
        ax.set_ylim(0, 105)
    if legend:
        ax.legend(frameon=False, ncol=3)
    style_axis(ax)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    sft = {model: read_json(path / "final_metrics.json") for model, path in SFT_DIRS.items()}
    sft_train = {
        model: read_jsonl(path / "train_metrics.jsonl") for model, path in SFT_DIRS.items()
    }
    sft_predictions = {
        model: read_jsonl(path / "final_validation_predictions.jsonl")
        for model, path in SFT_DIRS.items()
    }
    jepa = {model: read_json(path / "final_metrics.json") for model, path in JEPA_DIRS.items()}
    jepa_train = {model: read_jsonl(path / "train.jsonl") for model, path in JEPA_DIRS.items()}
    jepa_eval = {
        model: read_jsonl(path / "evaluation_checkpoints.jsonl")
        for model, path in JEPA_DIRS.items()
    }
    jepa_data = read_json(JEPA_DIRS["model4"] / "dataset_audit.json")
    data = read_json(DATA_AUDIT_PATH)

    manifest: list[dict] = []
    pdf_path = OUT / "research-chart-pack.pdf"

    def save(fig, filename: str, title: str, description: str, sources: list[str]) -> None:
        # Keep the report title above subplot titles in multi-panel charts.
        fig.suptitle(title, y=1.045, fontsize=15, fontweight="bold")
        add_source_note(fig, "Source: " + "; ".join(sources))
        path = OUT / filename
        fig.savefig(path, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        manifest.append(
            {
                "file": filename,
                "title": title,
                "description": description,
                "sources": sources,
            }
        )
        plt.close(fig)

    with PdfPages(pdf_path) as pdf:
        # 1. Final exact success.
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        models = ["model2", "model3", "model4"]
        vals = [100 * sft[m]["final_validation"]["exact_success_rate"] for m in models]
        bars = ax.bar([MODEL_NAMES[m] for m in models], vals, color=[COLORS[m] for m in models])
        annotate_bars(ax, bars, lambda v: f"{v:.1f}%")
        ax.set_ylabel("Exact success on 250 held-out examples")
        ax.set_ylim(0, 60)
        style_axis(ax)
        save(
            fig,
            "01-final-exact-success.png",
            "Final computer-use accuracy",
            "Model 2 clearly outperformed both JEPA-transfer models.",
            ["SFT final_metrics.json"],
        )

        # 2. Before and after SFT.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for index, model in enumerate(models):
            before = 100 * sft[model]["initial_validation"]["exact_success_rate"]
            after = 100 * sft[model]["final_validation"]["exact_success_rate"]
            ax.plot([0, 1], [before, after], marker="o", linewidth=3, color=COLORS[model])
            ax.text(-0.03, before, f"{before:.1f}%", ha="right", va="center")
            ax.text(1.03, after, f"{after:.1f}%  (+{after-before:.1f})", ha="left", va="center")
            ax.text(0.5, (before + after) / 2 + 1.1, SHORT_NAMES[model], ha="center")
        ax.set_xticks([0, 1], ["Before SFT", "After SFT"])
        ax.set_ylabel("Exact success (%)")
        ax.set_xlim(-0.22, 1.32)
        ax.set_ylim(0, 58)
        style_axis(ax)
        save(
            fig,
            "02-sft-before-after.png",
            "What SFT added to each starting point",
            "SFT helped all models, but it did not repair the large JEPA transfer loss.",
            ["SFT initial_validation", "SFT final_validation"],
        )

        # 3. Final metric profile.
        metrics = [
            ("Parse", "parse_rate"),
            ("Action type", "action_type_accuracy"),
            ("Exact", "exact_success_rate"),
            ("Mean score", "mean_action_score"),
            ("Macro score", "macro_action_score"),
        ]
        values = {
            model: [100 * sft[model]["final_validation"][key] for _, key in metrics]
            for model in models
        }
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        grouped_bars(ax, [label for label, _ in metrics], values, ylabel="Score (%)", percent_axis=True)
        save(
            fig,
            "03-final-metric-profile.png",
            "Final policy metric profile",
            "All models produced valid JSON, but the JEPA models often chose the wrong action or location.",
            ["SFT final_validation_metrics.json"],
        )

        # 4. Coordinate precision thresholds.
        thresholds = [("≤ 0.02", "hit_rate_at_0_02"), ("≤ 0.05", "hit_rate_at_0_05"), ("≤ 0.10", "hit_rate_at_0_1")]
        values = {
            model: [100 * sft[model]["final_validation"]["coordinate"][key] for _, key in thresholds]
            for model in models
        }
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        grouped_bars(ax, [x[0] for x in thresholds], values, ylabel="Coordinate hit rate (%)", percent_axis=True)
        ax.set_xlabel("Maximum normalized coordinate error")
        save(
            fig,
            "04-coordinate-precision.png",
            "Click precision at three tolerances",
            "Model 2 found screen locations much more accurately at every tolerance.",
            ["SFT final_validation coordinate metrics"],
        )

        # 5. Mean action score by action type.
        values = {
            model: [
                100 * sft[model]["final_validation"]["by_action"].get(action, {}).get("mean_action_score", np.nan)
                for action in ACTION_ORDER
            ]
            for model in models
        }
        fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
        grouped_bars(ax, ACTION_ORDER, values, ylabel="Mean action score (%)", percent_axis=True)
        ax.tick_params(axis="x", rotation=25)
        save(
            fig,
            "05-action-score-by-type.png",
            "Final quality by GUI action type",
            "Click dominates the dataset. Rare-action scores are unstable because their sample counts are small.",
            ["SFT final_validation by_action"],
        )

        # 6. Action-type classification accuracy.
        values = {
            model: [
                100 * sft[model]["final_validation"]["by_action"].get(action, {}).get("action_type_accuracy", np.nan)
                for action in ACTION_ORDER
            ]
            for model in models
        }
        fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
        grouped_bars(ax, ACTION_ORDER, values, ylabel="Correct action type (%)", percent_axis=True)
        ax.tick_params(axis="x", rotation=25)
        save(
            fig,
            "06-action-type-accuracy.png",
            "Did the model choose the correct kind of action?",
            "Model 2 was strongest overall. All models were weak on several rare actions.",
            ["SFT final_validation by_action"],
        )

        # 7. Mean score by operating system.
        systems = ["Darwin", "Ubuntu", "Windows"]
        values = {
            model: [100 * sft[model]["final_validation"]["by_system"][system]["mean_action_score"] for system in systems]
            for model in models
        }
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        grouped_bars(ax, systems, values, ylabel="Mean action score (%)", percent_axis=True)
        save(
            fig,
            "07-score-by-operating-system.png",
            "Final policy score by operating system",
            "Model 2 led on macOS, Ubuntu, and Windows.",
            ["SFT final_validation by_system"],
        )

        # 8. Domain score with validation sample size.
        domain_counts = Counter(row["domain"] for row in sft_predictions["model2"])
        top_domains = [name for name, _ in domain_counts.most_common(12)]
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True, gridspec_kw={"width_ratios": [3, 1]})
        y = np.arange(len(top_domains))
        height = 0.22
        for offset, model in zip([-height, 0, height], models, strict=True):
            by_domain = defaultdict(list)
            for row in sft_predictions[model]:
                by_domain[row["domain"]].append(row["score"])
            scores = [100 * np.mean(by_domain[name]) for name in top_domains]
            axes[0].barh(y + offset, scores, height, color=COLORS[model], label=SHORT_NAMES[model])
        axes[0].set_yticks(y, [textwrap.fill(name, 24) for name in top_domains])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Mean action score (%)")
        axes[0].legend(frameon=False, ncol=3)
        style_axis(axes[0], grid="x")
        counts = [domain_counts[name] for name in top_domains]
        bars = axes[1].barh(y, counts, color="#777777")
        axes[1].set_yticks(y, [])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Validation examples")
        for bar, count in zip(bars, counts, strict=True):
            axes[1].text(count + 0.3, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=8)
        style_axis(axes[1], grid="x")
        save(
            fig,
            "08-score-by-domain.png",
            "Final policy score by software domain",
            "The right panel shows sample size so that small-domain results are not mistaken for stable estimates.",
            ["SFT final_validation_predictions.jsonl"],
        )

        # 9. Confusion matrices.
        actions = ACTION_ORDER
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
        for ax, model in zip(axes, models, strict=True):
            matrix = np.zeros((len(actions), len(actions)))
            for row in sft_predictions[model]:
                target = row["target_action"]
                predicted = row.get("predicted_action")
                if target in actions and predicted in actions:
                    matrix[actions.index(target), actions.index(predicted)] += 1
            row_totals = matrix.sum(axis=1, keepdims=True)
            norm = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals > 0)
            image = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            ax.set_title(SHORT_NAMES[model])
            ax.set_xticks(range(len(actions)), actions, rotation=55, ha="right", fontsize=8)
            ax.set_yticks(range(len(actions)), actions, fontsize=8)
            ax.set_xlabel("Predicted action")
            if ax is axes[0]:
                ax.set_ylabel("Target action")
            for i in range(len(actions)):
                for j in range(len(actions)):
                    if matrix[i, j] and norm[i, j] >= 0.08:
                        ax.text(j, i, f"{100*norm[i,j]:.0f}", ha="center", va="center", fontsize=7,
                                color="white" if norm[i, j] > 0.55 else "black")
        fig.colorbar(image, ax=axes, label="Row share", shrink=0.8)
        save(
            fig,
            "09-action-confusion-matrices.png",
            "Which actions were confused?",
            "Rows are the correct actions. Columns are the model outputs. Values are percentages within each row.",
            ["SFT final_validation_predictions.jsonl"],
        )

        # 10. Per-example score distributions.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        bins = np.linspace(-0.025, 1.025, 22)
        for model in models:
            scores = [row["score"] for row in sft_predictions[model]]
            ax.hist(scores, bins=bins, histtype="step", linewidth=2.4, color=COLORS[model], label=SHORT_NAMES[model])
        ax.set_xlabel("Per-example action score")
        ax.set_ylabel("Examples")
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "10-example-score-distribution.png",
            "Distribution of final per-example scores",
            "Model 2 produced many more fully correct actions.",
            ["SFT final_validation_predictions.jsonl"],
        )

        # 11. Coordinate distance CDF.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for model in models:
            distances = sorted(
                row["coordinate_distance"]
                for row in sft_predictions[model]
                if row.get("coordinate_distance") is not None and row.get("type_correct")
            )
            if distances:
                ax.plot(distances, np.arange(1, len(distances) + 1) / len(distances), color=COLORS[model], linewidth=2.4, label=SHORT_NAMES[model])
        for threshold in [0.02, 0.05, 0.1]:
            ax.axvline(threshold, color="#777777", alpha=0.5, linestyle="--")
            ax.text(threshold, 0.03, f"{threshold:.2f}", rotation=90, ha="right", va="bottom", fontsize=8)
        ax.set_xlim(0, 0.65)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Normalized coordinate distance; lower is better")
        ax.set_ylabel("Share at or below distance")
        ax.legend(frameon=False)
        style_axis(ax, grid="both")
        save(
            fig,
            "11-coordinate-distance-cdf.png",
            "Coordinate error distribution when the action type was correct",
            "The Model 2 curve rises earlier, which means its predicted locations were closer to the targets.",
            ["SFT final_validation_predictions.jsonl"],
        )

        # 12. Small live monitor curve.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for model in models:
            steps, scores = [], []
            for step in [100, 200, 300, 400]:
                metrics = read_json(SFT_DIRS[model] / f"monitor_step_{step:06d}_metrics.json")
                steps.append(step)
                scores.append(100 * metrics["exact_success_rate"])
            ax.plot(steps, scores, marker="o", linewidth=2.4, color=COLORS[model], label=SHORT_NAMES[model])
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Exact success on fixed 32-example monitor (%)")
        ax.set_ylim(0, 75)
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "12-small-monitor-trajectory.png",
            "Small live-monitor trajectory",
            "This monitor was easier than the full validation set. It is a health signal, not the final result.",
            ["SFT monitor_step metrics; 32 examples"],
        )

        # 13. SFT training loss curves.
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for model in models:
            steps = [row["step"] for row in sft_train[model]]
            losses = [row["mean_loss"] for row in sft_train[model]]
            ax.plot(steps, moving_average(losses, 5), color=COLORS[model], linewidth=2.2, label=SHORT_NAMES[model])
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Recent mean training loss")
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "13-sft-training-loss.png",
            "SFT training loss",
            "Model 2 had the lowest overall training loss. Curves use a five-record moving average.",
            ["SFT train_metrics.jsonl"],
        )

        # 14. First 20 versus last 20 losses.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        x = np.arange(3)
        width = 0.34
        first = [sft[m]["first_20_loss"] for m in models]
        last = [sft[m]["last_20_loss"] for m in models]
        b1 = ax.bar(x - width / 2, first, width, label="First 20", color="#999999")
        b2 = ax.bar(x + width / 2, last, width, label="Last 20", color=[COLORS[m] for m in models])
        annotate_bars(ax, b1, lambda v: f"{v:.2f}")
        annotate_bars(ax, b2, lambda v: f"{v:.2f}")
        ax.set_xticks(x, [SHORT_NAMES[m] for m in models])
        ax.set_ylabel("Training loss")
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "14-sft-loss-reduction.png",
            "Loss reduction during SFT",
            "All models learned the supervised format, but lower training loss did not remove the JEPA transfer gap.",
            ["SFT final_metrics.json"],
        )

        # 15. Gradient norms in small multiples.
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
        for ax, model in zip(axes, models, strict=True):
            steps = [row["step"] for row in sft_train[model]]
            ax.plot(steps, [row["vision_lora_gradient_norm"] for row in sft_train[model]], color=COLORS[model], label="Vision LoRA")
            ax.plot(steps, [row["language_lora_gradient_norm"] for row in sft_train[model]], color="#555555", label="Language LoRA")
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_ylabel("Gradient norm")
            ax.set_title(SHORT_NAMES[model], loc="left", fontsize=10)
            style_axis(ax)
        axes[0].legend(frameon=False, ncol=2)
        axes[-1].set_xlabel("Optimizer step")
        save(
            fig,
            "15-sft-gradient-norms.png",
            "Visual and language LoRA gradient norms",
            "Both adapter parts received gradients throughout training. The poor JEPA result was not caused by zero gradients.",
            ["SFT train_metrics.jsonl"],
        )

        # 16. Learning-rate schedule.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for model in models:
            ax.plot(
                [row["step"] for row in sft_train[model]],
                [row["learning_rate"] for row in sft_train[model]],
                color=COLORS[model],
                linewidth=2,
                label=SHORT_NAMES[model],
            )
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Learning rate")
        ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "16-sft-learning-rate.png",
            "Matched SFT learning-rate schedule",
            "The three runs used the same warmup and cosine decay schedule.",
            ["SFT train_metrics.jsonl"],
        )

        # 17. GPU memory during SFT.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for model in models:
            ax.plot(
                [row["step"] for row in sft_train[model]],
                [row["peak_cuda_memory_gib"] for row in sft_train[model]],
                color=COLORS[model],
                linewidth=2,
                label=SHORT_NAMES[model],
            )
        ax.axhline(24, color="#777777", linestyle="--", label="L4 physical VRAM: 24 GiB")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Peak measured GPU memory (GiB)")
        ax.set_ylim(0, 26)
        ax.legend(frameon=False, ncol=2)
        style_axis(ax)
        save(
            fig,
            "17-sft-gpu-memory.png",
            "SFT GPU memory use",
            "Each full SFT run peaked at 13.84 GiB on a 24 GiB L4 GPU.",
            ["SFT train_metrics.jsonl", "Modal L4 specification"],
        )

        # 18. SFT resource panels.
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
        resource_values = [
            ("Elapsed time (min)", [minutes(sft[m]["elapsed_seconds"]) for m in models], lambda v: f"{v:.1f}"),
            ("Estimated cost (USD)", [sft[m]["estimated_modal_cost_usd"] for m in models], lambda v: f"${v:.2f}"),
            ("Peak GPU memory (GiB)", [sft[m]["peak_cuda_memory_gib"] for m in models], lambda v: f"{v:.2f}"),
        ]
        for ax, (label, vals, formatter) in zip(axes, resource_values, strict=True):
            bars = ax.bar([SHORT_NAMES[m] for m in models], vals, color=[COLORS[m] for m in models])
            annotate_bars(ax, bars, formatter)
            ax.set_ylabel(label)
            style_axis(ax)
        save(
            fig,
            "18-sft-time-cost-memory.png",
            "SFT time, cost, and peak GPU memory",
            "Resource use was almost equal, which supports the fairness of the model comparison.",
            ["SFT final_metrics.json"],
        )

        # 19. Trainable parameter composition.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        vision = [sft[m]["initialization_audit"]["vision_lora_parameters"] / 1e6 for m in models]
        language = [sft[m]["initialization_audit"]["language_lora_parameters"] / 1e6 for m in models]
        x = np.arange(3)
        b1 = ax.bar(x, vision, label="Vision LoRA", color="#56B4E9")
        b2 = ax.bar(x, language, bottom=vision, label="Language LoRA", color="#009E73")
        for index, total in enumerate(np.array(vision) + np.array(language)):
            ax.text(index, total + 0.08, f"{total:.2f}M", ha="center", fontsize=9)
        ax.set_xticks(x, [SHORT_NAMES[m] for m in models])
        ax.set_ylabel("Trainable parameters (millions)")
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "19-sft-trainable-parameters.png",
            "Trainable SFT parameter allocation",
            "All three models trained the same number of adapter parameters. Base Qwen parameters were frozen.",
            ["SFT initialization_audit.json"],
        )

        # 20. Training loss by action.
        action_loss: dict[str, list[float]] = {model: [] for model in models}
        for model in models:
            bucket: dict[str, list[float]] = defaultdict(list)
            for row in sft_train[model]:
                for action, value in row.get("mean_loss_by_action", {}).items():
                    bucket[action].append(value)
            action_loss[model] = [np.mean(bucket[action]) if bucket[action] else np.nan for action in ACTION_ORDER]
        fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
        grouped_bars(ax, ACTION_ORDER, action_loss, ylabel="Mean logged training loss")
        ax.tick_params(axis="x", rotation=25)
        save(
            fig,
            "20-sft-loss-by-action.png",
            "Training difficulty by action type",
            "Writing and rare actions were often harder than ordinary clicks.",
            ["SFT train_metrics.jsonl"],
        )

        # 21. Paired wins, losses, and ties.
        pairs = [("model2", "model3"), ("model2", "model4"), ("model3", "model4")]
        pair_labels = ["Model 2 vs 3", "Model 2 vs 4", "Model 3 vs 4"]
        wins_a, ties, wins_b = [], [], []
        for a, b in pairs:
            a_rows = {row["example_id"]: row["score"] for row in sft_predictions[a]}
            b_rows = {row["example_id"]: row["score"] for row in sft_predictions[b]}
            delta = [a_rows[key] - b_rows[key] for key in a_rows]
            wins_a.append(sum(value > 1e-12 for value in delta))
            ties.append(sum(abs(value) <= 1e-12 for value in delta))
            wins_b.append(sum(value < -1e-12 for value in delta))
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        x = np.arange(3)
        ax.bar(x, wins_a, label="First model wins", color="#0072B2")
        ax.bar(x, ties, bottom=wins_a, label="Tie", color="#BBBBBB")
        ax.bar(x, wins_b, bottom=np.array(wins_a) + np.array(ties), label="Second model wins", color="#D55E00")
        for i, total in enumerate(np.array(wins_a) + np.array(ties) + np.array(wins_b)):
            ax.text(i, total + 3, str(total), ha="center")
        ax.set_xticks(x, pair_labels)
        ax.set_ylabel("Held-out examples")
        ax.legend(frameon=False, ncol=3)
        style_axis(ax)
        save(
            fig,
            "21-paired-example-outcomes.png",
            "Paired wins and ties on the same 250 examples",
            "Most Model 3 versus Model 4 examples were ties. Model 2 had many more wins against both.",
            ["SFT final_validation_predictions.jsonl"],
        )

        # 22. Error decomposition.
        categories = ["Fully correct", "Correct type, wrong details", "Wrong action type", "Could not parse"]
        error_values = {model: [] for model in models}
        for model in models:
            counts = Counter()
            for row in sft_predictions[model]:
                if row["score"] >= 1 - 1e-12:
                    counts[categories[0]] += 1
                elif not row.get("parsed"):
                    counts[categories[3]] += 1
                elif row.get("type_correct"):
                    counts[categories[1]] += 1
                else:
                    counts[categories[2]] += 1
            error_values[model] = [100 * counts[name] / 250 for name in categories]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        bottom = np.zeros(3)
        category_colors = ["#009E73", "#E69F00", "#D55E00", "#777777"]
        for index, category in enumerate(categories):
            vals = [error_values[m][index] for m in models]
            ax.bar(range(3), vals, bottom=bottom, color=category_colors[index], label=category)
            bottom += vals
        ax.set_xticks(range(3), [SHORT_NAMES[m] for m in models])
        ax.set_ylabel("Validation examples (%)")
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, ncol=2)
        style_axis(ax)
        save(
            fig,
            "22-final-error-decomposition.png",
            "Where final policy errors came from",
            "All outputs parsed. The main failures were wrong action types and wrong action details.",
            ["SFT final_validation_predictions.jsonl"],
        )

        # 23. Data action mix.
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
        for ax, split in zip(axes, ["train", "validation"], strict=True):
            counts = data[f"{split}_by_action"]
            total = sum(counts.values())
            vals = [100 * counts.get(action, 0) / total for action in ACTION_ORDER]
            bars = ax.bar(ACTION_ORDER, vals, color="#0072B2")
            ax.set_title(f"{split.title()} set: {total:,} examples")
            ax.set_ylabel("Share (%)")
            ax.tick_params(axis="x", rotation=35)
            style_axis(ax)
            for bar, action in zip(bars, ACTION_ORDER, strict=True):
                count = counts.get(action, 0)
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, str(count), ha="center", fontsize=7)
        save(
            fig,
            "23-data-action-mix.png",
            "Action distribution in the SFT data",
            "Clicks were 65% of training data. Scroll had only 6 training and 2 validation examples.",
            ["AgentNet dataset_audit.json"],
        )

        # 24. Data OS balance.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        x = np.arange(3)
        train_share = [100 * data["train_by_system"][s] / data["train_examples"] for s in systems]
        val_share = [100 * data["validation_by_system"][s] / data["validation_examples"] for s in systems]
        b1 = ax.bar(x - 0.18, train_share, 0.36, label="Train", color="#0072B2")
        b2 = ax.bar(x + 0.18, val_share, 0.36, label="Validation", color="#E69F00")
        annotate_bars(ax, b1, lambda v: f"{v:.1f}%")
        annotate_bars(ax, b2, lambda v: f"{v:.1f}%")
        ax.set_xticks(x, systems)
        ax.set_ylabel("Share (%)")
        ax.set_ylim(0, 40)
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "24-data-os-balance.png",
            "Operating-system balance",
            "Train and validation data were nearly equal across macOS, Ubuntu, and Windows.",
            ["AgentNet dataset_audit.json"],
        )

        # 25. Domain mix.
        train_domains = data["train_by_domain"]
        chosen_domains = [name for name, _ in sorted(train_domains.items(), key=lambda item: item[1], reverse=True)[:15]]
        fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
        y = np.arange(len(chosen_domains))
        train_vals = [100 * train_domains[name] / data["train_examples"] for name in chosen_domains]
        val_vals = [100 * data["validation_by_domain"].get(name, 0) / data["validation_examples"] for name in chosen_domains]
        ax.barh(y - 0.18, train_vals, 0.36, color="#0072B2", label="Train")
        ax.barh(y + 0.18, val_vals, 0.36, color="#E69F00", label="Validation")
        ax.set_yticks(y, [textwrap.fill(name, 27) for name in chosen_domains])
        ax.invert_yaxis()
        ax.set_xlabel("Dataset share (%)")
        ax.legend(frameon=False)
        style_axis(ax, grid="x")
        save(
            fig,
            "25-data-domain-mix.png",
            "Largest software domains in the SFT data",
            "The data covered many domains, but the distribution was not uniform.",
            ["AgentNet dataset_audit.json"],
        )

        # 26. Selection rate from candidate pool.
        selected_actions = {
            action: data["train_by_action"].get(action, 0) + data["validation_by_action"].get(action, 0)
            for action in ACTION_ORDER
        }
        candidate_actions = data["candidate_examples_by_action"]
        rates = [100 * selected_actions[action] / candidate_actions[action] for action in ACTION_ORDER]
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        bars = ax.bar(ACTION_ORDER, rates, color="#0072B2")
        ax.set_yscale("log")
        ax.set_ylabel("Selected from candidate pool (%) — log scale")
        ax.tick_params(axis="x", rotation=30)
        for bar, rate in zip(bars, rates, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, rate * 1.15, f"{rate:.2f}%", ha="center", fontsize=8)
        style_axis(ax)
        save(
            fig,
            "26-data-selection-rate.png",
            "Sampling rate by action type",
            "Rare actions were sampled more heavily than clicks, but some still had very small absolute counts.",
            ["AgentNet candidate and selected counts"],
        )

        # 27. Dataset integrity panel.
        fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
        ax.axis("off")
        facts = [
            ("Stored images", f"{data['stored_images']:,}"),
            ("Unique hashes", f"{data['unique_stored_image_hashes']:,}"),
            ("Duplicates", str(data["duplicate_stored_images"])),
            ("Missing images", str(data["missing_images"])),
            ("Train–validation image overlap", str(data["exact_train_validation_image_hash_overlap"])),
            ("Task overlap", str(data["task_overlap"])),
        ]
        for index, (label, value) in enumerate(facts):
            x = (index % 3) / 3 + 0.02
            y = 0.62 if index < 3 else 0.18
            ax.text(x, y + 0.13, value, fontsize=24, fontweight="bold", transform=ax.transAxes)
            ax.text(x, y, textwrap.fill(label, 26), fontsize=10, transform=ax.transAxes)
        save(
            fig,
            "27-dataset-integrity.png",
            "Dataset integrity checks",
            "The saved SFT dataset had no missing images, duplicates, or exact split overlap.",
            ["AgentNet dataset_audit.json"],
        )

        # 28. JEPA before/after four-way accuracy.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for index, model in enumerate(["model3", "model4"]):
            before = 100 * jepa[model]["initial_validation"]["four_way_accuracy"]
            after = 100 * jepa[model]["final_validation"]["four_way_accuracy"]
            ax.plot([0, 1], [before, after], marker="o", linewidth=3, color=COLORS[model], label=SHORT_NAMES[model])
            ax.text(-0.03, before, f"{before:.1f}%", ha="right", va="center")
            ax.text(1.03, after, f"{after:.1f}% ({after-before:+.1f})", ha="left", va="center")
        ax.axhline(25, color="#777777", linestyle="--", label="Four-way chance")
        ax.set_xticks([0, 1], ["Before JEPA", "After JEPA"])
        ax.set_ylabel("Future-screen four-way accuracy (%)")
        ax.set_xlim(-0.18, 1.3)
        ax.set_ylim(15, 45)
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "28-jepa-before-after-accuracy.png",
            "What JEPA training changed",
            "Action-conditioned Model 4 improved future matching. The no-action control remained at chance.",
            ["JEPA final_metrics.json"],
        )

        # Utility for evaluation checkpoints.
        def validation_curve(model: str) -> list[dict]:
            allowed = {"initial_validation_monitor", "validation_monitor", "final_validation"}
            by_step: dict[int, dict] = {}
            for row in jepa_eval[model]:
                if row.get("evaluation") in allowed:
                    by_step[int(row["step"])] = row
            by_step[0] = jepa[model]["initial_validation"] | {"step": 0}
            by_step[int(jepa[model]["steps"])] = jepa[model]["final_validation"] | {"step": int(jepa[model]["steps"])}
            return [by_step[key] for key in sorted(by_step)]

        curves = {model: validation_curve(model) for model in ["model3", "model4"]}

        # 29. JEPA accuracy curve.
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for model in ["model3", "model4"]:
            ax.plot(
                [row["step"] for row in curves[model]],
                [100 * row["four_way_accuracy"] for row in curves[model]],
                marker="o",
                linewidth=2.2,
                color=COLORS[model],
                label=SHORT_NAMES[model],
            )
        ax.axhline(25, color="#777777", linestyle="--", label="Chance")
        ax.set_xlabel("JEPA optimizer step")
        ax.set_ylabel("Validation four-way accuracy (%)")
        ax.set_ylim(15, 45)
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "29-jepa-accuracy-curve.png",
            "JEPA future-screen accuracy over training",
            "Only the action-conditioned model moved clearly above chance.",
            ["JEPA evaluation_checkpoints.jsonl"],
        )

        # 30. JEPA loss curves.
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        for ax, model in zip(axes, ["model3", "model4"], strict=True):
            rows = [row for row in jepa_train[model] if "loss" in row]
            ax.plot([r["step"] for r in rows], moving_average([r["loss"] for r in rows], 7), color=COLORS[model], label="Total loss")
            ax.plot([r["step"] for r in rows], moving_average([r["regression_loss"] for r in rows], 7), color="#555555", label="Regression loss")
            if model == "model4":
                ax.plot([r["step"] for r in rows], moving_average([r["action_separation_loss"] for r in rows], 7), color="#009E73", label="Action-separation loss")
            ax.set_yscale("log")
            ax.set_ylabel("Loss — log scale")
            ax.set_title(SHORT_NAMES[model], loc="left", fontsize=10)
            ax.legend(frameon=False, ncol=3)
            style_axis(ax)
        axes[-1].set_xlabel("JEPA optimizer step")
        save(
            fig,
            "30-jepa-loss-curves.png",
            "JEPA training losses",
            "Both runs reduced latent regression loss. Model 4 also optimized action separation.",
            ["JEPA train.jsonl"],
        )

        # 31. Mean loss components.
        component_keys = [
            ("Regression", "mean_regression_loss"),
            ("Changed region", "mean_changed_region_loss"),
            ("Global", "mean_global_loss"),
            ("Variance", "mean_variance_loss"),
            ("Covariance", "mean_covariance_loss"),
            ("Relation", "mean_relation_loss"),
            ("Action separation", "mean_action_separation_loss"),
        ]
        fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
        x = np.arange(len(component_keys))
        width = 0.36
        for offset, model in zip([-width / 2, width / 2], ["model3", "model4"], strict=True):
            vals = [jepa[model][key] for _, key in component_keys]
            ax.bar(x + offset, vals, width, color=COLORS[model], label=SHORT_NAMES[model])
        ax.set_xticks(x, [label for label, _ in component_keys], rotation=25, ha="right")
        ax.set_ylabel("Mean component value")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "31-jepa-loss-components.png",
            "JEPA objective components",
            "Model 3 correctly had zero action-separation loss. Model 4 paid a large action-separation term.",
            ["JEPA final_metrics.json"],
        )

        # 32. Action sensitivity.
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for model in ["model3", "model4"]:
            rows = curves[model]
            ax.plot(
                [row["step"] for row in rows],
                [row.get("shuffled_minus_correct", np.nan) for row in rows],
                marker="o",
                linewidth=2.2,
                color=COLORS[model],
                label=SHORT_NAMES[model],
            )
        ax.axhline(0, color="#777777", linestyle="--")
        ax.set_xlabel("JEPA optimizer step")
        ax.set_ylabel("Shuffled-action distance minus correct-action distance")
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "32-jepa-action-sensitivity.png",
            "How much correct action information helped JEPA prediction",
            "A positive gap means correct actions produced closer predictions. Model 4 developed a small positive gap.",
            ["JEPA evaluation_checkpoints.jsonl"],
        )

        # 33. Target and predicted latent variance.
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for ax, model in zip(axes, ["model3", "model4"], strict=True):
            rows = curves[model]
            ax.plot([r["step"] for r in rows], [r["target_latent_variance"] for r in rows], marker="o", color="#0072B2", label="Target variance")
            ax.plot([r["step"] for r in rows], [r["predicted_latent_variance"] for r in rows], marker="s", color="#D55E00", label="Predicted variance")
            ax.set_yscale("log")
            ax.set_title(SHORT_NAMES[model])
            ax.set_xlabel("JEPA optimizer step")
            ax.set_ylabel("Latent variance — log scale")
            ax.legend(frameon=False)
            style_axis(ax)
        save(
            fig,
            "33-jepa-representation-variance.png",
            "Target and predicted representation variance",
            "Model 4's final predicted variance was very small even though target variance remained substantial.",
            ["JEPA evaluation_checkpoints.jsonl"],
        )

        # 34. Accuracy by changed-pixel bucket.
        buckets = ["0-1%", "1-5%", "5-20%", "20-50%", "50-100%"]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for model in ["model3", "model4"]:
            vals = [100 * jepa[model]["final_validation"]["by_changed_pixel_bucket"][bucket]["four_way_accuracy"] for bucket in buckets]
            ax.plot(buckets, vals, marker="o", linewidth=2.2, color=COLORS[model], label=SHORT_NAMES[model])
        ax.axhline(25, color="#777777", linestyle="--", label="Chance")
        ax.set_xlabel("Fraction of screen pixels changed")
        ax.set_ylabel("Final four-way accuracy (%)")
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "34-jepa-accuracy-by-screen-change.png",
            "JEPA accuracy by amount of screen change",
            "Action-conditioned performance varied strongly with how much of the screen changed.",
            ["JEPA final_validation by_changed_pixel_bucket"],
        )

        # 35. JEPA timing breakdown.
        phases = [
            ("Manifest", "dataset_manifest_seconds"),
            ("Data load", "data_load_seconds"),
            ("Model load", "model_load_seconds"),
            ("Initial eval", "initial_evaluation_seconds"),
            ("Training", "training_compute_seconds"),
            ("Checkpoints", "checkpoint_seconds"),
            ("Persistence", "volume_persistence_seconds"),
            ("Final eval", "final_evaluation_seconds"),
        ]
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        bottoms = np.zeros(2)
        phase_colors = plt.cm.tab20(np.linspace(0, 0.75, len(phases)))
        for color, (label, key) in zip(phase_colors, phases, strict=True):
            vals = [minutes(jepa[m]["timing"][key]) for m in ["model3", "model4"]]
            ax.bar(["Model 3", "Model 4"], vals, bottom=bottoms, label=label, color=color)
            bottoms += vals
        for index, total in enumerate(bottoms):
            ax.text(index, total + 1.5, f"{total:.1f} min", ha="center")
        ax.set_ylabel("Minutes")
        ax.legend(frameon=False, ncol=4, fontsize=8)
        style_axis(ax)
        save(
            fig,
            "35-jepa-time-breakdown.png",
            "JEPA runtime by phase",
            "Training dominated the JEPA wall time. Final evaluation was the next largest phase.",
            ["JEPA final_metrics timing"],
        )

        # 36. JEPA resource efficiency.
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.8), constrained_layout=True)
        entries = [
            ("Elapsed (min)", [minutes(jepa[m]["elapsed_seconds"]) for m in ["model3", "model4"]], lambda v: f"{v:.1f}"),
            ("Cost (USD)", [jepa[m]["estimated_modal_cost_usd"]["total_usd"] for m in ["model3", "model4"]], lambda v: f"${v:.2f}"),
            ("Peak GPU (GiB)", [jepa[m]["peak_cuda_memory_gib"] for m in ["model3", "model4"]], lambda v: f"{v:.2f}"),
            ("Steps / sec", [jepa[m]["training_steps_per_second"] for m in ["model3", "model4"]], lambda v: f"{v:.2f}"),
        ]
        for ax, (label, vals, formatter) in zip(axes, entries, strict=True):
            bars = ax.bar(["Model 3", "Model 4"], vals, color=[COLORS["model3"], COLORS["model4"]])
            annotate_bars(ax, bars, formatter)
            ax.set_ylabel(label)
            style_axis(ax)
        save(
            fig,
            "36-jepa-resource-efficiency.png",
            "JEPA resource use",
            "The matched no-action and action-conditioned Qwen JEPA runs used similar resources.",
            ["JEPA final_metrics.json"],
        )

        # 37. Architecture parameter split.
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        parts = ["Online visual LoRA", "Target visual LoRA", "Predictor + action head"]
        m3_parts = [jepa["model3"]["trainable_online_lora_parameters"], jepa["model3"]["target_lora_parameters"], jepa["model3"]["trainable_head_parameters"]]
        m4_parts = [jepa["model4"]["trainable_online_lora_parameters"], jepa["model4"]["target_lora_parameters"], jepa["model4"]["trainable_head_parameters"]]
        x = np.arange(2)
        bottoms = np.zeros(2)
        for index, part in enumerate(parts):
            vals = np.array([m3_parts[index], m4_parts[index]]) / 1e6
            ax.bar(x, vals, bottom=bottoms, label=part, color=["#56B4E9", "#999999", "#009E73"][index])
            bottoms += vals
        ax.set_xticks(x, ["Model 3", "Model 4"])
        ax.set_ylabel("Parameters (millions)")
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "37-jepa-parameter-composition.png",
            "JEPA adapter and head parameter composition",
            "The two JEPA controls had matched parameter counts. The full Qwen base remained frozen.",
            ["JEPA final_metrics parameter fields"],
        )

        # 38. Pilot landscape.
        pilot_rows = []
        for path in (ROOT / "artifacts").rglob("final_metrics.json"):
            try:
                record = read_json(path)
            except (json.JSONDecodeError, OSError):
                continue
            final = record.get("final_validation")
            if not isinstance(final, dict) or "four_way_accuracy" not in final:
                continue
            if "smoke" in str(path).lower():
                continue
            path_text = str(path).lower()
            if "same-app" in path_text:
                family = "Same-app diagnostic"
            elif "full-data" in path_text:
                family = "V-JEPA full-data"
            elif "qwen-fusion" in path_text:
                family = "V-JEPA + Qwen fusion"
            elif "independent-tiled-wide" in path_text:
                family = "V-JEPA wide tiled"
            elif "independent-tiled" in path_text:
                family = "V-JEPA independent tiled"
            elif "scaled-separation" in path_text:
                family = "V-JEPA scaled"
            elif "tiled-separation" in path_text:
                family = "V-JEPA tiled"
            elif "vjepa2-gui-pure" in path_text:
                family = "V-JEPA pure"
            elif "model3-model3_full" in path_text:
                family = "Qwen no-action full"
            elif "model4-model4_full" in path_text:
                family = "Qwen action full"
            else:
                continue
            seed = "s12" if "20260812" in path_text else "s11" if "20260811" in path_text else "main"
            pilot_rows.append((family, seed, 100 * final["four_way_accuracy"], record.get("steps", 0)))
        pilot_rows.sort(key=lambda item: item[2])
        fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
        y = np.arange(len(pilot_rows))
        labels = [f"{family} - {seed}" for family, seed, _, _ in pilot_rows]
        colors = ["#999999" if "Same-app" in family else "#0072B2" for family, _, _, _ in pilot_rows]
        bars = ax.barh(y, [row[2] for row in pilot_rows], color=colors)
        ax.axvline(25, color="#D55E00", linestyle="--", label="Chance")
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlabel("Final four-way accuracy (%)")
        ax.legend(frameon=False)
        style_axis(ax, grid="x")
        for bar, row in zip(bars, pilot_rows, strict=True):
            ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2, f"{row[2]:.1f}%", va="center", fontsize=8)
        save(
            fig,
            "38-jepa-pilot-landscape.png",
            "JEPA architecture search results",
            "Grey same-app diagnostics used an easier test and are not directly comparable with held-out-app runs.",
            ["All saved JEPA final_metrics.json files"],
        )

        # 39. Seed reliability for repeated V-JEPA pilots.
        repeat_groups: dict[str, list[float]] = defaultdict(list)
        for family, seed, score, _ in pilot_rows:
            if seed in {"s11", "s12"} and family in {
                "V-JEPA full-data",
                "Same-app diagnostic",
                "V-JEPA independent tiled",
                "V-JEPA scaled",
            }:
                repeat_groups[family].append(score)
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        families = list(repeat_groups)
        for index, family in enumerate(families):
            values_ = repeat_groups[family]
            ax.plot([index] * len(values_), values_, "o", color="#0072B2", markersize=8)
            ax.plot([index - 0.15, index + 0.15], [np.mean(values_)] * 2, color="#D55E00", linewidth=3)
            if len(values_) > 1:
                ax.vlines(index, min(values_), max(values_), color="#777777", linewidth=1.5)
        ax.axhline(25, color="#777777", linestyle="--", label="Chance")
        ax.set_xticks(range(len(families)), [textwrap.fill(name, 18) for name in families])
        ax.set_ylabel("Final four-way accuracy (%)")
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "39-vjepa-seed-reliability.png",
            "Variation across repeated V-JEPA pilot seeds",
            "Dots are individual seeds. Orange lines are two-seed means. Two seeds are still too few for a stable estimate.",
            ["Repeated V-JEPA final_metrics.json files"],
        )

        # 40. Action-conditioned final delta against Model 2.
        comparison = read_json(ROOT / "artifacts" / "sft-comparison" / "model4-minus-model2-bootstrap.json")
        deltas = comparison["by_action_mean_delta"]
        action_labels = [action for action in ACTION_ORDER if action in deltas]
        delta_vals = [100 * deltas[action] for action in action_labels]
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        bars = ax.bar(action_labels, delta_vals, color=["#009E73" if value > 0 else "#D55E00" for value in delta_vals])
        ax.axhline(0, color="#555555")
        ax.set_ylabel("Model 4 minus Model 2 mean score (points)")
        ax.tick_params(axis="x", rotation=30)
        for bar, value in zip(bars, delta_vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + (1 if value >= 0 else -1), f"{value:+.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
        style_axis(ax)
        save(
            fig,
            "40-model4-minus-model2-by-action.png",
            "Where action-conditioned JEPA lost or gained against Model 2",
            "Model 4 lost most heavily on clicks. Tiny positive values on rare actions are not stable evidence.",
            ["Paired SFT bootstrap comparison"],
        )

        # 41. End-to-end core cost.
        jepa_cost = {"model2": 0.0, "model3": jepa["model3"]["estimated_modal_cost_usd"]["total_usd"], "model4": jepa["model4"]["estimated_modal_cost_usd"]["total_usd"]}
        sft_cost = {model: sft[model]["estimated_modal_cost_usd"] for model in models}
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        x = np.arange(3)
        jvals = [jepa_cost[m] for m in models]
        svals = [sft_cost[m] for m in models]
        ax.bar(x, jvals, color="#E69F00", label="JEPA stage")
        ax.bar(x, svals, bottom=jvals, color="#0072B2", label="SFT stage")
        for index, total in enumerate(np.array(jvals) + np.array(svals)):
            ax.text(index, total + 0.08, f"${total:.2f}", ha="center")
        ax.set_xticks(x, [SHORT_NAMES[m] for m in models])
        ax.set_ylabel("Estimated Modal cost (USD)")
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "41-core-training-cost.png",
            "Core training cost per model",
            "This is training cost, not inference cost. Inference latency and cost were not measured.",
            ["JEPA and SFT final_metrics cost estimates"],
        )

        # 42. Outcome flow.
        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 7)
        ax.axis("off")
        rows = [
            ("model2", "Base Qwen", "No JEPA stage", "SFT", "51.2% policy"),
            ("model3", "Base Qwen", "No-action JEPA\n24.5% future match", "SFT", "18.8% policy"),
            ("model4", "Base Qwen", "Action JEPA\n38.0% future match", "SFT", "18.4% policy"),
        ]
        ys = [5.6, 3.5, 1.4]
        for (model, left, middle, sft_label, outcome), y in zip(rows, ys, strict=True):
            items = [(0.6, left), (4.3, middle), (8.2, sft_label), (11.2, outcome)]
            for x0, label in items:
                color = COLORS[model] if x0 == 11.2 else "#EEEEEE"
                text_color = "black" if x0 != 11.2 or model == "model3" else "white"
                box = FancyBboxPatch((x0, y - 0.5), 2.2, 1.0, boxstyle="round,pad=0.12", facecolor=color, edgecolor="#777777", linewidth=1)
                ax.add_patch(box)
                ax.text(x0 + 1.1, y, label, ha="center", va="center", color=text_color, fontsize=9)
            for start, end in [(2.8, 4.3), (6.5, 8.2), (10.4, 11.2)]:
                ax.add_patch(FancyArrowPatch((start, y), (end, y), arrowstyle="->", mutation_scale=14, color="#777777"))
            ax.text(0.1, y, SHORT_NAMES[model], ha="right", va="center", color=COLORS[model], fontweight="bold")
        ax.text(5.4, 6.55, "Future-match accuracy and policy accuracy are different tests.", ha="center", fontsize=10)
        save(
            fig,
            "42-model-pipeline-outcome.png",
            "How each model reached its final policy score",
            "The higher Model 4 JEPA score did not transfer into a better computer-use policy.",
            ["JEPA and SFT final_metrics.json"],
        )

        # 43. Research evidence scorecard.
        fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
        ax.axis("off")
        rows = [
            ("Higher policy accuracy", "Tested", "Not supported", "Model 4: 18.4%; Model 2: 51.2%"),
            ("Action information helps", "Tested", "Not supported", "Model 4 and Model 3 were statistically tied"),
            ("Cheaper inference", "Not tested", "Unknown", "No latency, FLOP, or serving-cost benchmark"),
            ("Cheaper training data", "Not isolated", "Unknown", "AgentNet supplied action labels; annotation cost was not compared"),
            ("Matched experimental controls", "Tested", "Supported", "Triplet audit passed every control check"),
        ]
        columns = [0.02, 0.35, 0.55, 0.72]
        headers = ["Research claim", "Coverage", "Result", "Evidence"]
        for x, header in zip(columns, headers, strict=True):
            ax.text(x, 0.93, header, transform=ax.transAxes, fontweight="bold", va="top")
        for index, row in enumerate(rows):
            y = 0.78 - index * 0.16
            for x, value in zip(columns, row, strict=True):
                ax.text(x, y, textwrap.fill(value, 34 if x == 0.72 else 25), transform=ax.transAxes, va="top", fontsize=9)
            ax.plot([0.02, 0.98], [y - 0.07, y - 0.07], transform=ax.transAxes, color="#DDDDDD", linewidth=1)
        save(
            fig,
            "43-research-evidence-scorecard.png",
            "Which parts of the original thesis were tested?",
            "The MVP tested accuracy and action conditioning. It did not yet test inference cost or annotation cost.",
            ["Triplet audit", "SFT results", "Recorded experiment scope"],
        )

        # 44. Measured and allocated memory.
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        labels = ["JEPA Model 3", "JEPA Model 4", "SFT Model 2", "SFT Model 3", "SFT Model 4"]
        measured_gpu = [jepa["model3"]["peak_cuda_memory_gib"], jepa["model4"]["peak_cuda_memory_gib"]] + [sft[m]["peak_cuda_memory_gib"] for m in models]
        allocated_host = [jepa["model3"]["config"]["memory_gib"], jepa["model4"]["config"]["memory_gib"]] + [24, 24, 24]
        x = np.arange(len(labels))
        b1 = ax.bar(x - 0.18, measured_gpu, 0.36, color="#0072B2", label="Measured peak GPU memory")
        b2 = ax.bar(x + 0.18, allocated_host, 0.36, color="#999999", label="Allocated host RAM")
        annotate_bars(ax, b1, lambda v: f"{v:.1f}")
        annotate_bars(ax, b2, lambda v: f"{v:.0f}")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel("GiB")
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "44-memory-summary.png",
            "GPU memory and allocated host RAM",
            "GPU memory was measured. Host RAM is the Modal allocation, not measured resident usage.",
            ["JEPA/SFT metrics and Modal function settings"],
        )

        # 45. Synthetic JEPA action mix.
        train_profile = jepa_data["train_profile"]
        validation_profile = jepa_data["validation_profile"]
        jepa_actions = ["click", "type", "scroll"]
        train_action_total = sum(train_profile["action_counts"].values())
        validation_action_total = sum(validation_profile["action_counts"].values())
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        x = np.arange(len(jepa_actions))
        train_values = [
            100 * train_profile["action_counts"][action] / train_action_total
            for action in jepa_actions
        ]
        validation_values = [
            100 * validation_profile["action_counts"][action] / validation_action_total
            for action in jepa_actions
        ]
        train_bars = ax.bar(x - 0.2, train_values, 0.4, label="Train", color="#0072B2")
        validation_bars = ax.bar(
            x + 0.2, validation_values, 0.4, label="Validation", color="#E69F00"
        )
        annotate_bars(ax, train_bars, lambda value: f"{value:.1f}%")
        annotate_bars(ax, validation_bars, lambda value: f"{value:.1f}%")
        ax.set_xticks(x, [action.title() for action in jepa_actions])
        ax.set_ylabel("Share of transitions (%)")
        ax.set_ylim(0, 80)
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "45-synthetic-jepa-action-mix.png",
            "Action mix in the synthetic JEPA data",
            "Clicks dominate both splits. Typing and scrolling supply smaller action groups.",
            ["Model 4 dataset_audit.json"],
        )

        # 46. Application split.
        train_apps = train_profile["application_transition_counts"]
        validation_apps = validation_profile["application_transition_counts"]
        app_rows = list(train_apps.items()) + list(validation_apps.items())
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        labels = [name.replace("_mock", "").replace("_", " ").title() for name, _ in app_rows]
        values = [count for _, count in app_rows]
        colors = ["#0072B2"] * len(train_apps) + ["#E69F00"] * len(validation_apps)
        bars = ax.barh(np.arange(len(labels)), values, color=colors)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_xlabel("Transitions")
        style_axis(ax, grid="x")
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value + 45,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,}",
                va="center",
                fontsize=8,
            )
        ax.text(
            0.99,
            0.03,
            "Blue: train   Orange: held-out validation",
            transform=ax.transAxes,
            ha="right",
        )
        save(
            fig,
            "46-synthetic-jepa-application-split.png",
            "Applications used for JEPA training and validation",
            "Jira and Slack were held out from JEPA training. This tests transfer to unseen applications.",
            ["Model 4 dataset_audit.json"],
        )

        # 47. Amount of visual change after each action.
        bucket_order = ["0-1%", "1-5%", "5-20%", "20-50%", "50-100%"]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        x = np.arange(len(bucket_order))
        train_change = [
            100 * train_profile["changed_pixel_fraction"]["buckets"][bucket] / train_action_total
            for bucket in bucket_order
        ]
        validation_change = [
            100
            * validation_profile["changed_pixel_fraction"]["buckets"][bucket]
            / validation_action_total
            for bucket in bucket_order
        ]
        ax.plot(x, train_change, "o-", linewidth=2.5, label="Train", color="#0072B2")
        ax.plot(
            x,
            validation_change,
            "s-",
            linewidth=2.5,
            label="Validation",
            color="#E69F00",
        )
        for index, value in enumerate(train_change):
            ax.text(index, value + 0.8, f"{value:.1f}%", ha="center", fontsize=8)
        for index, value in enumerate(validation_change):
            ax.text(index, value - 1.2, f"{value:.1f}%", ha="center", va="top", fontsize=8)
        ax.set_xticks(x, bucket_order)
        ax.set_xlabel("Screen pixels that changed after the action")
        ax.set_ylabel("Share of transitions (%)")
        ax.set_ylim(0, max(train_change + validation_change) + 5)
        ax.legend(frameon=False)
        style_axis(ax)
        save(
            fig,
            "47-synthetic-jepa-screen-change.png",
            "How much each action changed the screen",
            "Validation actions changed more of the screen on average than training actions.",
            ["Model 4 dataset_audit.json"],
        )

        # 48. Synthetic data scale and archive size.
        train_archive_gib = jepa_data["train_tar_manifest"]["total_bytes"] / 2**30
        validation_archive_gib = jepa_data["validation_tar_manifest"]["total_bytes"] / 2**30
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        split_labels = ["Train", "Validation"]
        transitions = [jepa_data["train_transitions"], jepa_data["validation_transitions"]]
        bundles = [jepa_data["train_bundles"], jepa_data["validation_bundles"]]
        x = np.arange(2)
        bars1 = axes[0].bar(
            x - 0.18, transitions, 0.36, label="Transitions", color="#0072B2"
        )
        bars2 = axes[0].bar(
            x + 0.18, bundles, 0.36, label="Four-action bundles", color="#E69F00"
        )
        axes[0].set_xticks(x, split_labels)
        axes[0].set_ylabel("Count")
        axes[0].legend(frameon=False)
        style_axis(axes[0])
        for bars in (bars1, bars2):
            annotate_bars(axes[0], bars, lambda value: f"{value:,.0f}")
        archive_bars = axes[1].bar(
            split_labels,
            [train_archive_gib, validation_archive_gib],
            color=["#0072B2", "#E69F00"],
        )
        annotate_bars(axes[1], archive_bars, lambda value: f"{value:.3f} GiB")
        axes[1].set_ylabel("Staged archive size (GiB)")
        axes[1].set_ylim(0, train_archive_gib * 1.18)
        style_axis(axes[1])
        save(
            fig,
            "48-synthetic-jepa-data-scale.png",
            "Synthetic JEPA data used by the full runs",
            "Both full JEPA runs used 30,668 training transitions and 2,000 validation transitions.",
            ["Model 4 dataset_audit.json"],
        )

        # 49. JEPA split integrity.
        fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
        ax.axis("off")
        rows = [
            ("Dataset version", jepa_data["dataset_id"], "Same saved data for Models 3 and 4"),
            ("Bundle overlap", str(jepa_data["bundle_overlap"]), "No action bundle appears in both splits"),
            (
                "Exact screenshot overlap",
                str(jepa_data["exact_screenshot_overlap"]),
                "No exact screen image appears in both splits",
            ),
            ("Train applications", str(len(train_profile["applications"])), "Eight applications"),
            (
                "Held-out applications",
                str(len(validation_profile["applications"])),
                "Jira and Slack appear only in validation",
            ),
        ]
        columns = [0.03, 0.36, 0.53]
        for x0, header in zip(columns, ["Check", "Value", "Meaning"], strict=True):
            ax.text(x0, 0.92, header, transform=ax.transAxes, fontweight="bold", va="top")
        for index, row in enumerate(rows):
            y = 0.77 - index * 0.15
            for x0, value in zip(columns, row, strict=True):
                ax.text(x0, y, value, transform=ax.transAxes, va="top", fontsize=9)
            ax.plot(
                [0.03, 0.97],
                [y - 0.06, y - 0.06],
                transform=ax.transAxes,
                color="#DDDDDD",
                linewidth=1,
            )
        save(
            fig,
            "49-synthetic-jepa-split-integrity.png",
            "JEPA train and validation split checks",
            "The split prevents exact screen reuse and holds out two complete applications.",
            ["Model 4 dataset_audit.json"],
        )

    # Machine-readable summary used to verify the charts.
    summary_rows = []
    for model in models:
        summary_rows.append(
            {
                "model": SHORT_NAMES[model],
                "initial_exact_success": sft[model]["initial_validation"]["exact_success_rate"],
                "final_exact_success": sft[model]["final_validation"]["exact_success_rate"],
                "final_mean_action_score": sft[model]["final_validation"]["mean_action_score"],
                "sft_elapsed_seconds": sft[model]["elapsed_seconds"],
                "sft_cost_usd": sft[model]["estimated_modal_cost_usd"],
                "sft_peak_gpu_gib": sft[model]["peak_cuda_memory_gib"],
                "jepa_final_four_way_accuracy": jepa[model]["final_validation"]["four_way_accuracy"] if model in jepa else "",
                "jepa_cost_usd": jepa[model]["estimated_modal_cost_usd"]["total_usd"] if model in jepa else 0,
            }
        )
    with (OUT / "core-metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps({"chart_count": len(manifest), "charts": manifest}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chart_count": len(manifest), "output_dir": str(OUT), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
