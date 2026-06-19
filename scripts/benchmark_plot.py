"""Generate glasstrace benchmark graphic across 4 models."""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

models = ["Qwen2.5\n0.5B", "Qwen2.5\n1.5B", "Qwen2.5\n3B", "SmolLM2\n1.7B"]
colors = ["#4C9BE8", "#4C9BE8", "#4C9BE8", "#E8824C"]

ms_per_token = [17.4, 26.6, 43.9, 23.0]
kv_growth = [0.21, 0.49, 0.63, 3.38]
lm_head_pct = [11.9, 10.7, 5.9, 3.7]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor("#0D1117")

for ax in axes:
    ax.set_facecolor("#161B22")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")


def bar_chart(ax, values, title, ylabel, fmt=".1f"):
    bars = ax.bar(models, values, color=colors, width=0.5, zorder=3)
    ax.set_title(title, color="white", fontsize=12, pad=12, fontweight="bold")
    ax.set_ylabel(ylabel, color="#8B949E", fontsize=10)
    ax.tick_params(axis="x", colors="white", labelsize=9)
    ax.tick_params(axis="y", colors="#8B949E", labelsize=9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:{fmt}}")
    )
    ax.grid(axis="y", color="#30363D", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:{fmt}}",
            ha="center",
            va="bottom",
            color="white",
            fontsize=9,
            fontweight="bold",
        )


bar_chart(axes[0], ms_per_token, "Decode Speed", "ms / token")
bar_chart(axes[1], kv_growth, "KV-Cache Growth", "MB (20 tokens)")
bar_chart(axes[2], lm_head_pct, "lm_head Share of Decode", "% of decode time")

blue = mpatches.Patch(color="#4C9BE8", label="Qwen 2.5 family")
orange = mpatches.Patch(color="#E8824C", label="SmolLM2 1.7B")
fig.legend(
    handles=[blue, orange],
    loc="lower center",
    ncol=2,
    frameon=False,
    labelcolor="white",
    fontsize=10,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "glasstrace benchmark — 4 models on T4 GPU (fp16, 20 decode tokens)",
    color="white",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(
    "figures/benchmark_graphic.png",
    dpi=180,
    bbox_inches="tight",
    pad_inches=0.3,
    facecolor="#0D1117",
)
plt.show()
print("Saved to figures/benchmark_graphic.png")
