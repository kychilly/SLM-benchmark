from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean scientific plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10})

# Data definition
data = [
    {
        "model": "LLM (Gemini 3.6)",
        "label": "Gemini 3.6",
        "acc": 80,
        "latency": 62,
        "tokens": 1280,
        "marker": "o",
        "color": "#1f77b4",
        "size": 140,
        "dominated": False,
    },
    {
        "model": "SLM (Phi-4-mini)",
        "label": "Phi-4-mini",
        "acc": 56,
        "latency": 29,
        "tokens": 296,
        "marker": "^",
        "color": "#ff7f0e",
        "size": 140,
        "dominated": False,
    },
    {
        "model": "SLM (Llama-3.2-3B - Dominated)",
        "label": "Llama-3.2-3B",
        "acc": 48,
        "latency": 31,
        "tokens": 363,
        "marker": "x",
        "color": "#d62728",
        "size": 160,
        "dominated": True,
    },
]

# Create figure with 2 subplots (widened aspect ratio to flatten the visual slope)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

# ----------------------------------------------------
# Subplot A: Accuracy vs. Latency
# ----------------------------------------------------

# Plot Pareto Frontier
frontier_latency = [data[1]["latency"], data[0]["latency"]]  # Phi4 -> Gemini
frontier_acc_lat = [data[1]["acc"], data[0]["acc"]]
ax1.plot(
    frontier_latency,
    frontier_acc_lat,
    color="#2ca02c",
    linestyle="--",
    linewidth=2,
    label="Pareto Frontier",
    zorder=1,
)

# Scatter points & Annotations
for pt in data:
    ax1.scatter(
        pt["latency"],
        pt["acc"],
        marker=pt["marker"],
        color=pt["color"],
        s=pt["size"],
        label=pt["model"],
        zorder=3,
    )
    annot_text = f"{pt['label']}\n({pt['acc']}%, {pt['latency']}s)"

    # Adjust offsets for clear placement from origin
    y_offset = -6 if pt["label"] == "Llama-3.2-3B" else 3
    ax1.annotate(
        annot_text,
        (pt["latency"], pt["acc"]),
        xytext=(pt["latency"] + 1.5, pt["acc"] + y_offset),
        fontweight="bold" if not pt["dominated"] else "normal",
        fontsize=9,
        color="#222222" if not pt["dominated"] else "#666666",
    )

ax1.set_title("A) Accuracy vs. Latency Pareto Frontier", fontweight="bold", pad=12)
ax1.set_xlabel("Latency (Seconds)", fontweight="bold")
ax1.set_ylabel("Accuracy (%)", fontweight="bold")

# Force 0 to Max scale
ax1.set_xlim(0, 70)
ax1.set_ylim(0, 100)

# ----------------------------------------------------
# Subplot B: Accuracy vs. Computational Cost
# ----------------------------------------------------

# Plot Pareto Frontier
frontier_tokens = [data[1]["tokens"], data[0]["tokens"]]  # Phi4 -> Gemini
frontier_acc_tok = [data[1]["acc"], data[0]["acc"]]
ax2.plot(
    frontier_tokens,
    frontier_acc_tok,
    color="#2ca02c",
    linestyle="--",
    linewidth=2,
    label="Pareto Frontier",
    zorder=1,
)

# Scatter points & Annotations
for pt in data:
    ax2.scatter(
        pt["tokens"],
        pt["acc"],
        marker=pt["marker"],
        color=pt["color"],
        s=pt["size"],
        label=pt["model"],
        zorder=3,
    )
    annot_text = f"{pt['label']}\n({pt['acc']}%, {pt['tokens']} tokens)"

    y_offset = -6 if pt["label"] == "Llama-3.2-3B" else 3
    ax2.annotate(
        annot_text,
        (pt["tokens"], pt["acc"]),
        xytext=(pt["tokens"] + 25, pt["acc"] + y_offset),
        fontweight="bold" if not pt["dominated"] else "normal",
        fontsize=9,
        color="#222222" if not pt["dominated"] else "#666666",
    )

ax2.set_title("B) Accuracy vs. Computational Cost Pareto Frontier", fontweight="bold", pad=12)
ax2.set_xlabel("Token Count (Cost Proxy)", fontweight="bold")

# Force 0 to Max scale
ax2.set_xlim(0, 1400)
ax2.set_ylim(0, 100)

# Apply unified legend
ax1.legend(loc="upper left", frameon=True)
ax2.legend(loc="upper left", frameon=True)

plt.tight_layout()

# Save plot to file
output_path = Path("pareto_frontier_zero_based.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Graph successfully saved to {output_path.resolve()}")

# Display plot window
plt.show()