import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set academic design style
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    }
)

# --- Data Setup (SLMs Only - AMC 8 First, AMC 12 Second) ---
exams = ["AMC 8", "AMC 12"]
x = np.arange(len(exams))
width = 0.35

# Model Specs (Reordered: [AMC 8, AMC 12])
# Phi-4-Mini: 56% (AMC8), 32% (AMC12) | 29s (AMC8), 88s (AMC12) | 296 (AMC8), 622 (AMC12)
# Llama3.2-3B: 48% (AMC8), 32% (AMC12) | 31s (AMC8), 137s (AMC12) | 363 (AMC8), 866 (AMC12)

phi_acc = [56, 32]
llama_acc = [48, 32]

phi_lat = [29, 88]
llama_lat = [31, 137]

phi_tok = [296, 622]
llama_tok = [363, 866]

# Custom Color Palette
c_phi = "#ff7f0e"  # Orange
c_llama = "#d62728"  # Red

# Create 1x3 Subplot Layout
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ==============================================================================
# PANEL 1: Accuracy Comparison (%)
# ==============================================================================
ax1 = axes[0]
rects1 = ax1.bar(
    x - width / 2, phi_acc, width, label="Phi-4-Mini", color=c_phi
)
rects2 = ax1.bar(
    x + width / 2, llama_acc, width, label="Llama3.2-3B", color=c_llama
)

ax1.set_ylabel("Accuracy (%)")
ax1.set_title("3.A) Accuracy Comparison")
ax1.set_xticks(x)
ax1.set_xticklabels(exams)
ax1.set_ylim(0, 70)
ax1.legend(loc="upper right")
ax1.bar_label(rects1, padding=3, fmt="%d%%", weight="bold")
ax1.bar_label(rects2, padding=3, fmt="%d%%", weight="bold")

# ==============================================================================
# PANEL 2: Latency Comparison (Seconds)
# ==============================================================================
ax2 = axes[1]
rects3 = ax2.bar(
    x - width / 2, phi_lat, width, label="Phi-4-Mini", color=c_phi
)
rects4 = ax2.bar(
    x + width / 2, llama_lat, width, label="Llama3.2-3B", color=c_llama
)

ax2.set_ylabel("Latency (Seconds)")
ax2.set_title("3.B) Latency Comparison")
ax2.set_xticks(x)
ax2.set_xticklabels(exams)
ax2.set_ylim(0, 160)
ax2.legend(loc="upper left")
ax2.bar_label(rects3, padding=3, fmt="%ds", weight="bold")
ax2.bar_label(rects4, padding=3, fmt="%ds", weight="bold")

# ==============================================================================
# PANEL 3: Token Cost Comparison
# ==============================================================================
ax3 = axes[2]
rects5 = ax3.bar(
    x - width / 2, phi_tok, width, label="Phi-4-Mini", color=c_phi
)
rects6 = ax3.bar(
    x + width / 2, llama_tok, width, label="Llama3.2-3B", color=c_llama
)

ax3.set_ylabel("Token Count (Cost Proxy)")
ax3.set_title("3.C) Computational Cost")
ax3.set_xticks(x)
ax3.set_xticklabels(exams)
ax3.set_ylim(0, 1000)
ax3.legend(loc="upper left")
ax3.bar_label(rects5, padding=3, fmt="%d", weight="bold")
ax3.bar_label(rects6, padding=3, fmt="%d", weight="bold")

# Overall Layout Adjustments
plt.suptitle(
    "Head-to-Head SLM Benchmark Comparison: Phi-4-Mini vs. Llama3.2-3B",
    fontsize=14,
    weight="bold",
    y=1.02,
)
plt.tight_layout()

# Save plot to file
plt.savefig("SLMGraphAnalysis.png", dpi=300, bbox_inches="tight")
print("Graph successfully saved to SLMGraphAnalysis.png")

# Display
plt.show()