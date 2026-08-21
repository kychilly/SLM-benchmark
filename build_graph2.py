import matplotlib.pyplot as plt

# Match Figure 1 exact style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Exact canvas dimensions matching Figure 1 (10x4, dpi=300)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

# AMC 8 Benchmark Data
models = ['Gemini 3.6 (LLM)', 'Phi-4-Mini (SLM)', 'Llama3.2-3B (SLM)']
accuracy = [80, 56, 48]
latency = [62, 29, 31]
tokens = [1280, 296, 363]

# --- PLOT 2.A: Accuracy vs. Latency ---
ax1.scatter(latency[0], accuracy[0], color='#1f77b4', s=140, label='LLM (Gemini 3.6)', zorder=5)
ax1.scatter(latency[1], accuracy[1], color='#ff7f0e', s=140, marker='^', label='SLM (Phi-4-Mini)', zorder=5)
ax1.scatter(latency[2], accuracy[2], color='#d62728', s=140, marker='x', label='SLM (Llama3.2-3B - Dominated)', zorder=5)

# Pareto Frontier Line
ax1.plot([29, 62], [56, 80], color='#2ca02c', linestyle='--', linewidth=2, label='Pareto Frontier', zorder=3)

# Point Annotations
ax1.annotate('Gemini 3.6\n(80%, 62s)', (62, 80), textcoords="offset points", xytext=(-10, -10), ha='right', weight='bold')
ax1.annotate('Phi-4-Mini\n(56%, 29s)', (29, 56), textcoords="offset points", xytext=(10, -5), weight='bold')
ax1.annotate('Llama3.2-3B\n(48%, 31s)', (31, 48), textcoords="offset points", xytext=(10, -18), color='#555555')

ax1.set_title('2.A) Accuracy vs. Latency Pareto Frontier', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Latency (Seconds)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_xlim(0, 75)
ax1.legend(loc='upper left', frameon=True)

# --- PLOT 2.B: Accuracy vs. Token Cost ---
ax2.scatter(tokens[0], accuracy[0], color='#1f77b4', s=140, label='LLM (Gemini 3.6)', zorder=5)
ax2.scatter(tokens[1], accuracy[1], color='#ff7f0e', s=140, marker='^', label='SLM (Phi-4-Mini)', zorder=5)
ax2.scatter(tokens[2], accuracy[2], color='#d62728', s=140, marker='x', label='SLM (Llama3.2-3B - Dominated)', zorder=5)

# Pareto Frontier Line
ax2.plot([296, 1280], [56, 80], color='#2ca02c', linestyle='--', linewidth=2, label='Pareto Frontier', zorder=3)

# Point Annotations
ax2.annotate('Gemini 3.6\n(80%, 1280 tokens)', (1280, 80), textcoords="offset points", xytext=(-10, -10), ha='right', weight='bold')
ax2.annotate('Phi-4-Mini\n(56%, 296 tokens)', (296, 56), textcoords="offset points", xytext=(10, -5), weight='bold')
ax2.annotate('Llama3.2-3B\n(48%, 363 tokens)', (363, 48), textcoords="offset points", xytext=(10, -18), color='#555555')

ax2.set_title('2.B) Accuracy vs. Computational Cost Pareto Frontier', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Token Count (Cost Proxy)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.set_xlim(0, 1400)
ax2.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('Figure 2.png', dpi=600)
print("Figure 2 exported cleanly as Figure 2.png")
plt.show()