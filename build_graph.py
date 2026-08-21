import matplotlib.pyplot as plt

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# Extended figure width (16x5) flattens the visual slope of the Pareto frontier
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

# Data from Table
models = ['Gemini 3.6 (LLM)', 'Phi-4-Mini (SLM)', 'Llama3.2-3B (SLM)']
accuracy = [72, 32, 32]
latency = [145, 88, 137]
tokens = [1240, 622, 866]

# --- PLOT 1: Accuracy vs. Latency ---
# Scatter Points
ax1.scatter(latency[0], accuracy[0], color='#1f77b4', s=140, label='LLM (Gemini 3.6)', zorder=5)
ax1.scatter(latency[1], accuracy[1], color='#ff7f0e', s=140, marker='^', label='SLM (Phi-4-Mini)', zorder=5)
ax1.scatter(latency[2], accuracy[2], color='#d62728', s=140, marker='x', label='SLM (Llama3.2-3B - Dominated)', zorder=5)

# Draw Pareto Frontier Line (Phi-4-mini -> Gemini 3.6)
ax1.plot([88, 145], [32, 72], color='#2ca02c', linestyle='--', linewidth=2, label='Pareto Frontier', zorder=3)

# Point Annotations
ax1.annotate('Gemini 3.6\n(72%, 145s)', (145, 72), textcoords="offset points", xytext=(-10, 10), ha='right', weight='bold')
ax1.annotate('Phi-4-Mini\n(32%, 88s)', (88, 32), textcoords="offset points", xytext=(10, -5), weight='bold')
ax1.annotate('Llama3.2-3B\n(32%, 137s)', (137, 32), textcoords="offset points", xytext=(10, -18), color='#555555')

# Axes Formatting (Anchored at 0 to Max)
ax1.set_title('1.A) Accuracy vs. Latency Pareto Frontier', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Latency (Seconds)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.set_xlim(0, 160)
ax1.legend(loc='upper left', frameon=True)

# --- PLOT 2: Accuracy vs. Token Cost ---
# Scatter Points
ax2.scatter(tokens[0], accuracy[0], color='#1f77b4', s=140, label='LLM (Gemini 3.6)', zorder=5)
ax2.scatter(tokens[1], accuracy[1], color='#ff7f0e', s=140, marker='^', label='SLM (Phi-4-Mini)', zorder=5)
ax2.scatter(tokens[2], accuracy[2], color='#d62728', s=140, marker='x', label='SLM (Llama3.2-3B - Dominated)', zorder=5)

# Draw Pareto Frontier Line
ax2.plot([622, 1240], [32, 72], color='#2ca02c', linestyle='--', linewidth=2, label='Pareto Frontier', zorder=3)

# Point Annotations
ax2.annotate('Gemini 3.6\n(72%, 1240 tokens)', (1240, 72), textcoords="offset points", xytext=(-10, 10), ha='right', weight='bold')
ax2.annotate('Phi-4-Mini\n(32%, 622 tokens)', (622, 32), textcoords="offset points", xytext=(10, -5), weight='bold')
ax2.annotate('Llama3.2:3B\n(32%, 866 tokens)', (866, 32), textcoords="offset points", xytext=(10, -18), color='#555555')

# Axes Formatting (Anchored at 0 to Max)
ax2.set_title('1.B) Accuracy vs. Computational Cost Pareto Frontier', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Token Count (Cost Proxy)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.set_xlim(0, 1400)
ax2.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('pareto_frontier_analysis.png', dpi=600)
print("AMC12 results printed in pareto_frontier_analysis.png")
plt.show()