import matplotlib.pyplot as plt
import numpy as np

# Apply clean Seaborn theme
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# High-density font rendering rules
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['text.antialiased'] = True

# 12x3.8 aspect ratio keeps text-to-bar proportions balanced across 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.8), dpi=300)

exams = ['AMC 8', 'AMC 12']
x = np.arange(len(exams))
width = 0.35

c_phi = '#ff7f0e'
c_llama = '#d62728'

# --- PANEL 3.A: Accuracy ---
rects1 = ax1.bar(x - width/2, [56, 32], width, label='Phi-4-Mini', color=c_phi)
rects2 = ax1.bar(x + width/2, [48, 32], width, label='Llama3.2-3B', color=c_llama)
ax1.set_title('3.A) Accuracy Comparison', fontsize=11, fontweight='bold', pad=10)
ax1.set_ylabel('Accuracy (%)', fontsize=10, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(exams, fontweight='bold', fontsize=10)
ax1.set_ylim(0, 75)
ax1.legend(loc='upper right', frameon=True, fontsize=9)
ax1.bar_label(rects1, padding=3, fmt='%d%%', weight='bold', fontsize=9)
ax1.bar_label(rects2, padding=3, fmt='%d%%', weight='bold', fontsize=9)

# --- PANEL 3.B: Latency ---
rects3 = ax2.bar(x - width/2, [29, 88], width, label='Phi-4-Mini', color=c_phi)
rects4 = ax2.bar(x + width/2, [31, 137], width, label='Llama3.2-3B', color=c_llama)
ax2.set_title('3.B) Latency Comparison', fontsize=11, fontweight='bold', pad=10)
ax2.set_ylabel('Latency (Seconds)', fontsize=10, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(exams, fontweight='bold', fontsize=10)
ax2.set_ylim(0, 165)
ax2.legend(loc='upper left', frameon=True, fontsize=9)
ax2.bar_label(rects3, padding=3, fmt='%ds', weight='bold', fontsize=9)
ax2.bar_label(rects4, padding=3, fmt='%ds', weight='bold', fontsize=9)

# --- PANEL 3.C: Token Cost ---
rects5 = ax3.bar(x - width/2, [296, 622], width, label='Phi-4-Mini', color=c_phi)
rects6 = ax3.bar(x + width/2, [363, 866], width, label='Llama3.2-3B', color=c_llama)
ax3.set_title('3.C) Computational Cost', fontsize=11, fontweight='bold', pad=10)
ax3.set_ylabel('Token Count (Cost Proxy)', fontsize=10, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(exams, fontweight='bold', fontsize=10)
ax3.set_ylim(0, 1050)
ax3.legend(loc='upper left', frameon=True, fontsize=9)
ax3.bar_label(rects5, padding=3, fmt='%d', weight='bold', fontsize=9)
ax3.bar_label(rects6, padding=3, fmt='%d', weight='bold', fontsize=9)

plt.tight_layout()

# 1. VECTOR EXPORT (Infinite resolution - recommended for Overleaf/LaTeX papers)
plt.savefig('Figure_3.pdf', format='pdf', bbox_inches='tight')

# 2. HIGH-RES RASTER EXPORT (600 DPI)
plt.savefig('Figure_3.png', format='png', dpi=600, bbox_inches='tight')

print("Exported Figure_3.pdf (Vector) and Figure_3.png (600 DPI PNG).")
plt.show()