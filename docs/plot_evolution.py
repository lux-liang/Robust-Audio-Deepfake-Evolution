#!/usr/bin/env python3
"""
Performance Evolution Visualization Script
Plots EER (%) across different phases of the project.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style (whitegrid-like without seaborn)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300

# Data
phases = ["Phase 3\n(MoE)", "Phase 4\n(Dual)", "Phase 5\n(BiMamba)", "Phase 6\n(Robust)"]
x_positions = np.arange(len(phases))

# Line 1: Clean Data (ASVspoof 19LA) - Blue
clean_data_eer = [23.0, 7.7, 4.49, 4.42]

# Line 2: Real-world/Compressed (MP3/AAC) - Red
compressed_data_eer = [45.0, 30.0, 20.0, 4.03]

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 7))

# Plot lines with markers
line1 = ax.plot(x_positions, clean_data_eer, 
                marker='o', markersize=10, linewidth=2.5, 
                color='#2E86AB', label='Clean Data (ASVspoof 19LA)', 
                markerfacecolor='#2E86AB', markeredgecolor='white', 
                markeredgewidth=2, zorder=3)

line2 = ax.plot(x_positions, compressed_data_eer, 
                marker='s', markersize=10, linewidth=2.5, 
                color='#A23B72', label='Real-world/Compressed (MP3/AAC)', 
                markerfacecolor='#A23B72', markeredgecolor='white', 
                markeredgewidth=2, zorder=3)

# Add value labels on points
for i, (clean_val, comp_val) in enumerate(zip(clean_data_eer, compressed_data_eer)):
    # Clean data labels (above)
    ax.text(i, clean_val + 1.5, f'{clean_val:.2f}%', 
            ha='center', va='bottom', fontsize=9, 
            color='#2E86AB', weight='bold')
    # Compressed data labels (below)
    ax.text(i, comp_val - 2.0, f'{comp_val:.2f}%', 
            ha='center', va='top', fontsize=9, 
            color='#A23B72', weight='bold')

# Add annotation for Phase 6 improvement
ax.annotate('Codec Augmentation + FGM', 
            xy=(3, compressed_data_eer[3]), 
            xytext=(2.3, 15),
            arrowprops=dict(arrowstyle='->', lw=2, color='#A23B72', 
                          connectionstyle='arc3,rad=0.2'),
            fontsize=11, weight='bold', color='#A23B72',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='#A23B72', linewidth=2, alpha=0.9))

# Customize axes
ax.set_xlabel('Project Phase', fontweight='bold', fontsize=13)
ax.set_ylabel('EER (%)', fontweight='bold', fontsize=13)
ax.set_title('Performance Evolution Across Phases\n(Lower EER is Better)', 
             fontweight='bold', fontsize=15, pad=20)
ax.set_xticks(x_positions)
ax.set_xticklabels(phases)
ax.set_ylim(0, 50)
ax.set_yticks(np.arange(0, 51, 5))
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, which='both')
ax.set_axisbelow(True)
ax.set_facecolor('#FAFAFA')  # Light gray background like whitegrid

# Add legend
legend = ax.legend(loc='upper right', frameon=True, 
                   fancybox=True, shadow=True, 
                   framealpha=0.95, edgecolor='gray')
legend.get_frame().set_linewidth(1)
legend.get_frame().set_facecolor('white')

# Add horizontal reference line at 5% (good performance threshold)
ax.axhline(y=5.0, color='green', linestyle=':', linewidth=2, 
           alpha=0.6, label='5% EER Threshold (Reference)', zorder=1)
ax.text(len(phases)-0.5, 5.5, '5% Threshold', 
        ha='right', va='bottom', fontsize=9, 
        color='green', style='italic', alpha=0.7)

# Highlight Phase 6 improvement area
ax.fill_between([2.5, 3.5], [0, 0], [50, 50], 
                alpha=0.1, color='green', zorder=0)
ax.text(3, 48, 'Phase 6\nBreakthrough', 
        ha='center', va='top', fontsize=10, 
        weight='bold', color='green', alpha=0.8)

# Tight layout
plt.tight_layout()

# Save figure
output_path = Path(__file__).parent / 'evolution_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"✓ Chart saved to: {output_path}")

# Also save as PDF for publication quality
output_path_pdf = Path(__file__).parent / 'evolution_chart.pdf'
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print(f"✓ Chart saved to: {output_path_pdf}")

plt.close()
print("\n✓ Visualization complete!")

