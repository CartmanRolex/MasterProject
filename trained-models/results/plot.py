import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Prepare the data
data = {
    'Model': [
        'ACT (100k steps, BS 8)', 'ACT (100k steps, BS 8)', 
        'ACT (100k steps, BS 8)', 'ACT (100k steps, BS 8)',
        'SmolVLA (40k steps, BS 32)', 'SmolVLA (40k steps, BS 32)', 
        'SmolVLA (40k steps, BS 32)', 'SmolVLA (40k steps, BS 32)'
    ],
    'Oranges Picked': [
        '3 Oranges', '2 Oranges', '1 Orange', '0 Oranges',
        '3 Oranges', '2 Oranges', '1 Orange', '0 Oranges'
    ],
    'Percentage': [
        29.0, 26.0, 23.0, 22.0,  # ACT data
        28.0, 13.0, 37.0, 22.0   # SmolVLA data
    ]
}

df = pd.DataFrame(data)

# 2. Set up the plotting style
sns.set_theme(style="whitegrid")
palette = sns.color_palette("colorblind", 2)

# 3. Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# 4. Create the grouped barplot
sns.barplot(
    data=df, 
    x='Oranges Picked', 
    y='Percentage', 
    hue='Model', 
    palette=palette, 
    ax=ax
)

# 5. Add percentages on top of the bars for exact readings
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)

# 6. Formatting titles and labels
ax.set_ylim(0, 45)  # Set slightly higher than max value (37%) for breathing room
ax.set_ylabel('Percentage of Episodes (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Performance (Oranges Picked)', fontsize=12, fontweight='bold')
ax.set_title('ACT vs SmolVLA: Orange Picking Performance', fontsize=14, fontweight='bold', pad=20)

# 7. Add annotation for Mean Steps for Success
mean_steps_text = (
    "Mean Steps for Full Success\n"
    "(3 Oranges + Return to Start):\n"
    "• ACT: 834.1 steps\n"
    "• SmolVLA: 826.4 steps"
)
# Place a text box in the upper right (adjust x, y coords if it overlaps with legend)
ax.text(
    0.97, 0.95, mean_steps_text, 
    transform=ax.transAxes, 
    fontsize=11,
    verticalalignment='top', 
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='gray')
)

# 8. Adjust legend position so it doesn't overlap the text box
plt.legend(title='Model architecture', loc='upper left')

# 9. Final layout adjustments and show
plt.tight_layout()
plt.show()