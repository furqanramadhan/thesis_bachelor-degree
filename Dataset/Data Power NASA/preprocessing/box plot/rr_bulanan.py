import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('hujan-preprocessing.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create month names in order (Dec - Jan - Feb - ... - Nov)
month_order = ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov']
month_names = {12: 'Dec', 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 
               6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov'}

# Map month numbers to names
df['Month_Name'] = df['Month'].map(month_names)

# Prepare data for boxplot
data_by_month = [df[df['Month'] == month]['RR_imputed'].values 
                 for month in [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 7))

# Create boxplot
bp = ax.boxplot(data_by_month, 
                labels=month_order,
                patch_artist=True,
                notch=False,
                showmeans=False,
                showfliers=True,
                widths=0.6)

# Customize colors - gradient from purple to yellow like the example
colors = ['#9B8BC4', '#A695C7', '#B19FCA', '#BCA9CD', '#C7B3D0', 
          '#88CCD3', '#93D4D6', '#9EDCD9', '#A9E4DC', '#B4ECDF',
          '#D4E8B4', '#DFE8A8']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(1)

# Customize median lines
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)

# Customize whiskers and caps
for whisker in bp['whiskers']:
    whisker.set_color('black')
    whisker.set_linewidth(1)
    
for cap in bp['caps']:
    cap.set_color('black')
    cap.set_linewidth(1)

# Customize fliers (outliers)
for flier in bp['fliers']:
    flier.set_marker('o')
    flier.set_markerfacecolor('white')
    flier.set_markeredgecolor('black')
    flier.set_markersize(5)
    flier.set_markeredgewidth(0.8)

# Customize plot
ax.set_xlabel('Month', fontsize=11, fontweight='normal')
ax.set_ylabel('Rainfall (mm)', fontsize=11, fontweight='normal')
ax.grid(True, alpha=0.3, linestyle='--', axis='y', color='gray')
ax.set_axisbelow(True)

# Set background color
fig.patch.set_facecolor('white')

# Customize ticks
plt.xticks(rotation=0, fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.savefig('monthly_rainfall_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== Monthly Rainfall Statistics (mm) ===\n")
for i, month in enumerate(month_order):
    month_num = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][i]
    data = df[df['Month'] == month_num]['RR_imputed']
    print(f"{month:>3}: Mean={data.mean():6.2f}, Median={data.median():6.2f}, "
          f"Min={data.min():6.2f}, Max={data.max():6.2f}, Std={data.std():6.2f}")