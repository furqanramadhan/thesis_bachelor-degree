import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('kelembaban-box-plot.csv')

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Extract year from date
df['Year'] = df['Date'].dt.year

# Get unique years and prepare data for boxplot
years = sorted(df['Year'].unique())
data_by_year = [df[df['Year'] == year]['RH_AVG_preprocessed'].values for year in years]

# Create figure and axis
fig, ax = plt.subplots(figsize=(16, 8))

# Create boxplot
bp = ax.boxplot(data_by_year, 
                labels=years,
                patch_artist=True,
                notch=False,
                showmeans=False,
                showfliers=True,
                widths=0.6)

# Create gradient colors for all years
n_years = len(years)
colors = plt.cm.viridis(np.linspace(0, 1, n_years))
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
ax.set_xlabel('Year', fontsize=12, fontweight='bold')
ax.set_ylabel('Daily Relative Humidity (%)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', axis='y', color='gray')
ax.set_axisbelow(True)

# Set background color
fig.patch.set_facecolor('white')

# Customize ticks
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.savefig('annual_daily_humidity_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n=== Annual Daily Relative Humidity Statistics (%) ===\n")
for year in years:
    data = df[df['Year'] == year]['RH_AVG_preprocessed']
    status = "(Partial)" if year == 2025 else "(Complete)"
    print(f"{year}: Mean={data.mean():6.2f}, Median={data.median():6.2f}, "
          f"Min={data.min():6.2f}, Max={data.max():6.2f}, Std={data.std():6.2f} {status}")