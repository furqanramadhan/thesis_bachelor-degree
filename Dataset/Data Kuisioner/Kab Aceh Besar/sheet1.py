import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
import numpy as np

COLOR_PALETTE = {
    'blue': '#1E88E5',
    'orange': '#FF6F00',
    'green': '#388E3C',
    'red': '#D32F2F',
    'purple': '#7B1FA2',
    'cyan': '#00ACC1',
    'yellow': '#FBC02D',
    'pink': '#C2185B',
    'teal': '#00897B',
    'indigo': '#3949AB',
}

# Load data
df = pd.read_csv("data kuisioner petani aceh besar_sheet1.csv")

# Debug: Check available columns
print("Available columns:")
print(df.columns.tolist())
print("\nSample data:")
print(df.head())

# Fungsi untuk mapping kabupaten berdasarkan ID petani
def get_kabupaten(id_petani):
    if pd.isna(id_petani):
        return np.nan
    
    id_str = str(id_petani).strip()
    
    if id_str.startswith('PTN'):
        try:
            id_num = int(id_str[3:])
        except (ValueError, IndexError):
            return 'Tidak Diketahui'
    else:
        try:
            id_num = int(id_petani)
        except:
            return 'Tidak Diketahui'
    
    if 1 <= id_num <= 50:
        return 'Aceh Besar'
    elif 51 <= id_num <= 92:
        return 'Aceh Jaya'
    elif 93 <= id_num <= 136:
        return 'Pidie'
    elif 137 <= id_num <= 156:
        return 'Aceh Utara'
    elif 157 <= id_num <= 196:
        return 'Bireuen'
    else:
        return 'Tidak Diketahui'

# Adjust column names based on CSV
id_col = 'id_petani'
kecamatan_col = 'kecamatan'

# Add kabupaten column
df['kabupaten'] = df[id_col].apply(get_kabupaten)

# Filter for Aceh Besar respondents only
df_aceh_besar = df[df['kabupaten'] == 'Aceh Besar'].copy()

print(f"\nTotal respondents from Aceh Besar: {len(df_aceh_besar)}")

# Count respondents by kecamatan
kecamatan_counts = df_aceh_besar[kecamatan_col].value_counts().sort_values(ascending=False)

print("\nRespondents by Kecamatan:")
print(kecamatan_counts)

# Create color list - different color for each kecamatan
colors = [COLOR_PALETTE['blue'], COLOR_PALETTE['orange'], COLOR_PALETTE['green']]
# If there are more than 3 kecamatan, extend colors
if len(kecamatan_counts) > 3:
    additional_colors = [COLOR_PALETTE['red'], COLOR_PALETTE['purple'], 
                        COLOR_PALETTE['cyan'], COLOR_PALETTE['yellow'],
                        COLOR_PALETTE['pink'], COLOR_PALETTE['teal'], 
                        COLOR_PALETTE['indigo']]
    colors.extend(additional_colors[:len(kecamatan_counts)-3])

# Create vertical bar plot
plt.figure(figsize=(10, 7))
bars = plt.bar(range(len(kecamatan_counts)), kecamatan_counts.values, 
               color=colors[:len(kecamatan_counts)], width=0.6, edgecolor='white', linewidth=1.5)

# Add value labels on top of bars
for i, (bar, value) in enumerate(zip(bars, kecamatan_counts.values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{int(value)}',
             ha='center', va='bottom',
             bbox=dict(
                 boxstyle="round,pad=0.4",
                 fc='white',
                 ec='gray',
                 lw=1,
                 alpha=0.95
             ),
             fontsize=11,
             fontweight='bold')

# Dynamic y-axis scaling
y_max = kecamatan_counts.max()
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 4
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

# Set x-axis labels
plt.xticks(range(len(kecamatan_counts)), kecamatan_counts.index, rotation=45, ha='right')

# Styling
plt.ylabel('Jumlah', fontsize=12)
plt.xlabel('Kecamatan', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig('01_responden_aceh_besar_per_kecamatan.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as: 01_responden_aceh_besar_per_kecamatan.png")

# Visualisasi 2: Varietas Padi per Kecamatan (Grouped Bar)
print("\nCreating grouped bar chart for varieties per kecamatan...")

# Dictionary varietas padi dengan hari panen
varietas_hari = {
    'Inpari': 111,
    'Tinggong': 75,
    'Inpari 32': 120,
    'Padi andin': 105,
    'Ciherang': 125,
    'Cibatu': 95,
    'CBD': 125,
    'Ngawos': 95,
    'Beulerang': 150,
    'Brangus': 150,
    'CBD 04': 100,
    'CBD Murni': 90,
    'Cibatu 05': 95,
    'Ciherang beruang': 87,
    'Inpari 42': 120,
    'Mekongga': 125,
    'Mustajab': 120,
    'Padi bojeng': 90,
    'Ramos': 110
}

# Fungsi untuk memisahkan varietas padi yang multiple
def pisahkan_varietas(varietas_text):
    if pd.isna(varietas_text):
        return ['Tidak Diketahui']
    
    # Convert to string dan clean
    varietas_str = str(varietas_text).strip()
    
    # Pisahkan berdasarkan berbagai separator
    separators = [' dan ', ',', ' & ', ';', ' + ', '/']
    varietas_list = [varietas_str]
    
    for separator in separators:
        new_list = []
        for item in varietas_list:
            if separator in item:
                new_list.extend([v.strip() for v in item.split(separator)])
            else:
                new_list.append(item)
        varietas_list = new_list
    
    # Clean empty strings
    varietas_list = [v for v in varietas_list if v and v.strip()]
    return varietas_list if varietas_list else ['Tidak Diketahui']

# Buat dataframe baru dengan varietas yang sudah dipisah
expanded_data = []
for idx, row in df_aceh_besar.iterrows():
    varietas_list = pisahkan_varietas(row['varietas_padi'])
    for varietas in varietas_list:
        new_row = row.copy()
        new_row['varietas_padi_clean'] = varietas
        expanded_data.append(new_row)

df_expanded = pd.DataFrame(expanded_data)

# Filter hanya varietas yang ada di dictionary
df_expanded = df_expanded[df_expanded['varietas_padi_clean'].isin(varietas_hari.keys())]

# Buat crosstab: kecamatan vs varietas
crosstab = pd.crosstab(df_expanded['kecamatan'], df_expanded['varietas_padi_clean'])

print("\nCrosstab Kecamatan vs Varietas:")
print(crosstab)

# Pilih varietas yang paling banyak ditanam (top 12)
top_varieties = df_expanded['varietas_padi_clean'].value_counts().head(12).index.tolist()
crosstab_filtered = crosstab[top_varieties]

# Urutkan kecamatan berdasarkan total
crosstab_filtered['total'] = crosstab_filtered.sum(axis=1)
crosstab_filtered = crosstab_filtered.sort_values('total', ascending=False)
crosstab_filtered = crosstab_filtered.drop('total', axis=1)

print("\nFiltered crosstab (top 12 varieties):")
print(crosstab_filtered)

# Setup plot
fig, ax = plt.subplots(figsize=(16, 8))

# Posisi bar
kecamatan_list = crosstab_filtered.index.tolist()
x_pos = np.arange(len(kecamatan_list))
bar_width = 0.065  # Lebar bar untuk setiap varietas

# Warna untuk setiap varietas
colors_map = {
    'Inpari 32': COLOR_PALETTE['blue'],
    'Ciherang': COLOR_PALETTE['green'],
    'Cibatu': COLOR_PALETTE['orange'],
    'Ngawos': COLOR_PALETTE['red'],
    'Beulerang': COLOR_PALETTE['purple'],
    'Brangus': COLOR_PALETTE['teal'],
    'CBD 04': COLOR_PALETTE['cyan'],
    'CBD Murni': COLOR_PALETTE['yellow'],
    'Mekongga': COLOR_PALETTE['pink'],
    'Mustajab': '#8B4513',
    'Ramos': '#708090',
    'Ciherang beruang': COLOR_PALETTE['indigo'],
    'Cibatu 05': '#FF69B4',
    'Inpari': '#9C27B0',
    'Padi bojeng': '#795548'
}

# Plot setiap varietas
bars_dict = {}
for i, varietas in enumerate(crosstab_filtered.columns):
    offset = (i - len(crosstab_filtered.columns)/2) * bar_width + bar_width/2
    values = crosstab_filtered[varietas].values
    color = colors_map.get(varietas, COLOR_PALETTE['blue'])
    
    bars = ax.bar(x_pos + offset, values, bar_width, 
                  label=f'{varietas} ({varietas_hari[varietas]} hari)',
                  color=color, edgecolor='white', linewidth=0.5)
    
    # Add value labels on top of bars - FIXED
    for j, (bar, value) in enumerate(zip(bars, values)):  # Changed: use 'values' instead of 'kecamatan_counts.values'
        if value > 0:  # Only show label if value is greater than 0
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,  # Reduced offset from 0.5 to 0.15
                    f'{int(value)}',
                    ha='center', va='bottom',
                    bbox=dict(
                        boxstyle="round,pad=0.3",  # Reduced padding from 0.4 to 0.3
                        fc='white',
                        ec='gray',
                        lw=0.8,
                        alpha=0.95
                    ),
                    fontsize=7,  # Reduced font size from 11 to 7 for better fit
                    fontweight='bold')

# Styling
ax.set_xlabel('Kecamatan', fontsize=12, fontweight='bold')
ax.set_ylabel('Jumlah Petani', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(kecamatan_list, fontsize=11, fontweight='bold')
ax.legend(title='Varietas Padi', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Set y-axis dengan ruang lebih untuk labels
y_max = crosstab_filtered.max().max()
if y_max <= 10:
    interval = 1
    y_limit = y_max + 3  # Tambah ruang untuk labels
elif y_max <= 20:
    interval = 2
    y_limit = y_max + 4
else:
    interval = 2
    y_limit = y_max + 4

y_ticks = np.arange(0, y_limit + interval, interval)
ax.set_yticks(y_ticks)
ax.set_ylim(0, y_limit)

plt.tight_layout()

# Save plot
plt.savefig('02_varietas_padi_per_kecamatan_aceh_besar.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as: 02_varietas_padi_per_kecamatan_aceh_besar.png")