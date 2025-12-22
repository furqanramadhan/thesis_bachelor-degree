import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')  # Tambahkan line ini
matplotlib.use('Agg')  
import seaborn as sns
import numpy as np
import re

COLOR_PALETTE = {
    'primary_blue': '#1E88E5',      # Biru utama
    'secondary_blue': '#42A5F5',    # Biru sekunder
    'green': '#388E3C',             # Hijau
    'red': '#D32F2F',               # Merah
    'orange': '#FF6F00',            # Orange
    'purple': '#7B1FA2',            # Ungu
    'teal': '#00897B',              # Teal
    'amber': '#FFA000',             # Amber
    'brown': '#5D4037',             # Coklat
    'cyan': '#00ACC1',              # Cyan
    'indigo': '#3949AB',            # Indigo
    'pink': '#C2185B',              # Pink
    'lime': '#689F38',              # Lime
    'deep_orange': '#E64A19',       # Deep Orange
    'gray': '#757575'               # Abu-abu
}

# Load data manajemen usaha
df_manajemen = pd.read_csv("kuisioner_tanam_padi - ManajemenUsaha.csv")

# Debug: Periksa struktur data
print("Kolom yang tersedia dalam dataset manajemen usaha:")
print(df_manajemen.columns.tolist())
print("\nShape dataset:", df_manajemen.shape)
print("\nSample data:")
print(df_manajemen.head())

# Fungsi untuk mapping kabupaten berdasarkan ID petani
def get_kabupaten(id_petani):
    if pd.isna(id_petani):
        return np.nan
    
    # Convert to string and extract numeric part
    id_str = str(id_petani).strip()
    
    # Handle PTN prefix format (PTN001, PTN002, etc.)
    if id_str.startswith('PTN'):
        try:
            # Extract numeric part after PTN
            id_num = int(id_str[3:])
        except (ValueError, IndexError):
            return 'Tidak Diketahui'
    else:
        # Try direct numeric conversion
        try:
            id_num = int(id_petani)
        except:
            return 'Tidak Diketahui'
    
    # Map to kabupaten based on numeric ID
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
df_manajemen['kabupaten'] = df_manajemen['id_petani'].apply(get_kabupaten)

# Visualisasi 1 : Distribusi Penggunaan Pembajakan Lahan Modern berdasarkan Kabupaten

# Debug: Cek data pembajakan lahan modern
print("\nStatistik pembajakan lahan modern:")
print(df_manajemen['pembajakan_lahan_modern'].describe())
print("\nUnique values pembajakan lahan modern:")
print(df_manajemen['pembajakan_lahan_modern'].value_counts())

# Bersihkan data pembajakan lahan modern
def clean_pembajakan(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data pembajakan lahan modern
df_manajemen['pembajakan_clean'] = df_manajemen['pembajakan_lahan_modern'].apply(clean_pembajakan)

# Debug: Cek distribusi pembajakan lahan modern setelah cleaning
print("\nDistribusi pembajakan lahan modern setelah cleaning:")
print(df_manajemen['pembajakan_clean'].value_counts())

# Membuat crosstab untuk pembajakan lahan modern berdasarkan kabupaten
pembajakan_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['pembajakan_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
pembajakan_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_pembajakan = [p for p in pembajakan_order if p in pembajakan_by_kabupaten.columns]
pembajakan_by_kabupaten = pembajakan_by_kabupaten[available_pembajakan]

print(f"\nStatus pembajakan lahan modern yang akan ditampilkan: {available_pembajakan}")

# Gunakan warna yang meaningful untuk pembajakan lahan modern
# Hijau untuk Ya (positif), merah untuk Tidak (negatif), abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['green'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Visualisasi: Distribusi Penggunaan Pembajakan Lahan Modern berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Membuat plot
ax = pembajakan_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_pembajakan)]
)

total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )

# Percantik plot
plt.title('Distribusi Penggunaan Pembajakan Lahan Modern berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Pembajakan Modern', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = pembajakan_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('21_distribusi_pembajakan_lahan_modern_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi pembajakan lahan modern selesai!")


# Visualisasi 2: Distribusi Penggunaan Pengairan Sumur Bor berdasarkan Kabupaten

# Debug: Cek data pengairan sumur bor
print("\nStatistik pengairan sumur bor:")
print(df_manajemen['pengairan_sumur_bor'].describe())
print("\nUnique values pengairan sumur bor:")
print(df_manajemen['pengairan_sumur_bor'].value_counts())

# Bersihkan data pengairan sumur bor
def clean_sumur_bor(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data pengairan sumur bor
df_manajemen['sumur_bor_clean'] = df_manajemen['pengairan_sumur_bor'].apply(clean_sumur_bor)

# Debug: Cek distribusi pengairan sumur bor setelah cleaning
print("\nDistribusi pengairan sumur bor setelah cleaning:")
print(df_manajemen['sumur_bor_clean'].value_counts())

# Membuat crosstab untuk pengairan sumur bor berdasarkan kabupaten
sumur_bor_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['sumur_bor_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
sumur_bor_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_sumur_bor = [s for s in sumur_bor_order if s in sumur_bor_by_kabupaten.columns]
sumur_bor_by_kabupaten = sumur_bor_by_kabupaten[available_sumur_bor]

print(f"\nStatus pengairan sumur bor yang akan ditampilkan: {available_sumur_bor}")

# Gunakan warna yang meaningful untuk pengairan sumur bor
# Biru untuk Ya (positif - berhubungan dengan air), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['primary_blue'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
plt.figure(figsize=(12, 6))

ax = sumur_bor_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_sumur_bor)]
)

total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )


# Percantik plot
plt.title('Distribusi Penggunaan Pengairan Sumur Bor berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Pengairan Sumur Bor', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = sumur_bor_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('22_distribusi_pengairan_sumur_bor_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 2 selesai!")


# Visualisasi 3: Distribusi Penggunaan Pengairan Pompa Air berdasarkan Kabupaten

# Debug: Cek data pengairan pompa air
print("\nStatistik pengairan pompa air:")
print(df_manajemen['pengairan_pompa_air'].describe())
print("\nUnique values pengairan pompa air:")
print(df_manajemen['pengairan_pompa_air'].value_counts())

# Bersihkan data pengairan pompa air
def clean_pompa_air(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data pengairan pompa air
df_manajemen['pompa_air_clean'] = df_manajemen['pengairan_pompa_air'].apply(clean_pompa_air)

# Debug: Cek distribusi pengairan pompa air setelah cleaning
print("\nDistribusi pengairan pompa air setelah cleaning:")
print(df_manajemen['pompa_air_clean'].value_counts())

# Membuat crosstab untuk pengairan pompa air berdasarkan kabupaten
pompa_air_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['pompa_air_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
pompa_air_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_pompa_air = [p for p in pompa_air_order if p in pompa_air_by_kabupaten.columns]
pompa_air_by_kabupaten = pompa_air_by_kabupaten[available_pompa_air]

print(f"\nStatus pengairan pompa air yang akan ditampilkan: {available_pompa_air}")

# Gunakan warna yang meaningful untuk pengairan pompa air
# Biru lebih muda untuk Ya (positif - berhubungan dengan air), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['secondary_blue'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]
  # Lighter blue to differentiate from sumur bor

# Membuat plot
plt.figure(figsize=(12, 6))

ax = pompa_air_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_pompa_air)]
)

# Tambahkan label jumlah pada setiap bar
total_count = sum([rect.get_height() for container in ax.containers for rect in container])
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )

# Percantik plot
plt.title('Distribusi Penggunaan Pengairan Pompa Air berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Pengairan Pompa Air', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = pompa_air_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('23_distribusi_pengairan_pompa_air_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 3 selesai!")

# Visualisasi 4: Distribusi Penggunaan Penyemprotan Pompa Tangan berdasarkan Kabupaten

# Debug: Cek data penyemprotan pompa tangan
print("\nStatistik penyemprotan pompa tangan:")
print(df_manajemen['penyemprotan_pompa_tangan'].describe())
print("\nUnique values penyemprotan pompa tangan:")
print(df_manajemen['penyemprotan_pompa_tangan'].value_counts())

# Bersihkan data penyemprotan pompa tangan
def clean_pompa_tangan(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data penyemprotan pompa tangan
df_manajemen['pompa_tangan_clean'] = df_manajemen['penyemprotan_pompa_tangan'].apply(clean_pompa_tangan)

# Debug: Cek distribusi penyemprotan pompa tangan setelah cleaning
print("\nDistribusi penyemprotan pompa tangan setelah cleaning:")
print(df_manajemen['pompa_tangan_clean'].value_counts())

# Membuat crosstab untuk penyemprotan pompa tangan berdasarkan kabupaten
pompa_tangan_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['pompa_tangan_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
pompa_tangan_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_pompa_tangan = [p for p in pompa_tangan_order if p in pompa_tangan_by_kabupaten.columns]
pompa_tangan_by_kabupaten = pompa_tangan_by_kabupaten[available_pompa_tangan]

print(f"\nStatus penyemprotan pompa tangan yang akan ditampilkan: {available_pompa_tangan}")

# Gunakan warna yang meaningful untuk penyemprotan pompa tangan
# Hijau untuk Ya (positif - berhubungan dengan aktivitas pertanian), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['lime'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]
 # Light green for hand pump spraying

# Membuat plot
plt.figure(figsize=(12, 6))

ax = pompa_tangan_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_pompa_tangan)]
)

total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )

# Percantik plot
plt.title('Distribusi Penggunaan Penyemprotan Pompa Tangan berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Pompa Tangan', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = pompa_tangan_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('24_distribusi_penyemprotan_pompa_tangan_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 4 selesai!")

# Visualisasi 5: Distribusi Penggunaan Penyemprotan Pompa Elektrik berdasarkan Kabupaten

# Debug: Cek data penyemprotan pompa elektrik
print("\nStatistik penyemprotan pompa elektrik:")
print(df_manajemen['penyemprotan_pompa_elektrik'].describe())
print("\nUnique values penyemprotan pompa elektrik:")
print(df_manajemen['penyemprotan_pompa_elektrik'].value_counts())

# Bersihkan data penyemprotan pompa elektrik
def clean_pompa_elektrik(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data penyemprotan pompa elektrik
df_manajemen['pompa_elektrik_clean'] = df_manajemen['penyemprotan_pompa_elektrik'].apply(clean_pompa_elektrik)

# Debug: Cek distribusi penyemprotan pompa elektrik setelah cleaning
print("\nDistribusi penyemprotan pompa elektrik setelah cleaning:")
print(df_manajemen['pompa_elektrik_clean'].value_counts())

# Membuat crosstab untuk penyemprotan pompa elektrik berdasarkan kabupaten
pompa_elektrik_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['pompa_elektrik_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
pompa_elektrik_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_pompa_elektrik = [p for p in pompa_elektrik_order if p in pompa_elektrik_by_kabupaten.columns]
pompa_elektrik_by_kabupaten = pompa_elektrik_by_kabupaten[available_pompa_elektrik]

print(f"\nStatus penyemprotan pompa elektrik yang akan ditampilkan: {available_pompa_elektrik}")

# Gunakan warna yang meaningful untuk penyemprotan pompa elektrik
# Hijau tua untuk Ya (positif - teknologi lebih maju), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['green'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
plt.figure(figsize=(12, 6))

ax = pompa_elektrik_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_pompa_elektrik)]
)

total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )

# Percantik plot
plt.title('Distribusi Penggunaan Penyemprotan Pompa Elektrik berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Pompa Elektrik', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = pompa_elektrik_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('25_distribusi_penyemprotan_pompa_elektrik_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 5 selesai!")

# Visualisasi 6: Distribusi Penggunaan Mesin Potong Panen berdasarkan Kabupaten

# Debug: Cek data panen mesin potong
print("\nStatistik panen mesin potong:")
print(df_manajemen['panen_mesin_potong'].describe())
print("\nUnique values panen mesin potong:")
print(df_manajemen['panen_mesin_potong'].value_counts())

# Bersihkan data panen mesin potong
def clean_mesin_potong(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data panen mesin potong
df_manajemen['mesin_potong_clean'] = df_manajemen['panen_mesin_potong'].apply(clean_mesin_potong)

# Debug: Cek distribusi panen mesin potong setelah cleaning
print("\nDistribusi panen mesin potong setelah cleaning:")
print(df_manajemen['mesin_potong_clean'].value_counts())

# Membuat crosstab untuk panen mesin potong berdasarkan kabupaten
mesin_potong_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['mesin_potong_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
mesin_potong_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_mesin_potong = [p for p in mesin_potong_order if p in mesin_potong_by_kabupaten.columns]
mesin_potong_by_kabupaten = mesin_potong_by_kabupaten[available_mesin_potong]

print(f"\nStatus panen mesin potong yang akan ditampilkan: {available_mesin_potong}")

# Gunakan warna yang meaningful untuk panen mesin potong
# Kuning keemasan untuk Ya (positif - berhubungan dengan panen/gandum), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['amber'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
plt.figure(figsize=(12, 6))

ax = mesin_potong_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_mesin_potong)]
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )


# Percantik plot
plt.title('Distribusi Penggunaan Mesin Potong Panen berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Mesin Potong Panen', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = mesin_potong_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('26_distribusi_panen_mesin_potong_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 6 selesai!")

# Visualisasi 7: Distribusi Penggunaan Internet berdasarkan Kabupaten

# Debug: Cek data penggunaan internet
print("\nStatistik penggunaan internet:")
print(df_manajemen['pakai_internet'].describe())
print("\nUnique values penggunaan internet:")
print(df_manajemen['pakai_internet'].value_counts())

# Bersihkan data penggunaan internet
def clean_internet(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data penggunaan internet
df_manajemen['internet_clean'] = df_manajemen['pakai_internet'].apply(clean_internet)

# Debug: Cek distribusi penggunaan internet setelah cleaning
print("\nDistribusi penggunaan internet setelah cleaning:")
print(df_manajemen['internet_clean'].value_counts())

# Membuat crosstab untuk penggunaan internet berdasarkan kabupaten
internet_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['internet_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
internet_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_internet = [i for i in internet_order if i in internet_by_kabupaten.columns]
internet_by_kabupaten = internet_by_kabupaten[available_internet]

print(f"\nStatus penggunaan internet yang akan ditampilkan: {available_internet}")

# Gunakan warna yang meaningful untuk penggunaan internet
# Biru terang untuk Ya (positif - berhubungan dengan teknologi/internet), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['primary_blue'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
plt.figure(figsize=(12, 6))

ax = internet_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_internet)]
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])
# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)
# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )



# Percantik plot
plt.title('Distribusi Penggunaan Internet berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Penggunaan Internet', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = internet_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('27_distribusi_penggunaan_internet_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 7 selesai!")

# Visualisasi 8: Distribusi Petani yang Mendapatkan Informasi dari Penyuluh berdasarkan Kabupaten

# Debug: Cek data info dari penyuluh
print("\nStatistik info dari penyuluh:")
print(df_manajemen['info_penyuluh'].describe())
print("\nUnique values info dari penyuluh:")
print(df_manajemen['info_penyuluh'].value_counts())

# Bersihkan data info dari penyuluh
def clean_info_penyuluh(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data info dari penyuluh
df_manajemen['info_penyuluh_clean'] = df_manajemen['info_penyuluh'].apply(clean_info_penyuluh)

# Debug: Cek distribusi info dari penyuluh setelah cleaning
print("\nDistribusi info dari penyuluh setelah cleaning:")
print(df_manajemen['info_penyuluh_clean'].value_counts())

# Membuat crosstab untuk info dari penyuluh berdasarkan kabupaten
penyuluh_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['info_penyuluh_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
penyuluh_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_penyuluh = [p for p in penyuluh_order if p in penyuluh_by_kabupaten.columns]
penyuluh_by_kabupaten = penyuluh_by_kabupaten[available_penyuluh]

print(f"\nStatus info dari penyuluh yang akan ditampilkan: {available_penyuluh}")

# Gunakan warna yang meaningful untuk info dari penyuluh
# Hijau kebiruan untuk Ya (positif - berhubungan dengan pengetahuan/pembelajaran), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['teal'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]
 # Teal color for knowledge/information

# Membuat plot
plt.figure(figsize=(12, 6))

ax = penyuluh_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_penyuluh)]
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )


# Percantik plot
plt.title('Distribusi Petani yang Mendapatkan Informasi dari Penyuluh berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Info dari Penyuluh', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = penyuluh_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('28_distribusi_info_penyuluh_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 8 selesai!")

# Visualisasi 9: Distribusi Petani yang Mendapatkan Informasi dari Keuchik berdasarkan Kabupaten

# Debug: Cek data info dari keuchik
print("\nStatistik info dari keuchik:")
print(df_manajemen['info_keuchik'].describe())
print("\nUnique values info dari keuchik:")
print(df_manajemen['info_keuchik'].value_counts())

# Bersihkan data info dari keuchik
def clean_info_keuchik(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data info dari keuchik
df_manajemen['info_keuchik_clean'] = df_manajemen['info_keuchik'].apply(clean_info_keuchik)

# Debug: Cek distribusi info dari keuchik setelah cleaning
print("\nDistribusi info dari keuchik setelah cleaning:")
print(df_manajemen['info_keuchik_clean'].value_counts())

# Membuat crosstab untuk info dari keuchik berdasarkan kabupaten
keuchik_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['info_keuchik_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
keuchik_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_keuchik = [k for k in keuchik_order if k in keuchik_by_kabupaten.columns]
keuchik_by_kabupaten = keuchik_by_kabupaten[available_keuchik]

print(f"\nStatus info dari keuchik yang akan ditampilkan: {available_keuchik}")

# Gunakan warna yang meaningful untuk info dari keuchik
# Ungu untuk Ya (positif - berhubungan dengan kepemimpinan lokal), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['purple'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
plt.figure(figsize=(12, 6))

ax = keuchik_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_keuchik)]
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )

# Percantik plot
plt.title('Distribusi Petani yang Mendapatkan Informasi dari Keuchik berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Info dari Keuchik', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = keuchik_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('29_distribusi_info_keuchik_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 9 selesai!")

# Visualisasi 10: Distribusi Petani yang Mendapatkan Informasi dari Keujrun Blang berdasarkan Kabupaten

# Debug: Cek data info dari keujrun blang
print("\nStatistik info dari keujrun blang:")
print(df_manajemen['info_keujrun_blang'].describe())
print("\nUnique values info dari keujrun blang:")
print(df_manajemen['info_keujrun_blang'].value_counts())

# Bersihkan data info dari keujrun blang
def clean_info_keujrun(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data info dari keujrun blang
df_manajemen['info_keujrun_clean'] = df_manajemen['info_keujrun_blang'].apply(clean_info_keujrun)

# Debug: Cek distribusi info dari keujrun blang setelah cleaning
print("\nDistribusi info dari keujrun blang setelah cleaning:")
print(df_manajemen['info_keujrun_clean'].value_counts())

# Membuat crosstab untuk info dari keujrun blang berdasarkan kabupaten
keujrun_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['info_keujrun_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
keujrun_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_keujrun = [k for k in keujrun_order if k in keujrun_by_kabupaten.columns]
keujrun_by_kabupaten = keujrun_by_kabupaten[available_keujrun]

print(f"\nStatus info dari keujrun blang yang akan ditampilkan: {available_keujrun}")

# Gunakan warna yang meaningful untuk info dari keujrun blang
# Biru kehijauan untuk Ya (positif - berhubungan dengan pengetahuan lokal), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['teal'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
plt.figure(figsize=(12, 6))

ax = keujrun_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_keujrun)]
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )


# Percantik plot
plt.title('Distribusi Petani yang Mendapatkan Informasi dari Keujrun Blang berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Info dari Keujrun Blang', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = keujrun_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('30_distribusi_info_keujrun_blang_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 10 selesai!")

# Visualisasi 11: Distribusi Keanggotaan Kelompok Tani berdasarkan Kabupaten

# Debug: Cek data keanggotaan kelompok tani
print("\nStatistik keanggotaan kelompok tani:")
print(df_manajemen['anggota_kelompok_tani'].describe())
print("\nUnique values keanggotaan kelompok tani:")
print(df_manajemen['anggota_kelompok_tani'].value_counts())

# Bersihkan data keanggotaan kelompok tani
def clean_kelompok_tani(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data keanggotaan kelompok tani
df_manajemen['kelompok_tani_clean'] = df_manajemen['anggota_kelompok_tani'].apply(clean_kelompok_tani)

# Debug: Cek distribusi keanggotaan kelompok tani setelah cleaning
print("\nDistribusi keanggotaan kelompok tani setelah cleaning:")
print(df_manajemen['kelompok_tani_clean'].value_counts())

# Membuat crosstab untuk keanggotaan kelompok tani berdasarkan kabupaten
kelompok_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['kelompok_tani_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
kelompok_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_kelompok = [k for k in kelompok_order if k in kelompok_by_kabupaten.columns]
kelompok_by_kabupaten = kelompok_by_kabupaten[available_kelompok]

print(f"\nStatus keanggotaan kelompok tani yang akan ditampilkan: {available_kelompok}")

# Gunakan warna yang meaningful untuk keanggotaan kelompok tani
# Oranye untuk Ya (positif - berhubungan dengan komunitas/kelompok), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['orange'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
plt.figure(figsize=(12, 6))

ax = kelompok_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_kelompok)]
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )


# Percantik plot
plt.title('Distribusi Keanggotaan Kelompok Tani berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Anggota Kelompok Tani', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = kelompok_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('31_distribusi_anggota_kelompok_tani_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 11 selesai!")

# Visualisasi 12: Distribusi Pengetahuan tentang Kalender Tanam berdasarkan Kabupaten

# Debug: Cek data pengetahuan tentang kalender tanam
print("\nStatistik pengetahuan tentang kalender tanam:")
print(df_manajemen['tahu_katam'].describe())
print("\nUnique values pengetahuan tentang kalender tanam:")
print(df_manajemen['tahu_katam'].value_counts())

# Bersihkan data pengetahuan tentang kalender tanam
def clean_tahu_katam(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data pengetahuan tentang kalender tanam
df_manajemen['tahu_katam_clean'] = df_manajemen['tahu_katam'].apply(clean_tahu_katam)

# Debug: Cek distribusi pengetahuan tentang kalender tanam setelah cleaning
print("\nDistribusi pengetahuan tentang kalender tanam setelah cleaning:")
print(df_manajemen['tahu_katam_clean'].value_counts())

# Membuat crosstab untuk pengetahuan tentang kalender tanam berdasarkan kabupaten
katam_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['tahu_katam_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
katam_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_katam = [k for k in katam_order if k in katam_by_kabupaten.columns]
katam_by_kabupaten = katam_by_kabupaten[available_katam]

print(f"\nStatus pengetahuan tentang kalender tanam yang akan ditampilkan: {available_katam}")

# Gunakan warna yang meaningful untuk pengetahuan tentang kalender tanam
# Kuning untuk Ya (positif - berhubungan dengan pengetahuan jadwal), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['amber'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
plt.figure(figsize=(12, 6))

ax = katam_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_katam)]
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )


# Percantik plot
plt.title('Distribusi Pengetahuan tentang Kalender Tanam berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Tahu Kalender Tanam', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = katam_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('32_distribusi_tahu_kalender_tanam_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 12 selesai!")

# Visualisasi 13: Distribusi Pengetahuan tentang Pergeseran Musim berdasarkan Kabupaten

# Debug: Cek data pengetahuan tentang pergeseran musim
print("\nStatistik pengetahuan tentang pergeseran musim:")
print(df_manajemen['tahu_pergeseran_musim'].describe())
print("\nUnique values pengetahuan tentang pergeseran musim:")
print(df_manajemen['tahu_pergeseran_musim'].value_counts())

# Bersihkan data pengetahuan tentang pergeseran musim
def clean_pergeseran_musim(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data pengetahuan tentang pergeseran musim
df_manajemen['pergeseran_musim_clean'] = df_manajemen['tahu_pergeseran_musim'].apply(clean_pergeseran_musim)

# Debug: Cek distribusi pengetahuan tentang pergeseran musim setelah cleaning
print("\nDistribusi pengetahuan tentang pergeseran musim setelah cleaning:")
print(df_manajemen['pergeseran_musim_clean'].value_counts())

# Membuat crosstab untuk pengetahuan tentang pergeseran musim berdasarkan kabupaten
musim_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['pergeseran_musim_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
musim_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_musim = [m for m in musim_order if m in musim_by_kabupaten.columns]
musim_by_kabupaten = musim_by_kabupaten[available_musim]

print(f"\nStatus pengetahuan tentang pergeseran musim yang akan ditampilkan: {available_musim}")

# Gunakan warna yang meaningful untuk pengetahuan tentang pergeseran musim
# Biru muda untuk Ya (positif - berhubungan dengan perubahan iklim/cuaca), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['secondary_blue'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]
 # Light blue for climate/weather awareness

# Membuat plot
plt.figure(figsize=(12, 6))

ax = musim_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_musim)]
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )

# Percantik plot
plt.title('Distribusi Pengetahuan tentang Pergeseran Musim berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Tahu Pergeseran Musim', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = musim_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('33_distribusi_tahu_pergeseran_musim_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 13 selesai!")

# Visualisasi 14: Distribusi Respon Petani terhadap Pergeseran Musim berdasarkan Kabupaten

# Debug: Cek data respon pergeseran musim
print("\nStatistik respon pergeseran musim:")
print(df_manajemen['respon_pergeseran_musim'].describe())
print("\nUnique values respon pergeseran musim:")
print(df_manajemen['respon_pergeseran_musim'].value_counts())

# Bersihkan data respon pergeseran musim
def clean_respon_pergeseran(value):
    if pd.isna(value):
        return 'Tidak Ada Respon'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Ada Respon'
    
    # Standardisasi respon berdasarkan kata kunci
    cleaned_lower = cleaned.lower()
    
    if 'pengumuman desa' in cleaned_lower or 'desa' in cleaned_lower:
        return 'Ikut Pengumuman Desa'
    elif 'kelompok tani' in cleaned_lower or 'poktan' in cleaned_lower:
        return 'Ikut Kelompok Tani'
    elif 'penyuluh' in cleaned_lower:
        return 'Ikut Penyuluh'
    elif 'radio' in cleaned_lower or 'tv' in cleaned_lower or 'siaran' in cleaned_lower:
        return 'Ikut Siaran Radio/TV'
    elif 'mandiri' in cleaned_lower:
        return 'Mandiri'
    elif 'keujrun' in cleaned_lower or 'blang' in cleaned_lower:
        return 'Ikut Keujrun Blang'
    else:
        # Jika tidak cocok dengan kategori manapun, ambil sebagai respon lain
        return 'Respon Lainnya'

# Bersihkan data respon pergeseran musim
df_manajemen['respon_pergeseran_clean'] = df_manajemen['respon_pergeseran_musim'].apply(clean_respon_pergeseran)

# Debug: Cek distribusi respon pergeseran musim setelah cleaning
print("\nDistribusi respon pergeseran musim setelah cleaning:")
print(df_manajemen['respon_pergeseran_clean'].value_counts())

# Membuat crosstab untuk respon pergeseran musim berdasarkan kabupaten
respon_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['respon_pergeseran_clean'])

# Urutkan kolom berdasarkan preferensi (yang paling umum dulu)
respon_order = ['Ikut Pengumuman Desa', 'Ikut Kelompok Tani', 'Ikut Penyuluh', 
                'Mandiri', 'Ikut Keujrun Blang', 'Ikut Siaran Radio/TV', 
                'Respon Lainnya', 'Tidak Ada Respon']

# Ambil kolom yang tersedia dalam data
available_respon = [r for r in respon_order if r in respon_by_kabupaten.columns]

# Jika ada kolom yang tidak ada dalam respon_order, tambahkan di akhir
for col in respon_by_kabupaten.columns:
    if col not in available_respon:
        available_respon.append(col)

respon_by_kabupaten = respon_by_kabupaten[available_respon]

print(f"\nJenis respon yang akan ditampilkan: {available_respon}")

# Gunakan warna yang meaningful untuk respon pergeseran musim
# Palet warna yang berbeda untuk setiap jenis respon
color_map = {
    'Ikut Pengumuman Desa': COLOR_PALETTE['green'],
    'Ikut Kelompok Tani': COLOR_PALETTE['orange'],
    'Ikut Penyuluh': COLOR_PALETTE['primary_blue'],
    'Mandiri': COLOR_PALETTE['purple'],
    'Ikut Keujrun Blang': COLOR_PALETTE['brown'],
    'Ikut Siaran Radio/TV': COLOR_PALETTE['amber'],
    'Respon Lainnya': COLOR_PALETTE['cyan'],
    'Tidak Ada Respon': COLOR_PALETTE['red']
}
# Pastikan jumlah warna sesuai dengan jumlah kategori
colors = [color_map.get(resp, '#000000') for resp in available_respon]

# Membuat plot
plt.figure(figsize=(14, 8))

ax = respon_by_kabupaten.plot(
    kind='bar', 
    width=0.7,
    color=colors
)

# Tambahkan label jumlah pada setiap bar
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )


# Percantik plot
plt.title('Distribusi Respon Petani terhadap Pergeseran Musim berdasarkan Kabupaten', fontsize=16, pad=20)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Place legend outside the plot to the right
plt.legend(title='Jenis Respon', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = respon_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.7)  # Reduce right margin to make space for legend

plt.savefig('34_distribusi_respon_pergeseran_musim_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 14 selesai!")

# Visualisasi 15: Distribusi Pernah Gagal Tanam berdasarkan Kabupaten

# Debug: Cek data pernah gagal tanam
print("\n=== VISUALISASI 15: PERNAH GAGAL TANAM ===")
print("Statistik pernah gagal tanam:")
print(df_manajemen['pernah_gagal_tanam'].describe())
print("\nUnique values pernah gagal tanam:")
print(df_manajemen['pernah_gagal_tanam'].value_counts())

# Bersihkan data pernah gagal tanam
def clean_pernah_gagal(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data pernah gagal tanam
df_manajemen['pernah_gagal_clean'] = df_manajemen['pernah_gagal_tanam'].apply(clean_pernah_gagal)

# Debug: Cek distribusi pernah gagal tanam setelah cleaning
print("\nDistribusi pernah gagal tanam setelah cleaning:")
print(df_manajemen['pernah_gagal_clean'].value_counts())

# Membuat crosstab untuk pernah gagal tanam berdasarkan kabupaten
gagal_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['pernah_gagal_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
gagal_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_gagal = [g for g in gagal_order if g in gagal_by_kabupaten.columns]
gagal_by_kabupaten = gagal_by_kabupaten[available_gagal]

print(f"\nStatus pernah gagal tanam yang akan ditampilkan: {available_gagal}")

# Gunakan warna yang meaningful untuk pernah gagal tanam
# Merah untuk Ya (negatif - pernah gagal), hijau untuk Tidak (positif - tidak pernah gagal), abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['red'], COLOR_PALETTE['green'], COLOR_PALETTE['gray']]

# Membuat plot
plt.figure(figsize=(12, 6))

ax = gagal_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_gagal)]
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])
# Tambahkan label jumlah pada setiap bar
# Clear existing labels
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )


# Percantik plot
plt.title('Distribusi Pernah Gagal Tanam berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Pernah Gagal Tanam', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = gagal_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('35_distribusi_pernah_gagal_tanam_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 15 selesai!")


# Visualisasi 16: Distribusi Penyebab Gagal Tanam berdasarkan Kabupaten

# Debug: Cek data penyebab gagal tanam
print("\n=== VISUALISASI 16: PENYEBAB GAGAL TANAM ===")
print("Statistik penyebab gagal tanam:")
print(df_manajemen['penyebab_gagal'].describe())
print("\nUnique values penyebab gagal tanam:")
print(df_manajemen['penyebab_gagal'].value_counts())

# Bersihkan data penyebab gagal tanam
def clean_penyebab_gagal(value):
    if pd.isna(value):
        return 'Tidak Ada Data'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Ada Data'
    
    # Standardisasi penyebab berdasarkan kategori yang sudah ditentukan
    cleaned_lower = cleaned.lower()
    
    if 'banjir' in cleaned_lower:
        return 'Banjir'
    elif 'kemarau' in cleaned_lower:
        return 'Kemarau'
    elif 'suhu' in cleaned_lower and 'ekstrim' in cleaned_lower:
        return 'Suhu Ekstrim'
    elif 'angin' in cleaned_lower and ('kencang' in cleaned_lower or 'keras' in cleaned_lower):
        return 'Angin Kencang'
    elif 'hujan' in cleaned_lower and 'deras' in cleaned_lower:
        return 'Hujan Deras'
    elif 'hujan' in cleaned_lower and ('tidak menentu' in cleaned_lower or 'tak menentu' in cleaned_lower):
        return 'Hujan Tidak Menentu'
    else:
        # Jika tidak cocok dengan kategori manapun, kembalikan nilai asli yang sudah dibersihkan
        return cleaned.title()

# Bersihkan data penyebab gagal tanam
df_manajemen['penyebab_gagal_clean'] = df_manajemen['penyebab_gagal'].apply(clean_penyebab_gagal)

# Debug: Cek distribusi penyebab gagal tanam setelah cleaning
print("\nDistribusi penyebab gagal tanam setelah cleaning:")
print(df_manajemen['penyebab_gagal_clean'].value_counts())

# Membuat crosstab untuk penyebab gagal tanam berdasarkan kabupaten
penyebab_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['penyebab_gagal_clean'])

# Urutkan kolom berdasarkan preferensi (yang paling umum dulu)
penyebab_order = ['Kemarau', 'Banjir', 'Hujan Tidak Menentu', 'Hujan Deras', 
                  'Suhu Ekstrim', 'Angin Kencang', 'Tidak Ada Data']

# Ambil kolom yang tersedia dalam data
available_penyebab = [p for p in penyebab_order if p in penyebab_by_kabupaten.columns]

# Jika ada kolom yang tidak ada dalam penyebab_order, tambahkan di akhir
for col in penyebab_by_kabupaten.columns:
    if col not in available_penyebab:
        available_penyebab.append(col)

penyebab_by_kabupaten = penyebab_by_kabupaten[available_penyebab]

print(f"\nPenyebab gagal tanam yang akan ditampilkan: {available_penyebab}")

# Gunakan warna yang meaningful untuk penyebab gagal tanam
# Warna berdasarkan jenis bencana/masalah
colors = [COLOR_PALETTE['deep_orange'],  # Kemarau
          COLOR_PALETTE['primary_blue'],  # Banjir
          COLOR_PALETTE['purple'],        # Hujan Tidak Menentu
          COLOR_PALETTE['indigo'],        # Hujan Deras
          COLOR_PALETTE['orange'],        # Suhu Ekstrim
          COLOR_PALETTE['brown'],         # Angin Kencang
          COLOR_PALETTE['gray']]          # Tidak Ada Data

# Pastikan jumlah warna sesuai dengan jumlah kategori
colors = colors[:len(available_penyebab)]

# Membuat plot
plt.figure(figsize=(14, 8))

ax = penyebab_by_kabupaten.plot(
    kind='bar', 
    width=0.7,
    color=colors
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])
# Tambahkan label jumlah pada setiap bar
# Clear existing labels
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )


# Percantik plot
plt.title('Distribusi Penyebab Gagal Tanam berdasarkan Kabupaten', fontsize=16, pad=20)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Place legend outside the plot to the right
plt.legend(title='Penyebab Gagal Tanam', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = penyebab_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.7)  # Reduce right margin to make space for legend

plt.savefig('36_distribusi_penyebab_gagal_tanam_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 16 selesai!")


# Visualisasi 17: Distribusi Penggunaan Teknologi Lain berdasarkan Kabupaten

# Debug: Cek data teknologi lain
print("\n=== VISUALISASI 17: TEKNOLOGI LAIN ===")
print("Statistik teknologi lain:")
print(df_manajemen['teknologi_lain'].describe())
print("\nUnique values teknologi lain:")
print(df_manajemen['teknologi_lain'].value_counts())

# Bersihkan data teknologi lain
def clean_teknologi_lain(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data teknologi lain
df_manajemen['teknologi_lain_clean'] = df_manajemen['teknologi_lain'].apply(clean_teknologi_lain)

# Debug: Cek distribusi teknologi lain setelah cleaning
print("\nDistribusi teknologi lain setelah cleaning:")
print(df_manajemen['teknologi_lain_clean'].value_counts())

# Membuat crosstab untuk teknologi lain berdasarkan kabupaten
teknologi_by_kabupaten = pd.crosstab(df_manajemen['kabupaten'], df_manajemen['teknologi_lain_clean'])

# Urutkan kolom berdasarkan preferensi (Ya dulu, lalu Tidak)
teknologi_order = ['Ya', 'Tidak', 'Tidak Diketahui']
available_teknologi = [t for t in teknologi_order if t in teknologi_by_kabupaten.columns]
teknologi_by_kabupaten = teknologi_by_kabupaten[available_teknologi]

print(f"\nStatus teknologi lain yang akan ditampilkan: {available_teknologi}")

# Gunakan warna yang meaningful untuk teknologi lain
# Hijau untuk Ya (positif - menggunakan teknologi tambahan), merah untuk Tidak, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['green'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]


# Membuat plot
plt.figure(figsize=(12, 6))

ax = teknologi_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_teknologi)]
)
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
            
            # Create text with frame
            ax.annotate(
                label_text, 
                xy=(x, y),
                xytext=(0, 0),
                textcoords='offset points',
                ha='center', va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc='white',
                    ec='gray',
                    lw=1,
                    alpha=0.9
                ),
                fontsize=9,
                fontweight='bold'
            )


# Percantik plot
plt.title('Distribusi Penggunaan Teknologi Lain berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Teknologi Lain', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = teknologi_by_kabupaten.values.max()
print(f"\nNilai maksimum pada grafik: {y_max}")

# Buat interval yang dinamis berdasarkan nilai maksimum
if y_max <= 10:
    interval = 1
    y_limit = y_max + 2
elif y_max <= 25:
    interval = 2
    y_limit = y_max + 5
elif y_max <= 50:
    interval = 5
    y_limit = y_max + 5
else:
    interval = 10
    y_limit = y_max + 10

# Set y-axis dengan scaling dinamis
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

print(f"Y-axis range: 0 to {y_limit} with interval {interval}")

# Adjust figure layout to make room for the legend
plt.tight_layout()
plt.subplots_adjust(right=0.75)  # Reduce right margin to make space for legend

plt.savefig('37_distribusi_teknologi_lain_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 17 selesai!")
