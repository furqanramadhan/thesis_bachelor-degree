import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
import matplotlib
# matplotlib.use('TkAgg')  # Tambahkan line ini
matplotlib.use('Agg')  
import seaborn as sns
import numpy as np
import re

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
    'lime': '#689F38',
    'amber': '#FFA000',
    'brown': '#5D4037',
    'gray': '#757575'
}

# Load data periode tanam
df_periode = pd.read_csv("kuisioner_tanam_padi - PeriodeTanam.csv")

# Debug: Periksa struktur data
print("Kolom yang tersedia dalam dataset periode tanam:")
print(df_periode.columns.tolist())
print("\nShape dataset:", df_periode.shape)
print("\nSample data:")
print(df_periode.head())

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
df_periode['kabupaten'] = df_periode['id_petani'].apply(get_kabupaten)

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

df_periode['kabupaten'] = df_periode['id_petani'].apply(get_kabupaten)

# Fungsi untuk membersihkan data gunca
def clean_gunca(value):
    if pd.isna(value):
        return None
    
    cleaned = str(value).strip()
    
    if cleaned == '' or cleaned.lower() == 'nan' or cleaned == '-':
        return None
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None

# Fungsi untuk membersihkan data pengeluaran
def clean_pengeluaran_data(value):
    if pd.isna(value):
        return None

    cleaned = str(value).strip()

    if cleaned == '' or cleaned.lower() == 'nan' or cleaned == '-':
        return None

    # Hapus titik sebagai pemisah ribuan, lalu konversi ke float
    cleaned = cleaned.replace('.', '')
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# Debug: Cek data bulan tanam dan kabupaten
print("\nBulan tanam yang tersedia:")
print(df_periode['bulan_tanam'].value_counts())
print("\nKabupaten yang tersedia:")
print(df_periode['kabupaten'].value_counts())

# Visualisasi 1: Distribusi Bulan Tanam berdasarkan Kabupaten
plt.figure(figsize=(14, 10))

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Ambil data petani unik dan bulan tanam mereka
petani_bulan_tanam = df_periode[['id_petani', 'kabupaten', 'bulan_tanam']].drop_duplicates(subset=['id_petani'])

print(f"\nJumlah total petani unik: {len(petani_bulan_tanam)}")
print(f"Distribusi bulan tanam per petani unik:")
print(petani_bulan_tanam['bulan_tanam'].value_counts().sort_index())

# Membuat crosstab untuk bulan tanam berdasarkan kabupaten (data petani unik)
bulan_by_kabupaten = pd.crosstab(petani_bulan_tanam['bulan_tanam'], petani_bulan_tanam['kabupaten'])

# Urutkan bulan tanam secara logis (semua periode termasuk yang baru)
bulan_order = ['nov - jan', 'des - feb', 'feb - apr', 'mei - juli', 'agt - okt']
available_bulan = [b for b in bulan_order if b in bulan_by_kabupaten.index]

print(f"\nPeriode tanam yang tersedia dalam data: {available_bulan}")
print(f"Total periode berbeda: {len(available_bulan)}")

if available_bulan:
    bulan_by_kabupaten = bulan_by_kabupaten.loc[available_bulan]
else:
    # Jika tidak ada yang match, tampilkan semua yang ada
    print("Menggunakan semua periode yang tersedia dalam dataset")

# Gunakan warna yang menarik (tambahkan warna untuk periode baru)
colors = [COLOR_PALETTE['blue'], COLOR_PALETTE['orange'], COLOR_PALETTE['green'], 
          COLOR_PALETTE['purple'], COLOR_PALETTE['gray']]

# Membuat plot
ax = bulan_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(bulan_by_kabupaten.columns)]
)

# Calculate total count for percentages
total_count = petani_bulan_tanam['bulan_tanam'].count()

# Tambahkan label jumlah dan persentase
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
                
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5  # Adjust this value based on your data scale
                
            # Create text with frame
            text = ax.annotate(
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
plt.title('Distribusi Periode Tanam berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Bulan Tanam', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)
plt.legend(
    title='Kabupaten',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    framealpha=0.9,
    fontsize=10
)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = bulan_by_kabupaten.values.max()

# Create dynamic intervals based on the maximum value
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

# Set y-axis ticks with dynamic interval
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)  # Explicitly set the y-axis limits

plt.tight_layout(pad=2.0)
plt.savefig('07_distribusi_bulan_tanam_kabupaten_petani_unik.png', dpi=300)

print("\nVisualisasi 1 selesai!")

# Visualisasi 2: Distribusi Luas Tanam berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data luas tanam
print("\nStatistik luas tanam (m2):")
print(df_periode['luas_tanam_m2'].describe())
print("\nUnique values luas tanam:")
print(sorted(df_periode['luas_tanam_m2'].unique()))

# Fungsi untuk kategorisasi luas tanam
def kategorisasi_luas_tanam(luas):
    if pd.isna(luas):
        return 'Tidak Diketahui'
    elif luas <= 1000:
        return '≤ 1.000 m²'
    elif luas <= 2000:
        return '1.001-2.000 m²'
    elif luas <= 3000:
        return '2.001-3.000 m²'
    elif luas <= 5000:
        return '3.001-5.000 m²'
    elif luas <= 7000:
        return '5.001-7.000 m²'
    else:
        return '> 7.000 m²'
# Membuat kolom kategori luas tanam
df_periode['kategori_luas_tanam'] = df_periode['luas_tanam_m2'].apply(kategorisasi_luas_tanam)

# Debug: Cek distribusi kategori luas tanam
print("\nDistribusi kategori luas tanam:")
print(df_periode['kategori_luas_tanam'].value_counts())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Ambil data petani unik dan luas tanam mereka
petani_luas_tanam = df_periode[['id_petani', 'kabupaten', 'kategori_luas_tanam']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_luas_tanam)}")
print(f"Distribusi kategori luas tanam per petani unik:")
print(petani_luas_tanam['kategori_luas_tanam'].value_counts())

# Membuat crosstab untuk luas tanam berdasarkan kabupaten (data petani unik)
luas_by_kabupaten = pd.crosstab(petani_luas_tanam['kabupaten'], petani_luas_tanam['kategori_luas_tanam'])

# Urutkan kolom berdasarkan urutan luas
luas_order = ['≤ 1.000 m²', '1.001-2.000 m²', '2.001-3.000 m²', '3.001-5.000 m²', '5.001-7.000 m²', '> 7.000 m²', 'Tidak Diketahui']
available_luas = [l for l in luas_order if l in luas_by_kabupaten.columns]
luas_by_kabupaten = luas_by_kabupaten[available_luas]

# Gunakan warna gradasi untuk ukuran lahan
colors = [COLOR_PALETTE['green'], COLOR_PALETTE['yellow'], COLOR_PALETTE['orange'], 
          COLOR_PALETTE['red'], '#8B0000', COLOR_PALETTE['gray']]

# Membuat plot
ax = luas_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_luas)]
)

# Tambahkan label jumlah pada setiap bar
# Calculate total count first
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

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
plt.title('Distribusi Luas Tanam berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Changed from 'Jumlah Periode Tanam'
plt.xticks(rotation=45)
#plt.legend(title='Luas Tanam (m²)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(
    title='Luas Tanam (m²)',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    framealpha=0.9,
    fontsize=10
)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set y-axis
y_max = luas_by_kabupaten.values.max()
y_ticks = np.arange(0, min(y_max + 10, 50), 5)
plt.yticks(y_ticks)
plt.ylim(0, min(y_max + 5, 45))

plt.tight_layout()
plt.savefig('08_distribusi_luas_tanam_kabupaten_petani_unik.png', dpi=300)

print("\nVisualisasi 2 selesai!")

# Visualisasi 3: Distribusi Periode Panen berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data bulan panen
print("\nBulan panen yang tersedia:")
print(df_periode['bulan_panen'].value_counts())

# Fungsi untuk mengkategorikan bulan panen ke dalam range periode
def kategorisasi_periode_panen(bulan):
    if pd.isna(bulan):
        return 'Tidak Diketahui'
    
    bulan_str = str(bulan).strip().title()
    
    # Mapping bulan ke range periode (updated with des-feb)
    if bulan_str in ['Desember', 'Des', 'Januari', 'Jan', 'Februari', 'Feb']:
        return 'des - feb'
    elif bulan_str in ['Maret', 'Mar', 'April', 'Apr', 'Mei']:
        return 'mar - mei'
    elif bulan_str in ['Juni', 'Jun', 'Juli', 'Jul', 'Agustus', 'Ags', 'Aug']:
        return 'jun - agt'
    elif bulan_str in ['September', 'Sep', 'Oktober', 'Okt', 'Oct', 'November', 'Nov']:
        return 'sep - nov'
    else:
        return 'Tidak Diketahui'

# Kategorikan bulan panen ke dalam range periode
df_periode['periode_panen'] = df_periode['bulan_panen'].apply(kategorisasi_periode_panen)

# Debug: Cek hasil kategorisasi
print("\nPeriode panen setelah kategorisasi:")
print(df_periode['periode_panen'].value_counts())

# PENTING: Gunakan SEMUA data periode (TIDAK pakai drop_duplicates)
# Karena satu petani bisa punya 2 periode tanam
print(f"\nJumlah total periode panen: {len(df_periode)}")
print(f"Distribusi periode panen:")
print(df_periode['periode_panen'].value_counts())

# Membuat crosstab untuk periode panen berdasarkan kabupaten
# Kabupaten di sumbu X, periode panen sebagai kolom (legend)
panen_by_kabupaten = pd.crosstab(df_periode['kabupaten'], df_periode['periode_panen'])

# Urutkan kolom berdasarkan urutan periode (4 musim)
periode_order = ['des - feb', 'mar - mei', 'jun - agt', 'sep - nov']
available_periode = [p for p in periode_order if p in panen_by_kabupaten.columns]

# Tambahkan 'Tidak Diketahui' jika ada
if 'Tidak Diketahui' in panen_by_kabupaten.columns:
    available_periode.append('Tidak Diketahui')

print(f"\nPeriode panen yang akan ditampilkan: {available_periode}")

# Reorder dataframe berdasarkan urutan periode
if available_periode:
    panen_by_kabupaten = panen_by_kabupaten[available_periode]
else:
    print("Tidak ada data periode panen yang valid")

# Gunakan warna solid dari COLOR_PALETTE
colors = [COLOR_PALETTE['blue'], COLOR_PALETTE['orange'], COLOR_PALETTE['green'], 
          COLOR_PALETTE['purple'], COLOR_PALETTE['gray']]

# Membuat plot
ax = panen_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_periode)]
)

# Calculate total count untuk percentage calculation
total_count = df_periode['periode_panen'].count()

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
            y = height + 0.5
            
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
plt.title('Distribusi Periode Panen berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Periode Panen', fontsize=12)  # Changed label
plt.xticks(rotation=45)
plt.legend(title='Periode Panen', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = panen_by_kabupaten.values.max() if len(panen_by_kabupaten) > 0 else 10

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

# Set y-axis ticks with dynamic interval
y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

plt.tight_layout()
plt.savefig('09_distribusi_periode_panen_kabupaten.png', dpi=300)

print("\nVisualisasi 3 selesai!")
print(f"Total periode panen yang divisualisasikan: {total_count}")

# Visualisasi 4: Distribusi Luas Panen berdasarkan Kabupaten (Per Periode)
print("VISUALISASI 4: DISTRIBUSI LUAS PANEN BERDASARKAN KABUPATEN (PER PERIODE)")

plt.figure(figsize=(12, 6))

print("\nStatistik luas panen (m2):")
print(df_periode['luas_panen_m2'].describe())

def kategorisasi_luas_panen(luas):
    if pd.isna(luas):
        return 'Tidak Diketahui'
    elif luas <= 1000:
        return '≤ 1.000 m²'
    elif luas <= 2000:
        return '1.001-2.000 m²'
    elif luas <= 3000:
        return '2.001-3.000 m²'
    elif luas <= 5000:
        return '3.001-5.000 m²'
    elif luas <= 7000:
        return '5.001-7.000 m²'
    else:
        return '> 7.000 m²'

df_periode['kategori_luas_panen'] = df_periode['luas_panen_m2'].apply(kategorisasi_luas_panen)

print("\nDistribusi kategori luas panen (per periode):")
print(df_periode['kategori_luas_panen'].value_counts())
print(f"Jumlah total periode panen: {len(df_periode)}")

luas_panen_by_kabupaten = pd.crosstab(df_periode['kabupaten'], df_periode['kategori_luas_panen'])

luas_order = ['≤ 1.000 m²', '1.001-2.000 m²', '2.001-3.000 m²', '3.001-5.000 m²', '5.001-7.000 m²', '> 7.000 m²', 'Tidak Diketahui']
available_luas_panen = [l for l in luas_order if l in luas_panen_by_kabupaten.columns]
luas_panen_by_kabupaten = luas_panen_by_kabupaten[available_luas_panen]

colors = [COLOR_PALETTE['green'], COLOR_PALETTE['yellow'], COLOR_PALETTE['orange'], 
          COLOR_PALETTE['red'], '#8B0000', COLOR_PALETTE['gray']]

ax = luas_panen_by_kabupaten.plot(kind='bar', width=0.6, color=colors[:len(available_luas_panen)])

total_count = df_periode['kategori_luas_panen'].count()

for container in ax.containers:
    for rect in container:
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            x = rect.get_x() + rect.get_width() / 2
            y = height + 0.5
            ax.annotate(label_text, xy=(x, y), xytext=(0, 0), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', lw=1, alpha=0.9),
                        fontsize=9, fontweight='bold')

plt.title('Distribusi Luas Panen berdasarkan Kabupaten (Per Periode)', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Periode Panen', fontsize=12)
plt.xticks(rotation=45)
plt.legend(
    title='Luas Panen (m²)',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    framealpha=0.9,
    fontsize=10
)
plt.grid(axis='y', linestyle='--', alpha=0.7)

y_max = luas_panen_by_kabupaten.values.max() if not luas_panen_by_kabupaten.empty else 10
interval = 10 if y_max > 50 else 5 if y_max > 25 else 2
y_limit = y_max + 10
plt.yticks(np.arange(0, y_limit + interval, interval))
plt.ylim(0, y_limit)

plt.tight_layout()
plt.savefig('10_distribusi_luas_panen_kabupaten_per_periode.png', dpi=300)
print("\nVisualisasi 4 (Per Periode) selesai!")
print(f"Total periode yang divisualisasikan: {total_count}")

# Visualisasi 5: Distribusi Tingkat Produktivitas (Gunca) berdasarkan Kabupaten (Per Periode)
print("\n" + "="*80)
print("VISUALISASI 5: DISTRIBUSI PRODUKTIVITAS (GUNCA) BERDASARKAN KABUPATEN (PER PERIODE)")
print("="*80)

plt.figure(figsize=(12, 6))

df_periode['produksi_gunca_clean'] = df_periode['produksi_dalam_gunca'].apply(clean_gunca)

def kategorisasi_produksi_gunca(gunca):
    if pd.isna(gunca):
        return 'Tidak Diketahui'
    elif gunca <= 5:
        return 'Rendah (≤ 5)'
    elif gunca <= 10:
        return 'Sedang (6-10)'
    elif gunca <= 20:
        return 'Tinggi (11-20)'
    elif gunca <= 30:
        return 'Sangat Tinggi (21-30)'
    else:
        return 'Exceptional (> 30)'

df_periode['kategori_produksi_gunca'] = df_periode['produksi_gunca_clean'].apply(kategorisasi_produksi_gunca)

print("\nDistribusi kategori produksi gunca (per periode):")
print(df_periode['kategori_produksi_gunca'].value_counts())
print(f"Jumlah total periode: {len(df_periode)}")

produksi_by_kabupaten = pd.crosstab(df_periode['kabupaten'], df_periode['kategori_produksi_gunca'])

produktivitas_order = ['Rendah (≤ 5)', 'Sedang (6-10)', 'Tinggi (11-20)', 'Sangat Tinggi (21-30)', 'Exceptional (> 30)', 'Tidak Diketahui']
available_produktivitas = [p for p in produktivitas_order if p in produksi_by_kabupaten.columns]
produksi_by_kabupaten = produksi_by_kabupaten[available_produktivitas]

colors = [COLOR_PALETTE['blue'], COLOR_PALETTE['orange'], COLOR_PALETTE['yellow'], 
          COLOR_PALETTE['green'], '#1B5E20', COLOR_PALETTE['gray']]

ax = produksi_by_kabupaten.plot(kind='bar', width=0.6, color=colors[:len(available_produktivitas)])

total_count = df_periode['kategori_produksi_gunca'].count()

for container in ax.containers:
    for rect in container:
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            x = rect.get_x() + rect.get_width() / 2
            y = height + 0.5
            ax.annotate(label_text, xy=(x, y), xytext=(0, 0), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', lw=1, alpha=0.9),
                        fontsize=9, fontweight='bold')

plt.title('Distribusi Tingkat Produktivitas (Gunca) berdasarkan Kabupaten (Per Periode)', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Periode Panen', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Tingkat Produktivitas', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

y_max = produksi_by_kabupaten.values.max() if not produksi_by_kabupaten.empty else 10
interval = 10 if y_max > 50 else 5 if y_max > 25 else 2
y_limit = y_max + 10
plt.yticks(np.arange(0, y_limit + interval, interval))
plt.ylim(0, y_limit)

plt.tight_layout()
plt.savefig('11_distribusi_produktivitas_gunca_kabupaten_per_periode.png', dpi=300, bbox_inches='tight')
print("\nVisualisasi 5 (Per Periode) selesai!")
print(f"Total periode yang divisualisasikan: {total_count}")

# Visualisasi 6: Distribusi Range Pengeluaran berdasarkan Kabupaten (Per Periode)
print("\n" + "="*80)
print("VISUALISASI 6: DISTRIBUSI PENGELUARAN BERDASARKAN KABUPATEN (PER PERIODE)")
print("="*80)

plt.figure(figsize=(12, 6))

df_periode['pengeluaran_clean'] = df_periode['pengeluaran_rp'].apply(clean_pengeluaran_data)

def kategorisasi_pengeluaran(pengeluaran):
    if pd.isna(pengeluaran) or pengeluaran is None:
        return 'Tidak Diketahui'
    elif pengeluaran < 1000000:
        return '< 1 juta'
    elif pengeluaran < 3000000:
        return '1-3 juta'
    elif pengeluaran < 5000000:
        return '3-5 juta'
    elif pengeluaran < 8000000:
        return '5-8 juta'
    else:
        return '> 8 juta'

df_periode['kategori_pengeluaran'] = df_periode['pengeluaran_clean'].apply(kategorisasi_pengeluaran)

print("\nDistribusi kategori pengeluaran (per periode):")
print(df_periode['kategori_pengeluaran'].value_counts())
print(f"Jumlah total periode: {len(df_periode)}")

pengeluaran_by_kabupaten = pd.crosstab(df_periode['kabupaten'], df_periode['kategori_pengeluaran'])

pengeluaran_order = ['< 1 juta', '1-3 juta', '3-5 juta', '5-8 juta', '> 8 juta', 'Tidak Diketahui']
available_pengeluaran = [p for p in pengeluaran_order if p in pengeluaran_by_kabupaten.columns]
pengeluaran_by_kabupaten = pengeluaran_by_kabupaten[available_pengeluaran]

colors = [COLOR_PALETTE['green'], COLOR_PALETTE['yellow'], COLOR_PALETTE['orange'], 
          COLOR_PALETTE['red'], '#8B0000', COLOR_PALETTE['gray']]

ax = pengeluaran_by_kabupaten.plot(kind='bar', width=0.6, color=colors[:len(available_pengeluaran)])

total_count = df_periode['kategori_pengeluaran'].count()

for container in ax.containers:
    for rect in container:
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            x = rect.get_x() + rect.get_width() / 2
            y = height + 0.5
            ax.annotate(label_text, xy=(x, y), xytext=(0, 0), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', lw=1, alpha=0.9),
                        fontsize=9, fontweight='bold')

plt.title('Distribusi Range Pengeluaran berdasarkan Kabupaten (Per Periode)', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Periode Panen', fontsize=12)
plt.xticks(rotation=45)
plt.legend(
    title='Range Pengeluaran',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    framealpha=0.9,
    fontsize=10
)
plt.grid(axis='y', linestyle='--', alpha=0.7)

y_max = pengeluaran_by_kabupaten.values.max() if not pengeluaran_by_kabupaten.empty else 10
interval = 10 if y_max > 50 else 5 if y_max > 25 else 2
y_limit = y_max + 10
plt.yticks(np.arange(0, y_limit + interval, interval))
plt.ylim(0, y_limit)

plt.tight_layout()
plt.savefig('12_distribusi_pengeluaran_kabupaten_per_periode.png', dpi=300, bbox_inches='tight')
print("\nVisualisasi 6 (Per Periode) selesai!")
print(f"Total periode yang divisualisasikan: {total_count}")

print("SEMUA VISUALISASI SELESAI!")

# Visualisasi 7: Distribusi Harga Jual Gabah per Kg berdasarkan Kabupaten (dengan Range)
plt.figure(figsize=(12, 6))

# Bersihkan dan konversi data harga jual ke numeric
def clean_harga_data_improved(value):
    if pd.isna(value):
        return None
    
    cleaned = str(value).strip()
    
    if cleaned == '' or cleaned.lower() == 'nan' or cleaned == '-':
        return None
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None

# Bersihkan data harga jual
df_periode['harga_jual_clean'] = df_periode['harga_jual_perkg'].apply(clean_harga_data_improved)

# Kategorisasi harga ke dalam range (Opsi B: Range Rp 1.500)
def kategorisasi_harga_range(harga):
    if pd.isna(harga) or harga is None:
        return 'Tidak Diketahui'
    elif harga <= 6500:
        return 'Rp 5.000 - 6.500'
    elif harga <= 8000:
        return 'Rp 6.501 - 8.000'
    else:
        return 'Rp 8.001 - 9.500'

# Buat kolom kategori range harga
df_periode['kategori_harga_range'] = df_periode['harga_jual_clean'].apply(kategorisasi_harga_range)

# Debug: Cek distribusi kategori harga
print("\nDistribusi kategori harga (dengan range):")
print(df_periode['kategori_harga_range'].value_counts())

# Ambil data petani unik dan kategori harga mereka
petani_harga_jual = df_periode[['id_petani', 'kabupaten', 'kategori_harga_range']].drop_duplicates(subset=['id_petani'])

# Debug
print(f"\nJumlah total petani unik: {len(petani_harga_jual)}")
print(f"Distribusi kategori harga per petani unik:")
print(petani_harga_jual['kategori_harga_range'].value_counts())

# Membuat crosstab untuk kategori harga berdasarkan kabupaten
harga_by_kabupaten = pd.crosstab(petani_harga_jual['kabupaten'], petani_harga_jual['kategori_harga_range'])

# Urutkan kolom berdasarkan urutan range harga
harga_order = ['Rp 5.000 - 6.500', 'Rp 6.501 - 8.000', 'Rp 8.001 - 9.500', 'Tidak Diketahui']
available_harga = [h for h in harga_order if h in harga_by_kabupaten.columns]
harga_by_kabupaten = harga_by_kabupaten[available_harga]

# Gunakan warna gradasi untuk range harga (dari hijau ke merah)
colors = [COLOR_PALETTE['green'], COLOR_PALETTE['orange'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
ax = harga_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_harga)]
)

# Calculate total count
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Clear existing labels
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5
            
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
plt.title('Distribusi Range Harga Jual Gabah per Kg berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)
plt.legend(
    title='Range Harga Gabah per Kg',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    framealpha=0.9,
    fontsize=10
)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = harga_by_kabupaten.values.max()

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

y_ticks = np.arange(0, y_limit + interval, interval)
plt.yticks(y_ticks)
plt.ylim(0, y_limit)

plt.tight_layout()
plt.savefig('13_distribusi_harga_jual_gabah_kabupaten_petani_unik_range.png', dpi=300, bbox_inches='tight')


# Visualisasi 8: Distribusi Status Pengelolaan berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data status pengelolaan
print("\nStatistik status pengelolaan:")
print(df_periode['status_pengelolaan'].describe())
print("\nUnique values status pengelolaan:")
print(df_periode['status_pengelolaan'].value_counts())

# Bersihkan data status pengelolaan jika ada nilai kosong
def clean_status_pengelolaan(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data status pengelolaan
df_periode['status_pengelolaan_clean'] = df_periode['status_pengelolaan'].apply(clean_status_pengelolaan)

# Debug: Cek distribusi status pengelolaan setelah cleaning
print("\nDistribusi status pengelolaan setelah cleaning:")
print(df_periode['status_pengelolaan_clean'].value_counts())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Ambil data petani unik dan status pengelolaan mereka
petani_status_pengelolaan = df_periode[['id_petani', 'kabupaten', 'status_pengelolaan_clean']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_status_pengelolaan)}")
print(f"Distribusi status pengelolaan per petani unik:")
print(petani_status_pengelolaan['status_pengelolaan_clean'].value_counts())

# Membuat crosstab untuk status pengelolaan berdasarkan kabupaten (data petani unik)
status_by_kabupaten = pd.crosstab(petani_status_pengelolaan['kabupaten'], petani_status_pengelolaan['status_pengelolaan_clean'])

# Urutkan kolom berdasarkan preferensi (Milik sendiri dulu, lalu Bagi hasil)
status_order = ['Milik sendiri', 'Bagi hasil', 'Tidak Diketahui']
available_status = [s for s in status_order if s in status_by_kabupaten.columns]
status_by_kabupaten = status_by_kabupaten[available_status]

print(f"\nStatus pengelolaan yang akan ditampilkan: {available_status}")

# Gunakan warna yang meaningful untuk status pengelolaan
# Hijau untuk milik sendiri (positif), orange untuk bagi hasil, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['green'], COLOR_PALETTE['orange'], COLOR_PALETTE['gray']]

# Membuat plot
ax = status_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_status)]
)

total_count = petani_status_pengelolaan['status_pengelolaan_clean'].count()
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
plt.title('Distribusi Status Pengelolaan Lahan berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Changed from 'Jumlah Periode'
plt.xticks(rotation=45)

plt.legend(title='Pengelolaan', 
           loc='center left',           # Position at the center left
           bbox_to_anchor=(1, 0.5),     # Place it outside to the right
           framealpha=0.9,              # More opaque background for readability
           ncol=1,                      # Single column since we have space
           fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = status_by_kabupaten.values.max()
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

# IMPROVED: Add more bottom margin to accommodate the centered legend
plt.subplots_adjust(bottom=0.25)

plt.savefig('14_distribusi_status_pengelolaan_kabupaten_petani_unik.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 8 selesai!")

# Visualisasi 9: Distribusi Hasil Panen Dijual berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data hasil panen dijual
print("\nStatistik hasil panen dijual:")
print(df_periode['hasil_panen_dijual'].describe())
print("\nUnique values hasil panen dijual:")
print(df_periode['hasil_panen_dijual'].value_counts())

# Bersihkan data hasil panen dijual jika ada nilai kosong
def clean_hasil_panen_dijual(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data hasil panen dijual
df_periode['hasil_panen_dijual_clean'] = df_periode['hasil_panen_dijual'].apply(clean_hasil_panen_dijual)

# Debug: Cek distribusi hasil panen dijual setelah cleaning
print("\nDistribusi hasil panen dijual setelah cleaning:")
print(df_periode['hasil_panen_dijual_clean'].value_counts())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Ambil data petani unik dan hasil panen dijual mereka
petani_hasil_panen = df_periode[['id_petani', 'kabupaten', 'hasil_panen_dijual_clean']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_hasil_panen)}")
print(f"Distribusi hasil panen dijual per petani unik:")
print(petani_hasil_panen['hasil_panen_dijual_clean'].value_counts())

# Membuat crosstab untuk hasil panen dijual berdasarkan kabupaten (data petani unik)
hasil_by_kabupaten = pd.crosstab(petani_hasil_panen['kabupaten'], petani_hasil_panen['hasil_panen_dijual_clean'])

# Urutkan kolom berdasarkan preferensi (Ya seluruhnya, Ya sebagian, Tidak)
hasil_order = ['Ya seluruhnya', 'Ya sebagian', 'Tidak', 'Tidak Diketahui']
available_hasil = [h for h in hasil_order if h in hasil_by_kabupaten.columns]
hasil_by_kabupaten = hasil_by_kabupaten[available_hasil]

print(f"\nHasil panen dijual yang akan ditampilkan: {available_hasil}")

# Gunakan warna yang meaningful untuk hasil panen dijual
# Hijau tua untuk ya seluruhnya, hijau muda untuk ya sebagian, merah untuk tidak, abu-abu untuk tidak diketahui
colors = ['#1B5E20', COLOR_PALETTE['blue'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
ax = hasil_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_hasil)]
)

total_count = petani_hasil_panen['hasil_panen_dijual_clean'].count()

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
plt.title('Distribusi Hasil Panen Dijual berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Changed from 'Jumlah Periode'
plt.xticks(rotation=45)

plt.legend(title='Hasil Panen Dijual', 
           bbox_to_anchor=(1.05, 1),     # Position to the right of the plot
           loc='upper left',             # Anchor at the upper left of the bbox
           framealpha=0.8,               # Semi-transparent background
           ncol=1,                       # One column for better readability
           fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = hasil_by_kabupaten.values.max()
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

plt.tight_layout()
plt.savefig('15_distribusi_hasil_panen_dijual_kabupaten_petani_unik.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 9 selesai!")

# Visualisasi 10: Distribusi Frekuensi Tanam Padi dalam Setahun berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data jumlah tanam padi dalam setahun
print("\nStatistik jumlah tanam padi dalam setahun:")
print(df_periode['jml_tanam_padi_1th'].describe())
print("\nUnique values jumlah tanam padi dalam setahun:")
print(df_periode['jml_tanam_padi_1th'].value_counts().sort_index())

# Bersihkan data jumlah tanam
def clean_jml_tanam(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    try:
        # Convert to integer
        jml = int(float(value))
        return jml
    except (ValueError, TypeError):
        return 'Tidak Diketahui'

# Bersihkan data jumlah tanam
df_periode['jml_tanam_clean'] = df_periode['jml_tanam_padi_1th'].apply(clean_jml_tanam)

# Debug: Cek distribusi jumlah tanam setelah cleaning
print("\nDistribusi jumlah tanam setelah cleaning:")
print(df_periode['jml_tanam_clean'].value_counts().sort_index())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Karena kita ingin menghitung petani, bukan periode tanam

# Ambil data petani unik dan jumlah tanam mereka
petani_jml_tanam = df_periode[['id_petani', 'kabupaten', 'jml_tanam_clean']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_jml_tanam)}")
print(f"Distribusi jumlah tanam per petani unik:")
print(petani_jml_tanam['jml_tanam_clean'].value_counts().sort_index())

# Membuat crosstab untuk jumlah tanam berdasarkan kabupaten
jml_tanam_by_kabupaten = pd.crosstab(petani_jml_tanam['kabupaten'], petani_jml_tanam['jml_tanam_clean'])

# Urutkan kolom berdasarkan jumlah tanam (1, 2, 3, 4, 5)
# Convert kolom ke string untuk konsistensi jika ada 'Tidak Diketahui'
jml_tanam_by_kabupaten.columns = [str(col) for col in jml_tanam_by_kabupaten.columns]

# Tentukan urutan kolom
tanam_order = ['1', '2', '3', '4', '5', 'Tidak Diketahui']
available_tanam = [t for t in tanam_order if t in jml_tanam_by_kabupaten.columns]
jml_tanam_by_kabupaten = jml_tanam_by_kabupaten[available_tanam]

print(f"\nFrekuensi tanam yang akan ditampilkan: {available_tanam}")

# Gunakan warna yang bervariasi untuk jumlah tanam
# Gradasi dari kuning (1x) ke hijau tua (5x)
colors = [COLOR_PALETTE['yellow'], COLOR_PALETTE['lime'], COLOR_PALETTE['green'], 
          '#2E7D32', '#1B5E20', COLOR_PALETTE['gray']]
# Membuat plot
ax = jml_tanam_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_tanam)]
)

# Tambahkan label jumlah pada setiap bar
total_count = petani_jml_tanam['jml_tanam_clean'].count()

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
plt.title('Distribusi Frekuensi Tanam Padi dalam Setahun berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Note: Jumlah Petani, bukan Periode
plt.xticks(rotation=45)

plt.legend(title='Frekuensi Tanam per Tahun ', 
           bbox_to_anchor=(1.05, 1),     # Position to the right of the plot
           loc='upper left',             # Anchor at the upper left of the bbox
           framealpha=0.8,               # Semi-transparent background
           ncol=len(available_tanam),                       # One column for better readability
           fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = jml_tanam_by_kabupaten.values.max()
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

plt.tight_layout()
plt.savefig('16_distribusi_frekuensi_tanam_padi_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 10 selesai!")

# Visualisasi 11: Distribusi Jenis Lahan berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data jenis lahan
print("\nStatistik jenis lahan:")
print(df_periode['jenis_lahan'].describe())
print("\nUnique values jenis lahan:")
print(df_periode['jenis_lahan'].value_counts())

# Bersihkan dan validasi data jenis lahan
def clean_jenis_lahan(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    cleaned = str(value).strip()
    
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    # Validasi hanya 5 value yang diizinkan (petani hanya bisa pilih 1)
    valid_values = [
        'Sawah irigasi',
        'Sawah tadah hujan', 
        'Tegalan/ladang',
        'Kombinasi irigasi dan tadah hujan',
        'Kombinasi irigasi dan tegalan/ladang'
    ]
    
    if cleaned in valid_values:
        return cleaned
    else:
        return 'Tidak Diketahui'

# Bersihkan data jenis lahan
df_periode['jenis_lahan_clean'] = df_periode['jenis_lahan'].apply(clean_jenis_lahan)

# Debug: Cek distribusi jenis lahan setelah cleaning
print("\nDistribusi jenis lahan setelah cleaning:")
print(df_periode['jenis_lahan_clean'].value_counts())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Karena kita ingin menghitung petani, bukan periode tanam
petani_jenis_lahan = df_periode[['id_petani', 'kabupaten', 'jenis_lahan_clean']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_jenis_lahan)}")
print(f"Distribusi jenis lahan per petani unik:")
print(petani_jenis_lahan['jenis_lahan_clean'].value_counts())

# Membuat crosstab untuk jenis lahan berdasarkan kabupaten
lahan_by_kabupaten = pd.crosstab(petani_jenis_lahan['kabupaten'], petani_jenis_lahan['jenis_lahan_clean'])

# Urutkan kolom berdasarkan 5 value valid + tidak diketahui (HURUF KECIL!)
lahan_order = [
    'Sawah irigasi', 
    'Sawah tadah hujan', 
    'Tegalan/ladang', 
    'Kombinasi irigasi dan tadah hujan',
    'Kombinasi irigasi dan tegalan/ladang',
    'Tidak Diketahui'
]
available_lahan = [l for l in lahan_order if l in lahan_by_kabupaten.columns]

print(f"\nJenis lahan yang akan ditampilkan: {available_lahan}")
print(f"Jumlah kolom yang tersedia: {len(available_lahan)}")

# Filter hanya kolom yang tersedia
if available_lahan:
    lahan_by_kabupaten = lahan_by_kabupaten[available_lahan]
else:
    print("\n⚠️ WARNING: Tidak ada kolom yang match!")
    print(f"Kolom yang ada: {lahan_by_kabupaten.columns.tolist()}")

colors = [COLOR_PALETTE['blue'], COLOR_PALETTE['green'], COLOR_PALETTE['orange'], 
          COLOR_PALETTE['purple'], COLOR_PALETTE['cyan'], COLOR_PALETTE['gray']]

# DEBUG: Print sebelum plot
print(f"\n=== DEBUG INFO ===")
print(f"lahan_by_kabupaten shape: {lahan_by_kabupaten.shape}")
print(f"lahan_by_kabupaten empty? {lahan_by_kabupaten.empty}")
print(f"available_lahan: {available_lahan}")
print(f"DataFrame:\n{lahan_by_kabupaten}")
print("="*50)

# LANGSUNG PLOT - JANGAN PAKAI IF
ax = lahan_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_lahan)]
)

total_count = petani_jenis_lahan['jenis_lahan_clean'].count()

# Clear existing labels
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            label_text = f'{int(height)}'
            
            # Position for the annotation
            x = rect.get_x() + rect.get_width()/2
            y = height + 0.5
            
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
plt.title('Distribusi Jenis Lahan berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)

# Place legend outside the plot to the right to avoid covering bars
plt.legend(
    title='Jenis Lahan',
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    framealpha=0.9,
    fontsize=9
)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = lahan_by_kabupaten.values.max()
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

plt.tight_layout()
plt.savefig('17_distribusi_jenis_lahan_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 11 selesai!")

# Visualisasi 12: Distribusi Kecukupan Air berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data kecukupan air
print("\nStatistik kecukupan air:")
print(df_periode['air_multi_tanam'].describe())
print("\nUnique values kecukupan air:")
print(df_periode['air_multi_tanam'].value_counts())

# Bersihkan data kecukupan air
def clean_kecukupan_air(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data kecukupan air
df_periode['air_clean'] = df_periode['air_multi_tanam'].apply(clean_kecukupan_air)

# Debug: Cek distribusi kecukupan air setelah cleaning
print("\nDistribusi kecukupan air setelah cleaning:")
print(df_periode['air_clean'].value_counts())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Karena kita ingin menghitung petani, bukan periode tanam
petani_air = df_periode[['id_petani', 'kabupaten', 'air_clean']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_air)}")
print(f"Distribusi kecukupan air per petani unik:")
print(petani_air['air_clean'].value_counts())

# Membuat crosstab untuk kecukupan air berdasarkan kabupaten
air_by_kabupaten = pd.crosstab(petani_air['kabupaten'], petani_air['air_clean'])

# Urutkan kolom berdasarkan preferensi (Mencukupi dulu, lalu Tidak Mencukupi)
air_order = ['Mencukupi', 'Tidak Mencukupi', 'Tidak Diketahui']
available_air = [a for a in air_order if a in air_by_kabupaten.columns]
air_by_kabupaten = air_by_kabupaten[available_air]

print(f"\nStatus kecukupan air yang akan ditampilkan: {available_air}")

# Gunakan warna yang meaningful untuk kecukupan air
# Biru untuk mencukupi, merah untuk tidak mencukupi, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['blue'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]
# Membuat plot
ax = air_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_air)]
)

total_count = petani_air['air_clean'].count()

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
plt.title('Distribusi Kecukupan Air untuk Penanaman Padi berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Note: Jumlah Petani, bukan Periode
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Kecukupan Air', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = air_by_kabupaten.values.max()
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

plt.savefig('18_distribusi_kecukupan_air_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 12 selesai!")

# Visualisasi 13: Distribusi Penggunaan Pupuk berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data penggunaan pupuk
print("\nStatistik penggunaan pupuk:")
print(df_periode['pemupukan'].describe())
print("\nUnique values penggunaan pupuk:")
print(df_periode['pemupukan'].value_counts())

# Bersihkan data penggunaan pupuk
def clean_pemupukan(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    return cleaned

# Bersihkan data penggunaan pupuk
df_periode['pemupukan_clean'] = df_periode['pemupukan'].apply(clean_pemupukan)

# Debug: Cek distribusi penggunaan pupuk setelah cleaning
print("\nDistribusi penggunaan pupuk setelah cleaning:")
print(df_periode['pemupukan_clean'].value_counts())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Karena kita ingin menghitung petani, bukan periode tanam
petani_pemupukan = df_periode[['id_petani', 'kabupaten', 'pemupukan_clean']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_pemupukan)}")
print(f"Distribusi penggunaan pupuk per petani unik:")
print(petani_pemupukan['pemupukan_clean'].value_counts())

# Membuat crosstab untuk penggunaan pupuk berdasarkan kabupaten
pemupukan_by_kabupaten = pd.crosstab(petani_pemupukan['kabupaten'], petani_pemupukan['pemupukan_clean'])

# Urutkan kolom berdasarkan preferensi (Ada pemupukan dulu, lalu Tidak ada pemupukan)
pemupukan_order = ['Ada pemupukan', 'Tidak ada pemupukan', 'Tidak Diketahui']
available_pemupukan = [p for p in pemupukan_order if p in pemupukan_by_kabupaten.columns]
pemupukan_by_kabupaten = pemupukan_by_kabupaten[available_pemupukan]

print(f"\nStatus penggunaan pupuk yang akan ditampilkan: {available_pemupukan}")

# Gunakan warna yang meaningful untuk penggunaan pupuk
# Hijau untuk ada pemupukan, merah untuk tidak ada pemupukan, abu-abu untuk tidak diketahui
colors = [COLOR_PALETTE['blue'], COLOR_PALETTE['red'], COLOR_PALETTE['gray']]

# Membuat plot
ax = pemupukan_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_pemupukan)]
)

total_count = petani_pemupukan['pemupukan_clean'].count()

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
plt.title('Distribusi Penggunaan Pupuk berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Note: Jumlah Petani, bukan Periode
plt.xticks(rotation=45)

# Place legend outside the plot to the right
plt.legend(title='Penggunaan Pupuk', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=10)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = pemupukan_by_kabupaten.values.max()
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

plt.savefig('19_distribusi_penggunaan_pupuk_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 13 selesai!")

# Visualisasi 14: Distribusi Cara Pengendalian Gulma berdasarkan Kabupaten
plt.figure(figsize=(14, 6))  # Wider figure to accommodate the legend

# Debug: Cek data cara pengendalian gulma
print("\nStatistik cara pengendalian gulma:")
print(df_periode['cara_pengendalian_gulma'].describe())
print("\nUnique values cara pengendalian gulma:")
print(df_periode['cara_pengendalian_gulma'].value_counts())

# Bersihkan dan standardisasi data cara pengendalian gulma
def clean_pengendalian_gulma(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    # Standardize similar methods
    if 'Herbisida/Mencabut rumput' in cleaned or 'Herbisida dan Mencabut rumput' in cleaned:
        return 'Herbisida dan Mencabut rumput'
    
    if 'Mencabut rumbut dan Menyemprot rumput' in cleaned or 'Mencabut rumput dan Menyemprot rumput' in cleaned:
        return 'Mencabut rumput dan Menyemprot'
    
    if 'Tidak Ada' in cleaned or cleaned.lower() == 'tidak':
        return 'Tidak Ada'
    
    return cleaned

# Bersihkan data cara pengendalian gulma
df_periode['gulma_clean'] = df_periode['cara_pengendalian_gulma'].apply(clean_pengendalian_gulma)

# Debug: Cek distribusi cara pengendalian gulma setelah cleaning
print("\nDistribusi cara pengendalian gulma setelah cleaning:")
print(df_periode['gulma_clean'].value_counts())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Karena kita ingin menghitung petani, bukan periode tanam
petani_gulma = df_periode[['id_petani', 'kabupaten', 'gulma_clean']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_gulma)}")
print(f"Distribusi cara pengendalian gulma per petani unik:")
print(petani_gulma['gulma_clean'].value_counts())

# Membuat crosstab untuk cara pengendalian gulma berdasarkan kabupaten
gulma_by_kabupaten = pd.crosstab(petani_gulma['kabupaten'], petani_gulma['gulma_clean'])

# Urutkan kolom berdasarkan preferensi
gulma_order = [
    'Mencabut rumput', 
    'Herbisida', 
    'Herbisida dan Mencabut rumput',
    'Menyemprot rumput',
    'Mencabut rumput dan Menyemprot',
    'Mencabut rumput dan Membuang keong',
    'Tidak Ada',
    'Tidak Diketahui'
]
available_gulma = [g for g in gulma_order if g in gulma_by_kabupaten.columns]
gulma_by_kabupaten = gulma_by_kabupaten[available_gulma]

print(f"\nCara pengendalian gulma yang akan ditampilkan: {available_gulma}")

# Gunakan warna yang bervariasi untuk cara pengendalian gulma
# Palette yang membedakan antar metode dengan jelas
colors = [COLOR_PALETTE['green'], COLOR_PALETTE['pink'], COLOR_PALETTE['purple'], 
          COLOR_PALETTE['blue'], COLOR_PALETTE['amber'], COLOR_PALETTE['brown'], 
          COLOR_PALETTE['red'], COLOR_PALETTE['gray']]
# Membuat plot
ax = gulma_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_gulma)]
)

total_count = petani_gulma['gulma_clean'].count()

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
plt.title('Distribusi Cara Pengendalian Gulma berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Note: Jumlah Petani, bukan Periode
plt.xticks(rotation=45)

# Place legend outside the plot to the right with more space
plt.legend(title='Cara Pengendalian Gulma', 
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           framealpha=0.9,
           fontsize=9)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = gulma_by_kabupaten.values.max()
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

# Adjust figure layout to make room for the legend - need more space for longer labels
plt.tight_layout()
plt.subplots_adjust(right=0.7)  # Wider margin for the legend

plt.savefig('20_distribusi_pengendalian_gulma_kabupaten.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 14 selesai!")

# Visualisasi: Distribusi Jenis Pupuk berdasarkan Kabupaten (Pie Chart)

# Fungsi untuk normalisasi kombinasi pupuk
def normalize_pupuk_combination(value):
    """
    Normalisasi kombinasi pupuk agar urutan tidak mempengaruhi hasil
    """
    if pd.isna(value):
        return 'Tidak ada'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak ada'
    
    # Remove quotes if present
    cleaned = cleaned.replace('"', '').strip()
    
    # Split by comma
    pupuk_list = [p.strip() for p in cleaned.split(',')]
    
    # Remove empty strings
    pupuk_list = [p for p in pupuk_list if p]
    
    # Define standard order for sorting
    standard_order = [
        'Pupuk urea',
        'NPK',
        'Pupuk SP-36',
        'KCL',
        'Pupuk Kandang',
        'Pupuk kompos',
        'Phonska',
        'Mutiara',
        'ZA',
        'Tidak ada'
    ]
    
    # Sort pupuk_list based on standard_order
    def get_order_index(pupuk_name):
        try:
            return standard_order.index(pupuk_name)
        except ValueError:
            return 999  # Put unknown items at the end
    
    pupuk_list_sorted = sorted(pupuk_list, key=get_order_index)
    
    # Join back with comma and space
    normalized = ', '.join(pupuk_list_sorted)
    
    return normalized if normalized else 'Tidak ada'

# Bersihkan dan normalisasi data jenis pupuk
df_periode['jenis_pupuk_normalized'] = df_periode['jenis_pupuk'].apply(normalize_pupuk_combination)

# Debug: Cek distribusi jenis pupuk setelah normalisasi
print("\nDistribusi jenis pupuk setelah normalisasi:")
print(df_periode['jenis_pupuk_normalized'].value_counts())

# Ambil data petani unik dan jenis pupuk mereka
petani_jenis_pupuk = df_periode[['id_petani', 'kabupaten', 'jenis_pupuk_normalized']].drop_duplicates(subset=['id_petani'])

print(f"\nJumlah total petani unik: {len(petani_jenis_pupuk)}")

# Pilih kabupaten untuk divisualisasikan (contoh: Aceh Besar)
kabupaten_dipilih = 'Aceh Besar'

# Filter data untuk kabupaten yang dipilih
data_kabupaten = petani_jenis_pupuk[petani_jenis_pupuk['kabupaten'] == kabupaten_dipilih]

print(f"\nJumlah petani di {kabupaten_dipilih}: {len(data_kabupaten)}")
print(f"\nDistribusi jenis pupuk di {kabupaten_dipilih}:")
print(data_kabupaten['jenis_pupuk_normalized'].value_counts())

# Hitung frekuensi setiap kombinasi pupuk
pupuk_counts = data_kabupaten['jenis_pupuk_normalized'].value_counts()

# Buat pie chart
plt.figure(figsize=(12, 8))

# Siapkan data untuk pie chart
labels = pupuk_counts.index.tolist()
sizes = pupuk_counts.values.tolist()
total = sum(sizes)

# Siapkan warna solid dari COLOR_PALETTE
color_list = [
    COLOR_PALETTE['blue'],
    COLOR_PALETTE['red'],
    COLOR_PALETTE['cyan'],
    COLOR_PALETTE['orange'],
    COLOR_PALETTE['green'],
    COLOR_PALETTE['purple'],
    COLOR_PALETTE['yellow'],
    COLOR_PALETTE['pink'],
    COLOR_PALETTE['teal'],
    COLOR_PALETTE['indigo'],
    COLOR_PALETTE['lime'],
    COLOR_PALETTE['amber'],
    COLOR_PALETTE['brown'],
    COLOR_PALETTE['gray']
]

# Extend color list if needed
while len(color_list) < len(labels):
    color_list.extend(color_list)

colors = color_list[:len(labels)]

# Buat pie chart tanpa label (akan dibuat manual)
wedges, texts = plt.pie(
    sizes,
    labels=None,  # label manual, bukan di sini
    colors=colors,
    startangle=90,
    textprops={'fontsize': 10, 'weight': 'bold'}
)

# Tambahkan label jumlah di tengah tiap slice dengan bubble (frame)
for i, (wedge, size) in enumerate(zip(wedges, sizes)):
    # Hitung posisi tengah slice
    ang = (wedge.theta2 + wedge.theta1) / 2
    x = np.cos(np.deg2rad(ang)) * 0.6  # 0.6 untuk posisi di tengah radius
    y = np.sin(np.deg2rad(ang)) * 0.6
    
    # Tambahkan label jumlah dengan frame/bubble
    plt.annotate(
        f'{size}',
        xy=(x, y),
        xytext=(0, 0),
        textcoords='offset points',
        ha='center', va='center',
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

# Tambahkan legend di luar pie chart dengan nama kombinasi pupuk
plt.legend(
    labels,
    title='Jenis Pupuk',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=9,
    framealpha=0.9
)

# Judul
plt.title(f'Jenis Pupuk - Kabupaten {kabupaten_dipilih}', fontsize=14, weight='bold', pad=20)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

plt.tight_layout()
plt.savefig(f'21_jenis_pupuk_{kabupaten_dipilih.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')

print(f"\nVisualisasi Jenis Pupuk untuk {kabupaten_dipilih} selesai!")
print(f"Total petani: {total}")

# ===================================================================
# BONUS: Buat untuk semua kabupaten sekaligus
# ===================================================================

print("\n" + "="*80)
print("MEMBUAT PIE CHART UNTUK SEMUA KABUPATEN")
print("="*80)

# Daftar kabupaten unik
list_kabupaten = petani_jenis_pupuk['kabupaten'].unique()
list_kabupaten = [k for k in list_kabupaten if k != 'Tidak Diketahui']

print(f"\nKabupaten yang akan divisualisasikan: {list_kabupaten}")

# Loop untuk setiap kabupaten
for kabupaten in list_kabupaten:
    # Filter data untuk kabupaten ini
    data_kab = petani_jenis_pupuk[petani_jenis_pupuk['kabupaten'] == kabupaten]
    
    # Hitung frekuensi
    pupuk_counts_kab = data_kab['jenis_pupuk_normalized'].value_counts()
    
    # Skip jika tidak ada data
    if len(pupuk_counts_kab) == 0:
        print(f"Skipping {kabupaten} - tidak ada data")
        continue
    
    # Buat figure
    plt.figure(figsize=(12, 8))
    
    # Data untuk pie chart
    labels_kab = pupuk_counts_kab.index.tolist()
    sizes_kab = pupuk_counts_kab.values.tolist()
    total_kab = sum(sizes_kab)
    
    # Warna
    colors_kab = color_list[:len(labels_kab)]
    
    # Buat pie chart tanpa label (akan dibuat manual)
    wedges, texts = plt.pie(
        sizes_kab,
        labels=None,  # label manual via legend
        colors=colors_kab,
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'}
    )
    
    # Tambahkan label jumlah di tengah tiap slice dengan bubble
    for i, (wedge, size) in enumerate(zip(wedges, sizes_kab)):
        # Hitung posisi tengah slice
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(ang)) * 0.6  # 0.6 untuk posisi di tengah radius
        y = np.sin(np.deg2rad(ang)) * 0.6
        
        # Tambahkan label jumlah dengan frame/bubble
        plt.annotate(
            f'{size}',
            xy=(x, y),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center', va='center',
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
    
    # Tambahkan legend di luar pie chart
    plt.legend(
        labels_kab,
        title='Jenis Pupuk',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=9,
        framealpha=0.9
    )
    
    # Judul
    plt.title(f'Jenis Pupuk - Kabupaten {kabupaten}', fontsize=14, weight='bold', pad=20)
    plt.axis('equal')
    
    plt.tight_layout()
    
    # Simpan dengan nama file yang sesuai
    filename = f'99_jenis_pupuk_{kabupaten.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {kabupaten}: {total_kab} petani - {len(labels_kab)} kombinasi pupuk")


print(f"\nVisualisasi Jenis Pupuk untuk {kabupaten_dipilih} selesai!")
print(f"Total petani: {total}")


# Visualisasi: Distribusi Jenis Hama berdasarkan Kabupaten (Pie Chart)

# Fungsi untuk normalisasi kombinasi hama
def normalize_hama_combination(value):
    """
    Normalisasi kombinasi hama agar urutan tidak mempengaruhi hasil
    """
    if pd.isna(value):
        return 'Tidak ada'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak ada'
    
    # Remove quotes if present
    cleaned = cleaned.replace('"', '').strip()
    
    # Split by comma
    hama_list = [h.strip() for h in cleaned.split(',')]
    
    # Remove empty strings
    hama_list = [h for h in hama_list if h]
    
    # Define standard order for sorting
    standard_order = [
        'Tikus',
        'Wereng',
        'Tulo/burung pipit',
        'penggerek batang padi',
        'walang sangit',
        'Hama putih',
        'Belalang',
        'Kepinding',
        'Siput Murbai/Keong',
        'Tidak ada'
    ]
    
    # Sort hama_list based on standard_order
    def get_order_index(hama_name):
        try:
            return standard_order.index(hama_name)
        except ValueError:
            return 999  # Put unknown items at the end
    
    hama_list_sorted = sorted(hama_list, key=get_order_index)
    
    # Join back with comma and space
    normalized = ', '.join(hama_list_sorted)
    
    return normalized if normalized else 'Tidak ada'

# Bersihkan dan normalisasi data jenis hama
df_periode['jenis_hama_normalized'] = df_periode['jenis_hama'].apply(normalize_hama_combination)

# Debug: Cek distribusi jenis hama setelah normalisasi
print("\nDistribusi jenis hama setelah normalisasi:")
print(df_periode['jenis_hama_normalized'].value_counts())

# Ambil data petani unik dan jenis hama mereka
petani_jenis_hama = df_periode[['id_petani', 'kabupaten', 'jenis_hama_normalized']].drop_duplicates(subset=['id_petani'])

print(f"\nJumlah total petani unik: {len(petani_jenis_hama)}")

# ===================================================================
# BUAT PIE CHART UNTUK SEMUA KABUPATEN
# ===================================================================

print("\n" + "="*80)
print("MEMBUAT PIE CHART JENIS HAMA UNTUK SEMUA KABUPATEN")
print("="*80)

# Daftar kabupaten unik
list_kabupaten = petani_jenis_hama['kabupaten'].unique()
list_kabupaten = [k for k in list_kabupaten if k != 'Tidak Diketahui']

print(f"\nKabupaten yang akan divisualisasikan: {list_kabupaten}")

# Siapkan warna solid dari COLOR_PALETTE
color_list = [
    COLOR_PALETTE['blue'],
    COLOR_PALETTE['red'],
    COLOR_PALETTE['cyan'],
    COLOR_PALETTE['orange'],
    COLOR_PALETTE['green'],
    COLOR_PALETTE['purple'],
    COLOR_PALETTE['yellow'],
    COLOR_PALETTE['pink'],
    COLOR_PALETTE['teal'],
    COLOR_PALETTE['indigo'],
    COLOR_PALETTE['lime'],
    COLOR_PALETTE['amber'],
    COLOR_PALETTE['brown'],
    COLOR_PALETTE['gray']
]

# Loop untuk setiap kabupaten
for kabupaten in list_kabupaten:
    # Filter data untuk kabupaten ini
    data_kab = petani_jenis_hama[petani_jenis_hama['kabupaten'] == kabupaten]
    
    # Hitung frekuensi
    hama_counts_kab = data_kab['jenis_hama_normalized'].value_counts()
    
    # Skip jika tidak ada data
    if len(hama_counts_kab) == 0:
        print(f"Skipping {kabupaten} - tidak ada data")
        continue
    
    # Buat figure
    plt.figure(figsize=(12, 8))
    
    # Data untuk pie chart
    labels_kab = hama_counts_kab.index.tolist()
    sizes_kab = hama_counts_kab.values.tolist()
    total_kab = sum(sizes_kab)
    
    # Warna
    # Extend color list if needed
    colors_kab = color_list.copy()
    while len(colors_kab) < len(labels_kab):
        colors_kab.extend(color_list)
    colors_kab = colors_kab[:len(labels_kab)]
    
    # Buat pie chart tanpa label (akan dibuat manual)
    wedges, texts = plt.pie(
        sizes_kab,
        labels=None,  # label manual via legend
        colors=colors_kab,
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'}
    )
    
    # Tambahkan label jumlah di tengah tiap slice dengan bubble
    for i, (wedge, size) in enumerate(zip(wedges, sizes_kab)):
        # Hitung posisi tengah slice
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(ang)) * 0.6  # 0.6 untuk posisi di tengah radius
        y = np.sin(np.deg2rad(ang)) * 0.6
        
        # Tambahkan label jumlah dengan frame/bubble
        plt.annotate(
            f'{size}',
            xy=(x, y),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center', va='center',
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
    
    # Tambahkan legend di luar pie chart
    plt.legend(
        labels_kab,
        title='Jenis Hama',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=9,
        framealpha=0.9
    )
    
    # Judul
    plt.title(f'Jenis Hama - Kabupaten {kabupaten}', fontsize=14, weight='bold', pad=20)
    plt.axis('equal')
    
    plt.tight_layout()
    
    # Simpan dengan nama file yang sesuai
    filename = f'22_jenis_hama_{kabupaten.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {kabupaten}: {total_kab} petani - {len(labels_kab)} kombinasi hama")
    

# Visualisasi: Distribusi Jenis Penyakit Padi berdasarkan Kabupaten (Pie Chart)

# Fungsi untuk normalisasi kombinasi penyakit
def normalize_penyakit_combination(value):
    """
    Normalisasi kombinasi penyakit agar urutan tidak mempengaruhi hasil
    """
    if pd.isna(value):
        return 'TIDAK ADA'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'TIDAK ADA'
    
    # Handle various forms of "tidak ada"
    if cleaned.upper() in ['TIDAK ADA', 'TIDAK', 'TIDAKADA']:
        return 'TIDAK ADA'
    
    # Remove quotes if present
    cleaned = cleaned.replace('"', '').strip()
    
    # Split by comma
    penyakit_list = [p.strip() for p in cleaned.split(',')]
    
    # Remove empty strings
    penyakit_list = [p for p in penyakit_list if p]
    
    # Define standard order for sorting
    standard_order = [
        'Blas',
        'Hawar daun bakteri',
        'Tungro',
        'Bercak daun',
        'Hawar pelepah daun',
        'Busuk batang',
        'TIDAK ADA'
    ]
    
    # Sort penyakit_list based on standard_order
    def get_order_index(penyakit_name):
        try:
            return standard_order.index(penyakit_name)
        except ValueError:
            # Try case-insensitive match
            for idx, std_name in enumerate(standard_order):
                if std_name.upper() == penyakit_name.upper():
                    return idx
            return 999  # Put unknown items at the end
    
    penyakit_list_sorted = sorted(penyakit_list, key=get_order_index)
    
    # Join back with comma and space
    normalized = ', '.join(penyakit_list_sorted)
    
    return normalized if normalized else 'TIDAK ADA'

# Bersihkan dan normalisasi data jenis penyakit
df_periode['jenis_penyakit_normalized'] = df_periode['penyakit_padi'].apply(normalize_penyakit_combination)
# Debug: Cek distribusi jenis penyakit setelah normalisasi
print("\nDistribusi jenis penyakit setelah normalisasi:")
print(df_periode['jenis_penyakit_normalized'].value_counts())

# Ambil data petani unik dan jenis penyakit mereka
petani_jenis_penyakit = df_periode[['id_petani', 'kabupaten', 'jenis_penyakit_normalized']].drop_duplicates(subset=['id_petani'])

print(f"\nJumlah total petani unik: {len(petani_jenis_penyakit)}")

# ===================================================================
# BUAT PIE CHART UNTUK SEMUA KABUPATEN
# ===================================================================

print("\n" + "="*80)
print("MEMBUAT PIE CHART JENIS PENYAKIT PADI UNTUK SEMUA KABUPATEN")
print("="*80)

# Daftar kabupaten unik
list_kabupaten = petani_jenis_penyakit['kabupaten'].unique()
list_kabupaten = [k for k in list_kabupaten if k != 'Tidak Diketahui']

print(f"\nKabupaten yang akan divisualisasikan: {list_kabupaten}")

# Siapkan warna solid dari COLOR_PALETTE
color_list = [
    COLOR_PALETTE['blue'],
    COLOR_PALETTE['red'],
    COLOR_PALETTE['cyan'],
    COLOR_PALETTE['orange'],
    COLOR_PALETTE['green'],
    COLOR_PALETTE['purple'],
    COLOR_PALETTE['yellow'],
    COLOR_PALETTE['pink'],
    COLOR_PALETTE['teal'],
    COLOR_PALETTE['indigo'],
    COLOR_PALETTE['lime'],
    COLOR_PALETTE['amber'],
    COLOR_PALETTE['brown'],
    COLOR_PALETTE['gray']
]

# Loop untuk setiap kabupaten
for kabupaten in list_kabupaten:
    # Filter data untuk kabupaten ini
    data_kab = petani_jenis_penyakit[petani_jenis_penyakit['kabupaten'] == kabupaten]
    
    # Hitung frekuensi
    penyakit_counts_kab = data_kab['jenis_penyakit_normalized'].value_counts()
    
    # Skip jika tidak ada data
    if len(penyakit_counts_kab) == 0:
        print(f"Skipping {kabupaten} - tidak ada data")
        continue
    
    # Buat figure
    plt.figure(figsize=(12, 8))
    
    # Data untuk pie chart
    labels_kab = penyakit_counts_kab.index.tolist()
    sizes_kab = penyakit_counts_kab.values.tolist()
    total_kab = sum(sizes_kab)
    
    # Warna
    # Extend color list if needed
    colors_kab = color_list.copy()
    while len(colors_kab) < len(labels_kab):
        colors_kab.extend(color_list)
    colors_kab = colors_kab[:len(labels_kab)]
    
    # Buat pie chart tanpa label (akan dibuat manual)
    wedges, texts = plt.pie(
        sizes_kab,
        labels=None,  # label manual via legend
        colors=colors_kab,
        startangle=90,
        textprops={'fontsize': 10, 'weight': 'bold'}
    )
    
    # Tambahkan label jumlah di tengah tiap slice dengan bubble
    for i, (wedge, size) in enumerate(zip(wedges, sizes_kab)):
        # Hitung posisi tengah slice
        ang = (wedge.theta2 + wedge.theta1) / 2
        x = np.cos(np.deg2rad(ang)) * 0.6  # 0.6 untuk posisi di tengah radius
        y = np.sin(np.deg2rad(ang)) * 0.6
        
        # Tambahkan label jumlah dengan frame/bubble
        plt.annotate(
            f'{size}',
            xy=(x, y),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center', va='center',
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
    
    # Tambahkan legend di luar pie chart
    plt.legend(
        labels_kab,
        title='Jenis Penyakit Padi',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=9,
        framealpha=0.9
    )
    
    # Judul
    plt.title(f'Jenis Penyakit Padi - Kabupaten {kabupaten}', fontsize=14, weight='bold', pad=20)
    plt.axis('equal')
    
    plt.tight_layout()
    
    # Simpan dengan nama file yang sesuai
    filename = f'23_jenis_penyakit_{kabupaten.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ {kabupaten}: {total_kab} petani - {len(labels_kab)} kombinasi penyakit")