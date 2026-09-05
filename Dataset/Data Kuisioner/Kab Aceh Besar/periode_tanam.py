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
    'gray': '#757575',
}

# Load data
df = pd.read_csv("data kuisioner petani aceh besar_sheet2.csv")

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

# Fungsi untuk mapping kecamatan berdasarkan ID petani (khusus untuk Aceh Besar)
def get_kecamatan_aceh_besar(id_petani):
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
    
    # Mapping kecamatan untuk Aceh Besar (ID 1-50)
    if 1 <= id_num <= 20:
        return 'Montasik'
    elif 21 <= id_num <= 40:
        return 'Indrapuri'
    elif 41 <= id_num <= 50:
        return 'Darussalam'
    else:
        return 'Tidak Diketahui'

# Adjust column names based on CSV
id_col = 'id_petani'

# Add kabupaten column
df['kabupaten'] = df[id_col].apply(get_kabupaten)

# Add kecamatan column untuk Aceh Besar
df['kecamatan'] = df[id_col].apply(get_kecamatan_aceh_besar)

# Filter for Aceh Besar respondents only
df = df[df['kabupaten'] == 'Aceh Besar'].copy()

print(f"\nTotal data periode di Aceh Besar: {len(df)}")
print(f"\nDistribusi per kecamatan:")
print(df['kecamatan'].value_counts())

# ============================================================
# PLOT 4: DISTRIBUSI PERIODE TANAM PER KECAMATAN
# ============================================================

plt.figure(figsize=(12, 6))

# Debug: Cek data bulan tanam
print("\n=== PLOT 4: PERIODE TANAM PER KECAMATAN ===")
print("\nBulan tanam yang tersedia:")
print(df['bulan_tanam'].value_counts())

# Fungsi untuk mengkategorikan bulan tanam ke dalam range periode
def kategorisasi_periode_tanam(bulan):
    if pd.isna(bulan):
        return 'Tidak Diketahui'
    
    bulan_str = str(bulan).strip().lower()
    
    # Mapping bulan ke range periode
    if 'nov' in bulan_str or 'des' in bulan_str or 'jan' in bulan_str:
        return 'nov - jan'
    elif 'feb' in bulan_str or 'mar' in bulan_str or 'apr' in bulan_str:
        return 'feb - apr'
    elif 'mei' in bulan_str or 'jun' in bulan_str or 'jul' in bulan_str:
        return 'mei - jul'
    elif 'agt' in bulan_str or 'aug' in bulan_str or 'sep' in bulan_str or 'okt' in bulan_str:
        return 'agt - okt'
    else:
        return 'Tidak Diketahui'

# Kategorikan bulan tanam ke dalam range periode
df['periode_tanam'] = df['bulan_tanam'].apply(kategorisasi_periode_tanam)

# Debug: Cek hasil kategorisasi
print("\nPeriode tanam setelah kategorisasi:")
print(df['periode_tanam'].value_counts())

# PENTING: Gunakan SEMUA data periode (TIDAK pakai drop_duplicates)
# Karena satu petani bisa punya 2 periode tanam
print(f"\nJumlah total periode tanam: {len(df)}")
print(f"Distribusi periode tanam:")
print(df['periode_tanam'].value_counts())

# Membuat crosstab untuk periode tanam berdasarkan kecamatan
# Kecamatan di sumbu X, periode tanam sebagai kolom (legend)
tanam_by_kecamatan = pd.crosstab(df['kecamatan'], df['periode_tanam'])

# Urutkan kolom berdasarkan urutan periode (4 musim)
periode_order = ['nov - jan', 'feb - apr', 'mei - jul', 'agt - okt']
available_periode = [p for p in periode_order if p in tanam_by_kecamatan.columns]

# Tambahkan 'Tidak Diketahui' jika ada
if 'Tidak Diketahui' in tanam_by_kecamatan.columns:
    available_periode.append('Tidak Diketahui')

print(f"\nPeriode tanam yang akan ditampilkan: {available_periode}")

# Reorder dataframe berdasarkan urutan periode
if available_periode:
    tanam_by_kecamatan = tanam_by_kecamatan[available_periode]
else:
    print("Tidak ada data periode tanam yang valid")

# Urutkan kecamatan berdasarkan total periode tanam (descending)
tanam_by_kecamatan['total'] = tanam_by_kecamatan.sum(axis=1)
tanam_by_kecamatan = tanam_by_kecamatan.sort_values('total', ascending=False)
tanam_by_kecamatan = tanam_by_kecamatan.drop('total', axis=1)

# Gunakan warna solid dari COLOR_PALETTE
colors = [COLOR_PALETTE['blue'], COLOR_PALETTE['orange'], COLOR_PALETTE['green'], 
          COLOR_PALETTE['purple'], COLOR_PALETTE['gray']]

# Membuat plot
ax = tanam_by_kecamatan.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_periode)],
    ax=plt.gca()
)

# Add labels with count in frames
for container in ax.containers:
    for rect in container:
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
plt.xlabel('Kecamatan', fontsize=12, fontweight='bold')
plt.ylabel('Jumlah Periode Tanam', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')

# Place legend outside the plot to the right
plt.legend(
    title='Periode Tanam',
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    framealpha=0.9,
    fontsize=9,
    title_fontsize=10
)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Dynamic Y-axis scaling
y_max = tanam_by_kecamatan.values.max() if len(tanam_by_kecamatan) > 0 else 10
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
plt.savefig('04_distribusi_periode_tanam_kecamatan.png', dpi=300, bbox_inches='tight')