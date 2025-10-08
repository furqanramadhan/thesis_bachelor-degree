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
    # Extract nomor dari ID (PTN001 -> 1)
    try:
        # Handle special cases first
        if id_petani in ['PTN130', 'PTN131', 'PTN132', 'PTN133', 'PTN134', 
                         'PTN200', 'PTN201', 'PTN202', 'PTN203', 'PTN204', 
                         'PTN205', 'PTN206', 'PTN207', 'PTN208', 'PTN209']:
            return 'Aceh Besar'
        
        # Process regular pattern IDs
        nomor = int(id_petani.replace('PTN', ''))
        if 1 <= nomor <= 35:
            return 'Aceh Besar'
        elif 36 <= nomor <= 77:
            return 'Aceh Jaya'
        elif 78 <= nomor <= 109:
            return 'Pidie'
        elif 110 <= nomor <= 129:
            return 'Aceh Utara'
        else:
            return 'Tidak Diketahui'
    except:
        return 'Tidak Diketahui'
df_periode['kabupaten'] = df_periode['id_petani'].apply(get_kabupaten)

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
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']

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
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
                
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
plt.legend(title='Kabupaten')
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
colors = ['#32CD32', '#FFD700', '#FF8C00', '#DC143C', '#8B0000', '#808080']

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
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
plt.legend(title='Luas Tanam (m²)', loc='upper right')
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

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Ambil data petani unik dan periode panen mereka
petani_periode_panen = df_periode[['id_petani', 'kabupaten', 'periode_panen']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_periode_panen)}")
print(f"Distribusi periode panen per petani unik:")
print(petani_periode_panen['periode_panen'].value_counts())

# Membuat crosstab untuk periode panen berdasarkan kabupaten (data petani unik)
# PENTING: Kabupaten di sumbu X, periode panen sebagai kolom
panen_by_kabupaten = pd.crosstab(petani_periode_panen['kabupaten'], petani_periode_panen['periode_panen'])

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

# Gunakan warna yang menarik untuk periode panen (4 warna berbeda)
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#808080']

# Membuat plot
ax = panen_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_periode)]
)

# Tambahkan label jumlah pada setiap bar
# Calculate total count for percentage calculation
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Clear existing labels
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
plt.title('Distribusi Periode Panen berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Changed from 'Jumlah Periode Panen'
plt.xticks(rotation=45)
plt.legend(title='Periode Panen')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set y-axis dengan interval yang sesuai
y_max = panen_by_kabupaten.values.max() if len(panen_by_kabupaten) > 0 else 10
y_ticks = np.arange(0, min(y_max + 10, 50), 5)
plt.yticks(y_ticks)
plt.ylim(0, min(y_max + 5, 45))

plt.tight_layout()
plt.savefig('09_distribusi_periode_panen_kabupaten_petani_unik.png', dpi=300)

print("\nVisualisasi 3 selesai!")

# Visualisasi 4: Distribusi Luas Panen berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data luas panen
print("\nStatistik luas panen (m2):")
print(df_periode['luas_panen_m2'].describe())
print("\nUnique values luas panen:")
print(sorted(df_periode['luas_panen_m2'].unique()))

# Fungsi untuk kategorisasi luas panen
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

# Membuat kolom kategori luas panen
df_periode['kategori_luas_panen'] = df_periode['luas_panen_m2'].apply(kategorisasi_luas_panen)

# Debug: Cek distribusi kategori luas panen
print("\nDistribusi kategori luas panen:")
print(df_periode['kategori_luas_panen'].value_counts())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Ambil data petani unik dan luas panen mereka
petani_luas_panen = df_periode[['id_petani', 'kabupaten', 'kategori_luas_panen']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_luas_panen)}")
print(f"Distribusi kategori luas panen per petani unik:")
print(petani_luas_panen['kategori_luas_panen'].value_counts())

# Membuat crosstab untuk luas panen berdasarkan kabupaten (data petani unik)
luas_panen_by_kabupaten = pd.crosstab(petani_luas_panen['kabupaten'], petani_luas_panen['kategori_luas_panen'])

# Urutkan kolom berdasarkan urutan luas
luas_order = ['≤ 1.000 m²', '1.001-2.000 m²', '2.001-3.000 m²', '3.001-5.000 m²', '5.001-7.000 m²', '> 7.000 m²', 'Tidak Diketahui']
available_luas_panen = [l for l in luas_order if l in luas_panen_by_kabupaten.columns]
luas_panen_by_kabupaten = luas_panen_by_kabupaten[available_luas_panen]

# Gunakan warna gradasi untuk ukuran lahan (sama seperti luas tanam)
colors = ['#32CD32', '#FFD700', '#FF8C00', '#DC143C', '#8B0000', '#808080']

# Membuat plot
ax = luas_panen_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_luas_panen)]
)

# Calculate total count for percentage calculation
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Clear existing labels
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
plt.title('Distribusi Luas Panen berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Changed from 'Jumlah Periode Panen'
plt.xticks(rotation=45)
plt.legend(title='Luas Panen (m²)', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set y-axis
y_max = luas_panen_by_kabupaten.values.max()
y_ticks = np.arange(0, min(y_max + 10, 50), 5)
plt.yticks(y_ticks)
plt.ylim(0, min(y_max + 5, 45))

plt.tight_layout()
plt.savefig('10_distribusi_luas_panen_kabupaten_petani_unik.png', dpi=300)

print("\nVisualisasi 4 selesai!")

# Visualisasi 5: Distribusi Produksi Gunca berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data produksi gunca
print("\nUnique values produksi gunca (raw):")
print(df_periode['produksi_dalam_gunca'].unique())

# Bersihkan data produksi gunca
def clean_gunca(value):
    if pd.isna(value) or value == '-' or value == '' or value is None:
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

# Terapkan pembersihan data
df_periode['produksi_gunca_clean'] = df_periode['produksi_dalam_gunca'].apply(clean_gunca)

# Debug: Cek hasil pembersihan data
print("\nStatistik produksi gunca setelah dibersihkan:")
print(df_periode['produksi_gunca_clean'].describe())
print("\nUnique values produksi gunca setelah dibersihkan:")
print(sorted(df_periode['produksi_gunca_clean'].dropna().unique()))

# Fungsi untuk kategorisasi produksi gunca berdasarkan tingkat produktivitas
def kategorisasi_produksi_gunca(gunca):
    if pd.isna(gunca) or gunca is None:
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

# Membuat kolom kategori produksi gunca menggunakan data yang sudah dibersihkan
df_periode['kategori_produksi_gunca'] = df_periode['produksi_gunca_clean'].apply(kategorisasi_produksi_gunca)

# Debug: Cek distribusi kategori produksi gunca
print("\nDistribusi kategori produksi gunca:")
print(df_periode['kategori_produksi_gunca'].value_counts())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Ambil data petani unik dan kategori produksi gunca mereka
petani_produksi_gunca = df_periode[['id_petani', 'kabupaten', 'kategori_produksi_gunca']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_produksi_gunca)}")
print(f"Distribusi kategori produksi gunca per petani unik:")
print(petani_produksi_gunca['kategori_produksi_gunca'].value_counts())

# Membuat crosstab untuk produksi gunca berdasarkan kabupaten (data petani unik)
produksi_by_kabupaten = pd.crosstab(petani_produksi_gunca['kabupaten'], petani_produksi_gunca['kategori_produksi_gunca'])

# Urutkan kolom berdasarkan urutan produktivitas
produktivitas_order = ['Rendah (≤ 5)', 'Sedang (6-10)', 
                      'Tinggi (11-20)', 'Sangat Tinggi (21-30)', 
                      'Exceptional (> 30)', 'Tidak Diketahui']
available_produktivitas = [p for p in produktivitas_order if p in produksi_by_kabupaten.columns]
produksi_by_kabupaten = produksi_by_kabupaten[available_produktivitas]

# Gunakan warna gradasi dari merah (rendah) ke hijau tua (exceptional)
colors = ['#DC143C', '#FF8C00', '#FFD700', '#32CD32', '#228B22', '#808080']

# Membuat plot
ax = produksi_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_produktivitas)]
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
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
plt.title('Distribusi Tingkat Produktivitas (Gunca) berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Changed from 'Jumlah Periode Tanam'
plt.xticks(rotation=45)
plt.legend(title='Tingkat Produktivitas', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set y-axis
y_max = produksi_by_kabupaten.values.max()
y_ticks = np.arange(0, min(y_max + 10, 50), 5)
plt.yticks(y_ticks)
plt.ylim(0, min(y_max + 5, 45))

plt.tight_layout()
plt.savefig('11_distribusi_produktivitas_gunca_kabupaten_petani_unik.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 5 selesai!")

# Visualisasi 6: Distribusi Pengeluaran berdasarkan Kabupaten
plt.figure(figsize=(12, 6))

# Debug: Cek data pengeluaran
print("\nStatistik pengeluaran (Rp):")
print(df_periode['pengeluaran_rp'].describe())
print("\nSample pengeluaran:")
print(df_periode['pengeluaran_rp'].head(10))

# Bersihkan dan konversi data pengeluaran ke numeric
def clean_pengeluaran_data(value):
    if pd.isna(value):
        return None
    try:
        # Konversi ke string dulu, hapus whitespace
        cleaned = str(value).strip()
        if cleaned == '' or cleaned.lower() == 'nan':
            return None
        
        # Hapus titik sebagai pemisah ribuan dan konversi ke float
        cleaned = cleaned.replace('.', '')
        return float(cleaned)
    except (ValueError, TypeError):
        return None

# Bersihkan data pengeluaran
df_periode['pengeluaran_clean'] = df_periode['pengeluaran_rp'].apply(clean_pengeluaran_data)

print("\nData pengeluaran setelah dibersihkan:")
print(df_periode['pengeluaran_clean'].describe())

# Fungsi untuk kategorisasi pengeluaran berdasarkan range biaya
def kategorisasi_pengeluaran(pengeluaran):
    if pd.isna(pengeluaran) or pengeluaran is None:
        return 'Tidak Diketahui'
    elif pengeluaran < 1000000:  # < 1 juta
        return '< 1 juta'
    elif pengeluaran < 3000000:  # 1-3 juta
        return '1-3 juta'
    elif pengeluaran < 5000000:  # 3-5 juta
        return '3-5 juta'
    elif pengeluaran < 8000000:  # 5-8 juta
        return '5-8 juta'
    else:  # >= 8 juta
        return '> 8 juta'

# Membuat kolom kategori pengeluaran menggunakan data yang sudah dibersihkan
df_periode['kategori_pengeluaran'] = df_periode['pengeluaran_clean'].apply(kategorisasi_pengeluaran)

# Debug: Cek distribusi kategori pengeluaran
print("\nDistribusi kategori pengeluaran:")
print(df_periode['kategori_pengeluaran'].value_counts())

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Ambil data petani unik dan kategori pengeluaran mereka
petani_pengeluaran = df_periode[['id_petani', 'kabupaten', 'kategori_pengeluaran']].drop_duplicates(subset=['id_petani'])

# Debug: Cek jumlah petani unik
print(f"\nJumlah total petani unik: {len(petani_pengeluaran)}")
print(f"Distribusi kategori pengeluaran per petani unik:")
print(petani_pengeluaran['kategori_pengeluaran'].value_counts())

# Membuat crosstab untuk pengeluaran berdasarkan kabupaten (data petani unik)
pengeluaran_by_kabupaten = pd.crosstab(petani_pengeluaran['kabupaten'], petani_pengeluaran['kategori_pengeluaran'])

# Urutkan kolom berdasarkan urutan pengeluaran
pengeluaran_order = ['< 1 juta', '1-3 juta', '3-5 juta', '5-8 juta', '> 8 juta', 'Tidak Diketahui']
available_pengeluaran = [p for p in pengeluaran_order if p in pengeluaran_by_kabupaten.columns]
pengeluaran_by_kabupaten = pengeluaran_by_kabupaten[available_pengeluaran]

# Gunakan warna gradasi dari hijau (rendah) ke merah (tinggi)
colors = ['#32CD32', '#FFD700', '#FF8C00', '#DC143C', '#8B0000', '#808080']

# Membuat plot
ax = pengeluaran_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_pengeluaran)]
)

total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Tambahkan label jumlah pada setiap bar
# Tambahkan label jumlah pada setiap bar
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
plt.title('Distribusi Range Pengeluaran berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Changed from 'Jumlah Periode Tanam'
plt.xticks(rotation=45)
plt.legend(title='Range Pengeluaran', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set y-axis
y_max = pengeluaran_by_kabupaten.values.max()
y_ticks = np.arange(0, min(y_max + 10, 50), 5)
plt.yticks(y_ticks)
plt.ylim(0, min(y_max + 5, 45))

plt.tight_layout()
plt.savefig('12_distribusi_pengeluaran_kabupaten_petani_unik.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 6 selesai!")

# Visualisasi 7: Distribusi Harga Jual Gabah per Kg berdasarkan Kabupaten
plt.figure(figsize=(14, 6))

# Bersihkan dan konversi data harga jual ke numeric (improved handling)
def clean_harga_data_improved(value):
    if pd.isna(value):
        return None
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string, 'nan', or dash explicitly
    if cleaned == '' or cleaned.lower() == 'nan' or cleaned == '-':
        return None
    
    try:
        # Direct conversion since data is already in numeric format (6500, 7000, etc.)
        return float(cleaned)
    except (ValueError, TypeError):
        return None

# Bersihkan data harga jual dengan fungsi yang diperbaiki
df_periode['harga_jual_clean'] = df_periode['harga_jual_perkg'].apply(clean_harga_data_improved)

# Format harga untuk display tanpa kategorisasi - langsung pakai nilai exact
def format_harga_display(harga):
    if pd.isna(harga) or harga is None:
        return 'Tidak Diketahui'
    else:
        return f'Rp {int(harga):,}'.replace(',', '.')

# Membuat kolom harga display menggunakan nilai exact
df_periode['harga_display'] = df_periode['harga_jual_clean'].apply(format_harga_display)

# Penting: Kita perlu mengambil nilai unik per petani, bukan per periode
# Ambil data petani unik dan harga jual mereka
petani_harga_jual = df_periode[['id_petani', 'kabupaten', 'harga_display']].drop_duplicates(subset=['id_petani'])

# Membuat crosstab untuk harga jual exact berdasarkan kabupaten (data petani unik)
harga_by_kabupaten = pd.crosstab(petani_harga_jual['kabupaten'], petani_harga_jual['harga_display'])

# Urutkan kolom berdasarkan nilai numerik harga (dari rendah ke tinggi)
def extract_numeric_value(harga_str):
    if harga_str == 'Tidak Diketahui':
        return 999999  # Put at the end
    try:
        # Extract numeric value from "Rp 6.500" format
        return float(harga_str.replace('Rp ', '').replace('.', ''))
    except:
        return 999999

# Sort columns by numeric value
sorted_columns = sorted(harga_by_kabupaten.columns, key=extract_numeric_value)

# Remove 'Tidak Diketahui' from the sorted columns
if 'Tidak Diketahui' in sorted_columns:
    sorted_columns.remove('Tidak Diketahui')

# Filter dataframe to only include the sorted columns (excluding 'Tidak Diketahui')
harga_by_kabupaten = harga_by_kabupaten[sorted_columns]

# Gunakan warna solid untuk variasi warna setiap harga exact
solid_colors = [
    '#2ECC71',  # Green - 5000 (terendah)
    '#58D68D',  # Light Green - 6000
    '#F4D03F',  # Yellow - 6500  
    '#F39C12',  # Orange - 6700
    '#E67E22',  # Dark Orange - 6900
    '#E74C3C',  # Red - 7000
    '#C0392B',  # Dark Red - 8000
    '#922B21'   # Very Dark Red - 8500 (tertinggi)
]

# If there are more columns than colors, cycle through the colors
n_columns = len(harga_by_kabupaten.columns)
color_list = [solid_colors[i % len(solid_colors)] for i in range(n_columns)]

# Membuat plot
ax = harga_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=color_list
)

# Calculate total count for percentage calculation
total_count = sum([rect.get_height() for container in ax.containers for rect in container])

# Clear existing labels
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
plt.title('Distribusi Harga Jual Gabah per Kg berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Gabah per Kg', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dynamic Y-axis scaling
y_max = harga_by_kabupaten.values.max()

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

plt.tight_layout()
plt.savefig('13_distribusi_harga_jual_gabah_kabupaten_petani_unik.png', dpi=300, bbox_inches='tight')

print("\nVisualisasi 7 (Fixed - No Categorization) selesai!")

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
colors = ['#32CD32', '#FF8C00', '#808080']

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
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
colors = ['#228B22', '#32CD32', '#DC143C', '#808080']

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
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
colors = ['#FFC107', '#8BC34A', '#4CAF50', '#388E3C', '#1B5E20', '#808080']

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
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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

# Bersihkan dan standardisasi data jenis lahan
def clean_jenis_lahan(value):
    if pd.isna(value):
        return 'Tidak Diketahui'
    
    # Convert to string and strip whitespace
    cleaned = str(value).strip()
    
    # Handle empty string or dash
    if cleaned == '' or cleaned == '-':
        return 'Tidak Diketahui'
    
    # Standardize common terms
    if cleaned.lower() in ['sawah irigasi', 'irigasi']:
        return 'Sawah Irigasi'
    elif cleaned.lower() in ['sawah tadah hujan', 'tadah hujan']:
        return 'Sawah Tadah Hujan'
    elif cleaned.lower() in ['tegalan', 'ladang', 'tegalan/ladang']:
        return 'Tegalan/Ladang'
    
    # Split by comma and clean each component
    if ',' in cleaned:
        # Split and strip each component
        components = [item.strip() for item in cleaned.split(',')]
        
        # Sort components to ensure consistent ordering regardless of input order
        components.sort()
        
        # Check for specific combinations
        if set(components) == set(['Sawah irigasi', 'Sawah tadah hujan']) or \
           set(components) == set(['Irigasi', 'Tadah hujan']):
            return 'Kombinasi Irigasi dan Tadah Hujan'
        elif set(components) == set(['Sawah irigasi', 'Tegalan/ladang']) or \
             set(components) == set(['Irigasi', 'Tegalan/ladang']):
            return 'Kombinasi Irigasi dan Tegalan'
        elif set(components) == set(['Sawah tadah hujan', 'Tegalan/ladang']) or \
             set(components) == set(['Tadah hujan', 'Tegalan/ladang']):
            return 'Kombinasi Tadah Hujan dan Tegalan'
        else:
            return 'Kombinasi Lahan Lainnya'
    
    return cleaned

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

# Urutkan kolom berdasarkan preferensi
lahan_order = [
    'Sawah Irigasi', 
    'Sawah Tadah Hujan', 
    'Tegalan/Ladang', 
    'Kombinasi Irigasi dan Tadah Hujan', 
    'Kombinasi Irigasi dan Tegalan', 
    'Kombinasi Tadah Hujan dan Tegalan',
    'Kombinasi Lahan Lainnya',
    'Tidak Diketahui'
]
available_lahan = [l for l in lahan_order if l in lahan_by_kabupaten.columns]
lahan_by_kabupaten = lahan_by_kabupaten[available_lahan]

print(f"\nJenis lahan yang akan ditampilkan: {available_lahan}")

colors = ['#FF1744',  # Bright Red
          '#00E676',  # Bright Green
          '#FF6D00',  # Bright Orange  
          '#AA00FF',  # Bright Purple
          '#00B0FF',  # Bright Blue
          '#FFEA00',  # Bright Yellow
          '#795548',  # Brown
          '#424242',  # Dark Grey
          '#E91E63']  # Hot Pink

# Membuat plot
ax = lahan_by_kabupaten.plot(
    kind='bar', 
    width=0.6,
    color=colors[:len(available_lahan)]
)

total_count = petani_jenis_lahan['jenis_lahan_clean'].count()
for container in ax.containers:
    ax.bar_label(container, labels=[''] * len(container), padding=5)

# Add labels with count and percentage in frames
for container in ax.containers:
    for i, rect in enumerate(container):
        height = rect.get_height()
        if height > 0:
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
plt.title('Distribusi Jenis Lahan berdasarkan Kabupaten', fontsize=14)
plt.xlabel('Kabupaten', fontsize=12)
plt.ylabel('Jumlah Petani', fontsize=12)  # Note: Jumlah Petani, bukan Periode
plt.xticks(rotation=45)

# Place legend outside the plot to the right to avoid covering bars
plt.legend(loc='upper right',
           bbox_to_anchor=(0.98, 0.98),  # Slight offset from the corner
           framealpha=0.9,
           ncol=1,
           fontsize=9)
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
colors = ['#1E88E5', '#E53935', '#808080']

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
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
colors = ['#4CAF50', '#E53935', '#808080']

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
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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
colors = ['#4CAF50', '#E91E63', '#9C27B0', '#2196F3', '#FF9800', '#795548', '#F44336', '#808080']

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
            percentage = (height / total_count) * 100
            label_text = f'{int(height)}\n{percentage:.1f}%'
            
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