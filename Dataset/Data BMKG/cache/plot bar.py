import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Fungsi untuk mengkonversi format angka Indonesia ke float
def parse_indonesian_number(value):
    if value == "–" or value is None:
        return 0
    return float(value.replace('.', '').replace(',', '.'))

# Konfigurasi API
base_url = "https://webapi.bps.go.id/v1/api/interoperabilitas/datasource/simdasi"
params = {
    'id': 25,
    'id_tabel': 'd3ZjM280TU9FanlkdDRETUV5aVdndz09',
    'wilayah': 1100000,
    'key': 'd83593d2486e73d9e28f059008bcfdcc'
}

# Tahun yang akan diambil
years = range(2018, 2025)  # 2018-2024

# Dictionary untuk menyimpan data
all_data = []

# Target kabupaten yang ingin ditampilkan
target_kabupaten = ['Aceh Utara', 'Bireuen', 'Pidie', 'Aceh Besar', 'Aceh Jaya']

print("Mengambil data dari BPS API...")

# Ambil data untuk setiap tahun
for year in years:
    url = f"{base_url}/id/{params['id']}/tahun/{year}/id_tabel/{params['id_tabel']}/wilayah/{params['wilayah']}/key/{params['key']}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'OK' and len(data['data']) > 1:
            table_data = data['data'][1]['data']
            
            # Ambil data setiap kabupaten/kota
            for item in table_data:
                kabupaten = item['label']
                
                # Skip jika bukan kabupaten target
                if kabupaten not in target_kabupaten:
                    continue
                
                # Ambil produksi padi
                produksi_padi_str = item['variables']['zuxztj3b0i']['value']
                produksi_padi = parse_indonesian_number(produksi_padi_str)
                
                # Skip jika tidak ada data
                if produksi_padi == 0:
                    continue
                
                all_data.append({
                    'Tahun': year,
                    'Kabupaten': kabupaten,
                    'Produksi Padi (ton)': produksi_padi
                })
            
            print(f"✓ Data tahun {year} berhasil diambil")
        else:
            print(f"✗ Data tahun {year} tidak tersedia")
            
    except Exception as e:
        print(f"✗ Error saat mengambil data tahun {year}: {str(e)}")

# Konversi ke DataFrame
df = pd.DataFrame(all_data)

print(f"\nTotal data yang berhasil diambil: {len(df)} records")
print(f"Kabupaten/Kota: {df['Kabupaten'].nunique()}")
print(f"Tahun: {df['Tahun'].nunique()}")

# Buat visualisasi
if not df.empty:
    # Hitung rata-rata produksi per kabupaten
    avg_by_kab = df.groupby('Kabupaten')['Produksi Padi (ton)'].mean()
    
    # Filter hanya kabupaten target dan urutkan dari tinggi ke rendah
    avg_by_kab = avg_by_kab.reindex(target_kabupaten).dropna()
    avg_by_kab = avg_by_kab.sort_values(ascending=True)  # ascending=True untuk horizontal bar (bottom to top)
    
    # Buat grafik batang horizontal
    plt.figure(figsize=(12, 8))
    
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    bars = plt.barh(range(len(avg_by_kab)), avg_by_kab.values, 
                    color=colors[:len(avg_by_kab)], edgecolor='black', linewidth=1)
    
    # Kustomisasi plot
    plt.yticks(range(len(avg_by_kab)), avg_by_kab.index, fontsize=12, fontweight='bold')
    plt.xlabel('Rata-rata Produksi Padi (ton)', fontsize=14, fontweight='bold')
    plt.title('Rata-rata Produksi Padi di 5 Kabupaten Aceh\n(2018-2024)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Tambahkan label nilai pada batang
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + max(avg_by_kab.values) * 0.01, 
                bar.get_y() + bar.get_height()/2.,
                f'{width:,.0f}',
                ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Atur margin dan layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.2)
    
    # Simpan dan tampilkan plot
    plt.savefig('rata_rata_produksi_padi_5_kabupaten_sorted.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Tampilkan statistik (diurutkan dari tinggi ke rendah)
    print("\n" + "="*60)
    print("RATA-RATA PRODUKSI PADI 5 KABUPATEN ACEH (2018-2024)")
    print("="*60)
    sorted_desc = avg_by_kab.sort_values(ascending=False)
    for i, (kab, prod) in enumerate(sorted_desc.items(), 1):
        print(f"{i}. {kab:20s} : {prod:>12,.2f} ton")
    
    print(f"\nKabupaten dengan produksi tertinggi: {avg_by_kab.idxmax()} ({avg_by_kab.max():,.2f} ton)")
    print(f"Kabupaten dengan produksi terendah : {avg_by_kab.idxmin()} ({avg_by_kab.min():,.2f} ton)")

else:
    print("\nTidak ada data yang berhasil diambil. Periksa koneksi API dan parameter.")