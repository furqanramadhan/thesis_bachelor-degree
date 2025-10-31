import pandas as pd
import os
from pathlib import Path

# Base directory
base_dir = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data Buoys"

# Lokasi-lokasi buoy
locations = ['0N90E', '4N90E', '8N90E']

def load_and_merge_location(location_path):
    """
    Menggabungkan semua file CSV untuk satu lokasi buoy
    """
    # Define file mappings
    files = {
        'sst': f'sst{location_path.lower()}_dy.csv',
        'rain': f'rain{location_path.lower()}_dy.csv',
        'rh': f'rh{location_path.lower()}_dy.csv',
        'wind': f'w{location_path.lower()}_dy.csv',
        'rad': f'rad{location_path.lower()}_dy.csv'
    }
    
    csv_dir = Path(base_dir) / location_path / 'CSV'
    
    # Load SST sebagai base (karena biasanya paling lengkap)
    sst_file = csv_dir / files['sst']
    if not sst_file.exists():
        print(f"Warning: {sst_file} tidak ditemukan")
        return None
    
    df_combined = pd.read_csv(sst_file)
    df_combined['Date'] = pd.to_datetime(df_combined['Date'])
    
    # Merge Rain data
    rain_file = csv_dir / files['rain']
    if rain_file.exists():
        df_rain = pd.read_csv(rain_file)
        df_rain['Date'] = pd.to_datetime(df_rain['Date'])
        df_combined = df_combined.merge(
            df_rain[['Date', 'Prec']], 
            on='Date', 
            how='left'
        )
    else:
        print(f"Warning: {rain_file} tidak ditemukan")
        df_combined['Prec'] = None
    
    # Merge RH data
    rh_file = csv_dir / files['rh']
    if rh_file.exists():
        df_rh = pd.read_csv(rh_file)
        df_rh['Date'] = pd.to_datetime(df_rh['Date'])
        df_combined = df_combined.merge(
            df_rh[['Date', 'RH']], 
            on='Date', 
            how='left'
        )
    else:
        print(f"Warning: {rh_file} tidak ditemukan")
        df_combined['RH'] = None
    
    # Merge Wind data
    wind_file = csv_dir / files['wind']
    if wind_file.exists():
        df_wind = pd.read_csv(wind_file)
        df_wind['Date'] = pd.to_datetime(df_wind['Date'])
        df_combined = df_combined.merge(
            df_wind[['Date', 'WSPD']], 
            on='Date', 
            how='left'
        )
    else:
        print(f"Warning: {wind_file} tidak ditemukan")
        df_combined['WSPD'] = None
    
    # Merge Radiation data
    rad_file = csv_dir / files['rad']
    if rad_file.exists():
        df_rad = pd.read_csv(rad_file)
        df_rad['Date'] = pd.to_datetime(df_rad['Date'])
        df_combined = df_combined.merge(
            df_rad[['Date', 'SWRad']], 
            on='Date', 
            how='left'
        )
    else:
        print(f"Warning: {rad_file} tidak ditemukan")
        df_combined['SWRad'] = None
    
    # Pilih dan reorder kolom sesuai format yang diinginkan
    df_combined = df_combined[['Date', 'Year', 'Month', 'Day', 'SST', 'Prec', 'RH', 'WSPD', 'SWRad']]
    
    # Rename Date column untuk konsistensi
    df_combined.columns = ['Date', 'year', 'month', 'day', 'SST', 'Prec', 'RH', 'WSPD', 'SWRad']
    
    # Tambahkan kolom location
    df_combined['Location'] = location_path
    
    return df_combined

# Main process
all_data = []

for location in locations:
    print(f"\nMemproses lokasi: {location}")
    df_loc = load_and_merge_location(location)
    
    if df_loc is not None:
        print(f"  - Data loaded: {len(df_loc)} rows")
        all_data.append(df_loc)
    else:
        print(f"  - Gagal memuat data untuk {location}")

# Gabungkan semua lokasi
if all_data:
    df_final = pd.concat(all_data, ignore_index=True)
    
    # Hapus duplikat berdasarkan Date dan Location
    df_final = df_final.drop_duplicates(subset=['Date', 'Location'], keep='first')
    
    # Sort by Location dan Date
    df_final = df_final.sort_values(['Location', 'Date']).reset_index(drop=True)
    
    print(f"\n{'='*60}")
    print(f"Total data gabungan: {len(df_final)} rows")
    print(f"Rentang tanggal: {df_final['Date'].min()} s/d {df_final['Date'].max()}")
    print(f"Lokasi: {df_final['Location'].unique()}")
    print(f"\nContoh data:")
    print(df_final.head(10))
    
    # Simpan hasil per lokasi di direktori masing-masing
    print(f"\n{'='*60}")
    print("Menyimpan file per lokasi:")
    for location in locations:
        df_loc = df_final[df_final['Location'] == location].copy()
        df_loc = df_loc.drop(columns=['Location'])
        
        # Buat direktori output di lokasi masing-masing
        output_dir = Path(base_dir) / location / 'CSV' / 'COMBINED'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f'{location}.csv'
        df_loc.to_csv(output_file, index=False)
        print(f"✓ {location}: {output_file} ({len(df_loc)} rows)")
    
    # Statistik missing values
    print(f"\n{'='*60}")
    print("Missing values per kolom:")
    print(df_final.isnull().sum())
    
else:
    print("\nError: Tidak ada data yang berhasil dimuat!")