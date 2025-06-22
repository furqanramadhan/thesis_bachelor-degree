import pandas as pd
import os
from pathlib import Path

def read_csv_file(file_path):
    """
    Membaca file CSV dengan error handling
    """
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Convert Date column to datetime for proper sorting
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            print(f"Warning: File {file_path} tidak ditemukan")
            return None
    except Exception as e:
        print(f"Error membaca file {file_path}: {e}")
        return None

def merge_location_data(csv_directory, location):
    """
    Menggabungkan semua file CSV untuk satu lokasi
    """
    print(f"Memproses lokasi: {location}")
    print(f"CSV Directory: {csv_directory}")
    
    # 1. Baca file RH sebagai base
    rh_file = os.path.join(csv_directory, f"rh{location.lower()}_dy.csv")

    base_df = read_csv_file(rh_file)
    
    if base_df is None:
        print(f"Error: File RH untuk lokasi {location} tidak dapat dibaca")
        return None
    
    # Pilih kolom yang dibutuhkan dari RH
    result_df = base_df[['Date', 'Year', 'Month', 'Day', 'RH']].copy()
    
    # 2. Gabungkan file SST
    sst_file = os.path.join(csv_directory, f"sst{location.lower()}_dy.csv")
    sst_df = read_csv_file(sst_file)
    if sst_df is not None:
        sst_df = sst_df[['Date', 'SST']]
        result_df = pd.merge(result_df, sst_df, on='Date', how='left')
    else:
        result_df['SST'] = None
    
    # 3. Gabungkan file RAD
    rad_file = os.path.join(csv_directory, f"rad{location.lower()}_dy.csv")
    rad_df = read_csv_file(rad_file)
    if rad_df is not None:
        rad_df = rad_df[['Date', 'SWRad']]
        result_df = pd.merge(result_df, rad_df, on='Date', how='left')
    else:
        result_df['SWRad'] = None
    
    # 4. Gabungkan file RAIN
    rain_file = os.path.join(csv_directory, f"rain{location.lower()}_dy.csv")
    rain_df = read_csv_file(rain_file)
    if rain_df is not None:
        rain_df = rain_df[['Date', 'Prec']]
        result_df = pd.merge(result_df, rain_df, on='Date', how='left')
    else:
        result_df['Prec'] = None
    
    # 5. Gabungkan file T (temperature)
    t_file = os.path.join(csv_directory, f"t{location.lower()}_dy.csv")
    t_df = read_csv_file(t_file)
    if t_df is not None:
        # Ambil semua kolom TEMP_*.0m, tapi tidak ambil SST dari file ini
        temp_columns = ['Date'] + [col for col in t_df.columns if col.startswith('TEMP_') and col.endswith('.0m')]
        t_df = t_df[temp_columns]
        result_df = pd.merge(result_df, t_df, on='Date', how='left')
    else:
        # Tambahkan kolom temperature kosong jika file tidak ada
        temp_cols = ['TEMP_10.0m', 'TEMP_20.0m', 'TEMP_40.0m', 'TEMP_60.0m', 
                    'TEMP_80.0m', 'TEMP_100.0m', 'TEMP_120.0m', 'TEMP_140.0m', 
                    'TEMP_180.0m', 'TEMP_300.0m', 'TEMP_500.0m']
        for col in temp_cols:
            result_df[col] = None
    
    # 6. Gabungkan file W (wind)
    w_file = os.path.join(csv_directory, f"w{location.lower()}_dy.csv")
    w_df = read_csv_file(w_file)
    if w_df is not None:
        w_df = w_df[['Date', 'UWND', 'VWND', 'WSPD', 'WDIR']]
        result_df = pd.merge(result_df, w_df, on='Date', how='left')
    else:
        # Tambahkan kolom wind kosong jika file tidak ada
        wind_cols = ['UWND', 'VWND', 'WSPD', 'WDIR']
        for col in wind_cols:
            result_df[col] = None
    
    # 7. Tambahkan kolom Location
    result_df['Location'] = location
    
    print(f"Lokasi {location} berhasil diproses: {len(result_df)} records")
    return result_df

def main():
    """
    Fungsi utama untuk menggabungkan semua data buoys
    """
    # Pengaturan lokasi dan direktori
    locations = ["0N90E", "4N90E", "8N90E"]
    base_dir = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys'
    
    # Cek apakah direktori ada
    if not os.path.exists(base_dir):
        print(f"Error: Direktori {base_dir} tidak ditemukan")
        return
    
    all_data = []
    
    # Proses setiap lokasi
    for location in locations:
        # Tentukan csv_directory untuk setiap lokasi
        csv_directory = f'{base_dir}/{location}/CSV'
        
        # Cek apakah direktori CSV untuk lokasi ini ada
        if not os.path.exists(csv_directory):
            print(f"Warning: Direktori CSV untuk lokasi {location} tidak ditemukan: {csv_directory}")
            continue
        
        location_data = merge_location_data(csv_directory, location)
        if location_data is not None:
            all_data.append(location_data)
    
    if not all_data:
        print("Error: Tidak ada data yang berhasil diproses")
        return
    
    # Gabungkan semua data lokasi
    print("\nMenggabungkan semua data lokasi...")
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Konversi Location ke string dan urutan kustom
    final_df['Location'] = final_df['Location'].astype(str)
    locations_custom = ["0N90E", "4N90E", "8N90E"]  # pastikan urutan sesuai kebutuhan
    final_df['Location'] = pd.Categorical(
        final_df['Location'], 
        categories=locations_custom, 
        ordered=True
    )
    
    # Sort berdasarkan Location dulu, kemudian Date dengan reset index
    final_df = final_df.sort_values(['Date', 'Location']).reset_index(drop=True)
    
    # Atur urutan kolom sesuai yang diinginkan
    column_order = [
        'Date', 'Year', 'Month', 'Day', 'SWRad', 'Prec', 'RH', 'SST',
        'TEMP_10.0m', 'TEMP_20.0m', 'TEMP_40.0m', 'TEMP_60.0m', 'TEMP_80.0m',
        'TEMP_100.0m', 'TEMP_120.0m', 'TEMP_140.0m', 'TEMP_180.0m', 
        'TEMP_300.0m', 'TEMP_500.0m', 'UWND', 'VWND', 'WSPD', 'WDIR', 'Location'
    ]
    
    # Pastikan semua kolom ada, jika tidak ada buat kolom kosong
    for col in column_order:
        if col not in final_df.columns:
            final_df[col] = None
    
    final_df = final_df[column_order]
    
    # Convert Date back to string format untuk output
    final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')
    
    # Simpan ke file CSV
    output_file = os.path.join(base_dir, 'CSV/Buoys_Data_All.csv')
    final_df.to_csv(output_file, index=False)
    
    print(f"\nData berhasil digabungkan dan disimpan ke: {output_file}")
    print(f"Total records: {len(final_df)}")
    print(f"Kolom yang tersedia: {list(final_df.columns)}")
    
    # Tampilkan ringkasan data per lokasi
    print("\nRingkasan data per lokasi:")
    for location in locations:
        count = len(final_df[final_df['Location'] == location])
        print(f"- {location}: {count} records")
    
    # Tampilkan beberapa baris pertama sebagai preview
    print("\nPreview data (5 baris pertama):")
    print(final_df.head())

if __name__ == "__main__":
    main()