import os
import re
import numpy as np
import pandas as pd
import glob
from datetime import datetime

def convert_ascii_to_unified_csv(input_file, variable_type=None):
    """
    Mengkonversi file ASCII menjadi DataFrame dengan deteksi otomatis format.
    Khusus untuk temperature, hanya ambil 10m dan 20m depth.
    
    Parameters:
    input_file (str): Path ke file ASCII
    variable_type (str): Tipe variabel ('temp', 'sst', 'wind', 'rain', 'rh', 'rad')
    
    Returns:
    pandas.DataFrame: DataFrame dengan data yang sudah diproses
    """
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Auto-detect format jika variable_type tidak ditentukan
    if variable_type is None:
        variable_type = detect_variable_type(lines)
    
    print(f"📊 Memproses {os.path.basename(input_file)} sebagai tipe: {variable_type.upper()}")
    
    if variable_type == 'temp':
        return process_temperature_data(lines, input_file)
    elif variable_type == 'sst':
        return process_sst_data(lines, input_file)
    elif variable_type == 'wind':
        return process_wind_data(lines, input_file)
    else:  # rain, rh, rad
        return process_general_data(lines, input_file, variable_type)

def detect_variable_type(lines):
    """Deteksi otomatis tipe variabel berdasarkan konten file"""
    
    preview_text = ''.join(lines[:20]).upper()
    
    if 'DEPTH(M):' in preview_text and len([line for line in lines[:10] if 'DEPTH(M):' in line.upper()]):
        # Cek apakah ada multiple depth values
        for line in lines[:10]:
            if 'DEPTH(M):' in line.upper():
                depth_parts = line.split(':')[1].strip().split()
                if len([p for p in depth_parts if re.match(r'^\d+\.?\d*$', p)]) > 2:
                    return 'temp'
                else:
                    return 'sst'
    
    if any(var in preview_text for var in ['UWND', 'VWND', 'WSPD', 'WDIR']):
        return 'wind'
    elif 'RAIN' in preview_text:
        return 'rain'
    elif 'RH' in preview_text:
        return 'rh'
    elif 'RAD' in preview_text:
        return 'rad'
    else:
        return 'general'

def process_temperature_data(lines, input_file):
    """Proses data temperature - hanya ambil 10m dan 20m depth"""
    
    # Cari informasi kedalaman
    depth_line = None
    for line in lines:
        if 'Depth(M):' in line:
            depth_line = line.strip()
            break
    
    if not depth_line:
        print("❌ Tidak dapat menemukan informasi kedalaman")
        return pd.DataFrame()
    
    # Ekstrak nilai kedalaman
    depth_parts = depth_line.split(':')[1].strip().split()
    depth_values = []
    depth_indices = []
    
    for i, part in enumerate(depth_parts):
        try:
            depth = float(part)
            # Hanya ambil kedalaman 10m dan 20m
            if depth in [10.0, 20.0]:
                depth_values.append(f"TEMP_{depth}m")
                depth_indices.append(i)
        except ValueError:
            continue
    
    if not depth_values:
        print("⚠️ Tidak ditemukan data kedalaman 10m atau 20m")
        return pd.DataFrame()
    
    # Proses baris data
    data_rows = []
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            
            date = parts[0]
            time = parts[1]
            
            # Ambil nilai temperature sesuai indeks kedalaman yang diinginkan
            temp_values = parts[2:]
            
            row_data = {'YYYYMMDD': date, 'HHMM': time}
            for i, depth_name in enumerate(depth_values):
                if depth_indices[i] < len(temp_values):
                    row_data[depth_name] = temp_values[depth_indices[i]]
                else:
                    row_data[depth_name] = 'NaN'
            
            data_rows.append(row_data)
    
    df = pd.DataFrame(data_rows)
    return process_datetime_and_clean(df)

def process_sst_data(lines, input_file):
    """Proses data SST"""
    
    data_rows = []
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            
            if len(parts) >= 3:
                date = parts[0]
                time = parts[1]
                sst_value = parts[2]
                
                data_rows.append({
                    'YYYYMMDD': date,
                    'HHMM': time,
                    'SST': sst_value
                })
    
    df = pd.DataFrame(data_rows)
    return process_datetime_and_clean(df)

def process_wind_data(lines, input_file):
    """Proses data angin"""
    
    # Cari header data
    header_line = None
    for i, line in enumerate(lines):
        if 'YYYYMMDD' in line and 'HHMM' in line:
            header_line = line
            break
    
    if not header_line:
        print("❌ Tidak dapat menemukan header")
        return pd.DataFrame()
    
    headers = header_line.strip().split()
    valid_headers = ['YYYYMMDD', 'HHMM', 'UWND', 'VWND', 'WSPD', 'WDIR']
    
    # Proses data
    data_rows = []
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            
            row_data = {}
            for i, header in enumerate(headers):
                if i < len(parts) and header in valid_headers:
                    row_data[header] = parts[i]
            
            if len(row_data) >= 2:
                data_rows.append(row_data)
    
    df = pd.DataFrame(data_rows)
    return process_datetime_and_clean(df)

def process_general_data(lines, input_file, variable_type):
    """Proses data umum (rain, rh, rad)"""
    
    # Cari header
    header_line = None
    for line in lines:
        if 'YYYYMMDD' in line and 'HHMM' in line:
            header_line = line
            break
    
    if not header_line:
        print("❌ Tidak dapat menemukan header")
        return pd.DataFrame()
    
    headers = header_line.strip().split()
    
    # Mapping variabel utama
    main_var_map = {
        'rain': 'RAIN',
        'rh': 'RH', 
        'rad': 'RAD'
    }
    
    # Proses data
    data_rows = []
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            
            if len(parts) >= 3:
                row_data = {'YYYYMMDD': parts[0], 'HHMM': parts[1]}
                
                # Ambil nilai variabel utama
                main_var = main_var_map.get(variable_type, variable_type.upper())
                if main_var in headers:
                    var_index = headers.index(main_var)
                    if var_index < len(parts):
                        row_data[main_var] = parts[var_index]
                elif len(parts) >= 3:
                    # Jika tidak ditemukan header, ambil kolom ke-3
                    row_data[main_var] = parts[2]
                
                data_rows.append(row_data)
    
    df = pd.DataFrame(data_rows)
    return process_datetime_and_clean(df)

def process_datetime_and_clean(df):
    """Proses datetime dan bersihkan data"""
    
    if df.empty:
        return df
    
    # Gabungkan tanggal dan waktu
    if 'YYYYMMDD' in df.columns and 'HHMM' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['YYYYMMDD'] + ' ' + df['HHMM'], 
                                       format='%Y%m%d %H%M', errors='coerce')
        
        # Buat komponen tanggal
        df['Date'] = df['Timestamp'].dt.date
        df['Year'] = df['Timestamp'].dt.year
        df['Month'] = df['Timestamp'].dt.month
        df['Day'] = df['Timestamp'].dt.day

        
        # Hapus kolom asli
        df.drop(['YYYYMMDD', 'HHMM', 'Timestamp'], axis=1, inplace=True)
        
        # Atur urutan kolom
        date_cols = ['Date', 'Year', 'Month', 'Day']
        other_cols = [col for col in df.columns if col not in date_cols]
        df = df[date_cols + other_cols]
    
    # Bersihkan missing values
    for col in df.columns:
        if col not in ['Date', 'Year', 'Month', 'Day']:
            df[col] = df[col].apply(lambda x: np.nan if re.match(r'^-9\.9+$|^-999\.9+$', str(x)) else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def process_single_location(location_path, location_name):
    """
    Memproses semua file ASCII dari satu lokasi dan menggabungkannya
    
    Parameters:
    location_path (str): Path ke direktori lokasi
    location_name (str): Nama lokasi (contoh: '0N90E')
    
    Returns:
    pandas.DataFrame: DataFrame gabungan untuk satu lokasi
    """
    
    ascii_path = os.path.join(location_path, 'ASCII')
    if not os.path.exists(ascii_path):
        print(f"❌ Direktori ASCII tidak ditemukan: {ascii_path}")
        return pd.DataFrame()
    
    # Pemetaan file berdasarkan nama
    file_patterns = {
        'rad': f'rad{location_name.lower()}_dy.ascii',
        'rain': f'rain{location_name.lower()}_dy.ascii', 
        'rh': f'rh{location_name.lower()}_dy.ascii',
        'sst': f'sst{location_name.lower()}_dy.ascii',
        'temp': f't{location_name.lower()}_dy.ascii',
        'wind': f'w{location_name.lower()}_dy.ascii'
    }
    
    dataframes = []
    
    print(f"\n🔄 Memproses lokasi: {location_name}")
    print("=" * 40)
    
    for var_type, file_pattern in file_patterns.items():
        file_path = os.path.join(ascii_path, file_pattern)
        
        if os.path.exists(file_path):
            print(f"📁 Memproses: {file_pattern}")
            df = convert_ascii_to_unified_csv(file_path, var_type)
            
            if not df.empty:
                # Tambahkan kolom lokasi
                df['Location'] = location_name
                dataframes.append(df)
                print(f"✅ Berhasil memproses {len(df)} baris data")
            else:
                print(f"⚠️ Tidak ada data yang berhasil diproses dari {file_pattern}")
        else:
            print(f"❌ File tidak ditemukan: {file_pattern}")
    
    # Gabungkan semua DataFrame berdasarkan datetime
    if dataframes:
        print(f"\n🔗 Menggabungkan {len(dataframes)} dataset...")
        
        # Merge berdasarkan kolom datetime
        merged_df = dataframes[0]
        datetime_cols = ['Date', 'Year', 'Month', 'Day', 'Location']
        
        for df in dataframes[1:]:
            merged_df = pd.merge(merged_df, df, on=datetime_cols, how='outer')
        
        # Sortir berdasarkan tanggal
        merged_df = merged_df.sort_values(['Year', 'Month', 'Day'])
        merged_df.reset_index(drop=True, inplace=True)
        
        # PERUBAHAN UTAMA: Pindahkan kolom 'Location' ke ujung
        if 'Location' in merged_df.columns:
            cols = [col for col in merged_df.columns if col != 'Location']
            cols.append('Location')
            merged_df = merged_df[cols]
        
        print(f"✅ Berhasil menggabungkan data - Total baris: {len(merged_df)}")
        return merged_df
    
    return pd.DataFrame()

def process_all_locations_to_single_csv(base_directory, location_names, output_file='Buoys.csv'):
    """
    Memproses semua lokasi dan menggabungkannya menjadi satu file CSV
    
    Parameters:
    base_directory (str): Path ke direktori utama
    location_names (list): List nama lokasi
    output_file (str): Nama file output
    
    Returns:
    str: Path ke file CSV yang dihasilkan
    """
    
    all_dataframes = []
    
    print("\n🌊 KONVERSI DATA BUOY - SEMUA LOKASI KE SATU FILE 🌊")
    print("=" * 60)
    
    for location in location_names:
        location_path = os.path.join(base_directory, location)
        
        if os.path.exists(location_path):
            df = process_single_location(location_path, location)
            if not df.empty:
                all_dataframes.append(df)
            else:
                print(f"⚠️ Tidak ada data untuk lokasi: {location}")
        else:
            print(f"❌ Direktori lokasi tidak ditemukan: {location_path}")
    
    # Gabungkan semua lokasi
    if all_dataframes:
        print(f"\n🔗 Menggabungkan data dari {len(all_dataframes)} lokasi...")
        
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Sortir berdasarkan lokasi dan tanggal
        final_df = final_df.sort_values(['Location', 'Year', 'Month', 'Day'])
        final_df.reset_index(drop=True, inplace=True)
        
        # PERUBAHAN UTAMA: Pastikan kolom 'Location' di ujung setelah concat
        if 'Location' in final_df.columns:
            cols = [col for col in final_df.columns if col != 'Location']
            cols.append('Location')
            final_df = final_df[cols]
        
        # Tentukan path output
        output_path = os.path.join(base_directory, output_file)
        
        # Simpan ke CSV
        final_df.to_csv(output_path, index=False)
        
        # Tampilkan ringkasan
        print(f"\n📊 RINGKASAN HASIL:")
        print(f"✅ Total baris data: {len(final_df):,}")
        print(f"📍 Jumlah lokasi: {final_df['Location'].nunique()}")
        print(f"📅 Rentang tanggal: {final_df['Date'].min()} - {final_df['Date'].max()}")
        print(f"📁 File disimpan: {output_path}")
        
        # Tampilkan kolom yang tersedia
        print(f"\n📋 Kolom yang tersedia:")
        for i, col in enumerate(final_df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Tampilkan distribusi data per lokasi
        print(f"\n📈 Distribusi data per lokasi:")
        location_counts = final_df['Location'].value_counts()
        for location, count in location_counts.items():
            print(f"   {location}: {count:,} baris")
        
        return output_path
    
    else:
        print("❌ Tidak ada data yang berhasil diproses dari semua lokasi")
        return None
    

# Contoh penggunaan
if __name__ == "__main__":
    # Konfigurasi path dan lokasi - SUDAH DISESUAIKAN
    base_directory = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys'
    location_names = ['0N90E', '4N90E', '8N90E']  # Sesuai dengan folder yang ada
    
    # Proses semua lokasi menjadi satu file
    result_file = process_all_locations_to_single_csv(
        base_directory=base_directory,
        location_names=location_names,
        output_file='Buoys_TA.csv'  # Nama file output yang lebih spesifik
    )
    
    if result_file:
        print(f"\n🎉 Konversi berhasil! File tersimpan di: {result_file}")
        
        # Tampilkan preview data (opsional)
        import pandas as pd
        try:
            df_preview = pd.read_csv(result_file)
            print(f"\n📋 Preview 5 baris pertama:")
            print(df_preview.head())
            print(f"\n📊 Info dataset:")
            print(f"   Shape: {df_preview.shape}")
            print(f"   Memory usage: {df_preview.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        except Exception as e:
            print(f"⚠️ Tidak dapat menampilkan preview: {e}")
            
    else:
        print("\n❌ Konversi gagal!")
        
    print("\n" + "="*60)
    print("🚀 Script selesai dijalankan!")
    print("="*60)