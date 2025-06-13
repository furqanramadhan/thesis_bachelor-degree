import os
import re
import numpy as np
import pandas as pd
import glob
from datetime import datetime

# Quality dan Source Code Configurations
QUALITY_CODES = {
    0: "Datum Missing",
    1: "Highest Quality", 
    2: "Default Quality",
    3: "Adjusted Data",
    4: "Lower Quality", 
    5: "Sensor or Tube Failed",
    -9: "Special Adjustments",  # netcdf format
    'C': "Special Adjustments"  # ascii format
}

SOURCE_CODES = {
    0: "No Sensor, No Data",
    1: "Real Time (Telemetered Mode)",
    2: "Derived from Real Time", 
    3: "Temporally Interpolated from Real Time",
    4: "Source Code Inactive at Present",
    5: "Recovered from Instrument RAM (Delayed Mode)",
    6: "Derived from RAM",
    7: "Temporally Interpolated from RAM",
    8: "Spatially Interpolated from RAM"
}

# Default filter settings - bisa disesuaikan sesuai kebutuhan
DEFAULT_QUALITY_FILTER = {
    'accept_quality': [1, 2, 3],  # Terima quality 1, 2, 3
    'reject_quality': [0, 4, 5, -9, 'C'],  # Tolak quality 0, 4, 5, special adjustments
    'strict_mode': False  # Jika True, hanya terima quality 1
}

DEFAULT_SOURCE_FILTER = {
    'accept_source': [1, 2, 5, 6],  # Terima real-time dan recovered data
    'reject_source': [0, 3, 4, 7, 8],  # Tolak interpolated dan inactive
    'strict_mode': False  # Jika True, hanya terima source 1 dan 5
}

def standardize_quality_source_codes(df):
    """
    Standardize quality and source codes to consistent data types
    """
    if 'Quality_Code' in df.columns:
        # Convert all quality codes to string for consistency
        df['Quality_Code'] = df['Quality_Code'].astype(str)
        # Handle NaN values
        df['Quality_Code'] = df['Quality_Code'].replace(['nan', 'None'], np.nan)

    
    if 'Source_Code' in df.columns:
        # Convert all source codes to string for consistency
        df['Source_Code'] = df['Source_Code'].astype(str)
        # Handle NaN values
        df['Source_Code'] = df['Source_Code'].replace(['nan', 'None'], np.nan)
        

    
    return df

def convert_ascii_to_unified_csv(input_file, variable_type=None, quality_filter=None, source_filter=None):
    """
    Mengkonversi file ASCII menjadi DataFrame dengan deteksi otomatis format dan filtering berkualitas.
    
    Parameters:
    input_file (str): Path ke file ASCII
    variable_type (str): Tipe variabel ('temp', 'sst', 'wind', 'rain', 'rh', 'rad')
    quality_filter (dict): Konfigurasi filter kualitas data
    source_filter (dict): Konfigurasi filter sumber data
    
    Returns:
    pandas.DataFrame: DataFrame dengan data yang sudah diproses dan difilter
    """
    
    # Set default filters jika tidak diberikan
    if quality_filter is None:
        quality_filter = DEFAULT_QUALITY_FILTER.copy()
    if source_filter is None:
        source_filter = DEFAULT_SOURCE_FILTER.copy()
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Auto-detect format jika variable_type tidak ditentukan
    if variable_type is None:
        variable_type = detect_variable_type(lines)
    
    print(f"📊 Memproses {os.path.basename(input_file)} sebagai tipe: {variable_type.upper()}")
    
    if variable_type == 'temp':
        df = process_temperature_data(lines, input_file, quality_filter, source_filter)
    elif variable_type == 'sst':
        df = process_sst_data(lines, input_file, quality_filter, source_filter)
    elif variable_type == 'wind':
        df = process_wind_data(lines, input_file, quality_filter, source_filter)
    else:  # rain, rh, rad
        df = process_general_data(lines, input_file, variable_type, quality_filter, source_filter)
    
    # Standardize quality and source codes
    return standardize_quality_source_codes(df)

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

def extract_quality_source_codes(data_line):
    """
    Ekstrak quality dan source codes dari baris data
    Asumsi: format baris data mengandung quality dan source codes
    """
    parts = data_line.strip().split()
    
    # Cari quality dan source codes (biasanya di bagian akhir atau dengan pattern khusus)
    quality_code = None
    source_code = None
    
    # Pattern untuk mencari quality/source codes
    for part in parts:
        # Cek jika ada karakter 'C' untuk special adjustment
        if part == 'C':
            quality_code = 'C'
        # Cek jika ada pattern angka yang mungkin quality/source code
        elif re.match(r'^[0-8]$', part):
            if quality_code is None:
                quality_code = int(part)
            elif source_code is None:
                source_code = int(part)
    
    return quality_code, source_code

def is_data_acceptable(value, quality_code, source_code, quality_filter, source_filter):
    """
    Cek apakah data dapat diterima berdasarkan quality dan source codes
    """
    # Cek missing value patterns
    if pd.isna(value) or re.match(r'^-9\.9+$|^-999\.9+$', str(value)):
        return False
    
    # Cek quality code
    if quality_code is not None:
        if quality_filter['strict_mode'] and quality_code != 1:
            return False
        elif quality_code in quality_filter['reject_quality']:
            return False
        elif quality_filter['accept_quality'] and quality_code not in quality_filter['accept_quality']:
            return False
    
    # Cek source code
    if source_code is not None:
        if source_filter['strict_mode'] and source_code not in [1, 5]:
            return False
        elif source_code in source_filter['reject_source']:
            return False
        elif source_filter['accept_source'] and source_code not in source_filter['accept_source']:
            return False
    
    return True

def process_temperature_data(lines, input_file, quality_filter, source_filter):
    """Proses data temperature dengan quality filtering - hanya ambil 10m dan 20m depth"""
    
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
    
    # Proses baris data dengan quality filtering
    data_rows = []
    filtered_count = 0
    total_count = 0
    
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            total_count += 1
            
            date = parts[0]
            time = parts[1]
            
            # Ekstrak quality dan source codes
            quality_code, source_code = extract_quality_source_codes(line)
            
            # Ambil nilai temperature sesuai indeks kedalaman yang diinginkan
            temp_values = parts[2:]
            
            row_data = {
                'YYYYMMDD': date, 
                'HHMM': time,
                'Quality_Code': quality_code,
                'Source_Code': source_code
            }
            
            # Proses setiap kedalaman dengan quality check
            for i, depth_name in enumerate(depth_values):
                if depth_indices[i] < len(temp_values):
                    temp_value = temp_values[depth_indices[i]]
                    
                    if is_data_acceptable(temp_value, quality_code, source_code, quality_filter, source_filter):
                        row_data[depth_name] = temp_value
                    else:
                        row_data[depth_name] = np.nan
                        filtered_count += 1
                else:
                    row_data[depth_name] = np.nan
            
            data_rows.append(row_data)
    
    print(f"🔍 Quality Filter: {filtered_count}/{total_count} data points filtered")
    
    df = pd.DataFrame(data_rows)
    return process_datetime_and_clean(df)

def process_sst_data(lines, input_file, quality_filter, source_filter):
    """Proses data SST dengan quality filtering"""
    
    data_rows = []
    filtered_count = 0
    total_count = 0
    
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            total_count += 1
            
            if len(parts) >= 3:
                date = parts[0]
                time = parts[1]
                sst_value = parts[2]
                
                # Ekstrak quality dan source codes
                quality_code, source_code = extract_quality_source_codes(line)
                
                # Cek kualitas data
                if not is_data_acceptable(sst_value, quality_code, source_code, quality_filter, source_filter):
                    sst_value = np.nan
                    filtered_count += 1
                
                data_rows.append({
                    'YYYYMMDD': date,
                    'HHMM': time,
                    'SST': sst_value,
                    'Quality_Code': quality_code,
                    'Source_Code': source_code
                })
    
    print(f"🔍 Quality Filter: {filtered_count}/{total_count} data points filtered")
    
    df = pd.DataFrame(data_rows)
    return process_datetime_and_clean(df)

def process_wind_data(lines, input_file, quality_filter, source_filter):
    """Proses data angin dengan quality filtering"""
    
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
    
    # Proses data dengan quality filtering
    data_rows = []
    filtered_count = 0
    total_count = 0
    
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            total_count += 1
            
            # Ekstrak quality dan source codes
            quality_code, source_code = extract_quality_source_codes(line)
            
            row_data = {
                'Quality_Code': quality_code,
                'Source_Code': source_code
            }
            
            # Proses setiap header dengan quality check
            for i, header in enumerate(headers):
                if i < len(parts) and header in valid_headers:
                    value = parts[i]
                    
                    if header in ['YYYYMMDD', 'HHMM']:
                        row_data[header] = value
                    else:
                        # Apply quality filtering untuk data wind
                        if is_data_acceptable(value, quality_code, source_code, quality_filter, source_filter):
                            row_data[header] = value
                        else:
                            row_data[header] = np.nan
                            filtered_count += 1
            
            if len(row_data) >= 4:  # Minimal ada YYYYMMDD, HHMM, dan quality codes
                data_rows.append(row_data)
    
    print(f"🔍 Quality Filter: {filtered_count}/{total_count} data points filtered")
    
    df = pd.DataFrame(data_rows)
    return process_datetime_and_clean(df)

def process_general_data(lines, input_file, variable_type, quality_filter, source_filter):
    """Proses data umum (rain, rh, rad) dengan quality filtering"""
    
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
    
    # Proses data dengan quality filtering
    data_rows = []
    filtered_count = 0
    total_count = 0
    
    for line in lines:
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            parts = line.strip().split()
            total_count += 1
            
            if len(parts) >= 3:
                date = parts[0]
                time = parts[1]
                
                # Ekstrak quality dan source codes
                quality_code, source_code = extract_quality_source_codes(line)
                
                row_data = {
                    'YYYYMMDD': date, 
                    'HHMM': time,
                    'Quality_Code': quality_code,
                    'Source_Code': source_code
                }
                
                # Ambil nilai variabel utama dengan quality check
                main_var = main_var_map.get(variable_type, variable_type.upper())
                if main_var in headers:
                    var_index = headers.index(main_var)
                    if var_index < len(parts):
                        value = parts[var_index]
                        if is_data_acceptable(value, quality_code, source_code, quality_filter, source_filter):
                            row_data[main_var] = value
                        else:
                            row_data[main_var] = np.nan
                            filtered_count += 1
                elif len(parts) >= 3:
                    # Jika tidak ditemukan header, ambil kolom ke-3 dengan quality check
                    value = parts[2]
                    if is_data_acceptable(value, quality_code, source_code, quality_filter, source_filter):
                        row_data[main_var] = value
                    else:
                        row_data[main_var] = np.nan
                        filtered_count += 1
                
                data_rows.append(row_data)
    
    print(f"🔍 Quality Filter: {filtered_count}/{total_count} data points filtered")
    
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
        
        # Atur urutan kolom - quality dan source codes di awal
        date_cols = ['Date', 'Year', 'Month', 'Day']
        qc_cols = ['Quality_Code', 'Source_Code']
        other_cols = [col for col in df.columns if col not in date_cols + qc_cols]
        df = df[date_cols + qc_cols + other_cols]
    
    # Bersihkan missing values (sudah ditangani di quality filtering)
    for col in df.columns:
        if col not in ['Date', 'Year', 'Month', 'Day', 'Quality_Code', 'Source_Code']:
            df[col] = df[col].apply(lambda x: np.nan if re.match(r'^-9\.9+$|^-999\.9+$', str(x)) else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def process_single_location(location_path, location_name, quality_filter=None, source_filter=None):
    """
    Memproses semua file ASCII dari satu lokasi dengan quality filtering
    """
    
    # Set default filters
    if quality_filter is None:
        quality_filter = DEFAULT_QUALITY_FILTER.copy()
    if source_filter is None:
        source_filter = DEFAULT_SOURCE_FILTER.copy()
    
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
            df = convert_ascii_to_unified_csv(file_path, var_type, quality_filter, source_filter)
            
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
        
        # Standardize all dataframes first
        for i in range(len(dataframes)):
            dataframes[i] = standardize_quality_source_codes(dataframes[i])
        
        # Use concat instead of merge to avoid data type conflicts
        # First, add a temporary index for matching records
        datetime_cols = ['Date', 'Year', 'Month', 'Day']
        
        # Create a master datetime index from all dataframes
        all_dates = set()
        for df in dataframes:
            for _, row in df.iterrows():
                date_key = (row['Date'], row['Year'], row['Month'], row['Day'])
                all_dates.add(date_key)
        
        # Sort dates
        all_dates = sorted(list(all_dates))
        
        # Create master dataframe with all dates
        master_df = pd.DataFrame(all_dates, columns=datetime_cols)
        master_df['Location'] = location_name
        
        # Add Quality_Code and Source_Code columns with proper data types
        master_df['Quality_Code'] = np.nan
        master_df['Source_Code'] = np.nan
        master_df['Quality_Code'] = master_df['Quality_Code'].astype('object')
        master_df['Source_Code'] = master_df['Source_Code'].astype('object')
        
        # Merge each dataframe
        for df in dataframes:
            # Merge on datetime columns only, then handle quality codes separately
            merge_cols = datetime_cols + ['Location']
            master_df = pd.merge(master_df, df.drop(['Quality_Code', 'Source_Code'], axis=1), 
                               on=merge_cols, how='left')
            
            # Update quality and source codes where data exists
            for _, row in df.iterrows():
                mask = ((master_df['Date'] == row['Date']) & 
                       (master_df['Year'] == row['Year']) & 
                       (master_df['Month'] == row['Month']) & 
                       (master_df['Day'] == row['Day']) & 
                       (master_df['Location'] == row['Location']))
                
                if mask.any():
                    # Update quality and source codes only if they're not already set
                    if pd.isna(master_df.loc[mask, 'Quality_Code'].iloc[0]):
                        master_df.loc[mask, 'Quality_Code'] = row['Quality_Code']
                    if pd.isna(master_df.loc[mask, 'Source_Code'].iloc[0]):
                        master_df.loc[mask, 'Source_Code'] = row['Source_Code']
        
        # Sort and clean up
        merged_df = master_df.sort_values(['Year', 'Month', 'Day'])
        merged_df.reset_index(drop=True, inplace=True)
        
        # Move 'Location' column to the end
        if 'Location' in merged_df.columns:
            cols = [col for col in merged_df.columns if col != 'Location']
            cols.append('Location')
            merged_df = merged_df[cols]
        
        print(f"✅ Berhasil menggabungkan data - Total baris: {len(merged_df)}")
        return merged_df
    
    return pd.DataFrame()

def process_all_locations_to_single_csv(base_directory, location_names, output_file='Buoys.csv', 
                                      quality_filter=None, source_filter=None):
    """
    Memproses semua lokasi dengan quality filtering dan menggabungkannya menjadi satu file CSV
    """
    
    # Set default filters
    if quality_filter is None:
        quality_filter = DEFAULT_QUALITY_FILTER.copy()
    if source_filter is None:
        source_filter = DEFAULT_SOURCE_FILTER.copy()
    
    all_dataframes = []
    
    print("\n🌊 KONVERSI DATA BUOY DENGAN QUALITY FILTERING 🌊")
    print("=" * 60)
    print(f"🔍 Quality Filter: Accept {quality_filter['accept_quality']}, Reject {quality_filter['reject_quality']}")
    print(f"📡 Source Filter: Accept {source_filter['accept_source']}, Reject {source_filter['reject_source']}")
    print("=" * 60)
    
    for location in location_names:
        location_path = os.path.join(base_directory, location)
        
        if os.path.exists(location_path):
            df = process_single_location(location_path, location, quality_filter, source_filter)
            if not df.empty:
                all_dataframes.append(df)
            else:
                print(f"⚠️ Tidak ada data untuk lokasi: {location}")
        else:
            print(f"❌ Direktori lokasi tidak ditemukan: {location_path}")
    
    # Gabungkan semua lokasi
    if all_dataframes:
        print(f"\n🔗 Menggabungkan data dari {len(all_dataframes)} lokasi...")
        
        # Standardize all dataframes before concatenation
        for i in range(len(all_dataframes)):
            all_dataframes[i] = standardize_quality_source_codes(all_dataframes[i])
        
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Sortir berdasarkan lokasi dan tanggal
        final_df = final_df.sort_values(['Location', 'Year', 'Month', 'Day'])
        final_df.reset_index(drop=True, inplace=True)
        
        # Pastikan kolom 'Location' di ujung
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
        
        # Tampilkan statistik quality codes
        if 'Quality_Code' in final_df.columns:
            print(f"\n🔍 Distribusi Quality Codes:")
            quality_counts = final_df['Quality_Code'].value_counts()
            for code, count in quality_counts.items():
                # Convert code back to original type for lookup
                lookup_code = code
                if code != 'nan' and code is not None:
                    try:
                        lookup_code = int(code) if code != 'C' else 'C'
                    except:
                        lookup_code = code
                desc = QUALITY_CODES.get(lookup_code, "Unknown")
                print(f"   {code}: {count:,} ({desc})")
        
        # Tampilkan statistik source codes
        if 'Source_Code' in final_df.columns:
            print(f"\n📡 Distribusi Source Codes:")
            source_counts = final_df['Source_Code'].value_counts()
            for code, count in source_counts.items():
                # Convert code back to original type for lookup
                lookup_code = code
                if code != 'nan' and code is not None:
                    try:
                        lookup_code = int(code)
                    except:
                        lookup_code = code
                desc = SOURCE_CODES.get(lookup_code, "Unknown")
                print(f"   {code}: {count:,} ({desc})")
        
        # Tampilkan kolom yang tersedia
        print(f"\n📋 Kolom yang tersedia:")
        for i, col in enumerate(final_df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        return output_path
    
    else:
        print("❌ Tidak ada data yang berhasil diproses dari semua lokasi")
        return None
# Fungsi untuk membuat konfigurasi filter custom
def create_quality_filter(accept_quality=None, reject_quality=None, strict_mode=False):
    """Membuat konfigurasi filter kualitas custom"""
    return {
        'accept_quality': accept_quality or [1, 2, 3],
        'reject_quality': reject_quality or [0, 4, 5, -9, 'C'],
        'strict_mode': strict_mode
    }

def create_source_filter(accept_source=None, reject_source=None, strict_mode=False):
    """Membuat konfigurasi filter sumber custom"""
    return {
        'accept_source': accept_source or [1, 2, 5, 6],
        'reject_source': reject_source or [0, 3, 4, 7, 8],
        'strict_mode': strict_mode
    }

# Contoh penggunaan
if __name__ == "__main__":
    # Konfigurasi path dan lokasi
    base_directory = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys'
    location_names = ['0N90E', '4N90E', '8N90E']
    
    # Konfigurasi filter - bisa disesuaikan sesuai kebutuhan
    # Contoh 1: Filter default (recommended)
    quality_filter = DEFAULT_QUALITY_FILTER
    source_filter = DEFAULT_SOURCE_FILTER
    
    # Contoh 2: Filter strict (hanya data berkualitas tinggi)
    #quality_filter = create_quality_filter(accept_quality=[1], strict_mode=True)
    #source_filter = create_source_filter(accept_source=[1, 5], strict_mode=True)
    
    # Contoh 3: Filter custom
    # quality_filter = create_quality_filter(accept_quality=[1, 2], reject_quality=[0, 4, 5])
    # source_filter = create_source_filter(accept_source=[1, 2, 5], reject_source=[0, 3, 4, 7, 8])
    
    # Proses semua lokasi dengan quality filtering
    result_file = process_all_locations_to_single_csv(
        base_directory=base_directory,
        location_names=location_names,
        output_file='Buoys_TA_Filtered.csv',
        quality_filter=quality_filter,
        source_filter=source_filter
    )
    
    if result_file:
        print(f"\n🎉 Konversi berhasil! File tersimpan di: {result_file}")
        
        # Tampilkan preview data
        try:
            df_preview = pd.read_csv(result_file)
            print(f"\n📋 Preview 5 baris pertama:")
            print(df_preview.head())
            print(f"\n📊 Info dataset:")
            print(f"   Shape: {df_preview.shape}")
            print(f"   Memory usage: {df_preview.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Tampilkan data quality summary
            print(f"\n📈 Data Quality Summary:")
            for col in df_preview.columns:
                if col not in ['Date', 'Year', 'Month', 'Day', 'Quality_Code', 'Source_Code', 'Location']:
                    non_null_count = df_preview[col].count()
                    total_count = len(df_preview)
                    percentage = (non_null_count / total_count) * 100
                    print(f"   {col}: {non_null_count}/{total_count} ({percentage:.1f}%)")
                    
        except Exception as e:
            print(f"⚠️ Tidak dapat menampilkan preview: {e}")
            
    else:
        print("\n❌ Konversi gagal!")
        
    print("\n" + "="*60)
    print("🚀 Script selesai dijalankan!")
    print("="*60)