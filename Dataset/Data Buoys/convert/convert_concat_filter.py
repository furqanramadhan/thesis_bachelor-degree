import os
import re
import numpy as np
import pandas as pd
import glob
from datetime import datetime

# KONFIGURASI QUALITY CONTROL
ACCEPTABLE_QUALITY_CODES = [1, 2, 3]  # 1 = Highest Quality, 2 = Default Quality, 3 = Data Adjusted
ACCEPTABLE_SOURCE_CODES = [1, 2, 5, 6]  # Real Time, Derived RT, Recovered RAM, Derived RAM
EXCLUDE_QUALITY_CODES = [0, 4, 5]  # Missing, Lower Quality, Sensor Failed
EXCLUDE_SOURCE_CODES = [0, 4]  # No Sensor, Inactive

def convert_ascii_to_unified_csv(input_file, variable_type=None):
    """
    Mengkonversi file ASCII menjadi DataFrame dengan deteksi otomatis format.
    Khusus untuk temperature, hanya ambil 10m dan 20m depth.
    DITAMBAHKAN: Quality Control filtering
    
    Parameters:
    input_file (str): Path ke file ASCII
    variable_type (str): Tipe variabel ('temp', 'sst', 'wind', 'rain', 'rh', 'rad')
    
    Returns:
    pandas.DataFrame: DataFrame dengan data yang sudah diproses dan di-filter
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

def parse_quality_source_codes(value_str):
    """
    Parse value string yang mengandung quality dan source codes
    Format umum: VALUE_QC_SC atau VALUE QC SC
    
    Returns:
    tuple: (value, quality_code, source_code, is_valid)
    """
    
    if pd.isna(value_str) or str(value_str).strip() == '':
        return np.nan, None, None, False
    
    # Convert to string and clean
    value_str = str(value_str).strip()
    
    # Handle missing data indicators - DIPERLUAS untuk menangani lebih banyak format
    missing_indicators = [
        '-9', '-9.0', '-9.9', '-9.99', '-9.999',
        '-999', '-999.0', '-999.9', '-999.99', '-999.999',
        'NaN', 'nan', 'NULL', 'null', '',
        '-99', '-99.0', '-99.9', '-99.99', '-99.999'
    ]
    
    if value_str in missing_indicators:
        return np.nan, 0, 0, False
    
    # Cek jika value adalah angka missing (negatif dengan pola tertentu)
    try:
        temp_val = float(value_str.split('_')[0] if '_' in value_str else value_str.split()[0])
        if temp_val <= -9:  # Semua nilai <= -9 dianggap missing
            return np.nan, 0, 0, False
    except (ValueError, IndexError):
        pass
    
    # Pattern 1: VALUE_QC_SC (underscore separated)
    if '_' in value_str:
        parts = value_str.split('_')
        if len(parts) >= 3:
            try:
                value = float(parts[0])
                # Cek lagi jika value adalah missing indicator
                if value <= -9:
                    return np.nan, 0, 0, False
                quality_code = int(parts[1])
                source_code = int(parts[2])
                return value, quality_code, source_code, True
            except (ValueError, IndexError):
                pass
    
    # Pattern 2: VALUE QC SC (space separated, last 2 digits)
    parts = value_str.split()
    if len(parts) >= 3:
        try:
            value = float(parts[0])
            # Cek lagi jika value adalah missing indicator
            if value <= -9:
                return np.nan, 0, 0, False
            quality_code = int(parts[1])
            source_code = int(parts[2])
            return value, quality_code, source_code, True
        except (ValueError, IndexError):
            pass
    
    # Pattern 3: Single value with embedded codes (e.g., "25.31_2_1")
    if re.match(r'^-?\d+\.?\d*_\d_\d$', value_str):
        parts = value_str.split('_')
        try:
            value = float(parts[0])
            # Cek lagi jika value adalah missing indicator
            if value <= -9:
                return np.nan, 0, 0, False
            quality_code = int(parts[1])
            source_code = int(parts[2])
            return value, quality_code, source_code, True
        except (ValueError, IndexError):
            pass
    
    # Pattern 4: Just numeric value (assume default quality)
    try:
        value = float(value_str)
        # Cek lagi jika value adalah missing indicator
        if value <= -9:
            return np.nan, 0, 0, False
        return value, 2, 1, True  # Default quality, real-time source
    except ValueError:
        return np.nan, None, None, False

def apply_quality_control(df, data_columns):
    """
    Menerapkan quality control pada DataFrame
    
    Parameters:
    df (pandas.DataFrame): DataFrame input
    data_columns (list): List kolom data yang perlu di-filter
    
    Returns:
    pandas.DataFrame: DataFrame yang sudah di-filter
    """
    
    if df.empty:
        return df
    
    original_rows = len(df)
    filtered_rows = 0
    
    print(f"🔍 Menerapkan Quality Control pada {len(data_columns)} kolom data...")
    
    for col in data_columns:
        if col in df.columns:
            print(f"   📋 Memproses kolom: {col}")
            
            # Parse quality dan source codes
            parsed_data = df[col].apply(parse_quality_source_codes)
            
            # Ekstrak komponen
            df[f'{col}_value'] = parsed_data.apply(lambda x: x[0])
            df[f'{col}_quality'] = parsed_data.apply(lambda x: x[1])
            df[f'{col}_source'] = parsed_data.apply(lambda x: x[2])
            df[f'{col}_valid'] = parsed_data.apply(lambda x: x[3])
            
            # Apply quality control filters
            quality_mask = (
                (df[f'{col}_quality'].isin(ACCEPTABLE_QUALITY_CODES)) |
                (df[f'{col}_quality'].isna() & df[f'{col}_valid'])  # Handle default cases
            )
            
            source_mask = (
                (df[f'{col}_source'].isin(ACCEPTABLE_SOURCE_CODES)) |
                (df[f'{col}_source'].isna() & df[f'{col}_valid'])  # Handle default cases
            )
            
            exclude_quality_mask = ~df[f'{col}_quality'].isin(EXCLUDE_QUALITY_CODES)
            exclude_source_mask = ~df[f'{col}_source'].isin(EXCLUDE_SOURCE_CODES)
            
            # Combine all filters
            final_mask = quality_mask & source_mask & exclude_quality_mask & exclude_source_mask
            
            # Apply filter - set invalid data to NaN
            df.loc[~final_mask, f'{col}_value'] = np.nan
            
            # Replace original column with filtered values
            df[col] = df[f'{col}_value']
            
            # Count valid data points
            valid_count = df[col].notna().sum()
            total_count = len(df)
            
            print(f"      ✅ Data valid: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")
            
            # Clean up temporary columns
            temp_cols = [f'{col}_value', f'{col}_quality', f'{col}_source', f'{col}_valid']
            df.drop(temp_cols, axis=1, inplace=True, errors='ignore')
    
    # Remove rows where ALL data columns are NaN
    data_mask = df[data_columns].notna().any(axis=1)
    df_filtered = df[data_mask].copy()
    
    filtered_rows = len(df_filtered)
    removed_rows = original_rows - filtered_rows
    
    print(f"📊 Quality Control Summary:")
    print(f"   📈 Baris asli: {original_rows}")
    print(f"   ✅ Baris valid: {filtered_rows}")
    print(f"   ❌ Baris dihapus: {removed_rows} ({removed_rows/original_rows*100:.1f}%)")
    
    return df_filtered

def process_temperature_data(lines, input_file):
    """Proses data temperature - hanya ambil 10m dan 20m depth dengan quality control"""
    
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
            if depth in [10.0, 13.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 180.0, 300.0, 500.0]:
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
    df = process_datetime_and_clean(df)
    
    # Apply quality control
    if not df.empty:
        data_columns = [col for col in df.columns if col.startswith('TEMP_')]
        df = apply_quality_control(df, data_columns)
    
    return df

def process_sst_data(lines, input_file):
    """Proses data SST dengan quality control"""
    
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
    df = process_datetime_and_clean(df)
    
    # Apply quality control
    if not df.empty:
        data_columns = ['SST']
        df = apply_quality_control(df, data_columns)
    
    return df

def process_wind_data(lines, input_file):
    """Proses data angin dengan quality control"""
    
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
    df = process_datetime_and_clean(df)
    
    # Apply quality control
    if not df.empty:
        data_columns = ['UWND', 'VWND', 'WSPD', 'WDIR']
        data_columns = [col for col in data_columns if col in df.columns]
        df = apply_quality_control(df, data_columns)
    
    return df

def process_general_data(lines, input_file, variable_type):
    """Proses data umum (rain, rh, rad) dengan quality control"""
    
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
    df = process_datetime_and_clean(df)
    
    # Apply quality control
    if not df.empty:
        main_var = main_var_map.get(variable_type, variable_type.upper())
        if main_var in df.columns:
            data_columns = [main_var]
            df = apply_quality_control(df, data_columns)
    
    return df

def process_datetime_and_clean(df):
    """Proses datetime dan bersihkan data (tanpa quality control di sini)"""
    
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
    
    return df

def reorder_columns_properly(df):
    """
    Mengurutkan kolom dengan urutan yang benar:
    1. Date, Year, Month, Day
    2. RAD, RAIN, RH, SST
    3. Temperature berdasarkan kedalaman (ascending)
    4. UWND, VWND, WSPD, WDIR
    5. Location (terakhir)
    """
    
    if df.empty:
        return df
    
    # Definisikan urutan kolom
    ordered_columns = []
    
    # 1. Date columns
    date_cols = ['Date', 'Year', 'Month', 'Day']
    ordered_columns.extend([col for col in date_cols if col in df.columns])
    
    # 2. Environmental variables
    env_cols = ['RAD', 'RAIN', 'RH', 'SST']
    ordered_columns.extend([col for col in env_cols if col in df.columns])
    
    # 3. Temperature columns - sorted by depth
    temp_cols = [col for col in df.columns if col.startswith('TEMP_')]
    # Extract depth values and sort
    temp_depths = []
    for col in temp_cols:
        depth_str = col.replace('TEMP_', '').replace('m', '')
        try:
            depth = float(depth_str)
            temp_depths.append((depth, col))
        except ValueError:
            temp_depths.append((999, col))  # Put invalid depths at end
    
    # Sort by depth and add to ordered columns
    temp_depths.sort(key=lambda x: x[0])
    ordered_columns.extend([col for depth, col in temp_depths])
    
    # 4. Wind variables
    wind_cols = ['UWND', 'VWND', 'WSPD', 'WDIR']
    ordered_columns.extend([col for col in wind_cols if col in df.columns])
    
    # 5. Location (always last)
    if 'Location' in df.columns:
        ordered_columns.append('Location')
    
    # Add any remaining columns that weren't categorized
    remaining_cols = [col for col in df.columns if col not in ordered_columns]
    ordered_columns.extend(remaining_cols)
    
    # Return dataframe with reordered columns
    return df[ordered_columns]

def process_single_location(location_path, location_name):
    """
    Memproses semua file ASCII dari satu lokasi dan menggabungkannya
    DITAMBAHKAN: Quality control pada setiap dataset dan proper column ordering
    
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
                print(f"✅ Berhasil memproses {len(df)} baris data (setelah quality control)")
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
        
        # Atur urutan kolom dengan benar
        merged_df = reorder_columns_properly(merged_df)
        
        print(f"✅ Berhasil menggabungkan data - Total baris: {len(merged_df)}")
        return merged_df
    
    return pd.DataFrame()

def process_all_locations_to_single_csv(base_directory, location_names, output_file='Buoys_QC.csv'):
    """
    Memproses semua lokasi dan menggabungkannya menjadi satu file CSV
    DITAMBAHKAN: Quality control summary reporting dan proper column ordering
    
    Parameters:
    base_directory (str): Path ke direktori utama
    location_names (list): List nama lokasi
    output_file (str): Nama file output
    
    Returns:
    str: Path ke file CSV yang dihasilkan
    """
    
    all_dataframes = []
    
    print("\n🌊 KONVERSI DATA BUOY - DENGAN QUALITY CONTROL 🌊")
    print("=" * 60)
    print(f"🔍 Quality Control Settings:")
    print(f"   ✅ Acceptable Quality Codes: {ACCEPTABLE_QUALITY_CODES}")
    print(f"   ✅ Acceptable Source Codes: {ACCEPTABLE_SOURCE_CODES}")
    print(f"   ❌ Excluded Quality Codes: {EXCLUDE_QUALITY_CODES}")
    print(f"   ❌ Excluded Source Codes: {EXCLUDE_SOURCE_CODES}")
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
        
        # Atur urutan kolom dengan benar untuk dataset final
        final_df = reorder_columns_properly(final_df)
        
        # Tentukan path output
        output_path = os.path.join(base_directory, output_file)
        
        # Simpan ke CSV
        final_df.to_csv(output_path, index=False)
        
        # Tampilkan ringkasan
        print(f"\n📊 RINGKASAN HASIL (DENGAN QUALITY CONTROL):")
        print(f"✅ Total baris data: {len(final_df):,}")
        print(f"📍 Jumlah lokasi: {final_df['Location'].nunique()}")
        print(f"📅 Rentang tanggal: {final_df['Date'].min()} - {final_df['Date'].max()}")
        print(f"📁 File disimpan: {output_path}")
        
        # Tampilkan kolom yang tersedia dengan urutan yang benar
        print(f"\n📋 Kolom yang tersedia (urutan sudah diperbaiki):")
        for i, col in enumerate(final_df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Tampilkan distribusi data per lokasi
        print(f"\n📈 Distribusi data per lokasi:")
        location_counts = final_df['Location'].value_counts()
        for location, count in location_counts.items():
            print(f"   {location}: {count:,} baris")
        
        # Data quality summary
        print(f"\n🔍 Data Quality Summary:")
        data_cols = [col for col in final_df.columns if col not in ['Date', 'Year', 'Month', 'Day', 'Location']]
        for col in data_cols:
            if col in final_df.columns:
                total_points = len(final_df)
                valid_points = final_df[col].notna().sum()
                completeness = (valid_points / total_points) * 100
                print(f"   {col}: {valid_points:,}/{total_points:,} ({completeness:.1f}% complete)")
        
        return output_path
    
    else:
        print("❌ Tidak ada data yang berhasil diproses dari semua lokasi")
        return None

# Contoh penggunaan
if __name__ == "__main__":
    # Konfigurasi path dan lokasi
    base_directory = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys'
    location_names = ['0N90E', '4N90E', '8N90E']
    
    # Proses semua lokasi menjadi satu file dengan quality control
    result_file = process_all_locations_to_single_csv(
        base_directory=base_directory,
        location_names=location_names,
        output_file='Buoys_Data_All.csv'
    )
    
    if result_file:
        print(f"\n🎉 Konversi dengan Quality Control berhasil! File tersimpan di: {result_file}")
        
        # Tampilkan preview data (opsional)
        import pandas as pd
        try:
            df_preview = pd.read_csv(result_file)
            print(f"\n📋 Preview 5 baris pertama:")
            print(df_preview.head())
            print(f"\n📊 Info dataset:")
            print(f"   Shape: {df_preview.shape}")
            print(f"   Memory usage: {df_preview.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Quality assessment
            print(f"\n🎯 Quality Assessment:")
            data_cols = [col for col in df_preview.columns if col not in ['Date', 'Year', 'Month', 'Day', 'Location']]
            for col in data_cols:
                if col in df_preview.columns:
                    null_pct = (df_preview[col].isna().sum() / len(df_preview)) * 100
                    print(f"   {col}: {100-null_pct:.1f}% data availability")
                    
        except Exception as e:
            print(f"⚠️ Tidak dapat menampilkan preview: {e}")
            
    else:
        print("\n❌ Konversi gagal!")
        
    print("\n" + "="*60)
    print("🚀 Script dengan Quality Control selesai dijalankan!")
    print("="*60)