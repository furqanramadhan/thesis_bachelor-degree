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
    """
    Proses data temperature dengan penanganan dynamic indexing
    untuk mengatasi masalah perubahan struktur kedalaman
    """
    
    # Target kedalaman yang diinginkan (tanpa 13m)
    TARGET_DEPTHS = [10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 180.0, 300.0, 500.0]
    
    data_rows = []
    current_depth_info = None
    
    for line_num, line in enumerate(lines):
        line = line.strip()
        
        # Deteksi header kedalaman
        if 'Depth(M):' in line:
            current_depth_info = parse_depth_header(line)
            print(f"📏 Ditemukan header kedalaman pada baris {line_num + 1}")
            print(f"    Kedalaman tersedia: {list(current_depth_info.keys())}")
            continue
        
        # Proses baris data
        if re.match(r'^\s*\d{8}\s+\d{4}', line):
            if current_depth_info is None:
                print("⚠️ Data ditemukan tanpa header kedalaman, skip...")
                continue
                
            parts = line.split()
            if len(parts) < 3:
                continue
                
            date = parts[0]
            time = parts[1]
            
            # Ambil semua nilai data (mulai dari index 2)
            data_values = parts[2:]
            
            # Buat row data dengan dynamic mapping
            row_data = {'YYYYMMDD': date, 'HHMM': time}
            
            for target_depth in TARGET_DEPTHS:
                col_name = f"TEMP_{target_depth}m"
                
                if target_depth in current_depth_info:
                    # Ambil index yang benar untuk kedalaman ini
                    depth_index = current_depth_info[target_depth]
                    
                    # Pastikan index valid dalam data_values
                    if depth_index < len(data_values):
                        value = data_values[depth_index]
                        # Skip jika 13m (meskipun ada di header)
                        if target_depth != 13.0:
                            row_data[col_name] = value
                    else:
                        row_data[col_name] = 'NaN'
                else:
                    # Kedalaman tidak tersedia untuk baris ini
                    row_data[col_name] = 'NaN'
            
            data_rows.append(row_data)
    
    # Convert ke DataFrame
    df = pd.DataFrame(data_rows)
    df = process_datetime_and_clean(df)
    
    # Apply quality control
    if not df.empty:
        data_columns = [col for col in df.columns if col.startswith('TEMP_')]
        df = apply_quality_control(df, data_columns)
        
        print(f"✅ Berhasil memproses {len(df)} baris temperature data")
        print(f"    Kolom temperature: {[col for col in df.columns if col.startswith('TEMP_')]}")
    
    return df

def parse_depth_header(depth_line):
    """
    Parse header kedalaman dan return mapping depth -> data_index
    
    Returns:
    dict: {depth_value: data_index}
    """
    
    # Ekstrak bagian setelah "Depth(M):"
    depth_part = depth_line.split('Depth(M):')[1].strip()
    
    # Split dan ambil hanya angka kedalaman (sebelum "QUALITY SOURCE")
    parts = depth_part.split()
    
    depth_mapping = {}
    data_index = 0  # Index dalam array data (dimulai dari 0)
    
    for part in parts:
        # Stop jika mencapai "QUALITY" atau "SOURCE"
        if part.upper() in ['QUALITY', 'SOURCE', 'QQQQQQQQQQQQ', 'SSSSSSSSSSSS']:
            break
            
        try:
            depth_value = float(part)
            depth_mapping[depth_value] = data_index
            data_index += 1
        except ValueError:
            # Skip non-numeric parts
            continue
    
    return depth_mapping

def debug_temperature_structure(input_file):
    """
    Fungsi debug untuk melihat struktur data temperature
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    print(f"\n🔍 DEBUG: Analisis struktur {os.path.basename(input_file)}")
    print("=" * 50)
    
    current_depth_info = None
    sample_data_shown = False
    
    for line_num, line in enumerate(lines[:50], 1):  # Check first 50 lines
        line = line.strip()
        
        if 'Depth(M):' in line:
            current_depth_info = parse_depth_header(line)
            print(f"\n📏 Header kedalaman (baris {line_num}):")
            print(f"    Raw: {line}")
            print(f"    Parsed: {current_depth_info}")
            sample_data_shown = False
        
        elif re.match(r'^\s*\d{8}\s+\d{4}', line) and not sample_data_shown:
            if current_depth_info:
                parts = line.split()
                data_values = parts[2:]
                print(f"\n📊 Sample data (baris {line_num}):")
                print(f"    Raw: {line}")
                print(f"    Data values: {data_values[:min(len(data_values), 15)]}...")
                print(f"    Jumlah nilai: {len(data_values)}")
                
                # Show mapping untuk beberapa kedalaman target
                target_depths = [10.0, 13.0, 20.0, 40.0]
                for depth in target_depths:
                    if depth in current_depth_info:
                        idx = current_depth_info[depth]
                        if idx < len(data_values):
                            print(f"    {depth}m -> index {idx} -> value: {data_values[idx]}")
                        else:
                            print(f"    {depth}m -> index {idx} -> OUT OF RANGE!")
                    else:
                        print(f"    {depth}m -> NOT AVAILABLE")
                
                sample_data_shown = True


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

if __name__ == "__main__":
    # ============================================================================
    # KONFIGURASI UTAMA
    # ============================================================================
    
    # Path dan lokasi
    base_directory = '/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data Buoys'
    location_names = ['0N90E', '4N90E', '8N90E']
    output_filename = 'Buoys_Data_All.csv'
    
    # Banner informasi
    print("\n" + "="*80)
    print("🌊 BUOY DATA PROCESSOR - ADVANCED QUALITY CONTROL SYSTEM 🌊")
    print("="*80)
    print(f"📂 Base Directory: {base_directory}")
    print(f"📍 Locations: {', '.join(location_names)}")
    print(f"📄 Output File: {output_filename}")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # ============================================================================
    # VALIDASI AWAL
    # ============================================================================
    
    print("\n🔍 VALIDASI AWAL:")
    print("-" * 40)
    
    # Cek base directory
    if not os.path.exists(base_directory):
        print(f"❌ Base directory tidak ditemukan: {base_directory}")
        exit(1)
    else:
        print(f"✅ Base directory OK: {base_directory}")
    
    # Cek setiap lokasi
    valid_locations = []
    for location in location_names:
        location_path = os.path.join(base_directory, location)
        ascii_path = os.path.join(location_path, 'ASCII')
        
        if os.path.exists(location_path):
            if os.path.exists(ascii_path):
                # Count available ASCII files
                ascii_files = glob.glob(os.path.join(ascii_path, '*.ascii'))
                print(f"✅ {location}: {len(ascii_files)} file ASCII ditemukan")
                valid_locations.append(location)
            else:
                print(f"⚠️ {location}: Direktori ASCII tidak ditemukan")
        else:
            print(f"❌ {location}: Direktori lokasi tidak ditemukan")
    
    if not valid_locations:
        print("\n❌ Tidak ada lokasi valid yang ditemukan!")
        exit(1)
    
    print(f"\n📊 Total lokasi valid: {len(valid_locations)}/{len(location_names)}")
    
    # ============================================================================
    # AUTO DEBUG MODE UNTUK TEMPERATURE DATA
    # ============================================================================
    
    print("\n🐛 DEBUG MODE - Menganalisis struktur temperature data...")
    print("-" * 50)
    
    for location in valid_locations:
        temp_file = os.path.join(base_directory, location, 'ASCII', f't{location.lower()}_dy.ascii')
        if os.path.exists(temp_file):
            print(f"\n📏 Analisis struktur: {location}")
            debug_temperature_structure(temp_file)
        else:
            print(f"⚠️ File temperature tidak ditemukan untuk {location}")
    
    # ============================================================================
    # PROCESSING UTAMA
    # ============================================================================
    
    print(f"\n🚀 MEMULAI PROCESSING DATA...")
    print("-" * 40)
    
    start_time = datetime.now()
    
    # Proses semua lokasi
    result_file = process_all_locations_to_single_csv(
        base_directory=base_directory,
        location_names=valid_locations,  # Gunakan lokasi yang valid saja
        output_file=output_filename
    )
    
    end_time = datetime.now()
    processing_time = end_time - start_time
    
    # ============================================================================
    # HASIL DAN ANALISIS
    # ============================================================================
    
    if result_file:
        print(f"\n🎉 KONVERSI BERHASIL!")
        print("="*60)
        print(f"📁 File tersimpan: {result_file}")
        print(f"⏱️ Waktu processing: {processing_time}")
        print(f"🏁 Selesai pada: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # ========================================================================
        # ANALISIS MENDALAM
        # ========================================================================
        
        try:
            print("\n📊 LOADING DATA UNTUK ANALISIS...")
            df_preview = pd.read_csv(result_file)
            
            # Basic Info
            print(f"\n📋 INFORMASI DATASET:")
            print("-" * 30)
            print(f"   📏 Shape: {df_preview.shape}")
            print(f"   💾 Memory: {df_preview.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"   📅 Date Range: {df_preview['Date'].min()} → {df_preview['Date'].max()}")
            print(f"   📍 Locations: {', '.join(df_preview['Location'].unique())}")
            
            # Data Availability Analysis
            print(f"\n🎯 DATA AVAILABILITY ANALYSIS:")
            print("-" * 40)
            
            data_cols = [col for col in df_preview.columns if col not in ['Date', 'Year', 'Month', 'Day', 'Location']]
            availability_stats = {}
            
            for col in data_cols:
                if col in df_preview.columns:
                    total_points = len(df_preview)
                    valid_points = df_preview[col].notna().sum()
                    availability_pct = (valid_points / total_points) * 100
                    availability_stats[col] = availability_pct
                    
                    # Color coding untuk output
                    if availability_pct >= 80:
                        status = "✅"
                    elif availability_pct >= 60:
                        status = "🟡"
                    elif availability_pct >= 40:
                        status = "🟠"
                    else:
                        status = "🔴"
                    
                    print(f"   {status} {col:<12}: {availability_pct:6.1f}% ({valid_points:,}/{total_points:,})")
            
            # Location-wise Analysis
            print(f"\n📍 ANALISIS PER LOKASI:")
            print("-" * 30)
            
            for location in df_preview['Location'].unique():
                location_data = df_preview[df_preview['Location'] == location]
                date_range = f"{location_data['Date'].min()} → {location_data['Date'].max()}"
                print(f"   🏝️ {location}: {len(location_data):,} records ({date_range})")
                
                # Top 3 variables dengan data terlengkap untuk lokasi ini
                loc_availability = {}
                for col in data_cols:
                    if col in location_data.columns:
                        availability = (location_data[col].notna().sum() / len(location_data)) * 100
                        loc_availability[col] = availability
                
                top_vars = sorted(loc_availability.items(), key=lambda x: x[1], reverse=True)[:3]
                top_vars_str = ", ".join([f"{var}({pct:.0f}%)" for var, pct in top_vars])
                print(f"      📈 Best variables: {top_vars_str}")
            
            # Temperature Depth Analysis (jika ada)
            temp_cols = [col for col in df_preview.columns if col.startswith('TEMP_')]
            if temp_cols:
                print(f"\n🌡️ ANALISIS KEDALAMAN TEMPERATURE:")
                print("-" * 40)
                
                # Extract depth values dan sort
                depth_analysis = {}
                for col in temp_cols:
                    depth_str = col.replace('TEMP_', '').replace('m', '')
                    try:
                        depth = float(depth_str)
                        availability = (df_preview[col].notna().sum() / len(df_preview)) * 100
                        depth_analysis[depth] = availability
                    except ValueError:
                        continue
                
                # Sort by depth
                sorted_depths = sorted(depth_analysis.items())
                for depth, availability in sorted_depths:
                    status = "✅" if availability >= 70 else "🟡" if availability >= 50 else "🔴"
                    print(f"   {status} {depth:6.0f}m: {availability:6.1f}% available")
            
                # Quality Issues Detection
                print(f"\n🔍 DETEKSI MASALAH KUALITAS:")
                print("-" * 35)
                issues_found = False
            # Check for suspicious values
            for col in data_cols:
                if col in df_preview.columns and df_preview[col].dtype in ['float64', 'int64']:
                # Check for problematic patterns
                    problematic_patterns = [
                    ('Large numbers (>1000)', df_preview[col] > 1000),
                    ('Negative values', df_preview[col] < 0),
            # Fixed: Use simpler pattern to avoid regex warning about capture groups
                    ('Repeated digits', df_preview[col].astype(str).str.contains(r'\d{5,}', na=False, regex=True))
                    ]
        
            for pattern_name, mask in problematic_patterns:
                if mask.any():
                    count = mask.sum()
                    pct = (count / len(df_preview)) * 100
                    if pct > 0.1:  # Only report if >0.1%
                        print(f"   ⚠️ {col}: {count} {pattern_name} ({pct:.2f}%)")
                        issues_found = True

            if not issues_found:
                print("   ✅ Tidak ditemukan masalah kualitas yang signifikan")
            
            # Sample Data Preview
            print(f"\n📋 PREVIEW DATA (5 baris pertama):")
            print("-" * 50)
            
            # Select important columns for preview
            preview_cols = ['Date', 'Location']
            for col in ['RAD', 'RAIN', 'RH', 'SST']:
                if col in df_preview.columns:
                    preview_cols.append(col)
            
            # Add first few temperature columns
            temp_preview = [col for col in temp_cols[:3]]
            preview_cols.extend(temp_preview)
            
            # Add wind data if available
            for col in ['WSPD', 'WDIR']:
                if col in df_preview.columns:
                    preview_cols.append(col)
            
            # Display preview
            preview_data = df_preview[preview_cols].head()
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(preview_data.to_string(index=False))
            
            # Export Summary
            print(f"\n📤 EXPORT SUMMARY:")
            print("-" * 25)
            print(f"   📄 Filename: {os.path.basename(result_file)}")
            print(f"   📏 File size: {os.path.getsize(result_file) / 1024**2:.2f} MB")
            print(f"   🗂️ Columns: {len(df_preview.columns)}")
            print(f"   📊 Records: {len(df_preview):,}")
            
        except Exception as e:
            print(f"\n⚠️ Error saat analisis: {e}")
            print("   File berhasil dibuat tapi tidak dapat dianalisis")
            
    else:
        print(f"\n❌ KONVERSI GAGAL!")
        print("   Periksa log error di atas untuk detail masalah")
        
    # ============================================================================
    # PENUTUP
    # ============================================================================
        
    print("\n" + "="*80)
    print("🏁 BUOY DATA PROCESSING COMPLETED")
    print("="*80)
    print(f"🕐 Total Runtime: {processing_time}")
    print(f"📊 Quality Control: {'ENABLED' if result_file else 'FAILED'}")
    print(f"🎯 Status: {'SUCCESS' if result_file else 'FAILED'}")