import csv
import math
import re

# Load the dataset
file_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data Power NASA/Aceh Besar/Kec Indrapuri/POWER_Point_Daily_20050101_20250930_005d42N_095d45E_LST.csv"
output_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data Power NASA/preprocessing/deskriptif/statistik_deskriptif_nasa.txt"

# Variabel NASA POWER
variables = ['T2M_MIN', 'T2M_MAX', 'T2M', 'RH2M', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN', 'WS10M', 'WS10M_MAX', 'WD10M']
data_dict = {var: [] for var in variables}
invalid_counts = {var: 0 for var in variables}
total_counts = {var: 0 for var in variables}

# Metadata untuk laporan
metadata = {
    'location': '',
    'elevation': '',
    'start_date': '',
    'end_date': '',
    'lat': '',
    'lon': ''
}

def detect_header_end(file_path):
    """
    Deteksi akhir header NASA POWER menggunakan regex pattern
    Returns: number of lines to skip
    """
    header_end_pattern = r'-END HEADER-'
    
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                # Extract metadata
                if 'latitude' in line and 'longitude' in line:
                    lat_match = re.search(r'latitude\s+([\d.-]+)', line)
                    lon_match = re.search(r'longitude\s+([\d.-]+)', line)
                    if lat_match:
                        metadata['lat'] = lat_match.group(1)
                    if lon_match:
                        metadata['lon'] = lon_match.group(1)
                
                if 'elevation from MERRA-2' in line:
                    elev_match = re.search(r'=\s*([\d.]+)\s*meters', line)
                    if elev_match:
                        metadata['elevation'] = elev_match.group(1)
                
                if 'Dates' in line:
                    date_match = re.search(r'(\d{2}/\d{2}/\d{4})\s+through\s+(\d{2}/\d{2}/\d{4})', line)
                    if date_match:
                        metadata['start_date'] = date_match.group(1)
                        metadata['end_date'] = date_match.group(2)
                
                if re.search(header_end_pattern, line):
                    return line_num
        
        print("⚠️  Warning: '-END HEADER-' pattern tidak ditemukan, menggunakan fallback skip=10")
        return 10
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return 10

def is_invalid(val, col_name):
    """
    Cek apakah nilai invalid untuk NASA POWER data
    NASA menggunakan -999 untuk missing values
    """
    if val is None:
        return True
    if math.isnan(val):
        return True
    if val == -999.0:  # NASA POWER missing value
        return True
    # Untuk PRECTOTCORR, nilai 0 adalah valid (tidak hujan)
    # Untuk wind direction (WD10M), 0 adalah valid (North)
    return False

def percentile(N, percent, key=lambda x:x):
    """Calculate percentile"""
    if not N:
        return None
    k = (len(N)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c-k)
    d1 = key(N[int(c)]) * (k-f)
    return d0+d1

def calculate_skewness(data, mean_val, std_dev):
    """Calculate skewness"""
    if std_dev == 0 or len(data) < 3:
        return None
    n = len(data)
    skew = sum(((x - mean_val) / std_dev) ** 3 for x in data) / n
    return skew

def calculate_kurtosis(data, mean_val, std_dev):
    """Calculate kurtosis"""
    if std_dev == 0 or len(data) < 4:
        return None
    n = len(data)
    kurt = sum(((x - mean_val) / std_dev) ** 4 for x in data) / n - 3
    return kurt

# Detect header
print("🔍 Detecting NASA POWER header...")
skip_rows = detect_header_end(file_path)
print(f"✅ Header detection: Skipping {skip_rows} baris (hingga -END HEADER-)")

# Read CSV dengan skip header
with open(file_path, mode='r') as csv_file:
    # Skip header lines
    for _ in range(skip_rows):
        next(csv_file)
    
    reader = csv.DictReader(csv_file)
    row_count = 0
    
    for row in reader:
        row_count += 1
        for var in variables:
            if var in row:
                total_counts[var] += 1
                try:
                    val = float(row[var]) if row[var].strip() else None
                except ValueError:
                    val = None
                
                if is_invalid(val, var):
                    invalid_counts[var] += 1
                else:
                    data_dict[var].append(val)

print(f"✅ Data loaded: {row_count} records")

# Calculate statistics
results = []

for var in variables:
    valid_data = sorted(data_dict[var])
    n_valid = len(valid_data)
    
    if n_valid > 0:
        mean_val = sum(valid_data) / n_valid
        variance = sum((x - mean_val) ** 2 for x in valid_data) / n_valid
        std_dev = math.sqrt(variance)
        skewness = calculate_skewness(valid_data, mean_val, std_dev)
        kurtosis = calculate_kurtosis(valid_data, mean_val, std_dev)
        
        stats = {
            'Variable': var,
            'Jumlah Data': total_counts[var],
            'Data Invalid': invalid_counts[var],
            'Count': n_valid,
            'Minimum': valid_data[0],
            'Q1': percentile(valid_data, 0.25),
            'Median': percentile(valid_data, 0.5),
            'Mean': mean_val,
            'Q3': percentile(valid_data, 0.75),
            'Maksimum': valid_data[-1],
            'Std. Dev': std_dev,
            'Skewness': skewness,
            'Kurtosis': kurtosis
        }
    else:
        stats = {
            'Variable': var,
            'Jumlah Data': total_counts[var],
            'Data Invalid': invalid_counts[var],
            'Count': 0,
            'Minimum': None,
            'Q1': None,
            'Median': None,
            'Mean': None,
            'Q3': None,
            'Maksimum': None,
            'Std. Dev': None,
            'Skewness': None,
            'Kurtosis': None
        }
    results.append(stats)

# Unit mapping untuk setiap variabel
unit_mapping = {
    'T2M_MIN': '°C',
    'T2M_MAX': '°C',
    'T2M': '°C',
    'RH2M': '%',
    'PRECTOTCORR': 'mm/day',
    'ALLSKY_SFC_SW_DWN': 'MJ/m²/day',
    'WS10M': 'm/s',
    'WS10M_MAX': 'm/s',
    'WD10M': 'degrees'
}

# Variable name mapping (deskripsi lengkap)
var_descriptions = {
    'T2M_MIN': 'Temperature at 2 Meters Minimum',
    'T2M_MAX': 'Temperature at 2 Meters Maximum',
    'T2M': 'Temperature at 2 Meters (Average)',
    'RH2M': 'Relative Humidity at 2 Meters',
    'PRECTOTCORR': 'Precipitation Corrected',
    'ALLSKY_SFC_SW_DWN': 'All Sky Surface Shortwave Downward Irradiance',
    'WS10M': 'Wind Speed at 10 Meters',
    'WS10M_MAX': 'Wind Speed at 10 Meters Maximum',
    'WD10M': 'Wind Direction at 10 Meters'
}

# Write output file dengan format mirip preprocessing report
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("STATISTIK DESKRIPTIF DATASET NASA POWER (20 TAHUN)\n")
    f.write("="*70 + "\n\n")
    
    # Metadata section
    f.write("📊 DATASET OVERVIEW:\n")
    f.write(f"   • Sumber Data       : NASA POWER MERRA-2\n")
    f.write(f"   • Total Records     : {total_counts[variables[0]]:,}\n")
    if metadata['start_date'] and metadata['end_date']:
        f.write(f"   • Periode           : {metadata['start_date']} s/d {metadata['end_date']}\n")
    if metadata['lat'] and metadata['lon']:
        f.write(f"   • Koordinat         : {metadata['lat']}°N, {metadata['lon']}°E\n")
    if metadata['elevation']:
        f.write(f"   • Elevasi           : {metadata['elevation']} m\n")
    f.write(f"   • Resolusi          : Harian (Daily)\n")
    f.write(f"   • Jumlah Variabel   : {len(variables)}\n")
    f.write("\n")
    
    # Overall data quality
    total_data_points = sum(total_counts.values())
    total_valid = sum(stat['Count'] for stat in results)
    total_invalid = sum(stat['Data Invalid'] for stat in results)
    
    f.write("🔍 KUALITAS DATA KESELURUHAN:\n")
    f.write(f"   • Total Data Points : {total_data_points:,}\n")
    f.write(f"   • Data Valid        : {total_valid:,} ({total_valid/total_data_points*100:.2f}%)\n")
    f.write(f"   • Data Invalid      : {total_invalid:,} ({total_invalid/total_data_points*100:.2f}%)\n")
    f.write("\n")
    f.write("="*70 + "\n\n")
    
    # Detailed statistics per variable
    for stat in results:
        var = stat['Variable']
        unit = unit_mapping.get(var, '')
        desc = var_descriptions.get(var, var)
        
        f.write(f"📊 {var} - {desc}\n")
        f.write("=" * 70 + "\n")
        
        # Data counts
        f.write(f"Jumlah Data Total    : {stat['Jumlah Data']:>10,}\n")
        f.write(f"Data Invalid (-999)  : {stat['Data Invalid']:>10,}\n")
        f.write(f"Data Valid           : {stat['Count']:>10,}\n")
        
        if stat['Count'] > 0:
            valid_pct = (stat['Count'] / stat['Jumlah Data']) * 100
            invalid_pct = (stat['Data Invalid'] / stat['Jumlah Data']) * 100
            f.write(f"Persentase Valid     : {valid_pct:>10.2f}%\n")
            f.write(f"Persentase Invalid   : {invalid_pct:>10.2f}%\n")
            f.write("\n")
            
            # Descriptive statistics
            f.write(f"Minimum              : {stat['Minimum']:>10.2f} {unit}\n")
            f.write(f"Q1 (25%)             : {stat['Q1']:>10.2f} {unit}\n")
            f.write(f"Median (Q2)          : {stat['Median']:>10.2f} {unit}\n")
            f.write(f"Mean                 : {stat['Mean']:>10.2f} {unit}\n")
            f.write(f"Q3 (75%)             : {stat['Q3']:>10.2f} {unit}\n")
            f.write(f"Maksimum             : {stat['Maksimum']:>10.2f} {unit}\n")
            f.write(f"Standar Deviasi      : {stat['Std. Dev']:>10.2f} {unit}\n")
            
            # Range and IQR
            range_val = stat['Maksimum'] - stat['Minimum']
            iqr = stat['Q3'] - stat['Q1']
            f.write(f"Range                : {range_val:>10.2f} {unit}\n")
            f.write(f"IQR (Q3-Q1)          : {iqr:>10.2f} {unit}\n")
            
            # Skewness and Kurtosis
            if stat['Skewness'] is not None:
                f.write(f"Skewness             : {stat['Skewness']:>10.3f}\n")
            if stat['Kurtosis'] is not None:
                f.write(f"Kurtosis             : {stat['Kurtosis']:>10.3f}\n")
            
            # Interpretation
            f.write("\n📋 INTERPRETASI:\n")
            
            # Skewness interpretation
            if stat['Skewness'] is not None:
                if abs(stat['Skewness']) < 0.5:
                    f.write("   • Distribusi: Simetris (mendekati normal)\n")
                elif stat['Skewness'] > 0.5:
                    f.write("   • Distribusi: Positively skewed (ekor kanan lebih panjang)\n")
                else:
                    f.write("   • Distribusi: Negatively skewed (ekor kiri lebih panjang)\n")
            
            # Kurtosis interpretation
            if stat['Kurtosis'] is not None:
                if abs(stat['Kurtosis']) < 0.5:
                    f.write("   • Kurtosis: Mesokurtic (mendekati normal)\n")
                elif stat['Kurtosis'] > 0.5:
                    f.write("   • Kurtosis: Leptokurtic (lebih runcing dari normal)\n")
                else:
                    f.write("   • Kurtosis: Platykurtic (lebih datar dari normal)\n")
            
            # Variability
            cv = (stat['Std. Dev'] / stat['Mean']) * 100 if stat['Mean'] != 0 else 0
            f.write(f"   • Coefficient of Variation: {cv:.2f}%")
            if cv < 15:
                f.write(" (Variabilitas rendah)\n")
            elif cv < 30:
                f.write(" (Variabilitas sedang)\n")
            else:
                f.write(" (Variabilitas tinggi)\n")
        else:
            f.write("\n⚠️  TIDAK ADA DATA VALID\n")
        
        f.write("\n" + "="*70 + "\n\n")
    
    # Summary recommendations
    f.write("🎯 RINGKASAN & REKOMENDASI:\n")
    f.write("="*70 + "\n")
    
    # Check data quality issues
    high_invalid_vars = [s for s in results if s['Count'] > 0 and (s['Data Invalid'] / s['Jumlah Data']) > 0.05]
    
    if high_invalid_vars:
        f.write("⚠️  PERHATIAN - Variabel dengan Missing Data > 5%:\n")
        for s in high_invalid_vars:
            invalid_pct = (s['Data Invalid'] / s['Jumlah Data']) * 100
            f.write(f"   • {s['Variable']}: {invalid_pct:.2f}% missing\n")
        f.write("\n")
    else:
        f.write("✅ Semua variabel memiliki missing data < 5%\n\n")
    
    # Extreme values check for precipitation
    prec_stat = next((s for s in results if s['Variable'] == 'PRECTOTCORR'), None)
    if prec_stat and prec_stat['Count'] > 0:
        extreme_threshold = 50  # mm/day
        extreme_count = sum(1 for x in data_dict['PRECTOTCORR'] if x > extreme_threshold)
        f.write(f"⚡ CURAH HUJAN EKSTREM (>{extreme_threshold} mm/day):\n")
        f.write(f"   • Jumlah kejadian: {extreme_count:,} hari\n")
        f.write(f"   • Persentase: {(extreme_count/prec_stat['Count'])*100:.2f}%\n")
        f.write(f"   • Maksimum tercatat: {prec_stat['Maksimum']:.2f} mm/day\n\n")
    
    # Temperature range
    temp_min_stat = next((s for s in results if s['Variable'] == 'T2M_MIN'), None)
    temp_max_stat = next((s for s in results if s['Variable'] == 'T2M_MAX'), None)
    if temp_min_stat and temp_max_stat and temp_min_stat['Count'] > 0:
        f.write(f"🌡️  RENTANG TEMPERATUR:\n")
        f.write(f"   • Temperatur minimum absolut: {temp_min_stat['Minimum']:.2f}°C\n")
        f.write(f"   • Temperatur maksimum absolut: {temp_max_stat['Maksimum']:.2f}°C\n")
        f.write(f"   • Range total: {temp_max_stat['Maksimum'] - temp_min_stat['Minimum']:.2f}°C\n\n")
    
    f.write("📋 STATUS PREPROCESSING:\n")
    f.write("   • ✅ Data loading dan format detection\n")
    f.write("   • ✅ Statistik deskriptif lengkap\n")
    f.write("   • ✅ Identifikasi missing values (-999)\n")
    f.write("   • ✅ Analisis distribusi data\n")
    f.write("   • ⏳ Ready for advanced preprocessing\n")
    f.write("\n")
    f.write("="*70 + "\n")

print(f"✅ Statistik deskriptif NASA POWER telah disimpan di:\n   {output_path}")
print(f"\n📊 Summary:")
print(f"   • Total records: {total_counts[variables[0]]:,}")
print(f"   • Data valid: {total_valid:,} ({total_valid/total_data_points*100:.2f}%)")
print(f"   • Data invalid: {total_invalid:,} ({total_invalid/total_data_points*100:.2f}%)")