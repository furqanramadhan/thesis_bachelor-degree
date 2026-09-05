import csv
import math

# Load the dataset
file_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data BMKG/Lokasi/Kab. Aceh Besar/Stasiun Klimatologi Aceh/CSV/BMKG_Data_All.csv"
output_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data BMKG/preprocessing/deskripitf/statistik_deskriptif.txt"

variables = ['TN', 'TX', 'TAVG', 'RH_AVG', 'RR', 'SS', 'FF_X', 'DDD_X', 'FF_AVG']
data_dict = {var: [] for var in variables}
invalid_counts = {var: 0 for var in variables}
total_counts = {var: 0 for var in variables}

def is_invalid(val, col_name):
    # FIX 1: Cek None dulu sebelum isnan
    if val is None:
        return True
    if math.isnan(val):
        return True
    if val == 8888.0 or val == 9999.0:
        return True
    if val == 0 and col_name not in ["RR", "PRECTOTCORR"]:
        return True
    return False

with open(file_path, mode='r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
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

results = []

def percentile(N, percent, key=lambda x:x):
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

for var in variables:
    valid_data = sorted(data_dict[var])
    n_valid = len(valid_data)
    
    if n_valid > 0:
        mean_val = sum(valid_data) / n_valid
        # FIX 2: Variance calculation (bagi dengan n bukan n-1 untuk population variance)
        variance = sum((x - mean_val) ** 2 for x in valid_data) / n_valid
        std_dev = math.sqrt(variance)
        
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
            'Skewness': None  # Placeholder, hitung manual jika perlu
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
            'Skewness': None
        }
    results.append(stats)

# Format the output mirip preprocessing report
with open(output_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("STATISTIK DESKRIPTIF DATASET BMKG (ORIGINAL DATA)\n")
    f.write("="*60 + "\n\n")
    
    for stat in results:
        f.write(f"📊 VARIABEL: {stat['Variable']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"   • Jumlah Data Total : {stat['Jumlah Data']:,}\n")
        f.write(f"   • Data Invalid      : {stat['Data Invalid']:,}\n")
        f.write(f"   • Count (Valid)     : {stat['Count']:,}\n")
        
        if stat['Count'] > 0:
            f.write(f"   • Mean              : {stat['Mean']:.2f}\n")
            f.write(f"   • Std Dev           : {stat['Std. Dev']:.2f}\n")
            f.write(f"   • Min               : {stat['Minimum']:.2f}\n")
            f.write(f"   • Q1 (25%)          : {stat['Q1']:.2f}\n")
            f.write(f"   • Median (Q2)       : {stat['Median']:.2f}\n")
            f.write(f"   • Q3 (75%)          : {stat['Q3']:.2f}\n")
            f.write(f"   • Max               : {stat['Maksimum']:.2f}\n")
        else:
            f.write("   ⚠️  Tidak ada data valid\n")
        
        f.write("\n")

print(f"✅ Statistik deskriptif telah disimpan di {output_path}")