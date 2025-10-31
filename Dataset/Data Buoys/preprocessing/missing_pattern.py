import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Konfigurasi
base_dir = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data Buoys"
locations = ['0N90E', '4N90E', '8N90E']

def analyze_missing_patterns(df, location_name):
    """
    Analisis pola missing data untuk satu lokasi
    """
    print(f"\n{'='*80}")
    print(f"ANALISIS MISSING DATA: {location_name}")
    print(f"{'='*80}")
    
    # Konversi Date ke datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Info umum
    print(f"\n📊 INFORMASI UMUM:")
    print(f"Periode data: {df['Date'].min().date()} s/d {df['Date'].max().date()}")
    print(f"Total hari: {len(df)} hari")
    
    # Variables to analyze
    variables = ['SST', 'Prec', 'RH', 'WSPD', 'SWRad']
    
    print(f"\n📈 STATISTIK MISSING PER VARIABEL:")
    print(f"{'-'*80}")
    print(f"{'Variabel':<12} {'Missing':<12} {'Persen':<12} {'Gap Events':<15} {'Max Gap':<12}")
    print(f"{'-'*80}")
    
    gap_details = {}
    
    for var in variables:
        if var not in df.columns:
            print(f"{var:<12} {'N/A':<12} {'N/A':<12} {'N/A':<15} {'N/A':<12}")
            continue
            
        # Hitung missing
        missing_mask = df[var].isna()
        total_missing = missing_mask.sum()
        persen_missing = (total_missing / len(df)) * 100
        
        # Deteksi gap events (consecutive missing)
        gaps = []
        in_gap = False
        gap_start = None
        gap_length = 0
        
        for idx, is_missing in enumerate(missing_mask):
            if is_missing:
                if not in_gap:
                    gap_start = idx
                    gap_length = 1
                    in_gap = True
                else:
                    gap_length += 1
            else:
                if in_gap:
                    gaps.append({
                        'start_idx': gap_start,
                        'end_idx': idx - 1,
                        'start_date': df.loc[gap_start, 'Date'],
                        'end_date': df.loc[idx - 1, 'Date'],
                        'length': gap_length
                    })
                    in_gap = False
        
        # Jika masih dalam gap di akhir data
        if in_gap:
            gaps.append({
                'start_idx': gap_start,
                'end_idx': len(df) - 1,
                'start_date': df.loc[gap_start, 'Date'],
                'end_date': df.loc[len(df) - 1, 'Date'],
                'length': gap_length
            })
        
        num_gaps = len(gaps)
        max_gap = max([g['length'] for g in gaps]) if gaps else 0
        
        gap_details[var] = gaps
        
        print(f"{var:<12} {total_missing:<12} {persen_missing:<12.2f} {num_gaps:<15} {max_gap:<12}")
    
    # Detail per variabel
    for var in variables:
        if var not in gap_details or not gap_details[var]:
            continue
            
        print(f"\n{'='*80}")
        print(f"🔍 DETAIL GAP: {var}")
        print(f"{'='*80}")
        
        gaps = gap_details[var]
        
        # Distribusi gap length
        gap_lengths = [g['length'] for g in gaps]
        print(f"\nDistribusi Panjang Gap:")
        print(f"  1 hari      : {sum(1 for x in gap_lengths if x == 1)} events")
        print(f"  2-3 hari    : {sum(1 for x in gap_lengths if 2 <= x <= 3)} events")
        print(f"  4-7 hari    : {sum(1 for x in gap_lengths if 4 <= x <= 7)} events")
        print(f"  8-14 hari   : {sum(1 for x in gap_lengths if 8 <= x <= 14)} events")
        print(f"  15-30 hari  : {sum(1 for x in gap_lengths if 15 <= x <= 30)} events")
        print(f"  >30 hari    : {sum(1 for x in gap_lengths if x > 30)} events")
        
        # Top 10 gap terpanjang
        sorted_gaps = sorted(gaps, key=lambda x: x['length'], reverse=True)[:10]
        
        print(f"\n📉 Top 10 Gap Terpanjang:")
        print(f"{'-'*80}")
        print(f"{'No':<5} {'Start':<15} {'End':<15} {'Duration':<12} {'Tahun':<8}")
        print(f"{'-'*80}")
        
        for i, gap in enumerate(sorted_gaps, 1):
            start_date = gap['start_date'].strftime('%Y-%m-%d')
            end_date = gap['end_date'].strftime('%Y-%m-%d')
            duration = gap['length']
            year = gap['start_date'].year
            
            print(f"{i:<5} {start_date:<15} {end_date:<15} {duration:<12} {year:<8}")
    
    # Analisis temporal (per tahun)
    print(f"\n{'='*80}")
    print(f"📅 MISSING DATA PER TAHUN:")
    print(f"{'='*80}")
    
    df['Year'] = df['Date'].dt.year
    years = sorted(df['Year'].unique())
    
    print(f"\n{'Tahun':<8}", end='')
    for var in variables:
        if var in df.columns:
            print(f"{var:<12}", end='')
    print()
    print(f"{'-'*80}")
    
    for year in years:
        year_data = df[df['Year'] == year]
        print(f"{year:<8}", end='')
        
        for var in variables:
            if var in df.columns:
                missing_count = year_data[var].isna().sum()
                total_days = len(year_data)
                pct = (missing_count / total_days * 100) if total_days > 0 else 0
                print(f"{missing_count}({pct:.1f}%){' ':<3}", end='')
            else:
                print(f"{'N/A':<12}", end='')
        print()
    
    # Analisis temporal (per bulan)
    print(f"\n{'='*80}")
    print(f"📅 MISSING DATA PER BULAN (agregat semua tahun):")
    print(f"{'='*80}")
    
    df['Month'] = df['Date'].dt.month
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    print(f"\n{'Bulan':<8}", end='')
    for var in variables:
        if var in df.columns:
            print(f"{var:<12}", end='')
    print()
    print(f"{'-'*80}")
    
    for month in range(1, 13):
        month_data = df[df['Month'] == month]
        print(f"{month_names[month-1]:<8}", end='')
        
        for var in variables:
            if var in df.columns:
                missing_count = month_data[var].isna().sum()
                total_days = len(month_data)
                pct = (missing_count / total_days * 100) if total_days > 0 else 0
                print(f"{missing_count}({pct:.1f}%){' ':<3}", end='')
            else:
                print(f"{'N/A':<12}", end='')
        print()
    
    # Simultaneous missing
    print(f"\n{'='*80}")
    print(f"🔗 SIMULTANEOUS MISSING (variabel hilang bersamaan):")
    print(f"{'='*80}")
    
    # Check all combinations
    from itertools import combinations
    
    available_vars = [v for v in variables if v in df.columns]
    
    for r in range(2, len(available_vars) + 1):
        for combo in combinations(available_vars, r):
            mask = df[list(combo)].isna().all(axis=1)
            count = mask.sum()
            if count > 0:
                pct = (count / len(df)) * 100
                print(f"  {' + '.join(combo)}: {count} hari ({pct:.2f}%)")
    
    return gap_details

# Main process
print("\n" + "="*80)
print("MEMULAI ANALISIS POLA MISSING DATA BUOYS")
print("="*80)

for location in locations:
    combined_file = Path(base_dir) / location / 'CSV' / 'COMBINED' / f'{location}.csv'
    
    if not combined_file.exists():
        print(f"\n⚠️  File tidak ditemukan: {combined_file}")
        continue
    
    # Load data
    df = pd.read_csv(combined_file)
    
    # Analisis
    gap_details = analyze_missing_patterns(df, location)

print(f"\n{'='*80}")
print("✅ ANALISIS SELESAI")
print(f"{'='*80}")