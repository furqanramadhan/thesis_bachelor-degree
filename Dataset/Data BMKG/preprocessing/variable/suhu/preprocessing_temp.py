import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class Temperature_Analyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.tn_stats = {}
        self.tx_stats = {}

    def load_data(self):
        print("🌡️  TEMPERATURE STATISTICS ANALYZER - DATASET BMKG")
        print("="*55)

        try:
            print(f"📂 Loading data dari: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values('Date').reset_index(drop=True)

            print(f"✅ Data berhasil dimuat: {len(self.data)} records")
            print(f"📅 Periode: {self.data['Date'].min()} s/d {self.data['Date'].max()}")
            print(f"📋 Kolom tersedia: {list(self.data.columns)}")

            # Validasi kolom temperature
            missing_cols = []
            if 'TN' not in self.data.columns:
                missing_cols.append('TN')
            if 'TX' not in self.data.columns:
                missing_cols.append('TX')
            
            if missing_cols:
                raise ValueError(f"❌ Kolom temperature tidak ditemukan: {missing_cols}")

            print("✅ Kolom TN dan TX ditemukan")
            return True

        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False

    def analyze_missing_values(self, column_name, show_details=True):
        """
        Analisis missing values untuk kolom temperature tertentu (TN atau TX)
        """
        if show_details:
            print(f"\n=== ANALISIS MISSING VALUES {column_name.upper()} ===")

        if column_name not in self.data.columns:
            print(f"❌ Kolom {column_name} tidak ditemukan!")
            return None

        total_records = len(self.data)
        missing_nan = self.data[column_name].isna().sum()

        # Deteksi nilai khusus BMKG
        special_values = {}
        for val in [9999, 8888, -999, -9999]:
            count = (self.data[column_name] == val).sum()
            if count > 0:
                special_values[val] = count

        total_missing = missing_nan + sum(special_values.values())

        if show_details:
            print(f"📊 Total records: {total_records:,}")
            print(f"📊 Missing/NaN values: {missing_nan:,} ({missing_nan/total_records*100:.2f}%)")

            if special_values:
                print(f"📊 Nilai khusus ditemukan:")
                for val, count in special_values.items():
                    meaning = {9999: "Tidak ada pengukuran", 8888: "Data tidak terukur", 
                              -999: "Error value", -9999: "Error value"}
                    print(f"   • Nilai {val} ({meaning.get(val, 'Unknown')}): {count:,} ({count/total_records*100:.2f}%)")
            else:
                print("✅ Tidak ada nilai khusus yang terdeteksi")

            print(f"📊 Total missing data: {total_missing:,} ({total_missing/total_records*100:.2f}%)")

        # Generate valid data
        valid_data = self.data[column_name].copy()
        valid_data = valid_data.dropna()
        for val in special_values.keys():
            valid_data = valid_data[valid_data != val]

        valid_count = len(valid_data)
        
        if show_details:
            print(f"📊 Valid data: {valid_count:,} ({valid_count/total_records*100:.2f}%)")

        return {
            'total_records': total_records,
            'missing_count': total_missing,
            'valid_data': valid_data,
            'valid_count': valid_count,
            'missing_percentage': total_missing/total_records*100
        }

    def calculate_descriptive_statistics(self, column_name):
        """
        Menghitung statistik deskriptif lengkap untuk kolom temperature
        """
        print(f"\n=== STATISTIK DESKRIPTIF {column_name.upper()} ===")

        # Analisis missing values
        missing_analysis = self.analyze_missing_values(column_name, show_details=True)
        
        if missing_analysis is None or len(missing_analysis['valid_data']) == 0:
            print("⚠️ Tidak ada data valid untuk dianalisis")
            return None

        valid_data = missing_analysis['valid_data']
        
        # Hitung semua statistik yang diperlukan
        stats_dict = {
            'Jumlah Data': len(valid_data),
            'Jumlah Missing Value': missing_analysis['missing_count'],
            'Minimum': valid_data.min(),
            'Q1': valid_data.quantile(0.25),
            'Median': valid_data.median(),
            'Mean': valid_data.mean(), 
            'Q3': valid_data.quantile(0.75),
            'Maksimum': valid_data.max(),
            'Standar Deviasi': valid_data.std()
        }

        # Tampilkan dalam format tabel yang rapi
        print(f"\n📈 TABEL STATISTIK DESKRIPTIF {column_name.upper()}:")
        print("-" * 40)
        print(f"{'Statistik':<20} {'Nilai':<15}")
        print("-" * 40)
        
        for stat_name, value in stats_dict.items():
            if stat_name in ['Jumlah Data', 'Jumlah Missing Value']:
                print(f"{stat_name:<20} {int(value):<15,}")
            else:
                print(f"{stat_name:<20} {value:<15.2f}")
        
        print("-" * 40)

        # Informasi tambahan
        print(f"\n📊 INFORMASI TAMBAHAN:")
        range_val = stats_dict['Maksimum'] - stats_dict['Minimum']
        iqr = stats_dict['Q3'] - stats_dict['Q1']
        cv = (stats_dict['Standar Deviasi'] / stats_dict['Mean']) * 100 if stats_dict['Mean'] != 0 else 0
        
        print(f"   • Range: {range_val:.2f}°C")
        print(f"   • IQR (Interquartile Range): {iqr:.2f}°C")
        print(f"   • Coefficient of Variation: {cv:.2f}%")
        
        # Deteksi outliers menggunakan IQR method
        lower_bound = stats_dict['Q1'] - 1.5 * iqr
        upper_bound = stats_dict['Q3'] + 1.5 * iqr
        outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
        
        print(f"   • Batas outlier (IQR): {lower_bound:.2f}°C - {upper_bound:.2f}°C")
        print(f"   • Jumlah outliers: {len(outliers)} ({len(outliers)/len(valid_data)*100:.2f}%)")

        # Klasifikasi temperature (untuk konteks iklim Indonesia)
        if column_name.upper() == 'TN':  # Temperature minimum
            temp_categories = {
                'Sangat Dingin (<20°C)': valid_data < 20,
                'Dingin (20-23°C)': (valid_data >= 20) & (valid_data < 23),
                'Normal (23-26°C)': (valid_data >= 23) & (valid_data < 26),
                'Hangat (≥26°C)': valid_data >= 26
            }
        else:  # Temperature maximum  
            temp_categories = {
                'Normal (<30°C)': valid_data < 30,
                'Hangat (30-33°C)': (valid_data >= 30) & (valid_data < 33),
                'Panas (33-36°C)': (valid_data >= 33) & (valid_data < 36),
                'Sangat Panas (≥36°C)': valid_data >= 36
            }

        print(f"\n🌡️  DISTRIBUSI KATEGORI TEMPERATURE {column_name.upper()}:")
        for category, mask in temp_categories.items():
            count = mask.sum()
            percentage = count / len(valid_data) * 100
            print(f"   • {category}: {count:,} ({percentage:.1f}%)")

        return stats_dict

    def compare_tn_tx(self):
        """
        Perbandingan statistik antara TN dan TX
        """
        print(f"\n=== PERBANDINGAN STATISTIK TN vs TX ===")
        
        # Hitung statistik untuk kedua kolom
        tn_analysis = self.analyze_missing_values('TN', show_details=False)
        tx_analysis = self.analyze_missing_values('TX', show_details=False)
        
        if tn_analysis is None or tx_analysis is None:
            print("❌ Tidak dapat melakukan perbandingan - data tidak valid")
            return
        
        tn_data = tn_analysis['valid_data']
        tx_data = tx_analysis['valid_data']
        
        # Buat tabel perbandingan
        comparison_stats = {
            'TN (°C)': {
                'Jumlah Data': len(tn_data),
                'Missing Values': tn_analysis['missing_count'],
                'Mean': tn_data.mean(),
                'Median': tn_data.median(),
                'Std Dev': tn_data.std(),
                'Min': tn_data.min(),
                'Max': tn_data.max()
            },
            'TX (°C)': {
                'Jumlah Data': len(tx_data),
                'Missing Values': tx_analysis['missing_count'],
                'Mean': tx_data.mean(),
                'Median': tx_data.median(),
                'Std Dev': tx_data.std(),   
                'Min': tx_data.min(),
                'Max': tx_data.max()
            }
        }
        
        print(f"\n📊 TABEL PERBANDINGAN TN vs TX:")
        print("-" * 60)
        print(f"{'Statistik':<15} {'TN (°C)':<20} {'TX (°C)':<20}")
        print("-" * 60)
        
        stat_order = ['Jumlah Data', 'Missing Values', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
        for stat in stat_order:
            tn_val = comparison_stats['TN (°C)'][stat]
            tx_val = comparison_stats['TX (°C)'][stat]
            
            if stat in ['Jumlah Data', 'Missing Values']:
                print(f"{stat:<15} {int(tn_val):<20,} {int(tx_val):<20,}")
            else:
                print(f"{stat:<15} {tn_val:<20.2f} {tx_val:<20.2f}")
        
        print("-" * 60)
        
        # Analisis hubungan TN-TX
        if len(tn_data) > 0 and len(tx_data) > 0:
            # Cari data yang valid untuk kedua kolom pada tanggal yang sama
            valid_both = self.data.dropna(subset=['TN', 'TX'])
            for val in [9999, 8888, -999, -9999]:
                valid_both = valid_both[(valid_both['TN'] != val) & (valid_both['TX'] != val)]
            
            if len(valid_both) > 1:
                correlation = valid_both['TN'].corr(valid_both['TX'])
                avg_diurnal_range = (valid_both['TX'] - valid_both['TN']).mean()
                
                print(f"\n🔄 ANALISIS HUBUNGAN TN-TX:")
                print(f"   • Korelasi TN-TX: {correlation:.3f}")
                print(f"   • Rata-rata selisih (TX-TN): {avg_diurnal_range:.2f}°C")
                print(f"   • Data valid bersamaan: {len(valid_both):,} records")

    def seasonal_analysis_temperature(self):
        """
        Analisis pola musiman untuk temperature
        """
        print(f"\n=== ANALISIS MUSIMAN TEMPERATURE ===")
        
        # Filter data valid untuk TN dan TX
        valid_data = self.data.copy()
        for col in ['TN', 'TX']:
            valid_mask = (
                valid_data[col].notna() & 
                (valid_data[col] != 9999) & 
                (valid_data[col] != 8888) &
                (valid_data[col] != -999) & 
                (valid_data[col] != -9999)
            )
            valid_data = valid_data[valid_mask]
        
        if len(valid_data) == 0:
            print("⚠️ Tidak ada data valid untuk analisis musiman")
            return
        
        # Hitung statistik bulanan
        monthly_stats = valid_data.groupby('Month')[['TN', 'TX']].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun',
                      'Jul', 'Ags', 'Sep', 'Okt', 'Nov', 'Des']
        
        print(f"\n📅 STATISTIK BULANAN TEMPERATURE:")
        print("Bulan | TN Mean | TN Med | TX Mean | TX Med | Diurnal Range")
        print("-" * 65)
        
        for month in range(1, 13):
            if month in monthly_stats.index:
                tn_mean = monthly_stats.loc[month, ('TN', 'mean')]
                tn_med = monthly_stats.loc[month, ('TN', 'median')]
                tx_mean = monthly_stats.loc[month, ('TX', 'mean')]
                tx_med = monthly_stats.loc[month, ('TX', 'median')]
                diurnal = tx_mean - tn_mean
                
                print(f"{month_names[month-1]:5s} | {tn_mean:7.1f} | {tn_med:6.1f} | "
                      f"{tx_mean:7.1f} | {tx_med:6.1f} | {diurnal:13.1f}")
            else:
                print(f"{month_names[month-1]:5s} | {'N/A':7s} | {'N/A':6s} | "
                      f"{'N/A':7s} | {'N/A':6s} | {'N/A':13s}")
        
        # Identifikasi bulan ekstrem
        if not monthly_stats.empty:
            # TN
            warmest_tn_month = monthly_stats[('TN', 'mean')].idxmax()
            coolest_tn_month = monthly_stats[('TN', 'mean')].idxmin()
            
            # TX  
            warmest_tx_month = monthly_stats[('TX', 'mean')].idxmax()
            coolest_tx_month = monthly_stats[('TX', 'mean')].idxmin()
            
            print(f"\n🌡️  KARAKTERISTIK MUSIMAN:")
            print(f"   • TN tertinggi: {month_names[warmest_tn_month-1]} ({monthly_stats.loc[warmest_tn_month, ('TN', 'mean')]:.1f}°C)")
            print(f"   • TN terendah: {month_names[coolest_tn_month-1]} ({monthly_stats.loc[coolest_tn_month, ('TN', 'mean')]:.1f}°C)")
            print(f"   • TX tertinggi: {month_names[warmest_tx_month-1]} ({monthly_stats.loc[warmest_tx_month, ('TX', 'mean')]:.1f}°C)")
            print(f"   • TX terendah: {month_names[coolest_tx_month-1]} ({monthly_stats.loc[coolest_tx_month, ('TX', 'mean')]:.1f}°C)")

    def generate_summary_table(self):
        """
        Generate tabel ringkasan untuk kedua parameter temperature
        """
        print(f"\n" + "="*70)
        print("TABEL RINGKASAN STATISTIK DESKRIPTIF TEMPERATURE")
        print("="*70)
        
        # Hitung statistik untuk TN dan TX
        tn_analysis = self.analyze_missing_values('TN', show_details=False)
        tx_analysis = self.analyze_missing_values('TX', show_details=False)
        
        if tn_analysis is None or tx_analysis is None:
            print("❌ Tidak dapat generate tabel - data tidak valid")
            return
        
        tn_data = tn_analysis['valid_data']
        tx_data = tx_analysis['valid_data']
        
        # Buat tabel final sesuai format yang diminta
        final_stats = {
            'Statistik': [
                'Jumlah Data',
                'Jumlah Missing Value', 
                'Minimum',
                'Q1',
                'Median',
                'Mean',
                'Q3',
                'Maksimum',
                'Standar Deviasi'
            ],
            'TN (°C)': [
                len(tn_data),
                tn_analysis['missing_count'],
                tn_data.min(),
                tn_data.quantile(0.25),
                tn_data.median(),
                tn_data.mean(),
                tn_data.quantile(0.75), 
                tn_data.max(),
                tn_data.std()
            ],
            'TX (°C)': [
                len(tx_data),
                tx_analysis['missing_count'],
                tx_data.min(),
                tx_data.quantile(0.25),
                tx_data.median(),
                tx_data.mean(),
                tx_data.quantile(0.75),
                tx_data.max(),
                tx_data.std()
            ]
        }
        
        print(f"{'Statistik':<20} {'TN (°C)':<15} {'TX (°C)':<15}")
        print("-" * 50)
        
        for i, stat in enumerate(final_stats['Statistik']):
            tn_val = final_stats['TN (°C)'][i]
            tx_val = final_stats['TX (°C)'][i]
            
            if stat in ['Jumlah Data', 'Jumlah Missing Value']:
                print(f"{stat:<20} {int(tn_val):<15,} {int(tx_val):<15,}")
            else:
                print(f"{stat:<20} {tn_val:<15.2f} {tx_val:<15.2f}")
        
        print("-" * 50)
        print("🎯 Tabel statistik deskriptif lengkap selesai!")
        
        return final_stats
    
    def impute_temperature_seasonal_interpolation(self, column_name):
        """
        Implementasi Seasonal Linear Interpolation untuk Temperature (TN/TX)
        
        Strategi:
        1. Linear interpolation berdasarkan jarak waktu
        2. Seasonal awareness menggunakan rata-rata bulanan historis
        3. Validasi range berdasarkan batas musiman
        4. Fallback ke seasonal mean jika interpolation tidak memungkinkan
        
        Parameters:
        -----------
        column_name : str
            Nama kolom temperature ('TN' atau 'TX')
        
        Returns:
        --------
        dict : Dictionary berisi detail imputasi yang dilakukan
        """
        print(f"\n=== IMPUTASI SEASONAL LINEAR INTERPOLATION {column_name.upper()} ===")
        
        if column_name not in self.data.columns:
            print(f"❌ Kolom {column_name} tidak ditemukan!")
            return None
        
        # 1. IDENTIFIKASI MISSING VALUES
        missing_mask = (
            self.data[column_name].isna() |
            (self.data[column_name] == 9999) |
            (self.data[column_name] == 8888) |
            (self.data[column_name] == -999) |
            (self.data[column_name] == -9999)
        )
        
        missing_indices = self.data[missing_mask].index.tolist()
        
        if not missing_indices:
            print(f"✅ Tidak ada missing values untuk {column_name}")
            return {'imputed_count': 0}
        
        print(f"🔍 Ditemukan {len(missing_indices)} missing values untuk {column_name}")
        
        # 2. HITUNG SEASONAL REFERENCE (MONTHLY AVERAGES & RANGES)
        valid_data = self.data[~missing_mask].copy()
        
        # Monthly statistics untuk seasonal awareness
        monthly_stats = valid_data.groupby('Month')[column_name].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).to_dict('index')
        
        # Seasonal ranges untuk validasi (mean ± 2*std per bulan)
        seasonal_ranges = {}
        for month, stats in monthly_stats.items():
            if stats['count'] > 0:  # Pastikan ada data
                lower_bound = max(stats['min'], stats['mean'] - 2 * stats['std'])
                upper_bound = min(stats['max'], stats['mean'] + 2 * stats['std'])
                seasonal_ranges[month] = {
                    'mean': stats['mean'],
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'std': stats['std']
                }
        
        print(f"📊 Seasonal reference berhasil dihitung untuk {len(seasonal_ranges)} bulan")
        
        # 3. PROSES IMPUTASI PER MISSING VALUE
        imputed_results = []
        
        for idx in missing_indices:
            date = self.data.loc[idx, 'Date']
            month = self.data.loc[idx, 'Month']
            
            print(f"\n📅 Imputasi {column_name} untuk: {date.strftime('%Y-%m-%d')} (Bulan {month})")
            
            # Cari tetangga valid terdekat
            prev_idx = idx - 1
            next_idx = idx + 1
            
            # Cari tetangga valid sebelumnya (maksimal 7 hari ke belakang)
            search_limit = 7
            search_count = 0
            while (prev_idx >= 0 and search_count < search_limit and 
                (self.data.loc[prev_idx, column_name] in [np.nan, 9999, 8888, -999, -9999] or 
                    pd.isna(self.data.loc[prev_idx, column_name]))):
                prev_idx -= 1
                search_count += 1
            
            # Cari tetangga valid setelahnya (maksimal 7 hari ke depan)
            search_count = 0
            while (next_idx < len(self.data) and search_count < search_limit and 
                (self.data.loc[next_idx, column_name] in [np.nan, 9999, 8888, -999, -9999] or 
                    pd.isna(self.data.loc[next_idx, column_name]))):
                next_idx += 1
                search_count += 1
            
            imputed_value = None
            method_used = ""
            confidence = 0
            
            # STRATEGI IMPUTASI
            
            # Strategy A: LINEAR INTERPOLATION (OPTIMAL)
            if (prev_idx >= 0 and next_idx < len(self.data) and
                prev_idx != idx and next_idx != idx):
                
                prev_val = self.data.loc[prev_idx, column_name]
                next_val = self.data.loc[next_idx, column_name]
                prev_date = self.data.loc[prev_idx, 'Date']
                next_date = self.data.loc[next_idx, 'Date']
                
                # Linear interpolation berdasarkan jarak waktu
                total_days = (next_date - prev_date).days
                target_days = (date - prev_date).days
                
                if total_days > 0 and total_days <= 14:  # Maksimal gap 2 minggu
                    weight = target_days / total_days
                    interpolated_value = prev_val + (next_val - prev_val) * weight
                    
                    # Seasonal adjustment jika tersedia
                    if month in seasonal_ranges:
                        seasonal_mean = seasonal_ranges[month]['mean']
                        # Weighted adjustment: 70% interpolation + 30% seasonal
                        seasonal_adjusted = 0.7 * interpolated_value + 0.3 * seasonal_mean
                        imputed_value = seasonal_adjusted
                    else:
                        imputed_value = interpolated_value
                    
                    method_used = f"Linear Interpolation + Seasonal ({total_days}d gap)"
                    confidence = 95 if total_days <= 3 else (90 if total_days <= 7 else 80)
            
            # Strategy B: FORWARD FILL + SEASONAL ADJUSTMENT
            elif prev_idx >= 0 and prev_idx != idx:
                prev_val = self.data.loc[prev_idx, column_name]
                
                if month in seasonal_ranges:
                    seasonal_mean = seasonal_ranges[month]['mean']
                    # Weighted: 60% previous value + 40% seasonal mean
                    imputed_value = 0.6 * prev_val + 0.4 * seasonal_mean
                    method_used = "Forward Fill + Seasonal Adjustment"
                    confidence = 75
                else:
                    imputed_value = prev_val
                    method_used = "Forward Fill"
                    confidence = 60
            
            # Strategy C: BACKWARD FILL + SEASONAL ADJUSTMENT  
            elif next_idx < len(self.data) and next_idx != idx:
                next_val = self.data.loc[next_idx, column_name]
                
                if month in seasonal_ranges:
                    seasonal_mean = seasonal_ranges[month]['mean']
                    # Weighted: 60% next value + 40% seasonal mean
                    imputed_value = 0.6 * next_val + 0.4 * seasonal_mean
                    method_used = "Backward Fill + Seasonal Adjustment"
                    confidence = 75
                else:
                    imputed_value = next_val
                    method_used = "Backward Fill"
                    confidence = 60
            
            # Strategy D: SEASONAL MEAN (FALLBACK)
            elif month in seasonal_ranges:
                imputed_value = seasonal_ranges[month]['mean']
                method_used = "Seasonal Mean (Fallback)"
                confidence = 70
            
            # Strategy E: GLOBAL MEAN (LAST RESORT)
            else:
                imputed_value = valid_data[column_name].mean()
                method_used = "Global Mean (Last Resort)"
                confidence = 50
            
            # 4. VALIDASI RANGE MUSIMAN
            if imputed_value is not None:
                original_value = imputed_value
                
                # Validasi dengan range musiman jika tersedia
                if month in seasonal_ranges:
                    seasonal_range = seasonal_ranges[month]
                    
                    # Clip ke range musiman (lebih lenient: mean ± 3*std)
                    extended_lower = seasonal_range['mean'] - 3 * seasonal_range['std']
                    extended_upper = seasonal_range['mean'] + 3 * seasonal_range['std']
                    
                    imputed_value = max(extended_lower, min(extended_upper, imputed_value))
                    
                    # Peringatan jika ada clipping
                    if abs(imputed_value - original_value) > 0.1:
                        print(f"   ⚠️  Nilai disesuaikan ke range musiman: {original_value:.2f} → {imputed_value:.2f}")
                
                # Validasi dengan range global sebagai safety net
                global_min = valid_data[column_name].min()
                global_max = valid_data[column_name].max()
                imputed_value = max(global_min, min(global_max, imputed_value))
                
                # Update data
                original_val = self.data.loc[idx, column_name]
                self.data.loc[idx, column_name] = imputed_value
                
                # Record hasil imputasi
                imputed_results.append({
                    'date': date,
                    'month': month,
                    'original': original_val,
                    'imputed': imputed_value,
                    'method': method_used,
                    'confidence': confidence
                })
                
                print(f"   ✅ {original_val} → {imputed_value:.2f}°C | Method: {method_used} | Confidence: {confidence}%")
            
            else:
                print(f"   ❌ Gagal imputasi untuk {date}")
        
        # 5. SUMMARY IMPUTASI
        if imputed_results:
            print(f"\n📊 RINGKASAN IMPUTASI {column_name.upper()}:")
            print(f"   • Total diimputasi: {len(imputed_results)} values")
            print(f"   • Range imputasi: {min([r['imputed'] for r in imputed_results]):.1f}°C - {max([r['imputed'] for r in imputed_results]):.1f}°C")
            print(f"   • Rata-rata imputasi: {np.mean([r['imputed'] for r in imputed_results]):.2f}°C")
            
            # Confidence statistics
            avg_confidence = np.mean([r['confidence'] for r in imputed_results])
            print(f"   • Average confidence: {avg_confidence:.1f}%")
            
            # Method distribution
            methods = [r['method'] for r in imputed_results]
            method_counts = pd.Series(methods).value_counts()
            print(f"   • Method distribution:")
            for method, count in method_counts.items():
                print(f"     - {method}: {count} ({count/len(imputed_results)*100:.1f}%)")
        
        print(f"🚀 Imputasi {column_name} selesai! Data siap untuk analisis lanjutan.")
        
        return {
            'imputed_count': len(imputed_results),
            'imputed_details': imputed_results,
            'seasonal_ranges': seasonal_ranges,
            'success_rate': len(imputed_results) / len(missing_indices) * 100 if missing_indices else 100
    }

    def detect_outliers_iqr(self, column_name):
        """
        Deteksi dan ekstrak nilai outlier untuk kolom temperature menggunakan metode IQR
        """
        print(f"\n=== DETEKSI OUTLIER IQR UNTUK {column_name.upper()} ===")
        
        # Analisis data valid
        analysis = self.analyze_missing_values(column_name, show_details=False)
        if analysis is None:
            print("❌ Tidak ada data valid")
            return None

        data = analysis['valid_data']
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        print(f"📊 Total outliers: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
        print(f"   • Batas bawah: {lower_bound:.2f}°C")
        print(f"   • Batas atas : {upper_bound:.2f}°C")
        
        return outliers

    def save_results(self, output_path="preprocessing_log_temp.txt"):
        """
        Simpan hasil analisis ke file
        """
        print(f"\n💾 Menyimpan hasil ke: {output_path}")
        
        # Redirect output ke file
        original_stdout = sys.stdout
        with open(output_path, 'w', encoding='utf-8') as f:
            sys.stdout = f
            
            # Generate semua analisis ke file
            print("HASIL ANALISIS STATISTIK DESKRIPTIF TEMPERATURE BMKG")
            print("=" * 60)
            
            self.calculate_descriptive_statistics('TN')
            self.calculate_descriptive_statistics('TX')
            self.compare_tn_tx()
            self.seasonal_analysis_temperature()
            self.generate_summary_table()
        
        # Kembalikan stdout
        sys.stdout = original_stdout
        print(f"✅ Hasil berhasil disimpan ke: {output_path}")

def main():
    # Path ke dataset BMKG
    data_path = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV/BMKG_Data_All.csv"
    
    # Inisialisasi analyzer
    analyzer = Temperature_Analyzer(data_path)
    
    try:
        # Load data
        if not analyzer.load_data():
            return
        
        # Analisis statistik deskriptif untuk TN
        print("\n" + "🔥"*20 + " ANALISIS TN " + "🔥"*20)
        analyzer.calculate_descriptive_statistics('TN')
        
        # Analisis statistik deskriptif untuk TX  
        print("\n" + "☀️"*20 + " ANALISIS TX " + "☀️"*20)
        analyzer.calculate_descriptive_statistics('TX')
        
        # Perbandingan TN vs TX
        analyzer.compare_tn_tx()
        
        # Analisis musiman
        analyzer.seasonal_analysis_temperature()

        # Imputasi missing values dengan seasonal linear interpolation
        analyzer.impute_temperature_seasonal_interpolation('TN')
        analyzer.impute_temperature_seasonal_interpolation('TX')

        # Cek outliers TN
        tn_outliers = analyzer.detect_outliers_iqr('TN')

        # Cek outliers TX
        tx_outliers = analyzer.detect_outliers_iqr('TX')
        
        # Generate tabel ringkasan final
        final_table = analyzer.generate_summary_table()
        
        # Simpan hasil
        analyzer.save_results()
        
        print(f"\n🎉 ANALISIS TEMPERATURE SELESAI!")
        print(f"📊 Semua statistik deskriptif telah dihitung")
        print(f"📁 Check file output untuk detail lengkap")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Error dalam analisis: {str(e)}")
        return None

if __name__ == "__main__":
    result = main()