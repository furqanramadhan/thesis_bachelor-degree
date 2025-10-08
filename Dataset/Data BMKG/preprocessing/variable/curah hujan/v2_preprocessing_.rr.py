import pandas as pd
import numpy as np
import sys
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
from scipy.stats import boxcox
import warnings
warnings.filterwarnings('ignore')


class RainfallPreprocessor:
    def __init__(self, data):
        """
        Inisialisasi preprocessor untuk data curah hujan
        
        Parameters:
        data: DataFrame dengan kolom Date, RR, dan kolom lainnya
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.processed_data = None
        self.missing_stats = {}
        self.outliers = {}
        
    def load_and_prepare_data(self):
        """
        Load data dan persiapan awal
        """
        # Konversi Date ke datetime
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)

        if 'SS' in self.data.columns:
            self.data['SS'] = self.data['SS'].replace(9999, np.nan)
        
        # Buat kolom tambahan untuk analisis
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Day'] = self.data['Date'].dt.day
        
        print(f"Data loaded: {len(self.data)} records from {self.data['Date'].min()} to {self.data['Date'].max()}")
        return self.data
    
    def analyze_missing_values(self):
        """
        Analisis mendalam missing values dan nilai khusus
        """
        print("=== ANALISIS MISSING VALUES DAN NILAI KHUSUS ===")
        
        # Identifikasi nilai 8888 (tidak terukur)
        missing_8888 = (self.data['RR'] == 8888).sum()
        missing_nan = self.data['RR'].isna().sum()
        zero_values = (self.data['RR'] == 0).sum()
        total_records = len(self.data)
        
        print(f"Total records: {total_records}")
        print(f"Nilai 8888 (tidak terukur): {missing_8888} ({missing_8888/total_records*100:.1f}%)")
        print(f"Missing/NaN values: {missing_nan} ({missing_nan/total_records*100:.1f}%)")
        print(f"Zero values (tidak hujan): {zero_values} ({zero_values/total_records*100:.1f}%)")
        
        # Simpan statistik missing values
        self.missing_stats = {
            'total_records': total_records,
            'missing_8888': missing_8888,
            'missing_nan': missing_nan,
            'zero_values': zero_values,
            'missing_percentage': (missing_8888 + missing_nan) / total_records * 100
        }
        
        # Analisis missing values per tahun
        yearly_missing = self.data.groupby('Year').agg({
            'RR': [
                lambda x: (x == 8888).sum(),
                lambda x: x.isna().sum(),
                lambda x: (x == 0).sum(),
                'count'
            ]
        }).round(2)
        
        yearly_missing.columns = ['Missing_8888', 'Missing_NaN', 'Zero_Values', 'Total_Records']
        yearly_missing['Missing_Percentage'] = (yearly_missing['Missing_8888'] + yearly_missing['Missing_NaN']) / yearly_missing['Total_Records'] * 100
        
        print("\n=== MISSING VALUES PER TAHUN ===")
        print(yearly_missing)
        
        return yearly_missing
    
    def clean_rainfall_data(self):
        """
        Membersihkan data curah hujan dengan pendekatan yang disederhanakan
        """
        print("\n=== CLEANING DATA CURAH HUJAN ===")
        
        # Backup data original
        self.data['RR_original'] = self.data['RR'].copy()
        
        # Step 1: Handle missing data codes
        self.data['RR'] = self.data['RR'].replace(9999, np.nan)  # Missing data
        
        # Step 2: Estimate unmeasured rainfall (8888)
        self._estimate_unmeasured_rainfall()
        
        # Step 3: Identify and handle outliers
        self._identify_outliers()
        
        # Step 4: Print summary statistics
        self._print_cleaning_summary()
        
        print("✅ Data cleaning selesai\n")

    def _estimate_unmeasured_rainfall(self):
        """
        Estimasi data 8888 (tidak terukur) berdasarkan kondisi meteorologi dengan SS modifier
        """
        mask_8888 = self.data['RR'] == 8888
        count_8888 = mask_8888.sum()
        
        if count_8888 == 0:
            return
        
        print(f"Estimasi {count_8888} data tidak terukur (8888) dengan SS modifier...")
        
        # Mapping kondisi meteorologi ke kategori hujan
        conditions = {
            'heavy_rain': {'rh_min': 90, 'tn_max': 23, 'rain_range': (7, 20)},
            'moderate_rain': {'rh_min': 80, 'tn_max': 24, 'rain_range': (2, 10)},
            'light_rain': {'rh_min': 70, 'tn_max': 24, 'rain_range': (0.5, 5)},
            'dry': {'rh_min': 0, 'tn_max': 100, 'rain_range': (0, 1)}  # default
        }
        
        estimated_values = []
        
        for idx in self.data[mask_8888].index:
            row = self.data.loc[idx]
            rh = row.get('RH_AVG', 75)  # default humidity
            tn = row.get('TN', 25)      # default temperature
            ss = row.get('SS', 4.0)     # default sunshine duration
            
            # Determine rainfall category based on conditions
            category = 'dry'  # default
            for cat, cond in conditions.items():
                if cat != 'dry' and rh >= cond['rh_min'] and tn <= cond['tn_max']:
                    category = cat
                    break
            
            # Generate base value within category range
            min_val, max_val = conditions[category]['rain_range']
            base_estimated_value = np.random.uniform(min_val, max_val)
            
            # Apply SS modifier
            ss_modifier = self._calculate_ss_modifier(ss)
            estimated_value = base_estimated_value * ss_modifier
            
            estimated_values.append(estimated_value)
        
        # Apply estimates
        self.data.loc[mask_8888, 'RR'] = estimated_values
        self.data.loc[mask_8888, 'RR_estimation_method'] = 'meteorological_estimate_with_ss'
        self.data.loc[mask_8888, 'imputation_method'] = 'meteorological_estimate_with_ss'
        
        print(f"  → {len(estimated_values)} nilai berhasil diestimasi dengan SS modifier")

    def _calculate_ss_modifier(self, ss):
        """
        Hitung modifier berdasarkan sunshine duration
        SS rendah → kemungkinan hujan lebih tinggi
        SS tinggi → kemungkinan hujan lebih rendah
        """
        if pd.isna(ss):
            return 1.0  # neutral if SS is missing
        
        if ss < 1.5:
            return np.random.uniform(1.3, 1.5)  # boost estimation
        elif ss < 3.0:
            return np.random.uniform(1.1, 1.3)  # slight boost
        elif ss < 6.0:
            return np.random.uniform(0.9, 1.1)  # neutral
        elif ss < 8.0:
            return np.random.uniform(0.7, 0.9)  # reduce estimation
        else:
            return np.random.uniform(0.5, 0.7)  # strong reduction

    def _identify_outliers(self):
        """
        Identifikasi outlier ekstrem (>150mm/hari)
        """
        outlier_threshold = 150
        outliers = self.data[self.data['RR'] > outlier_threshold]
        
        print(f"Outlier ekstrem (>{outlier_threshold}mm): {len(outliers)} data")
        
        if len(outliers) > 0:
            print("  Dates dengan outlier:")
            for idx, row in outliers.head(5).iterrows():  # Show max 5
                print(f"    {row['Date']}: {row['RR']:.1f}mm")
        
        # Store outliers for analysis
        self.outliers = outliers
    
    def _print_cleaning_summary(self):
        """
        Print ringkasan hasil cleaning
        """
        valid_data = self.data['RR'].dropna()
        
        if len(valid_data) == 0:
            print("⚠️  Tidak ada data valid setelah cleaning")
            return
        
        print(f"\nRingkasan data setelah cleaning:")
        print(f"  Total data valid: {len(valid_data)}")
        print(f"  Mean: {valid_data.mean():.2f}mm")
        print(f"  Median: {valid_data.median():.2f}mm")
        print(f"  Min: {valid_data.min():.2f}mm")
        print(f"  Max: {valid_data.max():.2f}mm")
        
        # Rainfall categories distribution
        categories = self._categorize_rainfall(valid_data)
        print(f"\nDistribusi kategori hujan:")
        for cat, count in categories.items():
            pct = count / len(valid_data) * 100
            print(f"  {cat}: {count} ({pct:.1f}%)")

    def _categorize_rainfall(self, data):
        """
        Kategorisasi curah hujan berdasarkan intensitas harian sesuai standar BMKG.
        """
        return {
            'Berawan (0–0.4 mm)': ((data >= 0) & (data <= 0.4)).sum(),
            'Hujan ringan (0.5–19.9 mm)': ((data > 0.4) & (data <= 19.9)).sum(),
            'Hujan sedang (20–49.9 mm)': ((data >= 20) & (data <= 49.9)).sum(),
            'Hujan lebat (50–99.9 mm)': ((data >= 50) & (data <= 99.9)).sum(),
            'Hujan sangat lebat (100–150 mm)': ((data >= 100) & (data <= 150)).sum(),
            'Hujan ekstrem (>150 mm)': (data > 150).sum()
        }

    def seasonal_imputation(self):
        """
        Imputasi missing values dengan pendekatan yang disederhanakan
        """
        print("\n=== IMPUTASI MISSING VALUES ===")
        
        # Initialize imputed column
        self.data['RR_imputed'] = self.data['RR'].copy()
        missing_count = self.data['RR_imputed'].isna().sum()
        
        if missing_count == 0:
            print("Tidak ada missing values untuk diimputasi")
            return True
        
        print(f"Memproses {missing_count} missing values...")
        
        # Calculate monthly statistics for imputation
        monthly_stats = self._calculate_monthly_stats()
        
        # Identify missing data patterns
        missing_patterns = self._identify_missing_patterns()
        
        # Apply imputation strategy based on gap length
        total_imputed = 0
        
        for pattern in missing_patterns:
            gap_length = pattern['length']
            indices = pattern['indices']
            
            if gap_length <= 3:
                # Short gaps: linear interpolation
                imputed_count = self._impute_short_gaps(indices)
            elif gap_length <= 14:
                # Medium gaps: seasonal average
                imputed_count = self._impute_medium_gaps(indices, monthly_stats)
            else:
                # Long gaps: historical monthly median
                imputed_count = self._impute_long_gaps(indices, monthly_stats)
            
            total_imputed += imputed_count
        
        print(f"✅ {total_imputed} values berhasil diimputasi")
        
        # Print final summary
        self._print_imputation_summary()

        # Tandai sumber data berdasarkan RR_original
        self.data['RR_source'] = 'original'  # default
        self.data.loc[self.data['RR_original'] == 8888, 'RR_source'] = 'estimated_8888'
        self.data.loc[self.data['RR_original'].isna(), 'RR_source'] = 'imputed_missing'

        # Final return
        return self.data['RR_imputed'].isna().sum() == 0

    def _calculate_monthly_stats(self):
        """
        Hitung statistik bulanan untuk imputasi
        """
        monthly_stats = {}
        
        for month in range(1, 13):
            month_data = self.data[
                (self.data['Month'] == month) & 
                (self.data['RR'].notna())
            ]['RR']
            
            if len(month_data) > 0:
                monthly_stats[month] = {
                    'mean': month_data.mean(),
                    'median': month_data.median(),
                    'std': month_data.std(),
                    'count': len(month_data)
                }
            else:
                # Fallback to overall statistics
                overall_data = self.data['RR'].dropna()
                monthly_stats[month] = {
                    'mean': overall_data.mean() if len(overall_data) > 0 else 5.0,
                    'median': overall_data.median() if len(overall_data) > 0 else 2.0,
                    'std': overall_data.std() if len(overall_data) > 0 else 10.0,
                    'count': 0
                }
        
        return monthly_stats

    def _identify_missing_patterns(self):
        """
        Identifikasi pola missing data untuk strategi imputasi
        """
        missing_mask = self.data['RR_imputed'].isna()
        
        if not missing_mask.any():
            return []
        
        # Group consecutive missing values
        groups = []
        current_group = []
        
        for idx, is_missing in missing_mask.items():
            if is_missing:
                current_group.append(idx)
            else:
                if current_group:
                    groups.append({
                        'indices': current_group,
                        'length': len(current_group),
                        'start': current_group[0],
                        'end': current_group[-1]
                    })
                    current_group = []
        
        # Don't forget the last group
        if current_group:
            groups.append({
                'indices': current_group,
                'length': len(current_group),
                'start': current_group[0],
                'end': current_group[-1]
            })
        
        return groups

    def _impute_short_gaps(self, indices):
        """
        Imputasi gap pendek (≤3 hari) dengan linear interpolation
        """
        # Create temporary series for interpolation
        temp_series = self.data['RR_imputed'].copy()
        
        # Apply linear interpolation
        interpolated = temp_series.interpolate(method='linear', limit_direction='both')
        
        # Fill the gaps
        for idx in indices:
            if pd.isna(self.data.loc[idx, 'RR_imputed']):
                self.data.loc[idx, 'RR_imputed'] = max(0, interpolated.loc[idx])
                self.data.loc[idx, 'imputation_method'] = 'linear_interpolation'
        
        return len(indices)

    def _impute_medium_gaps(self, indices, monthly_stats):
        """
        Imputasi gap sedang (4-14 hari) dengan monthly average + noise + SS modifier
        """
        for idx in indices:
            if pd.isna(self.data.loc[idx, 'RR_imputed']):
                month = self.data.loc[idx, 'Month']
                
                # Get monthly statistics
                stats = monthly_stats[month]
                
                # Generate base value with some randomness
                base_value = stats['median']
                noise = np.random.normal(0, stats['std'] * 0.3)  # 30% of std as noise
                base_imputed_value = max(0, base_value + noise)
                
                # Apply SS modifier if available
                ss = self.data.loc[idx, 'SS'] if 'SS' in self.data.columns else 4.0
                if not pd.isna(ss):
                    ss_modifier = self._calculate_ss_modifier(ss)
                    imputed_value = base_imputed_value * ss_modifier
                else:
                    imputed_value = base_imputed_value
                
                self.data.loc[idx, 'RR_imputed'] = imputed_value
                self.data.loc[idx, 'imputation_method'] = 'monthly_average_with_ss'
        
        return len(indices)

    def _impute_long_gaps(self, indices, monthly_stats):
        """
        Imputasi gap panjang (>14 hari) dengan monthly median + SS modifier
        """
        for idx in indices:
            if pd.isna(self.data.loc[idx, 'RR_imputed']):
                month = self.data.loc[idx, 'Month']
                
                # Use monthly median for stability
                base_imputed_value = monthly_stats[month]['median']
                
                # Apply SS modifier if available
                ss = self.data.loc[idx, 'SS'] if 'SS' in self.data.columns else 4.0
                if not pd.isna(ss):
                    ss_modifier = self._calculate_ss_modifier(ss)
                    imputed_value = base_imputed_value * ss_modifier
                else:
                    imputed_value = base_imputed_value
                
                self.data.loc[idx, 'RR_imputed'] = imputed_value
                self.data.loc[idx, 'imputation_method'] = 'monthly_median_with_ss'
        
        return len(indices)

    def _print_imputation_summary(self):
        """
        Print ringkasan hasil imputasi
        """
        # Count imputation methods
        imputed_data = self.data[self.data.get('imputation_method', '').str.len() > 0]
        
        if len(imputed_data) > 0:
            print("\nMetode imputasi yang digunakan:")
            method_counts = imputed_data['imputation_method'].value_counts()
            for method, count in method_counts.items():
                print(f"  {method}: {count} values")
        
        # Final statistics
        final_data = self.data['RR_imputed'].dropna()
        if len(final_data) > 0:
            print(f"\nStatistik final:")
            print(f"  Total data: {len(final_data)}")
            print(f"  Mean: {final_data.mean():.2f}mm")
            print(f"  Median: {final_data.median():.2f}mm")
            print(f"  Missing values: {self.data['RR_imputed'].isna().sum()}")
            
            # Compare with original
            if 'RR_original' in self.data.columns:
                original_clean = self.data['RR_original'].replace([8888, 9999], np.nan).dropna()
                if len(original_clean) > 0:
                    mean_diff = abs(final_data.mean() - original_clean.mean())
                    print(f"  Deviasi dari original: {mean_diff:.2f}mm")

    def detect_outliers_advanced(self):
        """
        Deteksi outlier menggunakan metode Percentile 99% 
        Disesuaikan untuk data curah hujan tropis
        """
        print("\n=== DETEKSI OUTLIER (PERCENTILE 99%) ===")
        
        valid_data = self.data['RR_imputed'].dropna()

        if valid_data.empty:
            print("Tidak ada data valid untuk deteksi outlier.")
            return

        # Hitung threshold Percentile 99
        p99_threshold = valid_data.quantile(0.99)
        outlier_mask = self.data['RR_imputed'] > p99_threshold

        # Flag outlier di dataset
        self.data['is_outlier'] = outlier_mask

        # Simpan outliers
        p99_outliers = self.data.loc[outlier_mask, 'RR_imputed']
        self.outliers = {
            'percentile_99': p99_outliers,
            'bounds': {'p99_threshold': p99_threshold}
        }

        # Statistik dan output ringkas
        print(f"📊 Threshold Percentile 99%: > {p99_threshold:.2f} mm")
        print(f"✅ Total outliers flagged: {outlier_mask.sum()} dari {len(valid_data)} data")

        # Tampilkan contoh outlier
        if not p99_outliers.empty:
            print("\nContoh outliers terdeteksi:")
            for idx, value in p99_outliers.head(5).items():
                print(f"  - {value:.2f} mm (index: {idx})")

        print("\n💡 Menggunakan threshold P99 sebagai pendekatan adaptif untuk iklim tropis.")

    def summary_report(self):
        """
        Laporan ringkasan preprocessing
        """
        print("\n" + "="*60)
        print("LAPORAN RINGKASAN PREPROCESSING CURAH HUJAN")
        print("="*60)
        
        print(f"📊 DATASET OVERVIEW:")
        print(f"   • Total records: {len(self.data):,}")
        print(f"   • Periode: {self.data['Date'].min()} s/d {self.data['Date'].max()}")
        print(f"   • Rentang tahun: {self.data['Year'].max() - self.data['Year'].min() + 1} tahun")
        
        print(f"\n🔍 DATA QUALITY:")
        print(f"   • Missing values (8888): {self.missing_stats['missing_8888']:,} ({self.missing_stats['missing_8888']/len(self.data)*100:.1f}%)")
        print(f"   • Missing values (NaN): {self.missing_stats['missing_nan']:,}")
        print(f"   • Zero values: {self.missing_stats['zero_values']:,}")
        print(f"   • Valid data: {len(self.data) - self.missing_stats['missing_8888'] - self.missing_stats['missing_nan']:,}")
        
        valid_data = self.data['RR_imputed'].dropna()
        if len(valid_data) > 0:
            print(f"\n📉 STATISTIK RR_original (sebelum imputasi):")
            original_rr = self.data['RR_original'].replace([8888, 9999], np.nan).dropna()
            if len(original_rr) > 0:
                print(f"   • Count: {len(original_rr):,}")
                print(f"   • Mean: {original_rr.mean():.2f} mm")
                print(f"   • Std Dev: {original_rr.std():.2f} mm")
                print(f"   • Min: {original_rr.min():.2f} mm")
                print(f"   • Q1 (25%): {original_rr.quantile(0.25):.2f} mm")
                print(f"   • Median (Q2): {original_rr.median():.2f} mm")
                print(f"   • Q3 (75%): {original_rr.quantile(0.75):.2f} mm")
                print(f"   • Max: {original_rr.max():.2f} mm")
                print(f"   • Skewness: {original_rr.skew():.2f}")

                print(f"\n📉 STATISTIK RR_imputed (setelah imputasi):")
                print(f"   • Count: {len(valid_data):,}")
                print(f"   • Mean: {valid_data.mean():.2f} mm")
                print(f"   • Std Dev: {valid_data.std():.2f} mm")
                print(f"   • Min: {valid_data.min():.2f} mm")
                print(f"   • Q1 (25%): {valid_data.quantile(0.25):.2f} mm")
                print(f"   • Median (Q2): {valid_data.median():.2f} mm")
                print(f"   • Q3 (75%): {valid_data.quantile(0.75):.2f} mm")
                print(f"   • Max: {valid_data.max():.2f} mm")
                print(f"   • Skewness: {valid_data.skew():.2f}")
        else:
            print("   ⚠️ Tidak ada data valid dalam RR_original")
        
        if hasattr(self, 'outliers') and 'iqr' in self.outliers:
            print(f"\n⚠️  OUTLIER DETECTION:")
            print(f"   • IQR method: {len(self.outliers['iqr'])} outliers")
            print(f"   • Upper bound: {self.outliers['bounds']['upper']:.2f} mm")
        


def main():
    """
    Fungsi utama untuk menjalankan preprocessing
    """
    print("🌧️  PREPROCESSING CURAH HUJAN - DATASET BMKG")

    sys.stdout = open("v2_preprocessing_log_rainfall.txt", "w")

    try:
        data_path = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV/BMKG_Data_All.csv"  # atau "data_bmkg.xlsx"
        
        # Load data
        print(f"📂 Loading data dari: {data_path}")
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("Format file tidak didukung. Gunakan .csv atau .xlsx")
        
        # Validasi kolom yang diperlukan
        required_columns = ['Date', 'RR']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Error: Kolom berikut tidak ditemukan: {missing_columns}")
            print(f"📋 Kolom yang tersedia: {list(df.columns)}")
            return
        
        print(f"✅ Data berhasil dimuat: {len(df)} records")
        print(f"📋 Kolom tersedia: {list(df.columns)}")
        
        # Inisialisasi preprocessor
        preprocessor = RainfallPreprocessor(df)
        
        # FASE 1: Load dan persiapan data
        print("\n🔄 FASE 1: Loading dan Persiapan Data")
        preprocessor.load_and_prepare_data()
        
        # FASE 2: Analisis missing values
        print("\n🔄 FASE 2: Analisis Missing Values")
        preprocessor.analyze_missing_values()
        
        # FASE 3: Cleaning data
        print("\n🔄 FASE 3: Data Cleaning")
        preprocessor.clean_rainfall_data()
        
        # FASE 4: Imputasi musiman
        print("\n🔄 FASE 4: Imputasi Berbasis Musiman")
        preprocessor.seasonal_imputation()
        
        # FASE 5: Deteksi outlier
        print("\n🔄 FASE 5: Deteksi Outlier Advanced")
        preprocessor.detect_outliers_advanced()
        
        # FASE 6: Laporan ringkasan
        print("\n🔄 FASE 6: Laporan Ringkasan")
        preprocessor.summary_report()
        
        # Simpan hasil preprocessing
        output_path = "preprocessed_rainfall_data.csv"
        
        # Simpan hanya kolom terkait RR
        rr_columns = [
            'Date', 'Year', 'Month', 'Day',
            'RR_original', 'RR_estimation_method',
            'RR_imputed', 'imputation_method', 'is_outlier'
        ]
        preprocessor.data[rr_columns].to_csv(output_path, index=False)
        
        # Rekomendasi untuk Holt-Winters
        valid_data = preprocessor.data['RR_imputed'].dropna()
        
        if len(valid_data) > 0:
            # Cek seasonal pattern
            seasonal_strength = preprocessor.data.groupby('Month')['RR_imputed'].std().mean()
            print(f"   • Data tersedia untuk forecasting: {len(valid_data):,} records")
            print(f"   • Seasonal strength: {seasonal_strength:.2f}")
            
            
        print(f"📁 File output: {output_path}")
        
        return preprocessor
        
    except FileNotFoundError:
        print(f"❌ Error: File {data_path} tidak ditemukan!")
        print("📋 Pastikan file data BMKG tersedia dengan kolom:")
        print("   • Date: Tanggal observasi")
        print("   • RR: Curah hujan (mm)")

        return None
        
    except Exception as e:
        print(f"❌ Error dalam preprocessing: {str(e)}")
        return None
    
if __name__== "__main__":
    result = main() 