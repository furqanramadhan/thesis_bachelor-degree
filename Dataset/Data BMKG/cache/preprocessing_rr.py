import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
from scipy.stats import boxcox
import warnings
warnings.filterwarnings('ignore')

# Set style untuk plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
        Kategorisasi curah hujan berdasarkan intensitas
        """
        return {
            'Berawan (0mm)': (data == 0).sum(),
            'Hujan ringan (0-20mm)': ((data > 0) & (data <= 20)).sum(),
            'Hujan sedang (20-50mm)': ((data > 20) & (data <= 50)).sum(),
            'Hujan lebat (50-100mm)': ((data > 50) & (data <= 100)).sum(),
            'Hujan sangat lebat (>100mm)': (data > 100).sum()
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

    def transform_data(self):
        """
        Transformasi data untuk stabilisasi varians
        """
        print("\n=== TRANSFORMASI DATA ===")
        
        valid_data = self.data['RR_imputed'].dropna()
        
        if len(valid_data) == 0:
            print("Tidak ada data valid untuk transformasi")
            return
        
        # Log transformation (log(x+1) untuk handle zero values)
        self.data['RR_log'] = np.log1p(self.data['RR_imputed'])
        
        # Square root transformation
        self.data['RR_sqrt'] = np.sqrt(self.data['RR_imputed'])
        
        # Box-Cox transformation (hanya untuk nilai positif)
        positive_data = valid_data[valid_data > 0]
        if len(positive_data) > 0:
            try:
                # Fit Box-Cox transformation
                boxcox_data, lambda_param = boxcox(positive_data)
                print(f"Box-Cox lambda parameter: {lambda_param:.4f}")
                
                # Apply Box-Cox to all positive values
                self.data['RR_boxcox'] = np.nan
                positive_mask = (self.data['RR_imputed'] > 0) & (self.data['RR_imputed'].notna())
                self.data.loc[positive_mask, 'RR_boxcox'] = boxcox(self.data.loc[positive_mask, 'RR_imputed'], lmbda=lambda_param)
                
            except Exception as e:
                print(f"Box-Cox transformation failed: {e}")
                self.data['RR_boxcox'] = np.nan
        
        # Evaluasi transformasi
        print("\nEvaluasi transformasi (skewness):")
        print(f"  Original: {valid_data.skew():.4f}")
        print(f"  Log(x+1): {self.data['RR_log'].skew():.4f}")
        print(f"  Sqrt: {self.data['RR_sqrt'].skew():.4f}")
        if 'RR_boxcox' in self.data.columns:
            print(f"  Box-Cox: {self.data['RR_boxcox'].skew():.4f}")
    
    def stationarity_tests(self):
        """
        Test stationarity untuk time series
        """
        print("\n=== UJI STATIONARITY ===")
        
        # Buat time series dengan frekuensi harian
        ts_data = self.data.set_index('Date')['RR_imputed'].dropna()
        
        if len(ts_data) < 50:
            print("Data terlalu sedikit untuk uji stationarity")
            return
        
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(ts_data)
            print("Augmented Dickey-Fuller Test:")
            print(f"  ADF Statistic: {adf_result[0]:.6f}")
            print(f"  p-value: {adf_result[1]:.6f}")
            print(f"  Critical Values:")
            for key, value in adf_result[4].items():
                print(f"    {key}: {value:.6f}")
            
            if adf_result[1] <= 0.05:
                print("  → Data STASIONER (p-value ≤ 0.05)")
            else:
                print("  → Data NON-STASIONER (p-value > 0.05)")
                
        except Exception as e:
            print(f"ADF test failed: {e}")
        
        # KPSS test
        try:
            kpss_result = kpss(ts_data)
            print("\nKPSS Test:")
            print(f"  KPSS Statistic: {kpss_result[0]:.6f}")
            print(f"  p-value: {kpss_result[1]:.6f}")
            print(f"  Critical Values:")
            for key, value in kpss_result[3].items():
                print(f"    {key}: {value:.6f}")
            
            if kpss_result[1] >= 0.05:
                print("  → Data STASIONER (p-value ≥ 0.05)")
            else:
                print("  → Data NON-STASIONER (p-value < 0.05)")
                
        except Exception as e:
            print(f"KPSS test failed: {e}")
    

    def create_individual_plots(self, output_dir="rainfall_plots", save_plots=True):
        """
        Membuat plots individual untuk analisis dan menyimpannya ke direktori terpisah
        
        Parameters:
        output_dir (str): Direktori untuk menyimpan plots
        save_plots (bool): Apakah akan menyimpan plots ke file
        """
        import os
        
        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"📁 Direktori {output_dir} dibuat")
        
        print("\n=== MEMBUAT PLOTS INDIVIDUAL ===")
        
        # Setup style untuk semua plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Time Series Plot
        plt.figure(figsize=(16, 8))
        valid_data = self.data[self.data['RR_imputed'].notna()]

        if not valid_data.empty:
            # Urutkan data
            valid_data = valid_data.sort_values('Date')
            
            # Plot garis utama & moving average
            plt.plot(valid_data['Date'], valid_data['RR_imputed'], color='blue', alpha=0.7, linewidth=1.0, label='Curah Hujan')
            
            # Scatter hari hujan tinggi (>50mm)
            high_rain = valid_data[valid_data['RR_imputed'] > 50]
            if not high_rain.empty:
                plt.scatter(high_rain['Date'], high_rain['RR_imputed'], color='red', s=20, alpha=0.8, label='> 50 mm')

            # Garis vertikal awal tahun dan label tiap 2 tahun
            years = sorted(valid_data['Year'].unique())
            for i, y in enumerate(years):
                tgl_awal = pd.Timestamp(f"{y}-01-01")
                plt.axvline(tgl_awal, color='gray', linestyle='--', linewidth=0.6, alpha=0.6)
                if i % 2 == 0 or y == years[-1]:
                    plt.text(tgl_awal, plt.ylim()[1]*0.95, str(y), rotation=90, va='top', ha='right', fontsize=9, color='gray')

            # Sumbu X dan grid
            start_date, end_date = valid_data['Date'].min(), valid_data['Date'].max()
            date_ticks = pd.date_range(start=start_date, end=end_date, freq='YS')
            plt.xticks(date_ticks, [d.year for d in date_ticks], rotation=45)
            plt.gca().set_xticks(pd.date_range(start_date, end_date, freq='MS'), minor=True)
            plt.grid(True, alpha=0.3, which='major')
            plt.grid(True, alpha=0.1, which='minor')

            # Judul dan keterangan
            total_days = (end_date - start_date).days
            plt.title(f'Time Series Curah Hujan Harian (RR)\n{start_date:%d %b %Y} - {end_date:%d %b %Y}', fontsize=14, fontweight='bold')
            plt.xlabel('Tahun')
            plt.ylabel('Curah Hujan (mm)')
            plt.legend(loc='upper right', frameon=True)

            # Statistik pojok kiri atas
            plt.text(0.02, 0.98,
                    f'Data Valid: {len(valid_data):,} hari\nMaksimum: {valid_data["RR_imputed"].max():.1f}mm\nMinimum: {valid_data["RR_imputed"].min():.1f}mm',
                    transform=plt.gca().transAxes,
                    fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            plt.text(0.5, 0.5, 'Tidak ada data valid untuk ditampilkan',
                    ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{output_dir}/01_time_series_plot.png', dpi=300, bbox_inches='tight')
            print("✅ 01_time_series_plot.png saved")
        plt.show()


        # 2. Histogram dan Distribusi
        plt.figure(figsize=(10, 6))
        valid_rr = self.data['RR_imputed'].dropna()

        if not valid_rr.empty:
            # Filter upper limit untuk menghindari outlier ekstrem
            upper_limit = min(valid_rr.quantile(0.95) * 1.1, 100)
            filtered_rr = valid_rr[valid_rr <= upper_limit]

            # Tentukan bin width merata
            bin_width = 2  # 2 mm per bin
            bins = np.arange(0, upper_limit + bin_width, bin_width)

            # Plot histogram
            plt.hist(filtered_rr, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)

            # Label dan tata letak
            plt.title('Distribusi Curah Hujan Harian (Filtered ≤ {:.0f} mm)'.format(upper_limit), fontsize=13, fontweight='bold')
            plt.xlabel('Curah Hujan (mm)', fontsize=11)
            plt.ylabel('Frekuensi', fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=9)
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'Tidak ada data valid untuk ditampilkan',
                    ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

        # Simpan jika diminta
        if save_plots:
            plt.savefig(f'{output_dir}/02_histogram_distribusi.png', dpi=300, bbox_inches='tight')
            print("✅ 02_histogram_distribusi.png saved")
        plt.show()
        
        # 3. Boxplot Curah Hukan per Tahun
        plt.figure(figsize=(14, 6))
        yearly_data = []
        years = sorted(self.data['Year'].dropna().unique())

        for year in years:
            year_rr = self.data[self.data['Year'] == year]['RR_imputed'].dropna()
            if len(year_rr) > 0:
                yearly_data.append(year_rr)

        if yearly_data:
            plt.boxplot(yearly_data, labels=years, showfliers=True)
            plt.title('Boxplot Curah Hujan Harian per Tahun', fontsize=14, fontweight='bold')
            plt.xlabel('Tahun', fontsize=12)
            plt.ylabel('Curah Hujan Harian (mm)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        plt.tight_layout()

        if save_plots:
            plt.savefig(f'{output_dir}/03_boxplot_tahunan.png', dpi=300, bbox_inches='tight')
            print("✅ 03_boxplot_tahunan.png saved")
        plt.show()
        
        
        # 4. Pola Musiman (Bar Chart)
        plt.figure(figsize=(10, 6))
        seasonal_mean = self.data.groupby('Month')['RR_imputed'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                    'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
        
        plt.bar(range(1, 13), seasonal_mean.values, alpha=0.7, color='lightcoral')
        plt.title('Rata-rata Curah Hujan per Bulan', fontsize=14, fontweight='bold')
        plt.xlabel('Bulan', fontsize=12)
        plt.ylabel('Rata-rata Curah Hujan (mm)', fontsize=12)
        plt.xticks(range(1, 13), month_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Tambahkan nilai di atas setiap bar
        for i, v in enumerate(seasonal_mean.values):
            plt.text(i+1, v + max(seasonal_mean.values)*0.01, f'{v:.1f}', 
                    ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/04_pola_musiman.png', dpi=300, bbox_inches='tight')
            print("✅ 04_pola_musiman.png saved")
        plt.show()
        

        # 5. Trend Rata-rata Curah Hujan Tahunan - Simplified
        plt.figure(figsize=(14, 6))

        # Hitung rata-rata tahunan
        yearly_mean = self.data.groupby('Year')['RR_imputed'].mean()
        years = yearly_mean.index.astype(int)
        values = yearly_mean.values

        # Plot garis utama
        plt.plot(years, values, marker='o', linewidth=2, color='green', label='Rata-rata Tahunan')

        # Tambahkan trendline
        z = np.polyfit(years, values, 1)
        plt.plot(years, np.poly1d(z)(years), linestyle='--', color='red', label=f'Trend: {z[0]:.2f} mm/tahun')

        # Highlight tahun tertinggi dan terendah
        plt.scatter(years[np.argmax(values)], values.max(), color='red', s=80, label=f'Tertinggi: {years[np.argmax(values)]}')
        plt.scatter(years[np.argmin(values)], values.min(), color='blue', s=80, label=f'Terendah: {years[np.argmin(values)]}')

        # Label, sumbu, grid
        plt.title('Rata-rata Curah Hujan Tahunan', fontsize=14, fontweight='bold')
        plt.xlabel('Tahun')
        plt.ylabel('Rata-rata (mm)')
        plt.xticks(years[::max(1, len(years)//10)], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Statistik ringkasan di pojok
        text = f'''Periode: {years.min()} - {years.max()}
        Rata-rata: {values.mean():.1f} mm/hari
        Rentang: {values.min():.1f} – {values.max():.1f} mm/hari
        Trend: {"↑" if z[0]>0 else "↓"} {abs(z[0]):.2f} mm/tahun'''
        plt.text(0.02, 0.95, text, transform=plt.gca().transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Simpan jika diminta
        if save_plots:
            plt.savefig(f'{output_dir}/05_trend_tahunan.png', dpi=300, bbox_inches='tight')
            print("✅ 05_trend_tahunan.png saved")
        plt.show()

        
        # 6. Pola Missing Values
        plt.figure(figsize=(12, 6))
        yearly_missing = self.data.groupby('Year').agg({
            'RR_original': lambda x: (x == 8888).sum() + x.isna().sum(),
            'Date': 'count'
        })
        missing_pct = yearly_missing['RR_original'] / yearly_missing['Date'] * 100
        
        plt.bar(missing_pct.index, missing_pct.values, alpha=0.7, color='red')
        plt.title('Persentase Missing Values per Tahun', fontsize=14, fontweight='bold')
        plt.xlabel('Tahun', fontsize=12)
        plt.ylabel('Missing Values (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.legend()
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/06_missing_values_pattern.png', dpi=300, bbox_inches='tight')
            print("✅ 06_missing_values_pattern.png saved")
        plt.show()
        
        # 7. Transformasi Comparison
        plt.figure(figsize=(15, 5))
        
        # Original distribution
        plt.subplot(1, 3, 1)
        valid_original = self.data['RR_imputed'].dropna()
        plt.hist(valid_original, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribusi Original', fontsize=12, fontweight='bold')
        plt.xlabel('RR (mm)')
        plt.ylabel('Frekuensi')
        plt.grid(True, alpha=0.3)
        
        # Log transformation
        plt.subplot(1, 3, 2)
        if 'RR_log' in self.data.columns:
            valid_log = self.data['RR_log'].dropna()
            plt.hist(valid_log, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.title('Distribusi Log Transform', fontsize=12, fontweight='bold')
            plt.xlabel('Log(RR + 1)')
            plt.ylabel('Frekuensi')
            plt.grid(True, alpha=0.3)
        
        # Square root transformation
        plt.subplot(1, 3, 3)
        if 'RR_sqrt' in self.data.columns:
            valid_sqrt = self.data['RR_sqrt'].dropna()
            plt.hist(valid_sqrt, bins=30, alpha=0.7, color='orange', edgecolor='black')
            plt.title('Distribusi Sqrt Transform', fontsize=12, fontweight='bold')
            plt.xlabel('√RR')
            plt.ylabel('Frekuensi')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/07_transformasi_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ 07_transformasi_comparison.png saved")
        plt.show()
        
        # 8. Curah Hujan Bulanan Kumulatif
        plt.figure(figsize=(12, 6))
        monthly_cumsum = self.data.groupby(['Year', 'Month'])['RR_imputed'].sum().reset_index()
        monthly_cumsum['Date'] = pd.to_datetime(monthly_cumsum[['Year', 'Month']].assign(day=1))
        monthly_cumsum = monthly_cumsum.sort_values('Date')
        
        plt.plot(monthly_cumsum['Date'], monthly_cumsum['RR_imputed'], linewidth=2, color='darkblue')
        plt.title('Curah Hujan Bulanan Kumulatif', fontsize=14, fontweight='bold')
        plt.xlabel('Tanggal', fontsize=12)
        plt.ylabel('Curah Hujan Bulanan (mm)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Tambahkan moving average
        if len(monthly_cumsum) > 12:
            monthly_cumsum['MA_12'] = monthly_cumsum['RR_imputed'].rolling(window=12).mean()
            plt.plot(monthly_cumsum['Date'], monthly_cumsum['MA_12'], 
                    linewidth=2, color='red', alpha=0.7, label='Moving Average (12 bulan)')
            plt.legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/08_curah_hujan_kumulatif.png', dpi=300, bbox_inches='tight')
            print("✅ 08_curah_hujan_kumulatif.png saved")
        plt.show()
        
        # 9. Seasonal Decomposition
        plt.figure(figsize=(12, 10))
        try:
            # Buat monthly time series untuk decomposition
            monthly_ts = self.data.groupby(['Year', 'Month'])['RR_imputed'].mean().reset_index()
            monthly_ts['Date'] = pd.to_datetime(monthly_ts[['Year', 'Month']].assign(day=1))
            monthly_ts = monthly_ts.set_index('Date')['RR_imputed'].dropna()
            
            if len(monthly_ts) >= 24:  # Minimal 2 tahun
                decomposition = seasonal_decompose(monthly_ts, model='additive', period=12)
                
                # Plot components
                plt.subplot(4, 1, 1)
                plt.plot(decomposition.observed, linewidth=2, color='blue')
                plt.title('Observed', fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(4, 1, 2)
                plt.plot(decomposition.trend, linewidth=2, color='green')
                plt.title('Trend', fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(4, 1, 3)
                plt.plot(decomposition.seasonal, linewidth=2, color='orange')
                plt.title('Seasonal', fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                plt.subplot(4, 1, 4)
                plt.plot(decomposition.resid, linewidth=2, color='red')
                plt.title('Residual', fontsize=12, fontweight='bold')
                plt.grid(True, alpha=0.3)
                
                plt.suptitle('Seasonal Decomposition', fontsize=16, fontweight='bold')
                
            else:
                plt.text(0.5, 0.5, 'Data tidak cukup untuk decomposition\n(minimal 24 bulan)', 
                        ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)
                plt.title('Seasonal Decomposition', fontsize=14, fontweight='bold')
                
        except Exception as e:
            plt.text(0.5, 0.5, f'Decomposition error:\n{str(e)}', 
                    ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
            plt.title('Seasonal Decomposition', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/09_seasonal_decomposition.png', dpi=300, bbox_inches='tight')
            print("✅ 09_seasonal_decomposition.png saved")
        plt.show()
        
        # 10. Data Quality Summary 
        fig, ax = plt.subplots(figsize=(8, 7))

        # Hitung metrik kualitas data
        missing_8888 = self.missing_stats['missing_8888']
        missing_nan = self.missing_stats['missing_nan']
        zero_values = self.missing_stats['zero_values']
        valid_data = len(self.data) - missing_8888 - missing_nan
        total_data = len(self.data)
        missing_data = missing_8888 + missing_nan

        # Donut chart setup
        donut_vals = [valid_data, missing_data, zero_values]
        donut_labels = ['Valid', 'Missing', 'Zero']
        colors = ['#4CAF50', '#FF6B6B', '#FFD700']

        wedges, _, _ = ax.pie(donut_vals, labels=donut_labels, autopct='%1.1f%%',
                            startangle=90, pctdistance=0.85, colors=colors,
                            textprops={'fontsize': 10})
        ax.add_artist(plt.Circle((0, 0), 0.70, fc='white'))  # Buat donut

        ax.set_title('📊 Komposisi Kualitas Data', fontsize=14, fontweight='bold')

        # Ringkasan statistik
        score_valid = valid_data / total_data * 100

        summary_text = f'''Ringkasan:
        • Total: {total_data:,}
        • Valid: {valid_data:,} ({score_valid:.1f}%)
        • Missing: {missing_data:,} ({missing_data/total_data*100:.1f}%)
        • Zero: {zero_values:,} ({zero_values/total_data*100:.1f}%)
        '''

        plt.figtext(0.5, 0.01, summary_text, fontsize=10, fontfamily='monospace',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9))

        plt.tight_layout()

        # Simpan jika diminta
        if save_plots:
            plt.savefig(f'{output_dir}/10_data_quality_summary.png', dpi=300, bbox_inches='tight')
            print("✅ 10_data_quality_summary.png saved")
        plt.show()

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
            print(f"\n📈 STATISTIK DESKRIPTIF (setelah preprocessing):")
            print(f"   • Mean: {valid_data.mean():.2f} mm")
            print(f"   • Median: {valid_data.median():.2f} mm")
            print(f"   • Std Dev: {valid_data.std():.2f} mm")
            print(f"   • Min: {valid_data.min():.2f} mm")
            print(f"   • Max: {valid_data.max():.2f} mm")
            print(f"   • Skewness: {valid_data.skew():.2f}")
        
        if hasattr(self, 'outliers') and 'iqr' in self.outliers:
            print(f"\n⚠️  OUTLIER DETECTION:")
            print(f"   • IQR method: {len(self.outliers['iqr'])} outliers")
            print(f"   • Upper bound: {self.outliers['bounds']['upper']:.2f} mm")
        
        print(f"\n✅ PREPROCESSING STEPS COMPLETED:")
        print(f"   • ✓ Data cleaning & special values handling")
        print(f"   • ✓ Missing values analysis")
        print(f"   • ✓ Outlier detection")
        print(f"   • ✓ Data transformation")
        print(f"   • ✓ Stationarity testing")
        print(f"   • ✓ Comprehensive visualization")
        
        print(f"\n📋 REKOMENDASI UNTUK FORECASTING:")
        missing_pct = self.missing_stats['missing_percentage']
        if missing_pct > 50:
            print(f"   • ⚠️  Missing values tinggi ({missing_pct:.1f}%) - pertimbangkan aggregasi temporal")
        elif missing_pct > 20:
            print(f"   • ⚠️  Missing values sedang ({missing_pct:.1f}%) - gunakan imputasi hati-hati")
        else:
            print(f"   • ✅ Missing values rendah ({missing_pct:.1f}%) - data cukup baik")
        
        if len(valid_data) > 0 and valid_data.skew() > 2:
            print(f"   • ✅ Distribusi sangat skewed - gunakan transformasi log")
        elif len(valid_data) > 0 and valid_data.skew() > 1:
            print(f"   • ✅ Distribusi skewed - pertimbangkan transformasi")
        
        print(f"\n🎯 SIAP UNTUK HOLT-WINTERS FORECASTING!")
        print("="*60)

def main():
    """
    Fungsi utama untuk menjalankan preprocessing
    """
    print("🌧️  PREPROCESSING CURAH HUJAN - DATASET BMKG")
    print("="*50)

    sys.stdout = open("preprocessing_log.txt", "w")

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
        yearly_missing = preprocessor.analyze_missing_values()
        
        # FASE 3: Cleaning data
        print("\n🔄 FASE 3: Data Cleaning")
        preprocessor.clean_rainfall_data()
        
        # FASE 4: Imputasi musiman
        print("\n🔄 FASE 4: Imputasi Berbasis Musiman")
        preprocessor.seasonal_imputation()
        
        # FASE 5: Deteksi outlier
        print("\n🔄 FASE 5: Deteksi Outlier Advanced")
        preprocessor.detect_outliers_advanced()
        
        # FASE 6: Transformasi data
        print("\n🔄 FASE 6: Transformasi Data")
        preprocessor.transform_data()
        
        # FASE 7: Uji stationarity
        print("\n🔄 FASE 7: Uji Stationarity")
        preprocessor.stationarity_tests()
        
        # FASE 8: Visualisasi komprehensif
        print("\n🔄 FASE 8: Visualisasi Komprehensif")
        preprocessor.create_individual_plots()
        
        # FASE 9: Laporan ringkasan
        print("\n🔄 FASE 9: Laporan Ringkasan")
        preprocessor.summary_report()
        
        # Simpan hasil preprocessing
        output_path = "preprocessed_rainfall_data.csv"
        
        # Simpan hanya kolom terkait RR
        rr_columns = [
            'Date', 'Year', 'Month', 'Day',
            'RR_original', 'RR_estimation_method',
            'RR_imputed', 'imputation_method', 'is_outlier',
            'RR_log', 'RR_sqrt', 'RR_boxcox'
        ]
        preprocessor.data[rr_columns].to_csv(output_path, index=False)
        print(f"\n💾 Data hasil preprocessing disimpan ke: {output_path}")
        
        # Informasi kolom hasil preprocessing
        print(f"\n📊 KOLOM HASIL PREPROCESSING:")
        processed_columns = [
            'Date', 'Year', 'Month', 'Day',
            'RR_original', 'RR_imputed', 'RR_log', 'RR_sqrt',
            'is_outlier'
        ]
        
        available_columns = [col for col in processed_columns if col in preprocessor.data.columns]
        for col in available_columns:
            non_null_count = preprocessor.data[col].notna().sum()
            print(f"   • {col}: {non_null_count:,} non-null values")
        
        # Rekomendasi untuk Holt-Winters
        print(f"\n🎯 REKOMENDASI UNTUK HOLT-WINTERS:")
        valid_data = preprocessor.data['RR_imputed'].dropna()
        
        if len(valid_data) > 0:
            # Cek seasonal pattern
            seasonal_strength = preprocessor.data.groupby('Month')['RR_imputed'].std().mean()
            print(f"   • Data tersedia untuk forecasting: {len(valid_data):,} records")
            print(f"   • Seasonal strength: {seasonal_strength:.2f}")
            
            # Rekomendasi parameter Holt-Winters
            if seasonal_strength > 10:
                print(f"   • Gunakan Holt-Winters ADDITIVE (seasonal pattern kuat)")
            else:
                print(f"   • Gunakan Holt-Winters MULTIPLICATIVE (seasonal pattern lemah)")
            
            # Rekomendasi periode seasonal
            print(f"   • Seasonal period: 12 (bulanan) atau 365 (harian)")
            
            # Rekomendasi transformasi
            skewness = valid_data.skew()
            if skewness > 2:
                print(f"   • Gunakan RR_log untuk forecasting (skewness tinggi: {skewness:.2f})")
            elif skewness > 1:
                print(f"   • Pertimbangkan RR_sqrt untuk forecasting (skewness sedang: {skewness:.2f})")
            else:
                print(f"   • Gunakan RR_imputed untuk forecasting (skewness rendah: {skewness:.2f})")
        
        print(f"\n🎉 PREPROCESSING SELESAI!")
        print(f"📁 File output: {output_path}")
        print(f"🚀 Siap untuk implementasi Holt-Winters forecasting!")
        
        return preprocessor
        
    except FileNotFoundError:
        print(f"❌ Error: File {data_path} tidak ditemukan!")
        print("📋 Pastikan file data BMKG tersedia dengan kolom:")
        print("   • Date: Tanggal observasi")
        print("   • RR: Curah hujan (mm)")
        print("   • Kolom lain (opsional): RH_AVG, Temperature, dll.")
        return None
        
    except Exception as e:
        print(f"❌ Error dalam preprocessing: {str(e)}")
        print("🔍 Periksa kembali format data dan kolom yang tersedia")
        return None
    
if __name__== "__main__":
    result = main() 