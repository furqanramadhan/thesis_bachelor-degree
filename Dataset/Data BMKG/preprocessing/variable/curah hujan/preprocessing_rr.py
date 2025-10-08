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
        
     # 1. Time Series Plot - PERBAIKAN
        plt.style.use('default')  # Ubah ke default (background putih)
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'

        plt.figure(figsize=(16, 8))
        valid_data = self.data[self.data['RR_imputed'].notna()]

        if not valid_data.empty:
            # Urutkan data
            valid_data = valid_data.sort_values('Date')

            # Plot garis utama dengan linewidth yang lebih tebal
            plt.plot(valid_data['Date'], valid_data['RR_imputed'], 
                    color='blue', alpha=0.7, linewidth=2.0)  # Ubah dari 1.0 ke 2.0
            
            categories = {
                'Hujan Sedang (20–49.9 mm)': ((valid_data['RR_imputed'] >= 20) & (valid_data['RR_imputed'] <= 49.9)),
                'Hujan Lebat (50–99.9 mm)': ((valid_data['RR_imputed'] >= 50) & (valid_data['RR_imputed'] <= 99.9)),
                'Hujan Sangat Lebat (100–150 mm)': ((valid_data['RR_imputed'] >= 100) & (valid_data['RR_imputed'] <= 150)),
                'Hujan Ekstrem (>150 mm)': (valid_data['RR_imputed'] > 150)
            }
            colors = {
            'Hujan Sedang (20–49.9 mm)': '#06923E',
            'Hujan Lebat (50–99.9 mm)': '#FFDE63',
            'Hujan Sangat Lebat (100–150 mm)': '#DC2525',
            'Hujan Ekstrem (>150 mm)': '#222831'
            }

            for label, mask in categories.items():
                subset = valid_data[mask]
                if not subset.empty:
                    plt.scatter(subset['Date'], subset['RR_imputed'], label=label,
                            color=colors[label], s=45, alpha=0.9) 

            # Sumbu X dan grid
            start_date, end_date = valid_data['Date'].min(), valid_data['Date'].max()
            date_ticks = pd.date_range(start=start_date, end=end_date, freq='YS')
            plt.xticks(date_ticks, [d.year for d in date_ticks], rotation=0, ha='center', fontsize=12)  # Tambah fontsize
            plt.gca().set_xticks(pd.date_range(start_date, end_date, freq='MS'), minor=True)
            
            # Grid dengan style yang sama seperti plot #4
            plt.grid(True, axis='y', alpha=0.6, linestyle='-', linewidth=0.4, color='gray')
            plt.grid(True, axis='y', which='minor', alpha=0.3, linestyle=':', linewidth=0.5, color='lightgray')

            # Label dengan fontsize yang sama seperti plot #4
            plt.xlabel('Tahun', fontsize=12)
            plt.ylabel('Curah Hujan (mm)', fontsize=12)
            
            # Legend dengan style yang sama seperti plot #4
            plt.legend(loc='upper right', fontsize=10, frameon=True, fancybox=True, 
                    shadow=True, framealpha=0.9)  # Style sama seperti plot #4

        else:
            plt.text(0.5, 0.5, 'Tidak ada data valid untuk ditampilkan',
                    ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

        plt.tight_layout()
        plt.ylim(0, 200)

        # Set background axes menjadi putih (sama seperti plot #4)
        plt.gca().set_facecolor('white')

        if save_plots:
            plt.savefig(f'{output_dir}/01_time_series_plot.png', dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')  # Tambah facecolor dan edgecolor
            print("✅ 01_time_series_plot.png saved")
        plt.show()


       # 2. Histogram dan Distribusi 
        plt.figure(figsize=(12, 8))

        valid_rr = self.data['RR_imputed'].dropna()

        if not valid_rr.empty:
            # Definisi kategori hujan BMKG
            categories = [
                (0, 0.4, 'Berawan'),
                (0.5, 19.9, 'Hujan Ringan'),
                (20, 49.9, 'Hujan Sedang'),
                (50, 99.9, 'Hujan Lebat'),
                (100, 150, 'Hujan Sangat Lebat'),
                (150.1, 200, 'Hujan Ekstrem')
            ]
            
            category_counts = []
            category_labels = []
            category_positions = []
            
            for i, (min_val, max_val, label) in enumerate(categories):
                count = ((valid_rr >= min_val) & (valid_rr <= max_val)).sum()
                category_counts.append(count)
                category_labels.append(f'{label}')
                category_positions.append(i)
            
            # Plot bar horizontal
            bars = plt.barh(
                category_positions, category_counts,
                color=['green', 'skyblue', 'orange', 'red', 'darkred', 'purple'],
                edgecolor='black', alpha=0.8, height=0.6
            )
            
            # Kustomisasi visual
            plt.xlabel('Frekuensi', fontsize=11)
            plt.ylabel('Kategori Hujan', fontsize=11)
            plt.yticks(category_positions, category_labels)
            
            for i, (bar, count) in enumerate(zip(bars, category_counts)):
                percentage = (count / len(valid_rr)) * 100
                plt.text(bar.get_width() + max(category_counts)*0.01, 
                        bar.get_y() + bar.get_height()/2, 
                        f'{percentage:.1f}%', ha='left', va='center', fontweight='bold')
            
            plt.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()

            # Simpan jika diminta
            if save_plots:
                plt.savefig(f'{output_dir}/02_histogram_kategori.png', dpi=300, bbox_inches='tight')
                print("✅ 02_histogram_kategori.png saved")
        else:
            plt.text(0.5, 0.5, 'Tidak ada data valid untuk ditampilkan',
                    ha='center', va='center', fontsize=14, transform=plt.gca().transAxes)

        plt.show()
        
        # 3. Boxplot Curah Hujan Harian per Tahun
        plt.style.use('default')  # Background putih
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.figure(figsize=(14, 6))

        monthly_data = []
        months = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        # Kumpulkan data curah hujan harian untuk tiap bulan (tanpa tahun)
        for month in months:
            month_rr = self.data[self.data['Month'] == month]['RR_imputed'].dropna()
            if not month_rr.empty:
                monthly_data.append(month_rr)
            else:
                monthly_data.append(pd.Series(dtype=float))  # Tambahkan placeholder jika kosong

        # Plot jika ada data
        if any(len(m) > 0 for m in monthly_data):
            box = plt.boxplot(monthly_data, labels=[
                'Des', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei',
                'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov'
            ], patch_artist=True, showfliers=True)

            colors = plt.cm.Set3(np.linspace(0, 1, 12))
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            plt.xlabel('Bulan', fontsize=12)
            plt.ylabel('Curah Hujan Harian (mm)', fontsize=12)
            plt.grid(True, axis='y', alpha=0.6, linestyle='-', linewidth=0.4, color='gray')
            plt.grid(True, axis='y', which='minor', alpha=0.3, linestyle=':', linewidth=0.5, color='lightgray')
            plt.xticks(rotation=0, ha='center')

        plt.tight_layout()
        plt.ylim(0, 200)

        # Simpan jika diminta
        if save_plots:
            plt.savefig(f'{output_dir}/03_boxplot_bulanan.png', dpi=300, bbox_inches='tight')
            print("✅ 03_boxplot_bulanan.png saved")

        plt.show()

        # 4. Pola Curah Hujan Harian (Des–Nov) - MODIFIED
        print("🔄 Plot #4: Pola Curah Hujan Harian")

        # Set background menjadi putih
        plt.style.use('default')  # Reset ke style default
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'

        plt.figure(figsize=(16, 10))

        # Periode yang diinginkan: tahun individual + gabungan
        periods = {
            '2020': [2020],
            '2021': [2021],
            '2022': [2022],
            '2023': [2023],
            '2024': [2024],
            '2025': [2025],  # Hingga Juni 2025
            '2005-2025': list(range(2005, 2026))
        }

        # Warna solid pekat sesuai permintaan
        colors = {
            '2020': '#A16D28',    # Coklat
            '2021': '#DC143C',    # Merah
            '2022': '#228B22',    # Hijau
            '2023': '#0000FF',    # Biru
            '2024': '#FF8C00',    # Jingga
            '2025': '#FFD700',    # Emas
            '2005-2025': '#000000'  # Hitam
        }

        custom_labels = {
            '2020': 'Curah Hujan Harian 2020',
            '2021': 'Curah Hujan Harian 2021',
            '2022': 'Curah Hujan Harian 2022',
            '2023': 'Curah Hujan Harian 2023',
            '2024': 'Curah Hujan Harian 2024',
            '2025': 'Curah Hujan Harian 2025 (Jan–Juni)',
            '2005-2025': 'Rata-Rata Curah Hujan Harian 2005–2025'
        }

        # Filter data untuk batas maksimal 200mm
        max_rainfall_limit = 200

        for i, (label, years) in enumerate(periods.items()):
            period_data = self.data[self.data['Year'].isin(years)].copy()
            
            # Filter curah hujan > 200mm
            period_data = period_data[period_data['RR_imputed'] <= max_rainfall_limit]
            
            # Untuk 2025, filter hanya sampai Juni
            if label == '2025':
                period_data = period_data[period_data['Month'] <= 6]
            
            if period_data.empty:
                continue
            
            # Buat kolom day-of-year untuk plotting
            period_data['DayOfYear'] = period_data['Month'] * 30 + period_data.get('Day', 15)  # Approximation
            
            if label == '2005-2025':
                # Untuk periode gabungan, hitung rata-rata harian per day-of-year
                daily_avg = period_data.groupby('DayOfYear')['RR_imputed'].mean().reset_index()
                
                plt.plot(daily_avg['DayOfYear'], daily_avg['RR_imputed'], 
                        linewidth=2.5, color=colors[label], 
                        label=custom_labels[label], alpha=0.9)
            else:
                # Untuk tahun individual, plot semua data harian
                # Menggunakan scatter plot untuk menghindari garis yang terlalu padat
                plt.plot(period_data['DayOfYear'], period_data['RR_imputed'],
                        linewidth=1.2, color=colors[label], label=custom_labels[label], alpha=0.8)

        # Kustomisasi sumbu X untuk menampilkan nama bulan
        month_positions = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
                    'Jul', 'Ags', 'Sep', 'Okt', 'Nov', 'Des']

        plt.xticks(month_positions, month_names, rotation=0, ha='center')

        # Kustomisasi grafik
        plt.xlabel('Bulan', fontsize=12)
        plt.ylabel('Curah Hujan Harian (mm)', fontsize=12)

        # Set batas Y-axis ke 200mm
        plt.ylim(0, 200)

        # Grid 
        plt.grid(True, axis='y', alpha=0.6, linestyle='-', linewidth=0.4, color='gray')
        plt.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5, color='lightgray')

        # Urutkan legend berdasarkan urutan yang diinginkan
        legend_order = ['2005-2025','2020','2021','2022','2023','2024','2025']
        custom_legend_order = [custom_labels[k] for k in legend_order]  # Konversi ke label final

        handles, labels = plt.gca().get_legend_handles_labels()
        legend_dict = dict(zip(labels, handles))

        ordered_handles = [legend_dict[label] for label in custom_legend_order if label in legend_dict]
        ordered_labels = [label for label in custom_legend_order if label in legend_dict]

        plt.legend(ordered_handles, ordered_labels, loc='upper right', fontsize=10, 
                frameon=True, fancybox=True, shadow=True, framealpha=0.9)

        # Set background axes menjadi putih
        plt.gca().set_facecolor('white')

        plt.tight_layout()

        if save_plots:
            plt.savefig(f'{output_dir}/04_pola_curah_hujan_harian.png', dpi=300, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
            print("✅ 04_pola_curah_hujan_harian.png saved")

        plt.show()

        #  # 4. Pergeseran Pola Curah Hujan Bulanan (Des–Nov) - MODIFIED
        # print("🔄 Plot #4: Pergeseran Pola Curah Hujan Bulanan (Des–Nov)")

        # # Set background menjadi putih
        # plt.style.use('default')  # Reset ke style default
        # plt.rcParams['figure.facecolor'] = 'white'
        # plt.rcParams['axes.facecolor'] = 'white'

        # plt.figure(figsize=(14, 8))

        # # Periode yang diinginkan: tahun individual + gabungan
        # periods = {
        #     '2020': [2020],
        #     '2021': [2021],
        #     '2022': [2022],
        #     '2023': [2023],
        #     '2024': [2024],
        #     '2025': [2025],  # Hingga Juni 2025
        #     '2005-2025': list(range(2005, 2026))
        # }

        # month_order = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # month_names = ['Des', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei', 
        #             'Jun', 'Jul', 'Ags', 'Sep', 'Okt', 'Nov']

        # # Warna solid pekat sesuai permintaan
        # colors = {
        #     '2020': '#A16D28',    # Coklat
        #     '2021': '#DC143C',    # Merah
        #     '2022': '#228B22',    # Hijau
        #     '2023': '#0000FF',    # Biru
        #     '2024': '#FF8C00',    # Jingga
        #     '2025': '#FFD700',    # Ungu untuk 2025
        #     '2005-2025': '#000000'  # Hitam
        # }

        # custom_labels = {
        #     '2020': 'Total Curah Hujan 2020',
        #     '2021': 'Total Curah Hujan 2021',
        #     '2022': 'Total Curah Hujan 2022',
        #     '2023': 'Total Curah Hujan 2023',
        #     '2024': 'Total Curah Hujan 2024',
        #     '2025': 'Total Curah Hujan 2025 (Jan–Juni)',
        #     '2005-2025': 'Rata-Rata Curah Hujan 2005–2025'
        # }
        

        # line_styles = ['-'] * len(periods)  

        # shift_summary = {}

        # for i, (label, years) in enumerate(periods.items()):
        #     period_data = self.data[self.data['Year'].isin(years)]
            
        #     # Untuk 2025, filter hanya sampai Juni
        #     if label == '2025':
        #         period_data = period_data[period_data['Month'] <= 6]
            
        #     if period_data.empty:
        #         continue
            
        #     # Hitung TOTAL curah hujan bulanan (bukan rata-rata)
        #     monthly_total = period_data.groupby('Month')['RR_imputed'].sum()
            
        #     # Untuk periode gabungan 2005-2025, hitung rata-rata dari total tahunan
        #     if label == '2005-2025':
        #         # Hitung total per tahun per bulan, lalu rata-rata
        #         yearly_monthly_total = period_data.groupby(['Year', 'Month'])['RR_imputed'].sum().reset_index()
        #         monthly_avg_total = yearly_monthly_total.groupby('Month')['RR_imputed'].mean()
        #         monthly_values = [monthly_avg_total.get(m, 0) for m in month_order]
        #     else:
        #         monthly_values = [monthly_total.get(m, 0) for m in month_order]
            
        #     # Untuk 2025, hanya tampilkan Jan-Jun (tidak termasuk Des)
        #     if label == '2025':
        #         # Buat array dengan NaN untuk semua bulan
        #         full_monthly_values = [np.nan] * 12
        #         for idx, month in enumerate(month_order):
        #             if 1 <= month <= 6:  # Hanya isi Jan-Jun untuk 2025
        #                 full_monthly_values[idx] = monthly_values[idx] if idx < len(monthly_values) else np.nan
        #         monthly_values = full_monthly_values
            
        #     # Plot garis untuk semua periode
        #     if label == '2005-2025':
        #         plt.plot(month_names, monthly_values, marker='o', linewidth=3.5, markersize=8,
        #                 color=colors[label], linestyle=line_styles[0], label=custom_labels[label], alpha=0.9)
        #     elif label == '2025':
        #         # Untuk 2025, plot hanya bagian yang memiliki data (Jan-Jun)
        #         jan_idx = month_names.index('Jan')
        #         jun_idx = month_names.index('Jun')
        #         plt.plot(month_names[jan_idx:jun_idx+1], monthly_values[jan_idx:jun_idx+1], 
        #                 marker='o', linewidth=2.5, markersize=6,
        #                 color=colors[label], linestyle=line_styles[0], label=custom_labels[label], alpha=0.8)
        #     else:
        #         plt.plot(month_names, monthly_values, marker='o', linewidth=2.5, markersize=6,
        #                 color=colors[label], linestyle=line_styles[0], label=custom_labels[label], alpha=0.8)
            
        #     max_idx = np.argmax(monthly_values)
        #     min_idx = np.argmin(monthly_values)            
           
        #     # Simpan ringkasan untuk analisis pergeseran
        #     shift_summary[label] = {
        #         'max_month': month_order[max_idx],
        #         'max_value': monthly_values[max_idx],
        #         'min_month': month_order[min_idx],
        #         'min_value': monthly_values[min_idx],
        #         'total': sum(monthly_values)
        #     }

        # # Kustomisasi grafik
        # plt.xlabel('Bulan', fontsize=12)
        # plt.ylabel('Curah Hujan (mm)', fontsize=12)

        # # Grid 
        # plt.grid(True, axis='y', alpha=0.6, linestyle='-', linewidth=0.4, color='gray')
        # plt.grid(True, axis='y', which='minor', alpha=0.3, linestyle=':', linewidth=0.5, color='lightgray')

        # # Urutkan legend berdasarkan urutan yang diinginkan
        # legend_order = ['2005-2025','2020','2021','2022','2023','2024','2025']
        # custom_legend_order = [custom_labels[k] for k in legend_order]  # Konversi ke label final

        # handles, labels = plt.gca().get_legend_handles_labels()
        # legend_dict = dict(zip(labels, handles))

        # ordered_handles = [legend_dict[label] for label in custom_legend_order if label in legend_dict]
        # ordered_labels = [label for label in custom_legend_order if label in legend_dict]


        # plt.legend(ordered_handles, ordered_labels, loc='upper right', fontsize=10, 
        #         frameon=True, fancybox=True, shadow=True, framealpha=0.9)

        # # Tulisan horizontal
        # plt.xticks(rotation=0, ha='center')

        # plt.ylim(bottom=0)

        # # Set background axes menjadi putih
        # plt.gca().set_facecolor('white')

        # plt.tight_layout()

        # if save_plots:
        #     plt.savefig(f'{output_dir}/04_pergeseran_curah_hujan.png', dpi=300, bbox_inches='tight', 
        #                 facecolor='white', edgecolor='none')
        #     print("✅ 04_pergeseran_curah_hujan.png saved")

        # plt.show()


        # #5. Heatmap Season
        # print("🔄 Plot #5: Seasonal Heatmap Analysis")
        # plt.style.use('default')
        # plt.rcParams['figure.facecolor'] = 'white'
        # plt.rcParams['axes.facecolor'] = 'white'

        # fig, ax = plt.subplots(figsize=(14, 8))

        # # Definisi season
        # seasons = {
        #     'DJF': [12, 1, 2], 'JFM': [1, 2, 3], 'FMA': [2, 3, 4], 'MAM': [3, 4, 5],
        #     'AMJ': [4, 5, 6], 'MJJ': [5, 6, 7], 'JJA': [6, 7, 8], 'JAS': [7, 8, 9],
        #     'ASO': [8, 9, 10], 'SON': [9, 10, 11], 'OND': [10, 11, 12], 'NDJ': [11, 12, 1]
        # }

        # season_names = list(seasons.keys())
        # years = sorted(self.data['Year'].dropna().unique())
        # years = [y for y in years if y < 2025]

        # # Buat matrix untuk heatmap
        # heatmap_data = np.zeros((len(years), len(season_names)))

        # for i, year in enumerate(years):
        #     year_data = self.data[self.data['Year'] == year]
            
        #     for j, (season_name, months) in enumerate(seasons.items()):
        #         season_total = 0
        #         for month in months:
        #             month_data = year_data[year_data['Month'] == month]['RR_imputed']
        #             if not month_data.empty:
        #                 season_total += month_data.sum()
                
        #         heatmap_data[i, j] = season_total

        # # Buat heatmap
        # im = ax.imshow(
        #     heatmap_data, 
        #     cmap='Reds',                 # 🔴 Ganti ke warna merah
        #     aspect='auto', 
        #     interpolation='nearest',
        #     vmin=0, vmax=1600             # 🔒 Batasi skala 0–800 mm
        # )

        # # Set ticks dan labels
        # ax.set_xticks(range(len(season_names)))
        # ax.set_xticklabels(season_names, fontsize=11)
        # ax.set_yticks(range(len(years)))  # Tampilkan semua tahun
        # ax.set_yticklabels([str(int(y)) for y in years], fontsize=10)

        # cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        # cbar.set_label('Total Curah Hujan (mm)', fontsize=11)
        # cbar.ax.tick_params(labelsize=10)

        # # Labels
        # ax.set_xlabel('Season', fontsize=12)
        # plt.tight_layout()

        # if save_plots:
        #     plt.savefig(f'{output_dir}/05_seasonal_heatmap.png', dpi=300, 
        #                 bbox_inches='tight', facecolor='white', edgecolor='none')
        #     print("✅ 05_seasonal_heatmap.png saved")

        # plt.show()


        #5. Heatmap Season
        print("🔄 Plot #5: Seasonal Heatmap Analysis")
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'

        fig, ax = plt.subplots(figsize=(14, 8))

        # Definisi season
        seasons = {
            'DJF': [12, 1, 2], 'JFM': [1, 2, 3], 'FMA': [2, 3, 4], 'MAM': [3, 4, 5],
            'AMJ': [4, 5, 6], 'MJJ': [5, 6, 7], 'JJA': [6, 7, 8], 'JAS': [7, 8, 9],
            'ASO': [8, 9, 10], 'SON': [9, 10, 11], 'OND': [10, 11, 12], 'NDJ': [11, 12, 1]
        }

        season_names = list(seasons.keys())
        years = sorted(self.data['Year'].dropna().unique())
        years = [y for y in years if y < 2025]

        # Buat matrix untuk heatmap
        heatmap_data = np.zeros((len(years), len(season_names)))

        for i, year in enumerate(years):
            year_data = self.data[self.data['Year'] == year]
            
            for j, (season_name, months) in enumerate(seasons.items()):
                season_total = 0
                for month in months:
                    month_data = year_data[year_data['Month'] == month]['RR_imputed']
                    if not month_data.empty:
                        season_total += month_data.sum()
                
                heatmap_data[i, j] = season_total

        # 🎯 Adaptive Percentile Scaling
        # Hitung percentile 95 dari data untuk vmax yang optimal
        valid_data = heatmap_data[heatmap_data > 0]  # Exclude zero values
        if len(valid_data) > 0:
            vmax_adaptive = np.percentile(valid_data, 95)
            vmin_adaptive = 0
            print(f"📊 Adaptive scaling: vmin={vmin_adaptive:.1f}mm, vmax={vmax_adaptive:.1f}mm")
        else:
            vmax_adaptive = 800  # Fallback value
            vmin_adaptive = 0
            print("⚠️ Using fallback scaling values")

        # Buat heatmap dengan adaptive scaling
        im = ax.imshow(
            heatmap_data, 
            cmap='Reds',                     # 🔴 Colormap merah
            aspect='auto', 
            interpolation='nearest',
            vmin=vmin_adaptive,              # 🎯 Adaptive vmin
            vmax=vmax_adaptive               # 🎯 Adaptive vmax (P95)
        )

        # Set ticks dan labels
        ax.set_xticks(range(len(season_names)))
        ax.set_xticklabels(season_names, fontsize=11)
        ax.set_yticks(range(len(years)))  # Tampilkan semua tahun
        ax.set_yticklabels([str(int(y)) for y in years], fontsize=10)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Total Curah Hujan (mm)', fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        # 📈 Tambahkan info scaling di title
        # plt.title(f'Seasonal Rainfall Heatmap (Scale: 0-{vmax_adaptive:.0f}mm, P95)', 
        #         fontsize=13, pad=15)

        # Labels
        ax.set_xlabel('Season', fontsize=12)
        plt.tight_layout()

        if save_plots:
            plt.savefig(f'{output_dir}/05_seasonal_heatmap.png', dpi=300, 
                        bbox_inches='tight', facecolor='white', edgecolor='none')
            print("✅ 05_seasonal_heatmap.png saved")

        plt.show()

        # 6. Data Quality Summary 
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
            plt.savefig(f'{output_dir}/06_data_quality_summary.png', dpi=300, bbox_inches='tight')
            print("✅ 06_data_quality_summary.png saved")
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
        
        print(f"\n✅ PREPROCESSING STEPS COMPLETED:")
        print(f"   • ✓ Data cleaning & special values handling")
        print(f"   • ✓ Missing values analysis")
        print(f"   • ✓ Outlier detection")
        print(f"   • ✓ Visualisasi utama")
        
        print(f"\n📋 REKOMENDASI UNTUK FORECASTING:")
        missing_pct = self.missing_stats['missing_percentage']
        if missing_pct > 50:
            print(f"   • ⚠️  Missing values tinggi ({missing_pct:.1f}%) - pertimbangkan aggregasi temporal")
        elif missing_pct > 20:
            print(f"   • ⚠️  Missing values sedang ({missing_pct:.1f}%) - gunakan imputasi hati-hati")
        else:
            print(f"   • ✅ Missing values rendah ({missing_pct:.1f}%) - data cukup baik")
        
        print(f"\n🎯 SIAP UNTUK HOLT-WINTERS FORECASTING!")
        print("="*60)

def main():
    """
    Fungsi utama untuk menjalankan preprocessing
    """
    print("🌧️  PREPROCESSING CURAH HUJAN - DATASET BMKG")
    print("="*50)

    sys.stdout = open("preprocessing_log_rainfall.txt", "w")

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
        
        # FASE 6: Visualisasi komprehensif
        print("\n🔄 FASE 6: Visualisasi Komprehensif")
        preprocessor.create_individual_plots()
        
        # FASE 7: Laporan ringkasan
        print("\n🔄 FASE 7: Laporan Ringkasan")
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
                
        # Informasi kolom hasil preprocessing
        print(f"\n📊 KOLOM HASIL PREPROCESSING:")
        processed_columns = [
            'Date', 'Year', 'Month', 'Day',
            'RR_original', 'RR_imputed', 'is_outlier'
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
            
            
        print(f"\n🎉 PREPROCESSING SELESAI!")
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