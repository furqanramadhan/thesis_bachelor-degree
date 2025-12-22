import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import os
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class Temperature_Analyzer:
    def __init__(self, data_path, output_dir=None):
        self.data_path = data_path
        self.output_dir = output_dir or os.path.dirname(data_path)
        self.data = None
        self.tn_stats = {}
        self.tx_stats = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created output directory: {self.output_dir}")

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
    
    def calculate_tavg_statistics_after_imputation(self):
        """
        Menghitung statistik deskriptif lengkap untuk TAVG setelah proses imputasi dan outlier treatment
        """
        print(f"\n=== STATISTIK DESKRIPTIF TAVG SETELAH IMPUTASI ===")
        
        if 'TAVG' not in self.data.columns:
            print("❌ Kolom TAVG tidak ditemukan! Jalankan calculate_and_validate_tavg() terlebih dahulu")
            return None
        
        # 1. ANALISIS DATA KOSONG/MISSING
        total_records = len(self.data)
        
        # Identifikasi missing values (NaN dan kode error BMKG)
        missing_mask = (
            self.data['TAVG'].isna() |
            (self.data['TAVG'] == 9999) |
            (self.data['TAVG'] == 8888) |
            (self.data['TAVG'] == -999) |
            (self.data['TAVG'] == -9999)
        )
        
        missing_count = missing_mask.sum()
        valid_count = total_records - missing_count
        
        # 2. FILTER DATA VALID UNTUK STATISTIK
        valid_tavg = self.data[~missing_mask]['TAVG']
        
        if len(valid_tavg) == 0:
            print("❌ Tidak ada data TAVG valid untuk analisis statistik")
            return None
        
        # 3. HITUNG STATISTIK DESKRIPTIF
        stats_dict = {
            'Jumlah Data': valid_count,
            'Jumlah Data Kosong': missing_count,
            'Minimum': valid_tavg.min(),
            'Q1': valid_tavg.quantile(0.25),
            'Median': valid_tavg.median(),
            'Mean': valid_tavg.mean(), 
            'Q3': valid_tavg.quantile(0.75),
            'Maksimum': valid_tavg.max(),
            'Standar Deviasi': valid_tavg.std()
        }
        
        # 4. TAMPILKAN HASIL DALAM TABEL RAPI
        print(f"\n📈 TABEL STATISTIK DESKRIPTIF TAVG (SETELAH IMPUTASI):")
        print("-" * 45)
        print(f"{'Statistik':<20} {'Nilai':<20}")
        print("-" * 45)
        
        for stat_name, value in stats_dict.items():
            if stat_name in ['Jumlah Data', 'Jumlah Data Kosong']:
                percentage = (value / total_records * 100) if total_records > 0 else 0
                print(f"{stat_name:<20} {int(value):,} ({percentage:.1f}%)")
            else:
                print(f"{stat_name:<20} {value:.2f}°C")
        
        print("-" * 45)
        
        # 5. INFORMASI TAMBAHAN DAN VALIDASI KUALITAS
        print(f"\n📊 INFORMASI TAMBAHAN TAVG:")
        
        # Range dan variabilitas
        range_val = stats_dict['Maksimum'] - stats_dict['Minimum']
        iqr = stats_dict['Q3'] - stats_dict['Q1']
        cv = (stats_dict['Standar Deviasi'] / stats_dict['Mean']) * 100 if stats_dict['Mean'] != 0 else 0
        
        print(f"   • Range: {range_val:.2f}°C ({stats_dict['Minimum']:.1f}°C - {stats_dict['Maksimum']:.1f}°C)")
        print(f"   • IQR (Interquartile Range): {iqr:.2f}°C")
        print(f"   • Coefficient of Variation: {cv:.2f}%")
        
        # Deteksi outliers potensial setelah treatment
        lower_bound = stats_dict['Q1'] - 1.5 * iqr
        upper_bound = stats_dict['Q3'] + 1.5 * iqr
        potential_outliers = valid_tavg[(valid_tavg < lower_bound) | (valid_tavg > upper_bound)]
        
        print(f"   • Batas outlier (IQR): {lower_bound:.2f}°C - {upper_bound:.2f}°C")
        print(f"   • Outliers potensial tersisa: {len(potential_outliers)} ({len(potential_outliers)/len(valid_tavg)*100:.2f}%)")
        
        # 6. KLASIFIKASI TEMPERATURE UNTUK KONTEKS IKLIM INDONESIA
        print(f"\n🌡️  DISTRIBUSI KATEGORI TAVG (KONTEKS IKLIM TROPIS):")
        
        temp_categories = {
            'Sejuk (<24°C)': valid_tavg < 24,
            'Normal (24-27°C)': (valid_tavg >= 24) & (valid_tavg < 27),
            'Hangat (27-30°C)': (valid_tavg >= 27) & (valid_tavg < 30), 
            'Panas (30-33°C)': (valid_tavg >= 30) & (valid_tavg < 33),
            'Sangat Panas (≥33°C)': valid_tavg >= 33
        }
        
        for category, mask in temp_categories.items():
            count = mask.sum()
            percentage = count / len(valid_tavg) * 100
            print(f"   • {category}: {count:,} ({percentage:.1f}%)")
        
        # 7. VALIDASI KUALITAS DATA SETELAH PREPROCESSING
        print(f"\n✅ VALIDASI KUALITAS DATA TAVG:")
        
        # Data completeness
        completeness = (valid_count / total_records) * 100
        if completeness >= 95:
            completeness_status = "Excellent"
        elif completeness >= 90:
            completeness_status = "Good" 
        elif completeness >= 80:
            completeness_status = "Acceptable"
        else:
            completeness_status = "Poor"
        
        print(f"   • Data Completeness: {completeness:.1f}% ({completeness_status})")
        
        # Temperature range validation
        if 22 <= stats_dict['Mean'] <= 32 and 20 <= stats_dict['Minimum'] <= 36 and stats_dict['Maksimum'] <= 38:
            range_status = "Valid (dalam range iklim tropis)"
        else:
            range_status = "Perlu review (di luar range normal)"
        
        print(f"   • Temperature Range: {range_status}")
        
        # Variability check
        if 1 <= stats_dict['Standar Deviasi'] <= 3:
            variability_status = "Normal (variabilitas wajar)"
        elif stats_dict['Standar Deviasi'] < 1:
            variability_status = "Rendah (mungkin terlalu uniform)"
        else:
            variability_status = "Tinggi (variabilitas besar)"
        
        print(f"   • Variabilitas: {variability_status}")
        
        # 8. PERBANDINGAN DENGAN TN DAN TX (JIKA TERSEDIA)
        if 'TN' in self.data.columns and 'TX' in self.data.columns:
            print(f"\n🔄 VALIDASI HUBUNGAN TN-TAVG-TX:")
            
            # Filter data yang valid untuk semua kolom
            valid_all_mask = (
                self.data['TN'].notna() & self.data['TX'].notna() & (~missing_mask) &
                (self.data['TN'] != 9999) & (self.data['TX'] != 9999) &
                (self.data['TN'] != 8888) & (self.data['TX'] != 8888)
            )
            
            if valid_all_mask.sum() > 0:
                valid_all_data = self.data[valid_all_mask]
                
                # Check logical relationship: TN < TAVG < TX
                tn_tavg_violations = (valid_all_data['TN'] >= valid_all_data['TAVG']).sum()
                tavg_tx_violations = (valid_all_data['TAVG'] >= valid_all_data['TX']).sum()
                
                print(f"   • Records dengan TN ≥ TAVG: {tn_tavg_violations} ({tn_tavg_violations/len(valid_all_data)*100:.2f}%)")
                print(f"   • Records dengan TAVG ≥ TX: {tavg_tx_violations} ({tavg_tx_violations/len(valid_all_data)*100:.2f}%)")
                
                # Check if TAVG is approximately (TN+TX)/2
                calculated_tavg = (valid_all_data['TN'] + valid_all_data['TX']) / 2
                tavg_diff = (valid_all_data['TAVG'] - calculated_tavg).abs()
                avg_diff = tavg_diff.mean()
                
                print(f"   • Rata-rata deviasi dari (TN+TX)/2: {avg_diff:.3f}°C")
                
                if avg_diff < 0.5:
                    calc_status = "Excellent consistency"
                elif avg_diff < 1.0:
                    calc_status = "Good consistency"
                else:
                    calc_status = "Needs review"
                    
                print(f"   • Status konsistensi: {calc_status}")
        
        # 9. SUMMARY AKHIR
        print(f"\n🎯 RINGKASAN STATISTIK TAVG:")
        print(f"   • Dataset: {total_records:,} records total")
        print(f"   • Valid data: {valid_count:,} ({completeness:.1f}%)")
        print(f"   • Temperature range: {stats_dict['Minimum']:.1f}°C - {stats_dict['Maksimum']:.1f}°C")
        print(f"   • Central tendency: Mean={stats_dict['Mean']:.1f}°C, Median={stats_dict['Median']:.1f}°C")
        print(f"   • Variability: SD={stats_dict['Standar Deviasi']:.2f}°C, CV={cv:.1f}%")
        
        if len(potential_outliers) == 0:
            print(f"   • Quality: Excellent (no outliers remaining)")
        elif len(potential_outliers) < 10:
            print(f"   • Quality: Good (minimal outliers: {len(potential_outliers)})")
        else:
            print(f"   • Quality: Acceptable ({len(potential_outliers)} outliers remain)")
        
        print(f"\n✅ Analisis statistik TAVG setelah imputasi selesai!")
        
        return {
            'statistics': stats_dict,
            'quality_metrics': {
                'completeness': completeness,
                'completeness_status': completeness_status,
                'range_status': range_status,
                'variability_status': variability_status,
                'outliers_remaining': len(potential_outliers),
                'total_records': total_records,
                'valid_records': valid_count
            },
            'temperature_distribution': {cat: mask.sum() for cat, mask in temp_categories.items()}
        }

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

    def calculate_and_validate_tavg(self):
        """
        Menghitung TAVG dari TN dan TX yang sudah diimputasi, 
        validasi dengan TAVG original, dan replace jika valid
        """
        print(f"\n=== KALKULASI DAN VALIDASI TAVG ===")
        
        # 1. PASTIKAN TN DAN TX SUDAH VALID
        if 'TN' not in self.data.columns or 'TX' not in self.data.columns:
            print("❌ TN atau TX tidak ditemukan! Lakukan imputasi terlebih dahulu.")
            return None
        
        # Filter data yang valid untuk TN dan TX
        valid_mask = (
            self.data['TN'].notna() & 
            self.data['TX'].notna() &
            (self.data['TN'] != 9999) & (self.data['TN'] != 8888) &
            (self.data['TX'] != 9999) & (self.data['TX'] != 8888) &
            (self.data['TN'] != -999) & (self.data['TN'] != -9999) &
            (self.data['TX'] != -999) & (self.data['TX'] != -9999)
        )
        
        valid_data = self.data[valid_mask].copy()
        
        if len(valid_data) == 0:
            print("❌ Tidak ada data TN-TX yang valid untuk kalkulasi TAVG")
            return None
        
        # 2. HITUNG TAVG_CALCULATED
        self.data['TAVG_calculated'] = (self.data['TN'] + self.data['TX']) / 2
        tavg_calculated = valid_data['TN'] + valid_data['TX']
        tavg_calculated = tavg_calculated / 2
        
        print(f"✅ TAVG_calculated berhasil dihitung untuk {len(valid_data)} records")
        print(f"   • Range TAVG_calculated: {tavg_calculated.min():.2f}°C - {tavg_calculated.max():.2f}°C")
        print(f"   • Mean TAVG_calculated: {tavg_calculated.mean():.2f}°C")
        
        # 3. VALIDASI DENGAN TAVG ORIGINAL (JIKA ADA)
        validation_results = {}
        
        if 'TAVG' in self.data.columns:
            print(f"\n📊 VALIDASI DENGAN TAVG ORIGINAL:")
            
            # Filter data yang memiliki TAVG original valid
            tavg_original_valid = (
                valid_data['TAVG'].notna() &
                (valid_data['TAVG'] != 9999) & 
                (valid_data['TAVG'] != 8888) &
                (valid_data['TAVG'] != -999) & 
                (valid_data['TAVG'] != -9999)
            )
            
            comparison_data = valid_data[tavg_original_valid].copy()
            
            if len(comparison_data) > 0:
                tavg_original = comparison_data['TAVG']
                tavg_calc_subset = (comparison_data['TN'] + comparison_data['TX']) / 2
                
                # Statistik perbandingan
                correlation = tavg_original.corr(tavg_calc_subset)
                mean_diff = (tavg_calc_subset - tavg_original).mean()
                rmse = np.sqrt(((tavg_calc_subset - tavg_original) ** 2).mean())
                mae = (tavg_calc_subset - tavg_original).abs().mean()
                
                print(f"   • Records dengan TAVG original valid: {len(comparison_data):,}")
                print(f"   • Correlation (original vs calculated): {correlation:.4f}")
                print(f"   • Mean difference: {mean_diff:.3f}°C")
                print(f"   • RMSE: {rmse:.3f}°C")  
                print(f"   • MAE: {mae:.3f}°C")
                
                # Kategori validasi
                if correlation >= 0.95 and rmse <= 1.0:
                    validation_status = "EXCELLENT - Safe to replace"
                    confidence = "Very High"
                elif correlation >= 0.90 and rmse <= 1.5:
                    validation_status = "GOOD - Acceptable to replace"
                    confidence = "High"
                elif correlation >= 0.80 and rmse <= 2.0:
                    validation_status = "MODERATE - Consider with caution"
                    confidence = "Medium"
                else:
                    validation_status = "POOR - Manual review needed"
                    confidence = "Low"
                
                print(f"   • Validation Status: {validation_status}")
                print(f"   • Confidence Level: {confidence}")
                
                validation_results = {
                    'correlation': correlation,
                    'rmse': rmse,
                    'mae': mae,
                    'mean_diff': mean_diff,
                    'status': validation_status,
                    'confidence': confidence,
                    'comparison_count': len(comparison_data)
                }
                
            else:
                print("   ⚠️  Tidak ada TAVG original yang valid untuk perbandingan")
        else:
            print("   ℹ️  Kolom TAVG original tidak ditemukan - menggunakan calculated")
        
        # 4. DECISION MAKING - REPLACE TAVG
        replace_decision = True
        
        if validation_results:
            if validation_results['confidence'] in ['Very High', 'High']:
                replace_decision = True
                print(f"\n✅ DECISION: Replace TAVG dengan TAVG_calculated")
            elif validation_results['confidence'] == 'Medium':
                replace_decision = True
                print(f"\n⚠️  DECISION: Replace TAVG dengan TAVG_calculated (dengan catatan)")
            else:
                replace_decision = False
                print(f"\n❌ DECISION: TIDAK replace TAVG - perlu review manual")
        else:
            print(f"\n✅ DECISION: Gunakan TAVG_calculated (no original for comparison)")
        
        # 5. EXECUTE REPLACEMENT
        if replace_decision:
            # Backup TAVG original jika ada
            if 'TAVG' in self.data.columns:
                self.data['TAVG_original_backup'] = self.data['TAVG'].copy()
                print(f"   💾 TAVG original disimpan ke kolom 'TAVG_original_backup'")
            
            # Replace dengan calculated
            self.data['TAVG'] = self.data['TAVG_calculated'].copy()
            print(f"   🔄 TAVG berhasil di-replace dengan TAVG_calculated")
            
            # Generate final statistics
            final_tavg = self.data[valid_mask]['TAVG']
            print(f"\n📈 STATISTIK FINAL TAVG:")
            print(f"   • Count: {len(final_tavg):,}")
            print(f"   • Mean: {final_tavg.mean():.2f}°C")
            print(f"   • Std: {final_tavg.std():.2f}°C") 
            print(f"   • Min: {final_tavg.min():.2f}°C")
            print(f"   • Max: {final_tavg.max():.2f}°C")
            
        else:
            print(f"   ⏭️  TAVG tidak di-replace - gunakan data original")
            # 👉 Tambahan: isi NaN dengan TAVG_calculated
            self.data['TAVG'] = self.data['TAVG'].fillna(self.data['TAVG_calculated'])
        
        
        return {
            'tavg_calculated': True,
            'replacement_done': replace_decision,
            'validation_results': validation_results,
            'final_count': len(valid_data)
        }
        

    # MODIFIKASI 2: Tambah TAVG analysis ke method yang sudah ada
    def calculate_descriptive_statistics_tavg(self):
        """
        Statistik deskriptif khusus untuk TAVG (copy dari method existing tapi untuk TAVG)
        """
        return self.calculate_descriptive_statistics('TAVG')

    def seasonal_analysis_all_temperature(self):
        """
        Analisis musiman untuk TN, TX, dan TAVG
        """
        print(f"\n=== ANALISIS MUSIMAN TEMPERATURE (TN, TX, TAVG) ===")
        
        # Filter data valid untuk semua kolom temperature
        temp_columns = ['TN', 'TX', 'TAVG']
        available_columns = [col for col in temp_columns if col in self.data.columns]
        
        if not available_columns:
            print("⚠️ Tidak ada kolom temperature yang tersedia")
            return
        
        valid_data = self.data.copy()
        for col in available_columns:
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
        
        # Hitung statistik bulanan untuk semua kolom temperature
        monthly_stats = valid_data.groupby('month')[available_columns].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun',
                    'Jul', 'Ags', 'Sep', 'Okt', 'Nov', 'Des']
        
        print(f"\n📅 STATISTIK BULANAN TEMPERATURE:")
        
        # Header tabel dinamis berdasarkan kolom yang tersedia
        if len(available_columns) == 3:  # TN, TX, TAVG semua ada
            print("Bulan | TN Mean | TX Mean | TAVG Mean | Diurnal Range")
            print("-" * 55)
            
            for month in range(1, 13):
                if month in monthly_stats.index:
                    tn_mean = monthly_stats.loc[month, ('TN', 'mean')]
                    tx_mean = monthly_stats.loc[month, ('TX', 'mean')] 
                    tavg_mean = monthly_stats.loc[month, ('TAVG', 'mean')]
                    diurnal = tx_mean - tn_mean
                    
                    print(f"{month_names[month-1]:5s} | {tn_mean:7.1f} | {tx_mean:7.1f} | "
                        f"{tavg_mean:9.1f} | {diurnal:13.1f}")
        else:
            # Fallback ke format original jika tidak semua kolom tersedia
            header = "Bulan |"
            for col in available_columns:
                header += f" {col} Mean |"
            print(header)
            print("-" * len(header))
            
            for month in range(1, 13):
                if month in monthly_stats.index:
                    row = f"{month_names[month-1]:5s} |"
                    for col in available_columns:
                        mean_val = monthly_stats.loc[month, (col, 'mean')]
                        row += f" {mean_val:7.1f} |"
                    print(row)

    def generate_summary_table(self):
        """
        Generate tabel ringkasan untuk TN, TX, dan TAVG
        """
        print(f"\n" + "="*80)
        print("TABEL RINGKASAN STATISTIK DESKRIPTIF TEMPERATURE (TN, TX, TAVG)")
        print("="*80)
        
        temp_columns = ['TN', 'TX', 'TAVG']
        available_columns = [col for col in temp_columns if col in self.data.columns]
        
        if not available_columns:
            print("❌ Tidak ada kolom temperature yang tersedia")
            return None
        
        stats_data = {}
        
        # Hitung statistik untuk setiap kolom yang tersedia
        for col in available_columns:
            analysis = self.analyze_missing_values(col, show_details=False)
            if analysis is not None and len(analysis['valid_data']) > 0:
                data = analysis['valid_data']
                stats_data[col] = {
                    'Jumlah Data': len(data),
                    'Jumlah Missing Value': analysis['missing_count'], 
                    'Minimum': data.min(),
                    'Q1': data.quantile(0.25),
                    'Median': data.median(),
                    'Mean': data.mean(),
                    'Q3': data.quantile(0.75),
                    'Maksimum': data.max(),
                    'Standar Deviasi': data.std()
                }
        
        if not stats_data:
            print("❌ Tidak ada data valid untuk generate tabel")
            return None
        
        # Print header
        header = f"{'Statistik':<20}"
        for col in available_columns:
            header += f" {col+' (°C)':<15}"
        print(header)
        print("-" * len(header))
        
        # Print rows
        stat_order = ['Jumlah Data', 'Jumlah Missing Value', 'Minimum', 'Q1', 'Median', 'Mean', 'Q3', 'Maksimum', 'Standar Deviasi']
        
        for stat in stat_order:
            row = f"{stat:<20}"
            for col in available_columns:
                if col in stats_data and stat in stats_data[col]:
                    value = stats_data[col][stat]
                    if stat in ['Jumlah Data', 'Jumlah Missing Value']:
                        row += f" {int(value):<15,}"
                    else:
                        row += f" {value:<15.2f}"
                else:
                    row += f" {'N/A':<15}"
            print(row)
        
        print("-" * len(header))
        print("🎯 Tabel statistik deskriptif lengkap selesai!")
        
        return stats_data
    
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
        monthly_stats = valid_data.groupby('month')[column_name].agg([
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
            month = self.data.loc[idx, 'month']
            
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
    
    def detect_outliers_domain_aware(self, column_name):
        """
        Deteksi outlier temperature dengan pendekatan domain-aware
        Menggabungkan validasi fisik, IQR seasonal, dan cross-variable validation
        """
        print(f"\n=== DETEKSI OUTLIER {column_name.upper()} (DOMAIN-AWARE) ===")
        
        analysis = self.analyze_missing_values(column_name, show_details=False)
        if analysis is None or len(analysis['valid_data']) == 0:
            print("⚠️ Tidak ada data valid untuk deteksi outlier")
            return None
        
        valid_data = analysis['valid_data']
        
        # Step 1: Domain Validation (Physical Bounds for Tropical Indonesia)
        print("🔍 Step 1: Validasi Domain Fisik")
        
        if column_name.upper() == 'TN':
            # Minimum temperature bounds for tropical coastal Indonesia
            physical_lower = 15.0  # Extremely rare below this
            physical_upper = 30.0  # Very unusual above this for minimum temp
            normal_range = "18-28°C"
        elif column_name.upper() == 'TX':
            # Maximum temperature bounds for tropical coastal Indonesia  
            physical_lower = 25.0  # Extremely rare below this
            physical_upper = 42.0  # Very unusual above this for maximum temp
            normal_range = "28-38°C"
        else:  # TAVG
            physical_lower = 20.0
            physical_upper = 36.0
            normal_range = "22-34°C"
        
        # Get full dataset for this column
        full_data = self.data[column_name].copy()
        
        physical_outliers = (
            (full_data < physical_lower) | 
            (full_data > physical_upper) |
            (full_data == 999) |
            (full_data == 9999) |
            (full_data == -999) |
            (full_data == -9999)
        ) & full_data.notna()
        
        physical_count = physical_outliers.sum()
        print(f"   • Physical outliers (di luar {normal_range} atau kode error): {physical_count}")
        
        if physical_count > 0:
            outlier_values = full_data[physical_outliers].unique()
            print(f"   • Nilai physical outliers: {outlier_values}")
        
        # Step 2: Seasonal IQR Detection
        print("\n🔍 Step 2: Deteksi IQR dengan Seasonal Adjustment")
        
        # Add season column if not exists
        if 'season' not in self.data.columns:
            self.data['season'] = self.data['month'].map({
                12: 'DJF', 1: 'DJF', 2: 'DJF',  # Wet season
                3: 'MAM', 4: 'MAM', 5: 'MAM',   # Transition 1
                6: 'JJA', 7: 'JJA', 8: 'JJA',   # Dry season
                9: 'SON', 10: 'SON', 11: 'SON'  # Transition 2
            })
        
        seasonal_outliers = pd.Series(False, index=self.data.index)
        seasonal_stats = {}
        
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            season_mask = (self.data['season'] == season) & (full_data.notna()) & (~physical_outliers)
            season_data = full_data[season_mask]
            
            if len(season_data) < 10:  # Skip if insufficient data
                continue
                
            q1 = season_data.quantile(0.25)
            q3 = season_data.quantile(0.75)
            iqr = q3 - q1
            
            # IQR bounds with 1.5 factor (standard for temperature)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # But respect physical bounds
            lower_bound = max(lower_bound, physical_lower + 1)
            upper_bound = min(upper_bound, physical_upper - 1)
            
            season_outliers_mask = (
                (full_data < lower_bound) |
                (full_data > upper_bound)
            ) & season_mask
            
            seasonal_outliers = seasonal_outliers | season_outliers_mask
            
            seasonal_stats[season] = {
                'count': len(season_data),
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers': season_outliers_mask.sum()
            }
            
            print(f"   • {season}: Q1={q1:.1f}°C, Q3={q3:.1f}°C, IQR={iqr:.1f}°C")
            print(f"     Bounds: {lower_bound:.1f}°C - {upper_bound:.1f}°C, Outliers: {season_outliers_mask.sum()}")
        
        # Step 3: Cross-Variable Validation (TN vs TX relationship)
        print("\n🔍 Step 3: Cross-Variable Validation")
        
        cross_variable_outliers = pd.Series(False, index=self.data.index)
        
        if column_name.upper() in ['TN', 'TX'] and 'TN' in self.data.columns and 'TX' in self.data.columns:
            # Check TN < TX relationship and diurnal range
            valid_both_mask = (
                self.data['TN'].notna() & self.data['TX'].notna() &
                (~physical_outliers) & 
                (self.data['TN'] != 9999) & (self.data['TX'] != 9999) &
                (self.data['TN'] != 8888) & (self.data['TX'] != 8888)
            )
            
            if valid_both_mask.sum() > 0:
                valid_both = self.data[valid_both_mask]
                
                # Rule 1: TN must be less than TX
                tn_greater_tx = (valid_both['TN'] >= valid_both['TX'])
                
                # Rule 2: Diurnal range should be between 3-12°C for tropical regions
                diurnal_range = valid_both['TX'] - valid_both['TN']
                abnormal_diurnal = (diurnal_range < 3) | (diurnal_range > 15)
                
                # Rule 3: Extreme temperature combinations
                if column_name.upper() == 'TN':
                    # Very high TN with normal TX (unusual combination)
                    high_tn_normal_tx = (valid_both['TN'] > 27) & (valid_both['TX'] < 31)
                    cross_var_mask = tn_greater_tx | abnormal_diurnal | high_tn_normal_tx
                else:  # TX
                    # Very low TX with normal TN (unusual combination)
                    low_tx_normal_tn = (valid_both['TX'] < 29) & (valid_both['TN'] > 24)
                    cross_var_mask = tn_greater_tx | abnormal_diurnal | low_tx_normal_tn
                
                # Map back to full dataset
                cross_variable_outliers.loc[valid_both_mask] = cross_var_mask
                
                cross_count = cross_variable_outliers.sum()
                print(f"   • Cross-variable outliers: {cross_count}")
                if cross_count > 0:
                    print(f"     - TN ≥ TX violations: {tn_greater_tx.sum()}")
                    print(f"     - Abnormal diurnal range: {abnormal_diurnal.sum()}")
        else:
            print("   • Cross-variable validation tidak tersedia (missing TN/TX data)")
        
        # Step 4: Combine all detections
        print("\n📊 Step 4: Ringkasan Deteksi Outlier")
        
        # Flag outliers in dataset
        self.data[f'is_physical_outlier_{column_name}'] = physical_outliers
        self.data[f'is_statistical_outlier_{column_name}'] = seasonal_outliers
        self.data[f'is_cross_variable_outlier_{column_name}'] = cross_variable_outliers
        
        # Combined outliers (any type)
        combined_outliers = physical_outliers | seasonal_outliers | cross_variable_outliers
        self.data[f'is_outlier_{column_name}'] = combined_outliers
        
        total_outliers = combined_outliers.sum()
        outlier_percentage = total_outliers / len(self.data) * 100
        
        print(f"   • Total outliers terdeteksi: {total_outliers} ({outlier_percentage:.2f}%)")
        print(f"     - Physical: {physical_count}")
        print(f"     - Statistical (IQR): {seasonal_outliers.sum()}")
        print(f"     - Cross-variable: {cross_variable_outliers.sum()}")
        
        # Store outlier statistics
        outlier_stats = {
            'seasonal_stats': seasonal_stats,
            'total_outliers': total_outliers,
            'physical_outliers': physical_count,
            'statistical_outliers': seasonal_outliers.sum(),
            'cross_variable_outliers': cross_variable_outliers.sum(),
            'physical_bounds': (physical_lower, physical_upper)
        }
        
        # Show sample outliers
        if total_outliers > 0:
            print(f"\n📋 Sample outlier terdeteksi (5 teratas):")
            outlier_samples = self.data[combined_outliers][
                ['Date', column_name, f'is_physical_outlier_{column_name}', 
                f'is_statistical_outlier_{column_name}', f'is_cross_variable_outlier_{column_name}']
            ].head()
            print(outlier_samples.to_string(index=False))
        
        print(f"\n✅ Deteksi outlier {column_name} selesai.")
        
        return outlier_stats

    def treat_outliers_gentle_capping(self, column_name):
        """
        Treatment outlier temperature dengan pendekatan gentle capping
        Mempertahankan pola musiman dan relationship antar variabel temperature
        """
        print(f"\n=== TREATMENT OUTLIER {column_name.upper()} (GENTLE CAPPING) ===")
        
        # Check if outlier detection has been run
        if f'is_outlier_{column_name}' not in self.data.columns:
            print("⚠️ Jalankan detect_outliers_domain_aware() terlebih dahulu")
            return False
        
        outlier_mask = self.data[f'is_outlier_{column_name}'] == True
        total_outliers = outlier_mask.sum()
        
        if total_outliers == 0:
            print("✅ Tidak ada outlier yang perlu di-treatment")
            return True
        
        # Backup original data
        self.data[f'{column_name}_original'] = self.data[column_name].copy()
        treated_count = 0
        
        print(f"🔧 Memproses {total_outliers} outlier...")
        
        # Get seasonal stats if available from detection
        seasonal_stats = {}
        if hasattr(self, 'seasonal_stats'):
            seasonal_stats = self.seasonal_stats
        else:
            # Recalculate if needed
            for season in ['DJF', 'MAM', 'JJA', 'SON']:
                season_mask = (self.data['season'] == season) & self.data[column_name].notna()
                if season_mask.sum() > 10:
                    season_data = self.data.loc[season_mask, column_name]
                    seasonal_stats[season] = {
                        'mean': season_data.mean(),
                        'std': season_data.std(),
                        'q10': season_data.quantile(0.10),
                        'q90': season_data.quantile(0.90)
                    }
        
        # Step 1: Handle Physical Outliers (highest priority)
        physical_mask = self.data[f'is_physical_outlier_{column_name}'] == True
        if physical_mask.sum() > 0:
            print(f"\n📌 Step 1: Treatment Physical Outliers ({physical_mask.sum()})")
            
            for idx in self.data[physical_mask].index:
                original_val = self.data.loc[idx, column_name]
                
                # Replace extreme values with NaN for re-imputation
                if (pd.isna(original_val) or original_val in [999, 9999, -999, -9999] or
                    original_val < 10 or original_val > 45):
                    self.data.loc[idx, column_name] = np.nan
                    self.data.loc[idx, f'treatment_method_{column_name}'] = 'physical_outlier_to_nan'
                    treated_count += 1
                    print(f"   📅 {self.data.loc[idx, 'Date'].strftime('%Y-%m-%d')}: {original_val} → NaN (physical outlier)")
        
        # Step 2: Handle Statistical Outliers with Seasonal Capping
        statistical_mask = (
            (self.data[f'is_statistical_outlier_{column_name}'] == True) & 
            (self.data[column_name].notna())
        )
        if statistical_mask.sum() > 0:
            print(f"\n📌 Step 2: Treatment Statistical Outliers ({statistical_mask.sum()})")
            
            for season, stats in seasonal_stats.items():
                season_mask = (self.data['season'] == season) & statistical_mask
                
                if season_mask.sum() == 0:
                    continue
                    
                print(f"   🌤️  Musim {season}: {season_mask.sum()} outlier")
                
                for idx in self.data[season_mask].index:
                    original_val = self.data.loc[idx, column_name]
                    
                    # Gentle capping to seasonal P10/P90 (preserve 80% of seasonal data)
                    if original_val < stats['q10']:
                        new_val = stats['q10']
                    elif original_val > stats['q90']:
                        new_val = stats['q90']
                    else:
                        continue
                    
                    self.data.loc[idx, column_name] = new_val
                    self.data.loc[idx, f'treatment_method_{column_name}'] = f'seasonal_capping_{season}'
                    treated_count += 1
                    
                    print(f"     📅 {self.data.loc[idx, 'Date'].strftime('%Y-%m-%d')}: {original_val:.1f}°C → {new_val:.1f}°C")
        
        # Step 3: Handle Cross-Variable Outliers (most gentle)
        cross_var_mask = (
            (self.data[f'is_cross_variable_outlier_{column_name}'] == True) & 
            (self.data[column_name].notna())
        )
        if cross_var_mask.sum() > 0:
            print(f"\n📌 Step 3: Review Cross-Variable Outliers ({cross_var_mask.sum()})")
            
            cross_extreme = 0
            
            for idx in self.data[cross_var_mask].index:
                original_val = self.data.loc[idx, column_name]
                
                # Only treat if it's an extreme relationship violation
                if column_name.upper() == 'TN':
                    # TN should not be higher than TX
                    if ('TX' in self.data.columns and 
                        self.data.loc[idx, 'TX'] is not np.nan and
                        original_val >= self.data.loc[idx, 'TX']):
                        # Set TN to TX - 2°C (minimum reasonable diurnal range)
                        new_val = self.data.loc[idx, 'TX'] - 2.0
                        self.data.loc[idx, column_name] = new_val
                        self.data.loc[idx, f'treatment_method_{column_name}'] = 'cross_variable_tn_correction'
                        treated_count += 1
                        cross_extreme += 1
                        print(f"     📅 {self.data.loc[idx, 'Date'].strftime('%Y-%m-%d')}: {original_val:.1f}°C → {new_val:.1f}°C (TN ≥ TX correction)")
                    else:
                        # Mark as reviewed but kept
                        self.data.loc[idx, f'treatment_method_{column_name}'] = 'cross_variable_reviewed_kept'
                
                elif column_name.upper() == 'TX':
                    # TX should not be lower than TN
                    if ('TN' in self.data.columns and 
                        self.data.loc[idx, 'TN'] is not np.nan and
                        original_val <= self.data.loc[idx, 'TN']):
                        # Set TX to TN + 3°C (minimum reasonable diurnal range)
                        new_val = self.data.loc[idx, 'TN'] + 3.0
                        self.data.loc[idx, column_name] = new_val
                        self.data.loc[idx, f'treatment_method_{column_name}'] = 'cross_variable_tx_correction'
                        treated_count += 1
                        cross_extreme += 1
                        print(f"     📅 {self.data.loc[idx, 'Date'].strftime('%Y-%m-%d')}: {original_val:.1f}°C → {new_val:.1f}°C (TX ≤ TN correction)")
                    else:
                        self.data.loc[idx, f'treatment_method_{column_name}'] = 'cross_variable_reviewed_kept'
            
            print(f"     💡 {cross_extreme} nilai dikoreksi, {cross_var_mask.sum() - cross_extreme} dipertahankan")
        
        # Step 4: Re-impute any new NaN values
        new_nan_count = self.data[column_name].isna().sum()
        if new_nan_count > 0:
            print(f"\n📌 Step 4: Re-imputasi {new_nan_count} nilai NaN hasil treatment")
            self.impute_temperature_seasonal_interpolation(column_name)
        
        # Step 5: Treatment Summary
        print(f"\n📊 RINGKASAN TREATMENT {column_name.upper()}:")
        
        treated_data = self.data[self.data.get(f'treatment_method_{column_name}', '').str.len() > 0]
        if len(treated_data) > 0:
            treatment_summary = treated_data[f'treatment_method_{column_name}'].value_counts()
            print("   📋 Methods used:")
            for method, count in treatment_summary.items():
                print(f"     • {method}: {count}")
        
        # Statistical comparison
        valid_original = self.data[f'{column_name}_original'].dropna()
        valid_treated = self.data[column_name].dropna()
        
        if len(valid_original) > 0 and len(valid_treated) > 0:
            print(f"\n📈 Perbandingan Before vs After:")
            print(f"   • Mean: {valid_original.mean():.2f}°C → {valid_treated.mean():.2f}°C")
            print(f"   • Median: {valid_original.median():.2f}°C → {valid_treated.median():.2f}°C")
            print(f"   • Std: {valid_original.std():.2f}°C → {valid_treated.std():.2f}°C")
            print(f"   • Range: {valid_original.min():.1f}-{valid_original.max():.1f}°C → {valid_treated.min():.1f}-{valid_treated.max():.1f}°C")
            
            # Check if treatment was gentle
            mean_change = abs(valid_treated.mean() - valid_original.mean())
            if mean_change < 0.5:
                print("   ✅ Treatment gentle: perubahan mean < 0.5°C")
            elif mean_change < 1.0:
                print(f"   ⚠️ Treatment moderate: perubahan mean {mean_change:.2f}°C")
            else:
                print(f"   🚨 Treatment signifikan: perubahan mean {mean_change:.2f}°C")
        
        print(f"\n✅ Treatment {column_name} selesai: {treated_count} nilai dimodifikasi")
        print("💡 Data siap untuk analisis dengan outlier yang sudah ditangani secara gentle")
        
        return True

    def process_all_temperature_outliers(self):
        """
        Process outlier detection and treatment for all temperature columns (TN, TX, TAVG)
        """
        print("\n" + "🌡️"*25 + " OUTLIER PROCESSING FOR ALL TEMPERATURE " + "🌡️"*25)
        
        temp_columns = ['TN', 'TX', 'TAVG']
        available_columns = [col for col in temp_columns if col in self.data.columns and self.data[col].notna().sum() > 0]
        
        if not available_columns:
            print("⚠️ Tidak ada kolom temperature yang valid untuk diproses")
            return False
        
        print(f"🎯 Processing outliers untuk: {available_columns}")
        
        # Process each temperature column
        all_results = {}
        
        for col in available_columns:
            print(f"\n{'='*20} PROCESSING {col.upper()} {'='*20}")
            
            # Step 1: Detect outliers
            outlier_stats = self.detect_outliers_domain_aware(col)
            
            if outlier_stats:
                # Step 2: Treat outliers
                treatment_success = self.treat_outliers_gentle_capping(col)
                
                all_results[col] = {
                    'detection_stats': outlier_stats,
                    'treatment_success': treatment_success
                }
            else:
                print(f"❌ Gagal memproses outlier untuk {col}")
                all_results[col] = {
                    'detection_stats': None,
                    'treatment_success': False
                }
        
        # Summary for all columns
        print(f"\n{'='*20} SUMMARY SEMUA KOLOM TEMPERATURE {'='*20}")
        
        total_outliers_detected = 0
        total_outliers_treated = 0
        
        for col, results in all_results.items():
            if results['detection_stats']:
                detected = results['detection_stats']['total_outliers']
                total_outliers_detected += detected
                
                if results['treatment_success']:
                    # Count treated outliers
                    treatment_col = f'treatment_method_{col}'
                    if treatment_col in self.data.columns:
                        treated = self.data[treatment_col].notna().sum()
                        total_outliers_treated += treated
                        print(f"✅ {col}: {detected} outliers detected, {treated} treated")
                    else:
                        print(f"⚠️ {col}: {detected} outliers detected, treatment status unknown")
                else:
                    print(f"❌ {col}: {detected} outliers detected, treatment failed")
            else:
                print(f"❌ {col}: detection failed")
        
        print(f"\n🎯 TOTAL SUMMARY:")
        print(f"   • Total outliers detected: {total_outliers_detected}")
        print(f"   • Total outliers treated: {total_outliers_treated}")
        print(f"   • Success rate: {(total_outliers_treated/total_outliers_detected*100) if total_outliers_detected > 0 else 0:.1f}%")
        
        return all_results
    
    def create_individual_plots(self, output_dir="temperature_plots", save_plots=True):
        """
        Membuat 3 plot individual untuk analisis TAVG preprocessing (SIMPLIFIED)
        """
        print("\n=== MEMBUAT VISUALISASI TAVG PREPROCESSING ===")
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Plots akan disimpan di: {output_dir}")
        
        # Check if TAVG exists
        if 'TAVG' not in self.data.columns:
            print("❌ Kolom TAVG tidak ditemukan! Jalankan calculate_and_validate_tavg() terlebih dahulu")
            return False
        
        # Pastikan data outlier sudah dideteksi untuk TAVG
        if 'is_outlier_TAVG' not in self.data.columns:
            print("⚠️ Menjalankan deteksi outlier TAVG terlebih dahulu...")
            self.detect_outliers_domain_aware('TAVG')
        
        valid_data = self.data['TAVG'].dropna()
        if len(valid_data) == 0:
            print("❌ Tidak ada data TAVG valid untuk plotting")
            return False
        
        # Set style dan color palette untuk temperature
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Updated color scheme - Blue for Before, Green for After
        colors = {
            'temp_before': '#2196F3',       # Blue untuk before treatment
            'temp_after': '#4CAF50',        # Green untuk after treatment  
            'outlier_stat': '#FF1744',      # Bright red untuk statistical outliers
            'outlier_cross': '#FF9800',     # Orange untuk cross-variable outliers
            'outlier_physical': '#9C27B0',  # Purple untuk physical outliers
            'season_djf': '#2196F3',        # Blue - Cool/wet season
            'season_mam': '#4CAF50',        # Green - Warming transition
            'season_jja': '#FF9800',        # Orange - Hot/dry season
            'season_son': '#9C27B0'         # Purple - Cooling transition
        }
        
        # ============================================================================
        # PLOT 1: TIME SERIES BEFORE-AFTER TREATMENT (SIMPLIFIED)
        # ============================================================================
        print("\n📊 Plot 1: Time Series Before-After Treatment")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Prepare data
        dates = self.data['Date']
        tavg_original = self.data.get('TAVG_original', self.data['TAVG'])
        tavg_treated = self.data['TAVG']
        
        # ========================
        # Subplot 1: Before (Original with outliers marked) - BLUE
        # ========================
        ax1.plot(dates, tavg_original, color=colors['temp_before'], alpha=0.7, linewidth=0.8, label='Data Original')
        
        # Mark outliers if available
        outlier_columns = ['is_physical_outlier_TAVG', 'is_statistical_outlier_TAVG', 'is_cross_variable_outlier_TAVG']
        outlier_found = False
        
        # Physical outliers (purple squares)
        if 'is_physical_outlier_TAVG' in self.data.columns:
            physical_mask = self.data['is_physical_outlier_TAVG'] == True
            if physical_mask.sum() > 0:
                ax1.scatter(dates[physical_mask], tavg_original[physical_mask], 
                        color=colors['outlier_physical'], s=25, alpha=0.8, 
                        label=f'Physical Outliers ({physical_mask.sum()})', marker='s')
                outlier_found = True
        
        # Statistical outliers (red circles)
        if 'is_statistical_outlier_TAVG' in self.data.columns:
            stat_mask = self.data['is_statistical_outlier_TAVG'] == True
            if stat_mask.sum() > 0:
                ax1.scatter(dates[stat_mask], tavg_original[stat_mask], 
                        color=colors['outlier_stat'], s=20, alpha=0.8, 
                        label=f'Statistical Outliers ({stat_mask.sum()})', marker='o')
                outlier_found = True
        
        # Cross-variable outliers (orange triangles)
        if 'is_cross_variable_outlier_TAVG' in self.data.columns:
            cross_mask = self.data['is_cross_variable_outlier_TAVG'] == True
            if cross_mask.sum() > 0:
                ax1.scatter(dates[cross_mask], tavg_original[cross_mask], 
                        color=colors['outlier_cross'], s=15, alpha=0.6,
                        label=f'Cross-Variable Outliers ({cross_mask.sum()})', marker='^')
                outlier_found = True
        
        if not outlier_found:
            print("   📝 No outliers detected or outlier columns not available")
        
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.set_title('Before Treatment', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
        ax1.legend(loc='upper right', fontsize=10)
        # ax1.set_yliax1.legem(20, 36)  # Appropriate for tropical temperature
        ax1.set_ylim(20, 36)
        ax1.set_facecolor('white')
        
        # ========================
        # Subplot 2: After (Treated data) - GREEN
        # ========================
        ax2.plot(dates, tavg_treated, color=colors['temp_after'], alpha=0.8, linewidth=0.8, label='Data After Treatment')
        
        # Highlight treated values (darker green)
        if 'treatment_method_TAVG' in self.data.columns:
            treated_mask = (
                self.data['treatment_method_TAVG'].notna() & 
                (self.data['treatment_method_TAVG'] != 'cross_variable_reviewed_kept') &
                (self.data['treatment_method_TAVG'] != '')
            )
            if treated_mask.sum() > 0:
                ax2.scatter(dates[treated_mask], tavg_treated[treated_mask], 
                        color='#2E7D32', s=25, alpha=0.9,  # Darker green for treated values
                        label=f'Treated Values ({treated_mask.sum()})', marker='s')
        
        ax2.set_xlabel('Tahun', fontsize=12)
        ax2.set_ylabel('Temperature (°C)', fontsize=12)
        ax2.set_title('After Treatment', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_ylim(20, 36)
        ax2.set_facecolor('white')
        
        # Set x-axis ticks
        start_date = dates.min()
        end_date = dates.max()
        date_ticks = pd.date_range(start=start_date, end=end_date, freq='YS')
        ax2.set_xticks(date_ticks)
        ax2.set_xticklabels([d.year for d in date_ticks], rotation=0, ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if save_plots:
            plot1_path = os.path.join(output_dir, "preprocessing_tavg_plot_01_timeseries_treatment.png")
            plt.savefig(plot1_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✅ Plot 1 saved: {plot1_path}")
        
        plt.show()
        
        # ============================================================================
        # PLOT 2: SEASONAL BOXPLOT (SIMPLIFIED)
        # ============================================================================
        print("\n📊 Plot 2: Seasonal Boxplot Analysis")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare monthly data (climatological year: Dec-Jan-Feb-...-Nov)
        month_order = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        month_labels = ['Des', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei',
                        'Jun', 'Jul', 'Ags', 'Sep', 'Okt', 'Nov']
        
        # Season color mapping for boxes (climatological seasons)
        season_month_map = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:0}  # Dec=0(DJF), etc.
        season_colors = [colors['season_djf'], colors['season_mam'], colors['season_jja'], colors['season_son']]
        
        # ========================
        # Subplot 1: Original Monthly Boxplot
        # ========================
        monthly_data_orig = []
        for month in month_order:
            month_mask = (self.data['month'] == month) & (tavg_original.notna())
            if month_mask.sum() > 0:
                monthly_data_orig.append(tavg_original[month_mask])
            else:
                monthly_data_orig.append(pd.Series(dtype=float))
        
        bp1 = ax1.boxplot(monthly_data_orig, labels=month_labels, patch_artist=True, showfliers=True)
        
        # Color by season
        for i, patch in enumerate(bp1['boxes']):
            if i < len(season_month_map):
                patch.set_facecolor(season_colors[season_month_map[i]])
                patch.set_alpha(0.7)
        
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.set_xlabel('Bulan', fontsize=12)
        ax1.set_title('Before Treatment', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.4)
        ax1.set_ylim(20, 36)
        ax1.set_facecolor('white')
        
        # ========================
        # Subplot 2: Treated Monthly Boxplot
        # ========================
        monthly_data_treated = []
        for month in month_order:
            month_mask = (self.data['month'] == month) & (tavg_treated.notna())
            if month_mask.sum() > 0:
                monthly_data_treated.append(tavg_treated[month_mask])
            else:
                monthly_data_treated.append(pd.Series(dtype=float))
        
        bp2 = ax2.boxplot(monthly_data_treated, labels=month_labels, patch_artist=True, showfliers=True)
        
        # Color by season
        for i, patch in enumerate(bp2['boxes']):
            if i < len(season_month_map):
                patch.set_facecolor(season_colors[season_month_map[i]])
                patch.set_alpha(0.7)
        
        ax2.set_ylabel('Temperature (°C)', fontsize=12)
        ax2.set_xlabel('Bulan', fontsize=12)
        ax2.set_title('After Treatment', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.4)
        ax2.set_ylim(20, 36)
        ax2.set_facecolor('white')
        
        plt.tight_layout()
        
        if save_plots:
            plot2_path = os.path.join(output_dir, "preprocessing_tavg_plot_02_seasonal_patterns.png")
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✅ Plot 2 saved: {plot2_path}")
        
        plt.show()
        
        # ============================================================================
        # PLOT 3: DISTRIBUTION COMPARISON (SIMPLIFIED)
        # ============================================================================
        print("\n📊 Plot 3: Distribution Analysis")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # ========================
        # Subplot 1: Original Distribution - BLUE
        # ========================
        ax1.hist(tavg_original.dropna(), bins=30, density=True, alpha=0.7, 
                color=colors['temp_before'], edgecolor='black', linewidth=0.5)

        # Add KDE
        from scipy.stats import gaussian_kde
        orig_kde = gaussian_kde(tavg_original.dropna())
        x_range = np.linspace(20, 36, 100)
        ax1.plot(x_range, orig_kde(x_range), color='#1565C0', linewidth=2.5)

        # Add mean and median lines
        ax1.axvline(tavg_original.mean(), color='red', linestyle='--', linewidth=2)
        ax1.axvline(tavg_original.median(), color='darkred', linestyle='--', linewidth=2)

        ax1.set_xlabel('Temperature (°C)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Before Treatment', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
        ax1.set_xlim(20, 36)
        ax1.set_ylim(0, ax1.get_ylim()[1])  # Auto adjust y-axis
        ax1.set_facecolor('white')

        # ========================
        # Subplot 2: Treated Distribution - GREEN
        # ========================
        ax2.hist(tavg_treated.dropna(), bins=30, density=True, alpha=0.7,
                color=colors['temp_after'], edgecolor='black', linewidth=0.5)

        # Add KDE for treated
        treated_kde = gaussian_kde(tavg_treated.dropna())
        ax2.plot(x_range, treated_kde(x_range), color='#2E7D32', linewidth=2.5)

        # Add mean and median lines
        ax2.axvline(tavg_treated.mean(), color='red', linestyle='--', linewidth=2)
        ax2.axvline(tavg_treated.median(), color='darkred', linestyle='--', linewidth=2)

        ax2.set_xlabel('Temperature (°C)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('After Treatment', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
        ax2.set_xlim(20, 36)
        ax2.set_ylim(0, ax2.get_ylim()[1])  # Auto adjust y-axis
        ax2.set_facecolor('white')

        plt.tight_layout()

        if save_plots:
            plot3_path = os.path.join(output_dir, "preprocessing_tavg_plot_03_distribution_outliers.png")
            plt.savefig(plot3_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✅ Plot 3 saved: {plot3_path}")

        plt.show()
        
        print("\n🎉 Semua plot TAVG berhasil dibuat!")
        print(f"📈 Plot 1: Time Series Before-After Treatment (Blue → Green)")
        print(f"📊 Plot 2: Seasonal Boxplot Analysis (Simplified)") 
        print(f"📋 Plot 3: Distribution Comparison (Blue → Green)")
        
        if save_plots:
            print(f"\n📁 Semua plot disimpan di: {output_dir}")
            
        return True

    def save_results(self, output_filename="preprocessing_log_temp.txt"):
        """
        Simpan hasil analisis ke file di output directory
        """
        output_path = os.path.join(self.output_dir, output_filename)
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
            self.seasonal_analysis_all_temperature()
            self.generate_summary_table()
        
        # Kembalikan stdout
        sys.stdout = original_stdout
        print(f"✅ Hasil berhasil disimpan ke: {output_path}")

def main():
    # Updated paths
    data_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data BMKG/Lokasi/Kab. Aceh Besar/Stasiun Klimatologi Aceh/CSV/BMKG_Data_All.csv"
    output_dir = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data BMKG/Lokasi/Kab. Aceh Besar/Stasiun Klimatologi Aceh/CSV CLEANED/suhu"
    
    # Inisialisasi analyzer dengan output directory
    analyzer = Temperature_Analyzer(data_path, output_dir)    
    try:
        # Load data
        if not analyzer.load_data():
            return

        # 🔥 TAHAP 1: Imputasi missing values TN dan TX
        print("\n" + "🔥"*20 + " TAHAP 1: IMPUTASI TN & TX " + "🔥"*20)
        analyzer.impute_temperature_seasonal_interpolation('TN')
        analyzer.impute_temperature_seasonal_interpolation('TX')

        # 🌡️ TAHAP 2: Kalkulasi dan validasi TAVG
        print("\n" + "🌡️"*20 + " TAHAP 2: KALKULASI TAVG " + "🌡️"*20)
        tavg_results = analyzer.calculate_and_validate_tavg()

        # 🚨 TAHAP 2.5: OUTLIER DETECTION & TREATMENT
        print("\n" + "🚨"*20 + " TAHAP 2.5: OUTLIER TREATMENT " + "🚨"*20)
        outlier_results = analyzer.process_all_temperature_outliers()

        # 📊 TAHAP 3: Analisis statistik deskriptif lengkap
        print("\n" + "📊"*20 + " TAHAP 3: ANALISIS STATISTIK " + "📊"*20)
        
        analyzer.calculate_descriptive_statistics('TN')
        analyzer.calculate_descriptive_statistics('TX') 
        
        # Tambahan: Analisis TAVG jika berhasil dikalkulasi
        if tavg_results and tavg_results.get('tavg_calculated'):
            analyzer.calculate_descriptive_statistics('TAVG')
        
        analyzer.compare_tn_tx()
        
        # Update seasonal analysis untuk include TAVG
        analyzer.seasonal_analysis_all_temperature()

        # Generate tabel ringkasan yang include TAVG
        analyzer.generate_summary_table()

        # 📈 TAHAP 3.5: STATISTIK TAVG SETELAH IMPUTASI (PERBAIKAN DI SINI)
        print("\n" + "📈"*20 + " TAHAP 3.5: STATISTIK TAVG FINAL " + "📈"*20)
        
        # ✅ PERBAIKAN: Tambahkan validasi dan error handling yang lebih jelas
        tavg_stats_final = None
        
        # Cek apakah TAVG tersedia di dataframe
        if 'TAVG' not in analyzer.data.columns:
            print("❌ Kolom TAVG tidak ditemukan di dataset!")
            print("   💡 Pastikan calculate_and_validate_tavg() telah dijalankan dengan sukses")
        
        # Cek apakah ada data TAVG yang valid
        elif analyzer.data['TAVG'].notna().sum() == 0:
            print("❌ Tidak ada data TAVG yang valid!")
            print("   💡 Semua nilai TAVG adalah NaN atau missing")
        
        # Jika semua kondisi OK, jalankan method
        else:
            print("🔍 Memulai perhitungan statistik TAVG final...")
            try:
                # ✅ INI YANG DIPERBAIKI: Pastikan method benar-benar dipanggil
                tavg_stats_final = analyzer.calculate_tavg_statistics_after_imputation()
                
                if tavg_stats_final is not None:
                    print("\n✅ Statistik TAVG final berhasil dihitung!")
                    print(f"   📊 Total records: {tavg_stats_final.get('quality_metrics', {}).get('total_records', 'N/A')}")
                    print(f"   📊 Valid records: {tavg_stats_final.get('quality_metrics', {}).get('valid_records', 'N/A')}")
                    print(f"   📊 Data completeness: {tavg_stats_final.get('quality_metrics', {}).get('completeness', 0):.1f}%")
                else:
                    print("⚠️ Method berhasil dijalankan tapi return None")
                    print("   💡 Cek implementasi calculate_tavg_statistics_after_imputation()")
            
            except Exception as e:
                print(f"❌ Error saat menghitung statistik TAVG: {str(e)}")
                print(f"   💡 Detail error: {type(e).__name__}")
                import traceback
                traceback.print_exc()
        
        # 🎨 TAHAP 4: VISUALISASI PREPROCESSING
        print("\n" + "🎨"*20 + " TAHAP 4: VISUALISASI PREPROCESSING " + "🎨"*20)
        
        if tavg_results and tavg_results.get('tavg_calculated'):
            plot_success = analyzer.create_individual_plots(
                output_dir=os.path.join(output_dir, "plots"), 
                save_plots=True
            )
            
            if plot_success:
                print("✅ Visualisasi TAVG preprocessing berhasil dibuat")
            else:
                print("❌ Gagal membuat visualisasi TAVG preprocessing")
        else:
            print("⚠️ TAVG tidak tersedia - skip visualisasi")
        
        # ✅ Simpan data final dengan TAVG
        output_cols = ['Date', 'Year', 'month', 'day', 'TN', 'TX']
        if 'TAVG' in analyzer.data.columns:
            output_cols.append('TAVG')
        
        if all(col in analyzer.data.columns for col in output_cols):
            output_csv_path = os.path.join(output_dir, "preprocessed_temperature_final.csv")
            analyzer.data[output_cols].to_csv(output_csv_path, index=False)
            print(f"💾 Data temperature final (TN, TX, TAVG) disimpan ke: {output_csv_path}")
            
        print(f"\n🎉 ANALISIS TEMPERATURE + TAVG + OUTLIER TREATMENT SELESAI!")
        
        # ✅ TAMBAHAN: Print summary TAVG stats jika tersedia
        if tavg_stats_final is not None:
            print("\n📋 RINGKASAN STATISTIK TAVG FINAL:")
            if 'statistics' in tavg_stats_final:
                stats = tavg_stats_final['statistics']
                print(f"   • Mean: {stats.get('Mean', 'N/A'):.2f}°C")
                print(f"   • Median: {stats.get('Median', 'N/A'):.2f}°C")
                print(f"   • Std Dev: {stats.get('Standar Deviasi', 'N/A'):.2f}°C")
                print(f"   • Range: {stats.get('Minimum', 'N/A'):.1f} - {stats.get('Maksimum', 'N/A'):.1f}°C")
        
        analyzer.save_results()
        return analyzer
        
    except Exception as e:
        print(f"❌ Error dalam analisis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()