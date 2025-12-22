import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class RH_AVG_Analyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.rh_stats = {}

    def load_data(self):
        print("\U0001f321️  RH_AVG STATISTICS ANALYZER - DATASET BMKG")
        print("="*50)

        try:
            print(f"\U0001f4c2 Loading data dari: {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values('Date').reset_index(drop=True)

            print(f"✅ Data berhasil dimuat: {len(self.data)} records")
            print(f"\U0001f4c5 Periode: {self.data['Date'].min()} s/d {self.data['Date'].max()}")
            print(f"\U0001f4cb Kolom tersedia: {list(self.data.columns)}")

            if 'RH_AVG' not in self.data.columns:
                raise ValueError("❌ Kolom RH_AVG tidak ditemukan!")

            return True

        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
        
    def impute_missing_rh_avg(self):
        """
        Implementasi Linear Interpolation dengan Seasonal Awareness
        """
        print("\n=== IMPUTASI MISSING VALUES RH_AVG ===")
        
        if 'RH_AVG' not in self.data.columns:
            print("❌ Kolom RH_AVG tidak ditemukan!")
            return
        
        # Identifikasi missing values
        missing_mask = (
            self.data['RH_AVG'].isna() |
            (self.data['RH_AVG'] == 9999) |
            (self.data['RH_AVG'] == 8888) |
            (self.data['RH_AVG'] == -999) |
            (self.data['RH_AVG'] == -9999)
        )
        
        missing_indices = self.data[missing_mask].index.tolist()
        
        if not missing_indices:
            print("✅ Tidak ada missing values yang perlu diimputasi")
            return
        
        print(f"🔍 Ditemukan {len(missing_indices)} missing values untuk diimputasi")
        
        # Hitung monthly averages untuk seasonal awareness
        valid_data = self.data[~missing_mask].copy()
        monthly_avg = valid_data.groupby('month')['RH_AVG'].mean()
        
        imputed_values = []
        
        for idx in missing_indices:
            date = self.data.loc[idx, 'Date']
            month = self.data.loc[idx, 'month']
            
            print(f"\n📅 Imputasi untuk tanggal: {date.strftime('%Y-%m-%d')}")
            
            # Strategy 1: Linear Interpolation dengan tetangga terdekat
            prev_idx = idx - 1
            next_idx = idx + 1
            
            # Cari tetangga valid sebelumnya
            while prev_idx >= 0 and (self.data.loc[prev_idx, 'RH_AVG'] in [np.nan, 9999, 8888, -999, -9999] or 
                                    pd.isna(self.data.loc[prev_idx, 'RH_AVG'])):
                prev_idx -= 1
                
            # Cari tetangga valid setelahnya  
            while next_idx < len(self.data) and (self.data.loc[next_idx, 'RH_AVG'] in [np.nan, 9999, 8888, -999, -9999] or 
                                            pd.isna(self.data.loc[next_idx, 'RH_AVG'])):
                next_idx += 1
            
            imputed_value = None
            method_used = ""
            
            # Jika kedua tetangga tersedia - Linear Interpolation
            if prev_idx >= 0 and next_idx < len(self.data):
                prev_val = self.data.loc[prev_idx, 'RH_AVG']
                next_val = self.data.loc[next_idx, 'RH_AVG']
                prev_date = self.data.loc[prev_idx, 'Date']
                next_date = self.data.loc[next_idx, 'Date']
                
                # Linear interpolation berdasarkan jarak waktu
                total_days = (next_date - prev_date).days
                target_days = (date - prev_date).days
                
                if total_days > 0:
                    weight = target_days / total_days
                    imputed_value = prev_val + (next_val - prev_val) * weight
                    method_used = f"Linear Interpolation ({prev_date.strftime('%m-%d')} to {next_date.strftime('%m-%d')})"
            
            # Jika hanya tetangga sebelumnya tersedia
            elif prev_idx >= 0:
                prev_val = self.data.loc[prev_idx, 'RH_AVG']
                seasonal_adj = monthly_avg[month]
                imputed_value = (prev_val + seasonal_adj) / 2
                method_used = f"Forward Fill + Seasonal ({prev_idx})"
                
            # Jika hanya tetangga setelahnya tersedia
            elif next_idx < len(self.data):
                next_val = self.data.loc[next_idx, 'RH_AVG']
                seasonal_adj = monthly_avg[month]
                imputed_value = (next_val + seasonal_adj) / 2
                method_used = f"Backward Fill + Seasonal ({next_idx})"
                
            # Fallback: Monthly average dengan seasonal pattern
            else:
                imputed_value = monthly_avg[month]
                method_used = "Monthly Average (Fallback)"
            
            # Validasi range (44-100% berdasarkan data historis)
            if imputed_value is not None:
                imputed_value = max(44.0, min(100.0, imputed_value))
                
                # Update data
                original_val = self.data.loc[idx, 'RH_AVG']
                self.data.loc[idx, 'RH_AVG'] = imputed_value
                
                imputed_values.append({
                    'date': date,
                    'original': original_val,
                    'imputed': imputed_value,
                    'method': method_used
                })
                
                print(f"   ✅ {original_val} → {imputed_value:.1f}% | Method: {method_used}")
            else:
                print(f"   ❌ Gagal imputasi untuk {date}")
        
        # Summary imputasi
        print(f"\n📊 RINGKASAN IMPUTASI:")
        print(f"   • Total diimputasi: {len(imputed_values)} values")
        print(f"   • Range imputasi: {min([v['imputed'] for v in imputed_values]):.1f}% - {max([v['imputed'] for v in imputed_values]):.1f}%")
        print(f"   • Rata-rata imputasi: {np.mean([v['imputed'] for v in imputed_values]):.1f}%")
        
        print("🚀 Imputasi selesai! Data siap untuk analisis lanjutan.")
        
        return imputed_values

  
    def analyze_missing_values(self, show_details=True):
        """
        Analisis missing values RH_AVG - Updated untuk post-imputasi
        """
        if show_details:
            print("\n=== ANALISIS MISSING VALUES RH_AVG (POST-IMPUTASI) ===")

        total_records = len(self.data)
        missing_nan = self.data['RH_AVG'].isna().sum()

        special_values = {}
        for val in [9999, 8888, -999, -9999]:
            count = (self.data['RH_AVG'] == val).sum()
            if count > 0:
                special_values[val] = count

        if show_details:
            print(f"📊 Total records: {total_records:,}")
            print(f"📊 Missing/NaN values: {missing_nan:,} ({missing_nan/total_records*100:.2f}%)")

            if special_values:
                print(f"📊 Nilai khusus ditemukan:")
                for val, count in special_values.items():
                    print(f"   • Nilai {val}: {count:,} ({count/total_records*100:.2f}%)")
            else:
                print("✅ Tidak ada nilai khusus yang terdeteksi")

        # Generate valid data setelah imputasi
        valid_data = self.data['RH_AVG'].dropna()
        for val in special_values.keys():
            valid_data = valid_data[valid_data != val]

        valid_count = len(valid_data)
        
        if show_details:
            print(f"📊 Valid data: {valid_count:,} ({valid_count/total_records*100:.2f}%)")
            
            if missing_nan == 0 and not special_values:
                print("🎉 Data cleaning berhasil! Semua missing values telah diimputasi.")

        return valid_data

   
    def descriptive_statistics(self):
        """
        Statistik deskriptif RH_AVG - Updated untuk data yang sudah diimputasi
        """
        print("\n=== STATISTIK DESKRIPTIF RH_AVG (POST-IMPUTASI) ===")

        # Analisis missing values dengan detail minimal karena sudah diimputasi
        rh_valid = self.analyze_missing_values(show_details=False)

        if len(rh_valid) == 0:
            print("⚠️ Tidak ada data RH_AVG yang valid untuk dianalisis")
            return None

        # Tampilkan info bahwa data sudah diimputasi
        total_records = len(self.data)
        imputed_count = total_records - len(rh_valid) if len(rh_valid) < total_records else 0
        
        if imputed_count == 0:
            print("✅ Menggunakan data yang telah diimputasi untuk analisis statistik")
        
        print(f"\n📈 STATISTIK DASAR:")
        print(f"   • Count: {len(rh_valid):,}")
        print(f"   • Mean: {rh_valid.mean():.2f}")
        print(f"   • Median: {rh_valid.median():.2f}")

        mode_series = rh_valid.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else "N/A"
        print(f"   • Mode: {mode_val}")

        print(f"   • Std Dev: {rh_valid.std():.2f}")
        print(f"   • Variance: {rh_valid.var():.2f}")
        print(f"   • Range: {rh_valid.max() - rh_valid.min():.2f}")

        print(f"\n📊 KUARTIL & PERCENTILES:")
        print(f"   • Min: {rh_valid.min():.2f}")
        print(f"   • Q1 (25%): {rh_valid.quantile(0.25):.2f}")
        print(f"   • Q2/Median (50%): {rh_valid.quantile(0.50):.2f}")
        print(f"   • Q3 (75%): {rh_valid.quantile(0.75):.2f}")
        print(f"   • Max: {rh_valid.max():.2f}")
        print(f"   • IQR: {rh_valid.quantile(0.75) - rh_valid.quantile(0.25):.2f}")

        print(f"\n🌡️  KATEGORI KELEMBABAN:")
        categories = {
            'Terlalu Kering (<45)': rh_valid < 45,
            'Ideal (45-65)': (rh_valid >= 45) & (rh_valid <= 65),
            'Terlalu Lembab (>65)': rh_valid > 65
        }

        for category, mask in categories.items():
            count = mask.sum()
            percentage = count / len(rh_valid) * 100
            print(f"   • {category}: {count:,} ({percentage:.1f}%)")

        # Update self.rh_stats untuk summary report
        self.rh_stats = {
            'count': len(rh_valid),
            'mean': rh_valid.mean(),
            'median': rh_valid.median(),
            'std': rh_valid.std(),
            'min': rh_valid.min(),
            'max': rh_valid.max(),
            'q1': rh_valid.quantile(0.25),
            'q3': rh_valid.quantile(0.75),
            'imputed': imputed_count
        }

        return rh_valid
    
    def detect_outliers_domain_aware(self):
        """
        Deteksi outlier RH_AVG dengan pendekatan domain-aware
        Menggabungkan validasi fisik, IQR seasonal, dan konteks meteorologi
        """
        print("\n=== DETEKSI OUTLIER RH_AVG (DOMAIN-AWARE) ===")
        
        valid_data = self.data['RH_AVG'].dropna()
        
        if valid_data.empty:
            print("⚠️ Tidak ada data valid untuk deteksi outlier")
            return
        
        # Step 1: Domain Validation (Physical Bounds)
        print("🔍 Step 1: Validasi Domain Fisik")
        physical_outliers = (
            (self.data['RH_AVG'] < 0) | 
            (self.data['RH_AVG'] > 100) |
            (self.data['RH_AVG'] == 999) |
            (self.data['RH_AVG'] == 9999) |
            (self.data['RH_AVG'] == -999)
        )
        
        physical_count = physical_outliers.sum()
        print(f"   • Outlier fisik (di luar 0-100% atau kode error): {physical_count}")
        
        if physical_count > 0:
            outlier_values = self.data.loc[physical_outliers, 'RH_AVG'].unique()
            print(f"   • Nilai outlier fisik: {outlier_values}")
        
        # Step 2: IQR-based Detection dengan Seasonal Adjustment
        print("\n🔍 Step 2: Deteksi IQR dengan Seasonal Adjustment")
        
        # Hitung IQR per musim (DJF, MAM, JJA, SON)
        self.data['season'] = self.data['month'].map({
            12: 'DJF', 1: 'DJF', 2: 'DJF',  # Musim Hujan
            3: 'MAM', 4: 'MAM', 5: 'MAM',   # Peralihan 1
            6: 'JJA', 7: 'JJA', 8: 'JJA',   # Musim Kering
            9: 'SON', 10: 'SON', 11: 'SON'  # Peralihan 2
        })
        
        seasonal_outliers = pd.Series(False, index=self.data.index)
        seasonal_stats = {}
        
        for season in ['DJF', 'MAM', 'JJA', 'SON']:
            season_mask = (self.data['season'] == season) & (self.data['RH_AVG'].notna())
            season_data = self.data.loc[season_mask, 'RH_AVG']
            
            if len(season_data) < 10:  # Skip jika data terlalu sedikit
                continue
                
            q1 = season_data.quantile(0.25)
            q3 = season_data.quantile(0.75)
            iqr = q3 - q1
            
            # IQR bounds dengan faktor 1.5 (moderate) untuk humidity
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Tapi tetap respect physical bounds
            lower_bound = max(lower_bound, 10)  # Min reasonable humidity
            upper_bound = min(upper_bound, 95)  # Max reasonable humidity
            
            season_outliers = (
                (self.data['RH_AVG'] < lower_bound) |
                (self.data['RH_AVG'] > upper_bound)
            ) & season_mask
            
            seasonal_outliers = seasonal_outliers | season_outliers
            
            seasonal_stats[season] = {
                'count': len(season_data),
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers': season_outliers.sum()
            }
            
            print(f"   • {season}: Q1={q1:.1f}%, Q3={q3:.1f}%, IQR={iqr:.1f}%")
            print(f"     Bounds: {lower_bound:.1f}% - {upper_bound:.1f}%, Outliers: {season_outliers.sum()}")
        
        # Step 3: Contextual Review dengan variabel meteorologi lain
        print("\n🔍 Step 3: Review Kontekstual")
        
        # Cek korelasi dengan suhu jika tersedia
        contextual_outliers = pd.Series(False, index=self.data.index)
        
        if 'TN' in self.data.columns and 'TX' in self.data.columns:
            # Rule: RH tinggi biasanya dengan suhu rendah, RH rendah dengan suhu tinggi
            high_rh_high_temp = (
                (self.data['RH_AVG'] > 85) & 
                ((self.data['TN'] > 26) | (self.data['TX'] > 32))
            )
            
            low_rh_low_temp = (
                (self.data['RH_AVG'] < 50) & 
                ((self.data['TN'] < 20) | (self.data['TX'] < 28))
            )
            
            contextual_outliers = high_rh_high_temp | low_rh_low_temp
            contextual_count = contextual_outliers.sum()
            
            print(f"   • Outlier kontekstual (RH vs Suhu): {contextual_count}")
            if contextual_count > 0:
                print(f"     - RH tinggi + Suhu tinggi: {high_rh_high_temp.sum()}")
                print(f"     - RH rendah + Suhu rendah: {low_rh_low_temp.sum()}")
        else:
            print("   • Data suhu tidak tersedia untuk validasi kontekstual")
        
        # Step 4: Kombinasi semua deteksi
        print("\n📊 Step 4: Ringkasan Deteksi Outlier")
        
        # Flag outliers di dataset
        self.data['is_physical_outlier'] = physical_outliers
        self.data['is_statistical_outlier'] = seasonal_outliers
        self.data['is_contextual_outlier'] = contextual_outliers
        
        # Kombinasi outlier (any type)
        combined_outliers = physical_outliers | seasonal_outliers | contextual_outliers
        self.data['is_outlier'] = combined_outliers
        
        total_outliers = combined_outliers.sum()
        outlier_percentage = total_outliers / len(self.data) * 100
        
        print(f"   • Total outlier terdeteksi: {total_outliers} ({outlier_percentage:.2f}%)")
        print(f"     - Fisik: {physical_count}")
        print(f"     - Statistik (IQR): {seasonal_outliers.sum()}")
        print(f"     - Kontekstual: {contextual_outliers.sum()}")
        
        # Simpan statistik untuk treatment
        self.outlier_stats = {
            'seasonal_stats': seasonal_stats,
            'total_outliers': total_outliers,
            'physical_outliers': physical_count,
            'statistical_outliers': seasonal_outliers.sum(),
            'contextual_outliers': contextual_outliers.sum()
        }
        
        # Show sample outliers
        if total_outliers > 0:
            print(f"\n📋 Sample outlier yang terdeteksi (5 teratas):")
            outlier_samples = self.data[combined_outliers][['Date', 'RH_AVG', 'is_physical_outlier', 
                                                        'is_statistical_outlier', 'is_contextual_outlier']].head()
            print(outlier_samples.to_string(index=False))
        
        print("\n✅ Deteksi outlier selesai. Siap untuk treatment.")
        
        return seasonal_stats

    def treat_outliers_gentle_capping(self):
        """
        Treatment outlier RH_AVG dengan pendekatan gentle capping
        Mempertahankan pola seasonal dan tidak menghilangkan informasi penting
        """
        print("\n=== TREATMENT OUTLIER RH_AVG (GENTLE CAPPING) ===")
        
        if not hasattr(self, 'outlier_stats'):
            print("⚠️ Jalankan detect_outliers_domain_aware() terlebih dahulu")
            return False
        
        if self.outlier_stats['total_outliers'] == 0:
            print("✅ Tidak ada outlier yang perlu di-treatment")
            return True
        
        # Backup data original
        self.data['RH_AVG_original'] = self.data['RH_AVG'].copy()
        treated_count = 0
        
        print(f"🔧 Memproses {self.outlier_stats['total_outliers']} outlier...")
        
        # Step 1: Handle Physical Outliers (paling prioritas)
        physical_mask = self.data['is_physical_outlier'] == True
        if physical_mask.sum() > 0:
            print(f"\n📌 Step 1: Treatment Physical Outliers ({physical_mask.sum()})")
            
            for idx in self.data[physical_mask].index:
                original_val = self.data.loc[idx, 'RH_AVG']
                
                # Replace dengan NaN untuk kemudian diimputasi
                if original_val in [999, 9999, -999] or original_val < 0 or original_val > 100:
                    self.data.loc[idx, 'RH_AVG'] = np.nan
                    self.data.loc[idx, 'treatment_method'] = 'physical_outlier_to_nan'
                    treated_count += 1
                    print(f"   📅 {self.data.loc[idx, 'Date'].strftime('%Y-%m-%d')}: {original_val} → NaN (physical outlier)")
        
        # Step 2: Handle Statistical Outliers dengan Seasonal Capping
        statistical_mask = (self.data['is_statistical_outlier'] == True) & (self.data['RH_AVG'].notna())
        if statistical_mask.sum() > 0:
            print(f"\n📌 Step 2: Treatment Statistical Outliers ({statistical_mask.sum()})")
            
            for season, stats in self.outlier_stats['seasonal_stats'].items():
                season_mask = (self.data['season'] == season) & statistical_mask
                
                if season_mask.sum() == 0:
                    continue
                    
                print(f"   🌤️  Musim {season}: {season_mask.sum()} outlier")
                
                for idx in self.data[season_mask].index:
                    original_val = self.data.loc[idx, 'RH_AVG']
                    
                    # Capping ke bounds yang reasonable
                    if original_val < stats['lower_bound']:
                        # Cap ke P10 musiman (lebih gentle dari lower_bound)
                        season_data = self.data[(self.data['season'] == season) & 
                                            (self.data['RH_AVG'].notna())]['RH_AVG']
                        new_val = max(season_data.quantile(0.10), 15)  # Min 15%
                        
                    elif original_val > stats['upper_bound']:
                        # Cap ke P90 musiman (lebih gentle dari upper_bound)  
                        season_data = self.data[(self.data['season'] == season) & 
                                            (self.data['RH_AVG'].notna())]['RH_AVG']
                        new_val = min(season_data.quantile(0.90), 95)  # Max 95%
                    
                    else:
                        continue
                    
                    self.data.loc[idx, 'RH_AVG'] = new_val
                    self.data.loc[idx, 'treatment_method'] = f'seasonal_capping_{season}'
                    treated_count += 1
                    
                    print(f"     📅 {self.data.loc[idx, 'Date'].strftime('%Y-%m-%d')}: {original_val:.1f}% → {new_val:.1f}%")
        
        # Step 3: Handle Contextual Outliers (paling ringan)
        contextual_mask = (self.data['is_contextual_outlier'] == True) & (self.data['RH_AVG'].notna())
        if contextual_mask.sum() > 0:
            print(f"\n📌 Step 3: Review Contextual Outliers ({contextual_mask.sum()})")
            
            # Untuk contextual outliers, kita hanya flag tanpa mengubah nilai
            # Karena mungkin merupakan kondisi cuaca ekstrem yang valid
            contextual_extreme = 0
            
            for idx in self.data[contextual_mask].index:
                original_val = self.data.loc[idx, 'RH_AVG']
                
                # Hanya treatment jika benar-benar ekstrem (di luar batas fisik yang masuk akal)
                if original_val > 98:
                    self.data.loc[idx, 'RH_AVG'] = 95  # Cap di 95%
                    self.data.loc[idx, 'treatment_method'] = 'contextual_extreme_cap'
                    treated_count += 1
                    contextual_extreme += 1
                    print(f"     📅 {self.data.loc[idx, 'Date'].strftime('%Y-%m-%d')}: {original_val:.1f}% → 95.0% (extreme cap)")
                elif original_val < 15:
                    self.data.loc[idx, 'RH_AVG'] = 20  # Cap di 20%
                    self.data.loc[idx, 'treatment_method'] = 'contextual_extreme_cap'
                    treated_count += 1
                    contextual_extreme += 1
                    print(f"     📅 {self.data.loc[idx, 'Date'].strftime('%Y-%m-%d')}: {original_val:.1f}% → 20.0% (extreme cap)")
                else:
                    # Mark sebagai reviewed tapi tidak diubah
                    self.data.loc[idx, 'treatment_method'] = 'contextual_reviewed_kept'
            
            print(f"     💡 {contextual_extreme} nilai di-cap, {contextual_mask.sum() - contextual_extreme} dipertahankan")
        
        # Step 4: Imputasi untuk nilai yang dijadikan NaN
        nan_after_treatment = self.data['RH_AVG'].isna().sum()
        if nan_after_treatment > 0:
            print(f"\n📌 Step 4: Imputasi {nan_after_treatment} nilai NaN hasil treatment")
            self.impute_missing_rh_avg()  # Gunakan method yang sudah ada
        
        # Step 5: Summary Treatment
        print(f"\n📊 RINGKASAN TREATMENT:")
        
        treated_data = self.data[self.data.get('treatment_method', '').str.len() > 0]
        if len(treated_data) > 0:
            treatment_summary = treated_data['treatment_method'].value_counts()
            print("   📋 Methods used:")
            for method, count in treatment_summary.items():
                print(f"     • {method}: {count}")
        
        # Statistik perbandingan
        valid_original = self.data['RH_AVG_original'].dropna()
        valid_treated = self.data['RH_AVG'].dropna()
        
        if len(valid_original) > 0 and len(valid_treated) > 0:
            print(f"\n📈 Perbandingan Before vs After:")
            print(f"   • Mean: {valid_original.mean():.2f}% → {valid_treated.mean():.2f}%")
            print(f"   • Median: {valid_original.median():.2f}% → {valid_treated.median():.2f}%")
            print(f"   • Std: {valid_original.std():.2f}% → {valid_treated.std():.2f}%")
            print(f"   • Range: {valid_original.min():.1f}-{valid_original.max():.1f}% → {valid_treated.min():.1f}-{valid_treated.max():.1f}%")
            
            # Cek apakah treatment terlalu agresif
            mean_change = abs(valid_treated.mean() - valid_original.mean())
            if mean_change < 2:
                print("   ✅ Treatment gentle: perubahan mean < 2%")
            else:
                print(f"   ⚠️ Treatment signifikan: perubahan mean {mean_change:.2f}%")
        
        print(f"\n✅ Treatment selesai: {treated_count} nilai dimodifikasi")
        print("💡 Data siap untuk analisis dengan outlier yang sudah ditangani secara gentle")
        
        return True

    def seasonal_analysis(self):
        print("\n=== ANALISIS MUSIMAN RH_AVG ===")

        valid_mask = self.data['RH_AVG'].notna()
        seasonal_data = self.data[valid_mask].copy()

        if len(seasonal_data) == 0:
            print("⚠️ Tidak ada data valid untuk analisis musiman")
            return

        monthly_stats = seasonal_data.groupby('month')['RH_AVG'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun',
                      'Jul', 'Ags', 'Sep', 'Okt', 'Nov', 'Des']

        print(f"\U0001f4c5 STATISTIK BULANAN:")
        print("Bulan | Count | Mean  | Median| Std   | Min   | Max")
        print("-" * 55)

        for month in range(1, 13):
            if month in monthly_stats.index:
                stats = monthly_stats.loc[month]
                print(f"{month_names[month-1]:5s} | {stats['count']:5.0f} | {stats['mean']:5.1f} | "
                      f"{stats['median']:5.1f} | {stats['std']:5.1f} | {stats['min']:5.1f} | {stats['max']:5.1f}")
            else:
                print(f"{month_names[month-1]:5s} | {'N/A':5s} | {'N/A':5s} | {'N/A':5s} | {'N/A':5s} | {'N/A':5s} | {'N/A':5s}")

        if not monthly_stats.empty:
            wettest_month = monthly_stats['mean'].idxmax()
            driest_month = monthly_stats['mean'].idxmin()

            print(f"\n🌧️  Bulan terlembab: {month_names[wettest_month-1]} ({monthly_stats.loc[wettest_month, 'mean']:.1f}%)")
            print(f"☀️  Bulan terkering: {month_names[driest_month-1]} ({monthly_stats.loc[driest_month, 'mean']:.1f}%)")

        return monthly_stats
    
    def show_missing_dates(self):
        """
        Menampilkan tanggal-tanggal yang memiliki nilai RH_AVG hilang (NaN atau nilai khusus seperti 9999)
        """
        print("\n=== TANGGAL DENGAN NILAI RH_AVG HILANG ===")
        
        if 'RH_AVG' not in self.data.columns or 'Date' not in self.data.columns:
            print("⚠️ Kolom 'RH_AVG' atau 'Date' tidak ditemukan!")
            return
        
        # Deteksi missing: NaN atau nilai ekstrem (misal 9999, 8888)
        missing_mask = (
            self.data['RH_AVG'].isna() |
            (self.data['RH_AVG'] == 9999) |
            (self.data['RH_AVG'] == 8888) |
            (self.data['RH_AVG'] == -999) |
            (self.data['RH_AVG'] == -9999)
        )
        
        missing_dates = self.data.loc[missing_mask, ['Date', 'RH_AVG']]
        
        if missing_dates.empty:
            print("✅ Tidak ada data hilang pada kolom RH_AVG.")
        else:
            print(f"🔍 Total tanggal dengan nilai hilang: {len(missing_dates)}")
            print(missing_dates.to_string(index=False))
            
   
    def create_individual_plots(self, output_dir="humidity_plots", save_plots=True):
        """
        Membuat 3 plot individual untuk analisis RH_AVG preprocessing (SIMPLIFIED)
        """
        print("\n=== MEMBUAT VISUALISASI RH_AVG PREPROCESSING ===")
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Plots akan disimpan di: {output_dir}")
        
        # Pastikan data outlier sudah dideteksi
        if not hasattr(self, 'outlier_stats'):
            print("⚠️ Menjalankan deteksi outlier terlebih dahulu...")
            self.detect_outliers_domain_aware()
        
        valid_data = self.data['RH_AVG'].dropna()
        if len(valid_data) == 0:
            print("❌ Tidak ada data valid untuk plotting")
            return
        
        # Set style dan color palette untuk humidity
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Color scheme untuk humidity
        colors = {
            'rh_main': '#4A90E2',      # Blue untuk humidity
            'rh_treated': '#2E8B57',   # Sea green untuk treated
            'outlier_stat': '#FF6B6B', # Red untuk statistical outliers
            'outlier_context': '#FFA500', # Orange untuk contextual outliers
            'season_djf': '#1E88E5',   # Blue - Wet season
            'season_mam': '#43A047',   # Green - Transition 1  
            'season_jja': '#FB8C00',   # Orange - Dry season
            'season_son': '#8E24AA'    # Purple - Transition 2
        }
        
        # ============================================================================
        # PLOT 1: TIME SERIES BEFORE-AFTER TREATMENT (SIMPLIFIED)
        # ============================================================================
        print("\n📊 Plot 1: Time Series Before-After Treatment")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Prepare data
        dates = self.data['Date']
        rh_original = self.data.get('RH_AVG_original', self.data['RH_AVG'])
        rh_treated = self.data['RH_AVG']
        
        # ========================
        # Subplot 1: Before (Original with outliers marked)
        # ========================
        ax1.plot(dates, rh_original, color=colors['rh_main'], alpha=0.7, linewidth=0.8, label='Data Original')
        
        # Mark outliers
        if hasattr(self, 'outlier_stats') and self.outlier_stats['total_outliers'] > 0:
            # Statistical outliers
            stat_mask = self.data.get('is_statistical_outlier', pd.Series(False, index=self.data.index))
            if stat_mask.sum() > 0:
                ax1.scatter(dates[stat_mask], rh_original[stat_mask], 
                        color=colors['outlier_stat'], s=20, alpha=0.8, 
                        label=f'Statistical Outliers ({stat_mask.sum()})', marker='o')
            
            # Contextual outliers  
            context_mask = self.data.get('is_contextual_outlier', pd.Series(False, index=self.data.index))
            if context_mask.sum() > 0:
                ax1.scatter(dates[context_mask], rh_original[context_mask], 
                        color=colors['outlier_context'], s=15, alpha=0.6,
                        label=f'Contextual Outliers ({context_mask.sum()})', marker='^')
        
        ax1.set_ylabel('Kelembaban Relatif (%)', fontsize=12)
        ax1.set_title('Before Treatment', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.set_ylim(40, 105)
        ax1.set_facecolor('white')
        
        # ========================
        # Subplot 2: After (Treated data)
        # ========================
        ax2.plot(dates, rh_treated, color=colors['rh_treated'], alpha=0.8, linewidth=0.8, label='Data After Treatment')
        
        # Highlight treated values
        if 'treatment_method' in self.data.columns:
            treated_mask = self.data['treatment_method'].notna() & (self.data['treatment_method'] != 'contextual_reviewed_kept')
            if treated_mask.sum() > 0:
                ax2.scatter(dates[treated_mask], rh_treated[treated_mask], 
                        color='red', s=25, alpha=0.9, 
                        label=f'Treated Values ({treated_mask.sum()})', marker='s')
        
        ax2.set_xlabel('Tahun', fontsize=12)
        ax2.set_ylabel('Kelembaban Relatif (%)', fontsize=12)
        ax2.set_title('After Treatment', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.set_ylim(40, 105)
        ax2.set_facecolor('white')
        
        # Set x-axis ticks
        start_date = dates.min()
        end_date = dates.max()
        date_ticks = pd.date_range(start=start_date, end=end_date, freq='YS')
        ax2.set_xticks(date_ticks)
        ax2.set_xticklabels([d.year for d in date_ticks], rotation=0, ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if save_plots:
            plot1_path = os.path.join(output_dir, "preprocessing_rh_plot_01_timeseries_treatment.png")
            plt.savefig(plot1_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✅ Plot 1 saved: {plot1_path}")
        
        plt.show()
        
        # ============================================================================
        # PLOT 2: SEASONAL BOXPLOT (SIMPLIFIED)
        # ============================================================================
        print("\n📊 Plot 2: Seasonal Boxplot Analysis")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare monthly data
        month_order = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        month_labels = ['Des', 'Jan', 'Feb', 'Mar', 'Apr', 'Mei',
                        'Jun', 'Jul', 'Ags', 'Sep', 'Okt', 'Nov']
        
        # Season color mapping for boxes
        season_month_map = {0:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:0}
        season_colors = [colors['season_djf'], colors['season_mam'], colors['season_jja'], colors['season_son']]
        
        # ========================
        # Subplot 1: Original Monthly Boxplot
        # ========================
        monthly_data_orig = []
        for month in month_order:
            month_mask = (self.data['month'] == month) & (rh_original.notna())
            if month_mask.sum() > 0:
                monthly_data_orig.append(rh_original[month_mask])
            else:
                monthly_data_orig.append(pd.Series(dtype=float))
        
        bp1 = ax1.boxplot(monthly_data_orig, labels=month_labels, patch_artist=True, showfliers=True)
        
        # Color by season
        for i, patch in enumerate(bp1['boxes']):
            if i < len(season_month_map):
                patch.set_facecolor(season_colors[season_month_map[i]])
                patch.set_alpha(0.7)
        
        ax1.set_ylabel('Kelembaban Relatif (%)', fontsize=12)
        ax1.set_xlabel('Bulan', fontsize=12)
        ax1.set_title('Before Treatment', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.4)
        ax1.set_ylim(40, 105)
        ax1.set_facecolor('white')
        
        # ========================
        # Subplot 2: Treated Monthly Boxplot
        # ========================
        monthly_data_treated = []
        for month in month_order:
            month_mask = (self.data['month'] == month) & (rh_treated.notna())
            if month_mask.sum() > 0:
                monthly_data_treated.append(rh_treated[month_mask])
            else:
                monthly_data_treated.append(pd.Series(dtype=float))
        
        bp2 = ax2.boxplot(monthly_data_treated, labels=month_labels, patch_artist=True, showfliers=True)
        
        # Color by season
        for i, patch in enumerate(bp2['boxes']):
            if i < len(season_month_map):
                patch.set_facecolor(season_colors[season_month_map[i]])
                patch.set_alpha(0.7)
        
        ax2.set_ylabel('Kelembaban Relatif (%)', fontsize=12)
        ax2.set_xlabel('Bulan', fontsize=12)
        ax2.set_title('After Treatment', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.4)
        ax2.set_ylim(40, 105)
        ax2.set_facecolor('white')
        
        plt.tight_layout()
        
        if save_plots:
            plot2_path = os.path.join(output_dir, "preprocessing_rh_plot_02_seasonal_patterns.png")
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✅ Plot 2 saved: {plot2_path}")
        
        plt.show()
        
        # ============================================================================
        # PLOT 3: DISTRIBUTION COMPARISON (SIMPLIFIED)
        # ============================================================================
        print("\n📊 Plot 3: Distribution Analysis")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # ========================
        # Subplot 1: Original Distribution
        # ========================
        ax1.hist(rh_original.dropna(), bins=50, density=True, alpha=0.7, 
                color=colors['rh_main'], label='Original Data', edgecolor='black', linewidth=0.5)
        
        # Add KDE
        from scipy.stats import gaussian_kde
        orig_kde = gaussian_kde(rh_original.dropna())
        x_range = np.linspace(40, 105, 100)
        ax1.plot(x_range, orig_kde(x_range), color='darkblue', linewidth=2.5, label='KDE')
        
        # Add mean and median lines
        ax1.axvline(rh_original.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {rh_original.mean():.1f}%')
        ax1.axvline(rh_original.median(), color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {rh_original.median():.1f}%')
        
        ax1.set_xlabel('Kelembaban Relatif (%)', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Before Treatment', fontsize=14, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
        ax1.set_xlim(40, 105)
        ax1.set_facecolor('white')
        
        # ========================
        # Subplot 2: Treated Distribution
        # ========================
        ax2.hist(rh_treated.dropna(), bins=50, density=True, alpha=0.7,
                color=colors['rh_treated'], label='Treated Data', edgecolor='black', linewidth=0.5)
        
        # Add KDE for treated
        treated_kde = gaussian_kde(rh_treated.dropna())
        ax2.plot(x_range, treated_kde(x_range), color='darkgreen', linewidth=2.5, label='KDE')
        
        # Add mean and median lines
        ax2.axvline(rh_treated.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {rh_treated.mean():.1f}%')
        ax2.axvline(rh_treated.median(), color='orange', linestyle='--', linewidth=2, 
                label=f'Median: {rh_treated.median():.1f}%')
        
        ax2.set_xlabel('Kelembaban Relatif (%)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('After Treatment', fontsize=14, fontweight='bold', pad=15)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.4)
        ax2.set_xlim(40, 105)
        ax2.set_facecolor('white')
        
        plt.tight_layout()
        
        if save_plots:
            plot3_path = os.path.join(output_dir, "preprocessing_rh_plot_03_distribution_outliers.png")
            plt.savefig(plot3_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"✅ Plot 3 saved: {plot3_path}")
        
        plt.show()
        
        print("\n🎉 Semua plot berhasil dibuat!")
        print(f"📈 Plot 1: Time Series Before-After Treatment (Simplified)")
        print(f"📊 Plot 2: Seasonal Boxplot Analysis (Simplified)") 
        print(f"📋 Plot 3: Distribution Comparison (Simplified)")
        
        if save_plots:
            print(f"\n📁 Semua plot disimpan di: {output_dir}")
            
        return True

    def summary_report(self):
        print("\n" + "="*60)
        print("LAPORAN RINGKASAN ANALISIS RH_AVG")
        print("="*60)

        if not self.rh_stats:
            print("⚠️ Jalankan descriptive_statistics() terlebih dahulu")
            return

        print(f"\U0001f4ca RINGKASAN STATISTIK:")
        print(f"   • Data valid: {self.rh_stats['count']:,} records")
        print(f"   • Rata-rata: {self.rh_stats['mean']:.2f}")
        print(f"   • Median: {self.rh_stats['median']:.2f}")
        print(f"   • Standar Deviasi: {self.rh_stats['std']:.2f}")
        print(f"   • Rentang: {self.rh_stats['min']:.1f} - {self.rh_stats['max']:.1f}")

        print(f"\n🎯 KARAKTERISTIK DISTRIBUSI:")
        iqr = self.rh_stats['q3'] - self.rh_stats['q1']
        lower_bound = self.rh_stats['q1'] - 1.5 * iqr
        upper_bound = self.rh_stats['q3'] + 1.5 * iqr
        print(f"   • Batas outlier IQR: {lower_bound:.1f} - {upper_bound:.1f}")

        print(f"\n💡 REKOMENDASI PREPROCESSING:")
        missing_pct = (len(self.data) - self.rh_stats['count']) / len(self.data) * 100
        if missing_pct > 10:
            print(f"   • ⚠️  Missing data: {missing_pct:.1f}% - perlu strategi imputasi")
        else:
            print(f"   • ✅ Missing data rendah: {missing_pct:.1f}%")

        if self.rh_stats['std'] < 10:
            quality = "stabil"
        elif self.rh_stats['std'] < 20:
            quality = "moderat"
        else:
            quality = "sangat bervariasi"
        print(f"   • 📈 Variabilitas data: {quality}")

        print("\n🚀 SIAP UNTUK PENGEMBANGAN PREPROCESSING!")
        print("="*60)

def main():
    data_path = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data BMKG/Lokasi/Kab. Aceh Besar/Stasiun Klimatologi Aceh/CSV/BMKG_Data_All.csv"
    
    # Setup direktori output
    output_dir = "/run/media/cryptedlm/local_d/Kuliah/Tugas Akhir/Dataset/Data BMKG/Lokasi/Kab. Aceh Besar/Stasiun Klimatologi Aceh/CSV CLEANED/kelembaban"
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = RH_AVG_Analyzer(data_path)

    # 🔄 Redirect stdout ke log file di direktori output
    log_file = open(os.path.join(output_dir, "preprocessing_log_rh.txt"), "w")
    sys.stdout = log_file

    try:
        if not analyzer.load_data():
            return

        # Step 1: Analisis missing values sebelum imputasi
        analyzer.show_missing_dates()
        
        # Step 2: Imputasi missing values
        imputed_results = analyzer.impute_missing_rh_avg()
        
        # Step 3: Verifikasi hasil imputasi
        if imputed_results:
            analyzer.show_missing_dates()
        
        # Step 4: Analisis statistik deskriptif dengan data yang sudah bersih
        valid_data = analyzer.descriptive_statistics()
        
        if valid_data is not None:
            # Step 5: Analisis musiman
            analyzer.seasonal_analysis()
            
            # Step 5a: Deteksi outlier
            analyzer.detect_outliers_domain_aware()

            # Step 5b: Treatment outlier  
            analyzer.treat_outliers_gentle_capping()

            # Step 5c: Re-run descriptive statistics dengan data yang sudah di-treatment
            print("\n🔄 Analisis ulang setelah outlier treatment...")
            valid_data_treated = analyzer.descriptive_statistics()
            
            # Step 5d: Buat visualisasi preprocessing
            print("\n🎨 Membuat visualisasi preprocessing...")
            plot_success = analyzer.create_individual_plots(
                output_dir=os.path.join(output_dir, "plots"), 
                save_plots=True
            )
            
            if plot_success:
                print("✅ Visualisasi berhasil dibuat")
            else:
                print("❌ Gagal membuat visualisasi")

            
            # Step 6: Laporan ringkasan final
            analyzer.summary_report()
            
            # Step 7: Summary imputasi (jika ada)
            if imputed_results:
                print("\n🔄 Ringkasan imputasi yang dilakukan...")
                print("="*50)
                print("DETAIL IMPUTASI YANG TELAH DILAKUKAN")
                print("="*50)
                for result in imputed_results:
                    print(f"📅 {result['date'].strftime('%Y-%m-%d')}: {result['original']} → {result['imputed']:.1f}% ({result['method']})")
                print("✅ Semua imputasi berhasil diterapkan!")

        return analyzer

    finally:
        # ✅ Kembalikan stdout dan tutup file log
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"📁 Log saved to {os.path.join(output_dir, 'preprocessing_log_rh.txt')}")

        # Step 8: Simpan hasil preprocessing ke CSV
        if analyzer and hasattr(analyzer, 'data'):
            try:
                print("\n🔄 Menyimpan hasil preprocessing...")
                
                # Buat copy data untuk output
                output_df = analyzer.data.copy()
                
                # Pastikan kolom datetime components tersedia
                if 'Date' in output_df.columns:
                    output_df['Year'] = output_df['Date'].dt.year
                    output_df['month'] = output_df['Date'].dt.month
                    output_df['day'] = output_df['Date'].dt.day
                
                # Rename RH_AVG yang sudah diimputasi untuk clarity
                if 'RH_AVG' in output_df.columns:
                    output_df['RH_AVG_preprocessed'] = output_df['RH_AVG']
                
                # Define output path
                output_path = os.path.join(output_dir, "preprocessed_humidity_data.csv")

                
                # Pilih kolom yang akan disimpan
                columns_to_save = ['Date', 'Year', 'month', 'day', 'RH_AVG_preprocessed']
                available_columns = [col for col in columns_to_save if col in output_df.columns]
                
                if available_columns:
                    # Simpan ke CSV
                    output_df[available_columns].to_csv(output_path, index=False)
                    
                    # Informasi hasil save
                    print(f"✅ File CSV berhasil disimpan: {output_path}")
                    print(f"📊 Jumlah records: {len(output_df):,}")
                    print(f"📋 Kolom tersimpan: {available_columns}")
                    
                    # Info tambahan jika ada imputasi
                    if imputed_results:
                        print(f"🔧 Termasuk {len(imputed_results)} nilai yang diimputasi")
                    
                    # Tampilkan sample data
                    print(f"\n📝 Sample data (5 baris pertama):")
                    print(output_df[available_columns].head().to_string(index=False))
                    
                else:
                    print("❌ Tidak ada kolom yang valid untuk disimpan")
                    
            except Exception as e:
                print(f"❌ Error saat menyimpan CSV: {str(e)}")
        
        else:
            print("❌ Data analyzer tidak tersedia untuk disimpan")

        print("\n🎉 Preprocessing selesai! Check file output dan log untuk detail lengkap.")

if __name__ == "__main__":
    result = main()