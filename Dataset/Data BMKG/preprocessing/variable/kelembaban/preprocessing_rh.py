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
        monthly_avg = valid_data.groupby('Month')['RH_AVG'].mean()
        
        imputed_values = []
        
        for idx in missing_indices:
            date = self.data.loc[idx, 'Date']
            month = self.data.loc[idx, 'Month']
            
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
        print(f"   • Mean: {rh_valid.mean():.2f}%")
        print(f"   • Median: {rh_valid.median():.2f}%")

        mode_series = rh_valid.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else "N/A"
        print(f"   • Mode: {mode_val}%")

        print(f"   • Std Dev: {rh_valid.std():.2f}%")
        print(f"   • Variance: {rh_valid.var():.2f}%²")
        print(f"   • Range: {rh_valid.max() - rh_valid.min():.2f}%")

        print(f"\n📊 KUARTIL & PERCENTILES:")
        print(f"   • Min: {rh_valid.min():.2f}%")
        print(f"   • Q1 (25%): {rh_valid.quantile(0.25):.2f}%")
        print(f"   • Q2/Median (50%): {rh_valid.quantile(0.50):.2f}%")
        print(f"   • Q3 (75%): {rh_valid.quantile(0.75):.2f}%")
        print(f"   • Max: {rh_valid.max():.2f}%")
        print(f"   • IQR: {rh_valid.quantile(0.75) - rh_valid.quantile(0.25):.2f}%")

        print(f"\n🌡️  KATEGORI KELEMBABAN:")
        categories = {
            'Terlalu Kering (<45%)': rh_valid < 45,
            'Ideal (45%-65%)': (rh_valid >= 45) & (rh_valid <= 65),
            'Terlalu Lembab (>65%)': rh_valid > 65
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

    def seasonal_analysis(self):
        print("\n=== ANALISIS MUSIMAN RH_AVG ===")

        valid_mask = self.data['RH_AVG'].notna()
        seasonal_data = self.data[valid_mask].copy()

        if len(seasonal_data) == 0:
            print("⚠️ Tidak ada data valid untuk analisis musiman")
            return

        monthly_stats = seasonal_data.groupby('Month')['RH_AVG'].agg([
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

    def summary_report(self):
        print("\n" + "="*60)
        print("LAPORAN RINGKASAN ANALISIS RH_AVG")
        print("="*60)

        if not self.rh_stats:
            print("⚠️ Jalankan descriptive_statistics() terlebih dahulu")
            return

        print(f"\U0001f4ca RINGKASAN STATISTIK:")
        print(f"   • Data valid: {self.rh_stats['count']:,} records")
        print(f"   • Rata-rata: {self.rh_stats['mean']:.2f}%")
        print(f"   • Median: {self.rh_stats['median']:.2f}%")
        print(f"   • Standar Deviasi: {self.rh_stats['std']:.2f}%")
        print(f"   • Rentang: {self.rh_stats['min']:.1f}% - {self.rh_stats['max']:.1f}%")

        print(f"\n🎯 KARAKTERISTIK DISTRIBUSI:")
        iqr = self.rh_stats['q3'] - self.rh_stats['q1']
        lower_bound = self.rh_stats['q1'] - 1.5 * iqr
        upper_bound = self.rh_stats['q3'] + 1.5 * iqr
        print(f"   • Batas outlier IQR: {lower_bound:.1f}% - {upper_bound:.1f}%")

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
    data_path = "/run/media/cryptedlm/localdisk/Kuliah/Tugas Akhir/Dataset/Data BMKG/Stasiun Klimatologi Aceh/CSV/BMKG_Data_All.csv"
    analyzer = RH_AVG_Analyzer(data_path)

    # 🔄 Redirect stdout ke log file
    log_file = open("preprocessing_log_rh.txt", "w")
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
        print("📁 Log saved to preprocessing_log_rh.txt")

        # Step 8: Simpan hasil preprocessing ke CSV
        if analyzer and hasattr(analyzer, 'data'):
            try:
                print("\n🔄 Menyimpan hasil preprocessing...")
                
                # Buat copy data untuk output
                output_df = analyzer.data.copy()
                
                # Pastikan kolom datetime components tersedia
                if 'Date' in output_df.columns:
                    output_df['Year'] = output_df['Date'].dt.year
                    output_df['Month'] = output_df['Date'].dt.month
                    output_df['Day'] = output_df['Date'].dt.day
                
                # Rename RH_AVG yang sudah diimputasi untuk clarity
                if 'RH_AVG' in output_df.columns:
                    output_df['RH_AVG_preprocessed'] = output_df['RH_AVG']
                
                # Define output path
                output_path = "preprocessed_humidity_data.csv"
                
                # Pilih kolom yang akan disimpan
                columns_to_save = ['Date', 'Year', 'Month', 'Day', 'RH_AVG_preprocessed']
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