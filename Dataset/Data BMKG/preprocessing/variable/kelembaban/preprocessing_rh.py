import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

    def analyze_missing_values(self):
        print("\n=== ANALISIS MISSING VALUES RH_AVG ===")

        total_records = len(self.data)
        missing_nan = self.data['RH_AVG'].isna().sum()

        special_values = {}
        for val in [9999, 8888, -999, -9999]:
            count = (self.data['RH_AVG'] == val).sum()
            if count > 0:
                special_values[val] = count

        print(f"\U0001f4ca Total records: {total_records:,}")
        print(f"\U0001f4ca Missing/NaN values: {missing_nan:,} ({missing_nan/total_records*100:.2f}%)")

        if special_values:
            print(f"\U0001f4ca Nilai khusus ditemukan:")
            for val, count in special_values.items():
                print(f"   • Nilai {val}: {count:,} ({count/total_records*100:.2f}%)")

        valid_data = self.data['RH_AVG'].dropna()
        for val in special_values.keys():
            valid_data = valid_data[valid_data != val]

        valid_count = len(valid_data)
        print(f"\U0001f4ca Valid data: {valid_count:,} ({valid_count/total_records*100:.2f}%)")

        return valid_data

    def descriptive_statistics(self):
        print("\n=== STATISTIK DESKRIPTIF RH_AVG ===")

        rh_valid = self.analyze_missing_values()

        if len(rh_valid) == 0:
            print("⚠️ Tidak ada data RH_AVG yang valid untuk dianalisis")
            return None

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

        print(f"\n\U0001f321️  KATEGORI KELEMBABAN:")
        categories = {
            'Sangat Kering (0-30%)': (rh_valid >= 0) & (rh_valid <= 30),
            'Kering (31-50%)': (rh_valid > 30) & (rh_valid <= 50),
            'Sedang (51-70%)': (rh_valid > 50) & (rh_valid <= 70),
            'Lembab (71-85%)': (rh_valid > 70) & (rh_valid <= 85),
            'Sangat Lembab (86-95%)': (rh_valid > 85) & (rh_valid <= 95),
            'Jenuh (>95%)': rh_valid > 95
        }

        for category, mask in categories.items():
            count = mask.sum()
            percentage = count / len(rh_valid) * 100
            print(f"   • {category}: {count:,} ({percentage:.1f}%)")

        self.rh_stats = {
            'count': len(rh_valid),
            'mean': rh_valid.mean(),
            'median': rh_valid.median(),
            'std': rh_valid.std(),
            'min': rh_valid.min(),
            'max': rh_valid.max(),
            'q1': rh_valid.quantile(0.25),
            'q3': rh_valid.quantile(0.75),
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

    def summary_report(self):
        print("\n" + "="*60)
        print("LAPORAN RINGKASAN ANALISIS RH_AVG")
        print("="*60)

        if not self.rh_stats:
            print("⚠️ Jalankan descriptive_statistics() terlebih dahulu")
            return

        print(f"\U0001f4ca RINGKASAN STATISTIK:")
        print(f"   • Data valid: {self.rh_stats['count']:,} records")
        print(f"   • Kelembaban rata-rata: {self.rh_stats['mean']:.2f}%")
        print(f"   • Kelembaban median: {self.rh_stats['median']:.2f}%")
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

    if not analyzer.load_data():
        return

    print("\n🔄 Menjalankan analisis statistik deskriptif...")
    valid_data = analyzer.descriptive_statistics()

    if valid_data is not None:
        print("\n🔄 Menjalankan analisis musiman...")
        analyzer.seasonal_analysis()

        print("\n🔄 Membuat laporan ringkasan...")
        analyzer.summary_report()

    return analyzer

if __name__ == "__main__":
    result = main()
