import pandas as pd  

# Baca CSV  
df = pd.read_csv("foler/NDVI_MODIS_AcehBesar_2005_2025.csv")  

# Konversi kolom date  
df["date"] = pd.to_datetime(df["date"])  
df["tanggal"] = df["date"].dt.strftime("%-d %b %Y")  
df["hari"] = df["date"].dt.day  
df["bulan"] = df["date"].dt.month  
df["tahun"] = df["date"].dt.year  

# Hapus kolom date  
df.drop(columns=["date"], inplace=True)  

# Pindahkan kolom geo ke akhir  
kolom_urut = [col for col in df.columns if col != ".geo"] + [".geo"]  
df = df[kolom_urut]  

# Simpan ke XLSX  
df.to_excel("output_NDVI.xlsx", index=False) 
