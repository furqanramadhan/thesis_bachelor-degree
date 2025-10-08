import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
import matplotlib
# matplotlib.use('TkAgg')  # Tambahkan line ini
matplotlib.use('Agg')  
import seaborn as sns
import numpy as np
import re

df = pd.read_csv("kuisioner_tanam_padi - Petani.csv")

# Debug: Periksa nama kolom yang tersedia
print("Kolom yang tersedia dalam dataset:")
print(df.columns.tolist())
print("\nShape dataset:", df.shape)
print("\nSample data:")
print(df.head())

# Gunakan nama kolom yang sesuai (sesuaikan dengan output debug di atas)
# Asumsi nama kolom berdasarkan kode Anda:
kabupaten_col = 'kab_kota'  # atau sesuaikan dengan nama kolom yang benar
gender_col = 'jenis_kelamin'  # atau sesuaikan dengan nama kolom yang benar
lahan_col = 'lahan_milik_sendiri'  # atau sesuaikan dengan nama kolom yang benar

# Periksa apakah kolom ada sebelum digunakan
required_columns = [kabupaten_col, gender_col, lahan_col]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"\nKolom yang tidak ditemukan: {missing_columns}")
    print("Silakan sesuaikan nama kolom dengan yang tersedia dalam dataset.")
else:
    # Visualisasi 1: Komposisi gender berdasarkan kabupaten
    gender_by_kabupaten = pd.crosstab(df[kabupaten_col], df[gender_col])
    
    plt.figure(figsize=(12, 6))
    ax = gender_by_kabupaten.plot(kind='bar', color=['skyblue', 'lightcoral'])


    total_count = df['jenis_kelamin'].count()

    for container in ax.containers:
        ax.bar_label(container, label=[''] * len(container), padding=5)

        # Then add custom framed annotations
    for container in ax.containers:
        for i, rect in enumerate(container):
            height = rect.get_height()
            if height > 0:
                percentage = (height / total_count) * 100
                label_text = f'{int(height)}\n{percentage:.1f}%'
                
                # Position for the annotation
                x = rect.get_x() + rect.get_width()/2
                y = height + 0.5  # Adjust this value based on your data scale
                
                # Create text with frame
                text = ax.annotate(
                    label_text, 
                    xy=(x, y),
                    xytext=(0, 0),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc='white',
                        ec='gray',
                        lw=1,
                        alpha=0.9
                    ),
                    fontsize=9,
                    fontweight='bold'
                )

    y_max = gender_by_kabupaten.values.max()

    # Create dynamic intervals based on the maximum value
    if y_max <= 10:
        interval = 1
        y_limit = y_max + 2
    elif y_max <= 25:
        interval = 2
        y_limit = y_max + 5
    elif y_max <= 50:
        interval = 5
        y_limit = y_max + 5
    else:
        interval = 10
        y_limit = y_max + 10

    # Set y-axis ticks with dynamic interval
    y_ticks = np.arange(0, y_limit + interval, interval)
    plt.yticks(y_ticks)
    plt.ylim(0, y_limit)  # Explicitly set the y-axis limits
    # You might need to adjust tight_layout or bottom margin to accommodate the multi-line labels
    plt.tight_layout(pad=2.0)
    plt.title('Komposisi Jenis Kelamin', fontsize=14)
    plt.xlabel('Kabupaten', fontsize=12)
    plt.ylabel('Jumlah Petani', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Jenis Kelamin')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('01_gender_by_kabupaten.png', dpi=300)


    # Visualisasi 2: Kepemilikan Lahan berdasarkan Kabupaten
    def kategorikan_lahan(text):
        if pd.isna(text):
            return 'Tidak Diketahui'
        elif 'Lahan milik sendiri' in str(text):
            return 'Lahan milik sendiri'
        else: 
            return "Lahan yang berasal dari pihak lain"

    df['kepemilikan_lahan'] = df[lahan_col].apply(kategorikan_lahan)

    lahan_by_kabupaten = pd.crosstab(df[kabupaten_col], df['kepemilikan_lahan'])

    plt.figure(figsize=(12, 6))
    ax = lahan_by_kabupaten.plot(kind='bar', color=['#66c2a5', '#fc8d62'])

    total_count = df['kepemilikan_lahan'].count()

    # Find the maximum value for dynamic y-axis scaling
    y_max = lahan_by_kabupaten.values.max()

    # Create the plot wihtout labels
    for container in ax.containers:
        ax.bar_label(container, labels=[''] * len(container), padding=5)

    for container in ax.containers:
        for i, rect in enumerate(container):
            height = rect.get_height()
            if height > 0:
                percentage = (height / total_count) * 100
                label_text = f'{int(height)}\n{percentage:.1f}%'
                
                # Position for the annotation
                x = rect.get_x() + rect.get_width()/2
                y = height + 0.5  # Adjust this value based on your data scale
                
                # Create text with frame
                text = ax.annotate(
                    label_text, 
                    xy=(x, y),
                    xytext=(0, 0),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc='white',
                        ec='gray',
                        lw=1,
                        alpha=0.9
                    ),
                    fontsize=9,
                    fontweight='bold'
                )


    # Create dynamic intervals based on the maximum value - FIXED INDENTATION
    if y_max <= 10:
        interval = 1
        y_limit = y_max + 2
    elif y_max <= 25:
        interval = 2
        y_limit = y_max + 5
    elif y_max <= 50:
        interval = 5
        y_limit = y_max + 5
    else:
        interval = 10
        y_limit = y_max + 10

    # Set y-axis ticks with dynamic interval
    y_ticks = np.arange(0, y_limit + interval, interval)
    plt.yticks(y_ticks)
    plt.ylim(0, y_limit)  # Explicitly set the y-axis limits

    plt.title('Kepemilikan Lahan berdasarkan Kabupaten', fontsize=14)
    plt.ylabel('Jumlah Petani', fontsize=12)
    plt.xlabel('Kabupaten', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Kepemilikan Lahan')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('02_kepemilikan_lahan_kabupaten.png', dpi=300)

    # Visualisasi 3 : Pendidikan Terakhir berdasarkan Kabupaten
    pendidikan_by_kabupaten = pd.crosstab(df[kabupaten_col], df['pendidikan_terakhir'])

    # Cek kategori pendidikan yang tersedia
    print("Kategori pendidikan dalam dataset:")
    print(df['pendidikan_terakhir'].unique())

    # Urutkan kategori pendidikan berdasarkan tingkatannya
    pendidikan_order = ['SD', 'SMP', 'SMA/SMK', 'D2', 'D3', 'S1', '-']

    # Filter dan urutkan kolom berdasarkan urutan pendidikan
    available_pendidikan = [p for p in pendidikan_order if p in pendidikan_by_kabupaten.columns]
    pendidikan_by_kabupaten = pendidikan_by_kabupaten[available_pendidikan]

    plt.figure(figsize=(12, 10))

    # Define your custom color palette
    custom_colors = ['#637AB9', '#FCB53B', '#556B2F', '#ED775A', '#660B05', '#BA487F', '#999999']
    # Make sure we have enough colors for all education categories
    colors_to_use = custom_colors[:len(available_pendidikan)]

    ax = pendidikan_by_kabupaten.plot(
        kind='barh',  # horizontal bar
        width=0.7,
        color=colors_to_use
    )
    
    total_count = df['pendidikan_terakhir'].count()


    for container in ax.containers:
        ax.bar_label(container, labels=[''] * len(container), padding=5)

    # Tambahkan label jumlah dan persentase dengan posisi yang lebih baik
    for container in ax.containers:
        for i, rect in enumerate(container):
            width = rect.get_width()
            if width > 0:
                percentage = (width / total_count) * 100
                label_text = f'{int(width)}\n{percentage:.1f}%'
                
                # Position for the annotation - adjusted with more offset
                y = rect.get_y() + rect.get_height()/2
                x = width + 1.0  # Increased offset from 0.5 to 1.0
                
                # Create text with frame
                text = ax.annotate(
                    label_text, 
                    xy=(x, y),
                    xytext=(8, 0),  # Increased horizontal offset from 5 to 8
                    textcoords='offset points',
                    ha='left', va='center',
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc='white',
                        ec='gray',
                        lw=1,
                        alpha=0.9
                    ),
                    fontsize=9,
                    fontweight='bold'
                )

    # Find the maximum value for dynamic x-axis scaling (horizontal)
    x_max = pendidikan_by_kabupaten.values.max()

    # Modified dynamic interval logic to use multiples of 5
    if x_max <= 5:
        interval = 1
        x_limit = 5
    elif x_max <= 10:
        interval = 2
        x_limit = 10
    elif x_max <= 20:
        interval = 5
        x_limit = 25
    elif x_max <= 50:
        interval = 5
        x_limit = (int(x_max / 5) + 1) * 5  # Round up to nearest multiple of 5
    else:
        interval = 10
        x_limit = (int(x_max / 10) + 1) * 10  # Round up to nearest multiple of 10

    # Set x-axis ticks with consistent interval
    x_ticks = np.arange(0, x_limit + interval, interval)
    plt.xticks(x_ticks)
    plt.xlim(0, x_limit)  # Explicitly set the x-axis limits

    plt.title('Pendidikan Terakhir berdasarkan Kabupaten', fontsize=14)
    plt.ylabel('Kabupaten', fontsize=12)
    plt.xlabel('Jumlah Petani', fontsize=12)
    plt.legend(title='Pendidikan Terakhir', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('03_pendidikan_by_kabupaten.png', dpi=300, bbox_inches='tight')

    # Visualisasi 4: Distribusi Umur berdasarkan Kabupaten (dalam range)
    plt.figure(figsize=(14, 6))  # Slightly wider for legend on the right

    # Fungsi untuk kategorisasi umur
    def kategorisasi_umur(umur):
        if umur <= 20:
            return '<20'
        elif umur <= 30:
            return '21-30'
        elif umur <= 40:
            return '31-40'
        elif umur <= 50:
            return '41-50'
        elif umur <= 60:
            return '51-60'
        else:
            return '60>'

    # Membuat kolom kategori umur
    df['kategori_umur'] = df['umur'].apply(kategorisasi_umur)

    # Membuat crosstab untuk umur berdasarkan kabupaten
    umur_by_kabupaten = pd.crosstab(df[kabupaten_col], df['kategori_umur'])

    # Urutkan kolom berdasarkan urutan umur yang ditetapkan
    umur_order = ['<20', '21-30', '31-40', '41-50', '51-60', '60>']

    # Tambahkan kolom yang tidak ada dalam data dengan nilai 0
    for category in umur_order:
        if category not in umur_by_kabupaten.columns:
            umur_by_kabupaten[category] = 0

    # Urutkan kolom sesuai dengan urutan yang ditetapkan
    umur_by_kabupaten = umur_by_kabupaten[umur_order]

    # Membuat plot dengan warna yang menarik
    colors = ['#52357B', '#F97A00', '#16610E', '#4300FF', '#FAA533', '#8C1007']
    ax = umur_by_kabupaten.plot(
        kind='bar', 
        width=0.6,
        color=colors[:len(umur_order)]
    )

    total_count = df['kategori_umur'].count()

    # Clear any existing bar labels first
    for container in ax.containers:
        ax.bar_label(container, labels=[''] * len(container), padding=5)

    # Add labels with frames to each bar - FIXED FOR VERTICAL BARS
    for container in ax.containers:
        for i, rect in enumerate(container):
            height = rect.get_height()  # Untuk vertical bar, gunakan get_height()
            if height > 0:
                percentage = (height / total_count) * 100
                label_text = f'{int(height)}\n{percentage:.1f}%'
                
                # Position for the annotation - FIXED positioning
                x = rect.get_x() + rect.get_width()/2  # Center horizontally
                y = height + 0.5  # Offset from bar top
                
                # Create text with frame
                ax.annotate(
                    label_text, 
                    xy=(x, y),
                    xytext=(0, 0),
                    textcoords='offset points',
                    ha='center', va='bottom',  # Center horizontally, bottom align vertically
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc='white',
                        ec='gray',
                        lw=1,
                        alpha=0.9
                    ),
                    fontsize=9,
                    fontweight='bold'
                )

    # Find the maximum value for dynamic y-axis scaling
    y_max = umur_by_kabupaten.values.max()

    if y_max <= 10:
        interval = 1
        y_limit = y_max + 2
    elif y_max <= 25:
        interval = 2
        y_limit = y_max + 5
    elif y_max <= 50:
        interval = 5
        y_limit = y_max + 5
    else:
        interval = 10
        y_limit = y_max + 10

    # Set y-axis with dynamic scaling
    y_ticks = np.arange(0, y_limit + interval, interval)
    plt.yticks(y_ticks)
    plt.ylim(0, y_limit)

    # Beautify plot
    plt.title('Distribusi Umur Petani berdasarkan Kabupaten', fontsize=14)
    plt.xlabel('Kabupaten', fontsize=12)
    plt.ylabel('Jumlah Petani', fontsize=12)
    plt.xticks(rotation=45)

    # Move legend outside of plot
    plt.legend(title='Kategori Umur', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('04_umur_range_by_kabupaten.png', dpi=300, bbox_inches='tight')

    # Visualisasi 5: Estimasi Mulai Bertani berdasarkan Kabupaten (dalam range)
    plt.figure(figsize=(14, 6))

    # Fungsi untuk kategorisasi tahun mulai bertani
    def kategorisasi_tahun_bertani(tahun):
        if pd.isna(tahun):
            return 'Tidak Diketahui'
        elif tahun <= 1990:
            return 'Sebelum 1990'
        elif tahun <= 2000:
            return '1991-2000'
        elif tahun <= 2010:
            return '2001-2010'
        elif tahun <= 2020:
            return '2011-2020'
        else:
            return '2021-Sekarang'

    # Membuat kolom kategori tahun mulai bertani
    # Sesuaikan nama kolom dengan yang ada di dataset
    tahun_mulai_col = 'tahun_mulai_bertani'  # Nama kolom yang benar
    df['kategori_tahun_bertani'] = df[tahun_mulai_col].apply(kategorisasi_tahun_bertani)

    # Membuat crosstab untuk tahun mulai bertani berdasarkan kabupaten
    tahun_by_kabupaten = pd.crosstab(df[kabupaten_col], df['kategori_tahun_bertani'])

    # Urutkan kolom berdasarkan urutan tahun
    tahun_order = ['Sebelum 1990', '1991-2000', '2001-2010', '2011-2020', '2021-Sekarang', 'Tidak Diketahui']
    available_tahun = [t for t in tahun_order if t in tahun_by_kabupaten.columns]
    tahun_by_kabupaten = tahun_by_kabupaten[available_tahun]

    # Membuat plot dengan warna yang menarik (gradasi dari lama ke baru)
    colors = ['#8B4513', '#FFD93D', '#DAA520', '#4682B4', '#2E8B57', '#A9A9A9']  # Coklat tua ke hijau, abu untuk tidak diketahui
    ax = tahun_by_kabupaten.plot(
        kind='bar', 
        width=0.6,
        color=colors[:len(available_tahun)]
    )

    # Calculate total count for percentages
    total_count = df['kategori_tahun_bertani'].count()

    for container in ax.containers:
            ax.bar_label(container, labels=[''] * len(container), padding=5)

    for container in ax.containers:
        for i, rect in enumerate(container):
            height = rect.get_height()
            if height > 0:
                percentage = (height / total_count) * 100
                label_text = f'{int(height)}\n{percentage:.1f}%'
                
                # Position for the annotation
                x = rect.get_x() + rect.get_width()/2
                y = height + 0.5  # Adjust this value based on your data scale
                
                # Create text with frame
                text = ax.annotate(
                    label_text, 
                    xy=(x, y),
                    xytext=(0, 0),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc='white',
                        ec='gray',
                        lw=1,
                        alpha=0.9
                    ),
                    fontsize=9,
                    fontweight='bold'
                )


    # Find the maximum value for dynamic y-axis scaling
    y_max = tahun_by_kabupaten.values.max()

    if y_max <= 10:
        interval = 1
        y_limit = y_max + 2
    elif y_max <= 25:
        interval = 2
        y_limit = y_max + 5
    elif y_max <= 50:
        interval = 5
        y_limit = y_max + 5
    else:
        interval = 10
        y_limit = y_max + 10

    # Set y-axis dengan scaling dinamis
    y_ticks = np.arange(0, y_limit + interval, interval)
    plt.yticks(y_ticks)
    plt.ylim(0, y_limit)

    # Percantik plot
    plt.title('Estimasi Mulai Bertani berdasarkan Kabupaten', fontsize=14)
    plt.xlabel('Kabupaten', fontsize=12)
    plt.ylabel('Jumlah Petani', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Periode Mulai Bertani', 
            bbox_to_anchor=(1.01, 1),  # Slightly closer to plot
            loc='upper left')    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('05_estimasi_mulai_bertani.png', dpi=300, bbox_inches='tight')

    print("Kategori tahun mulai bertani dalam dataset:")
    print(df['kategori_tahun_bertani'].value_counts())

    # Visualisasi 6: Varietas Padi berdasarkan Kabupaten
    plt.figure(figsize=(18, 12))

    # Dictionary varietas padi dengan hari panen
    varietas_hari = {
        'Inpari': '(111 hari)',
        'Tinggong': '(75 hari)',
        'Inpari 32': '(120 hari)',
        'Padi andin': '(105 hari)',
        'Ciherang': '(125 hari)',
        'Cibatu': '(95 hari)',
        'CBD': '(125 hari)',
        'Ngawos': '(95 hari)',
        'Beulerang': '(150 hari)',
        'Brangus': '(150 hari)',
        'CBD 04': '(100 hari)',
        'CBD Murni': '(90 hari)',
        'Cibatu 05': '(95 hari)',
        'Ciheurang beruang': '(87 hari)',
        'Inpari 42': '(120 hari)',
        'Mekongga': '(125 hari)',
        'Mustajab': '(120 hari)',
        'Padi bojeng': '(90 hari)',
        'Ramos': '(110 hari)'
    }

    # Fungsi untuk memisahkan varietas padi yang multiple
    def pisahkan_varietas(varietas_text):
        if pd.isna(varietas_text):
            return ['Tidak Diketahui']
        
        # Convert to string dan clean
        varietas_str = str(varietas_text).strip()
        
        # Pisahkan berdasarkan berbagai separator
        separators = [' dan ', ',', ' & ', ';', ' + ']
        varietas_list = [varietas_str]
        
        for separator in separators:
            new_list = []
            for item in varietas_list:
                if separator in item:
                    new_list.extend([v.strip() for v in item.split(separator)])
                else:
                    new_list.append(item)
            varietas_list = new_list
        
        # Clean empty strings dan duplicates
        varietas_list = [v for v in varietas_list if v and v.strip()]
        return list(set(varietas_list)) if varietas_list else ['Tidak Diketahui']

    # Buat dataframe baru dengan varietas yang sudah dipisah
    expanded_data = []
    for idx, row in df.iterrows():
        varietas_list = pisahkan_varietas(row['varietas_padi'])
        for varietas in varietas_list:
            new_row = row.copy()
            new_row['varietas_padi_clean'] = varietas
            expanded_data.append(new_row)

    df_expanded = pd.DataFrame(expanded_data)

    # Group varietas padi berdasarkan kabupaten
    varietas_by_kabupaten = pd.crosstab(df_expanded[kabupaten_col], df_expanded['varietas_padi_clean'])

    # Pilih varietas utama saja (yang memiliki informasi hari)
    main_varieties = list(varietas_hari.keys())
    available_varieties = [v for v in main_varieties if v in varietas_by_kabupaten.columns]

    # Tambahkan varietas yang tidak ada dalam list utama tapi ada di data
    other_varieties = [v for v in varietas_by_kabupaten.columns if v not in main_varieties]

    # Gabungkan untuk mendapatkan semua varietas yang akan ditampilkan
    all_varieties = available_varieties + other_varieties

    # Filter hanya varietas yang dipilih
    if available_varieties:
        varietas_by_kabupaten = varietas_by_kabupaten[all_varieties]

    # Warna-warna menarik untuk varietas padi
    colors = ['#4682B4', '#8FBC8F', '#DAA520', '#B22222', '#9370DB', 
            '#3CB371', '#FF8C00', '#4169E1', '#CD5C5C', '#2E8B57', 
            '#A9A9A9', '#6A5ACD', '#808000', '#FF4500', '#20B2AA',
            '#FF69B4', '#32CD32', '#8A2BE2', '#FF6347', '#00CED1']

    # Plot horizontal bar chart
    ax = varietas_by_kabupaten.plot(
        kind='barh',  # horizontal bar
        width=0.8,
        color=colors[:len(all_varieties)],
        figsize=(16, 10)
    )

    # Calculate total count for percentages
    total_count = df_expanded['varietas_padi_clean'].count()

    # Create the plot without labels
    for container in ax.containers:
        ax.bar_label(container, labels=[''] * len(container), padding=5)

    # Then add custom framed annotations - FIXED FOR HORIZONTAL BARS
    for container in ax.containers:
        for i, rect in enumerate(container):
            width = rect.get_width()
            if width > 0:
                percentage = (width / total_count) * 100
                label_text = f'{int(width)}\n{percentage:.1f}%'
                
                # Position for the annotation - adjusted with more offset
                y = rect.get_y() + rect.get_height()/2
                x = width + 1.0  # Increased offset from 0.5 to 1.0
                
                # Create text with frame
                text = ax.annotate(
                    label_text, 
                    xy=(x, y),
                    xytext=(8, 0),  # Increased horizontal offset from 5 to 8
                    textcoords='offset points',
                    ha='left', va='center',
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc='white',
                        ec='gray',
                        lw=1,
                        alpha=0.9
                    ),
                    fontsize=9,
                    fontweight='bold'
                )
    # Find the maximum value for dynamic x-axis scaling (horizontal)
    x_max = varietas_by_kabupaten.values.max()

    # Create dynamic intervals based on the maximum value
    if x_max <= 10:
        interval = 1
        x_limit = x_max + 2
    elif x_max <= 25:
        interval = 2
        x_limit = x_max + 5
    elif x_max <= 50:
        interval = 5
        x_limit = x_max + 5
    else:
        interval = 10
        x_limit = x_max + 10

    # Set x-axis ticks with dynamic interval
    x_ticks = np.arange(0, x_limit + interval, interval)
    plt.xticks(x_ticks)
    plt.xlim(0, x_limit)  # Explicitly set the x-axis limits

    # Tambahkan hari panen ke legend labels
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label in varietas_hari:
            new_labels.append(f'{label} {varietas_hari[label]}')
        else:
            new_labels.append(label)

    # Percantik plot
    plt.title('Varietas Padi berdasarkan Kabupaten', fontsize=14)
    plt.ylabel('Kabupaten', fontsize=12)
    plt.xlabel('Jumlah Petani', fontsize=12)

    # Legend with smaller font size and multiple columns if needed
    if len(new_labels) > 10:
        plt.legend(handles, new_labels, title='Varietas Padi', 
                bbox_to_anchor=(1.05, 1), loc='upper left', 
                fontsize=9, ncol=2)
    else:
        plt.legend(handles, new_labels, title='Varietas Padi', 
                bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.subplots_adjust(left=0.2) 
    plt.tight_layout(rect=[0, 0, 0.78, 1])  # Reserve more space for legend (22%)
    plt.subplots_adjust(left=0.2, right=0.78)  # Adjust right margin to make room for legend
    plt.savefig('06_varietas_padi_by_kabupaten.png', dpi=300, bbox_inches='tight')