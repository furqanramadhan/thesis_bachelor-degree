pdflatex -interaction=nonstopmode --shell-escape seminar-proposal.tex;
bibtex seminar-proposal.aux ;
pdflatex -interaction=nonstopmode --shell-escape seminar-proposal.tex;
pdflatex -interaction=nonstopmode --shell-escape seminar-proposal.tex;

# pdflatex -interaction=nonstopmode --shell-escape seminar-hasil.tex;
# bibtex seminar-hasil.aux ;
# pdflatex -interaction=nonstopmode --shell-escape seminar-hasil.tex;
# pdflatex -interaction=nonstopmode --shell-escape seminar-hasil.tex;


# pdflatex -interaction=nonstopmode --shell-escape skripsi.tex;
# bibtex skripsi.aux;
# pdflatex -interaction=nonstopmode --shell-escape skripsi.tex;
# pdflatex -interaction=nonstopmode --shell-escape skripsi.tex;

# Silahkan uncomment sesuai file yang ingin dicompile
# di dalam template ini terdapat bug, oleh karena itu kita perlu mengcompile beberapa kali untuk menghasilkan satu file
# pada bagian daftar isi, daftar tabel, daftar gambar, dan daftar lampiran tidak akan berubah jika kita hanya mengcompile sekali.
# Namun Karena saya sudah berpengalaman dengan LaTeX, ini bug ini bisa dianggap sebagai fitur, misalkan kita ingin menggantikan file pengesahan yang sudah ditanda tangani oleh dosen pembimbing, kita hanya perlu menggantikan file tersebut dan mengcompile sekali saja, maka file akan tergantikan dengan sendirinya. dan daftar isi, daftar tabel, daftar gambar, dan daftar lampiran tidak berubah. 
# Jadi, kita tidak perlu menghapus file pengesahan yang lama, cukup gantikan saja dengan file yang baru, dan compile sekali saja, maka file pengesahan akan tergantikan dengan sendirinya. (Dalam kondisi ini cukup compile sekali saja jangan berkali-kali, karena bisa saja list pada daftar isi hilang)


# Tetapi untuk awal-awal pembuatan tidak perlu khawatir, cukup ikut arahan sesuai readme saja

# Tips from me: Abdul Hafidh

# Okay If you have any question, you can ask me on my email: hafidhabdul4444@gmail.com with subject: [LaTeX] Question ?