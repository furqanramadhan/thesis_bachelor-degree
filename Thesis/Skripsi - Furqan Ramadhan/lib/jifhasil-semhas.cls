%
% Template Laporan Tugas Akhir: Seminar Hasil
%
% @author Abdul
% Template ini telah disesuaikan dengan aturan penulisan tugas akhir yang terdapat pada dokumen Panduan Tugas Akhir FMIPA Unsyiah tahun 2010.
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{jifhasiltheme}

\RequirePackage[english,indonesian]{babel}
\RequirePackage{indentfirst}
\RequirePackage{setspace}
\RequirePackage[T1]{fontenc}
\RequirePackage{times}
\RequirePackage{graphicx,latexsym}

\setlength{\paperwidth}{210mm}
\setlength{\paperheight}{297mm}
%\usepackage[a4paper,left=4cm,right=3cm,top=3cm,bottom=3cm]{geometry}
\usepackage[a4paper,left=3.5cm,right=3cm,top=3cm,bottom=3cm]{geometry}
\usepackage{verbatim,enumerate}
%tabularx untuk membuat tabel yang ukurannya bisa disesuaikan dengan lebar halaman
\usepackage{tabularx}

% Membantu penulisan notasi matematika terutama untuk dokumen dengan banyak rumus. it's me!
% \usepackage{amsmath}
\usepackage[fleqn]{amsmath}% <-- tambahkan fleqn di sini
\setlength{\mathindent}{1.5cm}
\usepackage{enumitem, mathtools} % Tambahkan di preamble
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{multirow}
\usepackage{float}
\usepackage{indentfirst}
\usepackage{amssymb}
\usepackage{siunitx} % Untuk format angka otomatis
\sisetup{round-mode=places, round-precision=3, group-separator={,}}


% Agar bisa menggunakan compactitem
\usepackage{paralist} %it's me!
\usepackage{changepage} % Untuk adjustwidth
% Agar bisa menulis code snippet
%\usepackage{listings} %it's me!

\newif\if@msthesis
\newif\if@msproposal
\DeclareOption{skripsi}{\@msthesistrue\@msproposalfalse}
\DeclareOption{proposal}{\@msthesisfalse\@msproposaltrue}

\RequirePackage[titles]{tocloft}

\newcommand*{\enableboldchapterintoc}{%
  \addtocontents{toc}{\string\renewcommand{\protect\cftchapfont}{\protect\normalfont\protect\bfseries}}
  \addtocontents{toc}{\string\renewcommand{\protect\cftchappagefont}{\protect\normalfont\protect}}
  \addtocontents{toc}{\protect\setlength{\cftbeforechapskip}{12pt}}
}

\newcommand*{\disableboldchapterintoc}{%
  \addtocontents{toc}{\string\renewcommand{\protect\cftchappagefont}{\protect\normalfont}}
  \addtocontents{toc}{\string\renewcommand{\protect\cftchapfont}{\protect\normalfont}}
  \addtocontents{toc}{\protect\setlength{\cftbeforechapskip}{0pt}}
}

\newif\if@singlesp
\newif\if@doublesp
\DeclareOption{satuspasi}{\@singlesptrue\@doublespfalse}
\DeclareOption{duaspasi}{\@singlespfalse\@doublesptrue}
\newif\if@onehalfsp
\@onehalfspfalse
\DeclareOption{satusetengahspasi}{\@onehalfsptrue}

\newif\if@langindo
\newif\if@langeng
\DeclareOption{indonesia}{\@langindotrue\@langengfalse}
\DeclareOption{inggris}{\@langindofalse\@langengtrue}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ExecuteOptions{jifhasil,satusetengahspasi,indonesia}
\ProcessOptions
\LoadClass[a4paper,12pt]{report}
\RequirePackage{natbib}
\bibliographystyle{style/jifbibstyle}
%\bibliographystyle{apalike}
%\RequirePackage{glossaries}%
%\renewcommand{\nomname}{\nomenclaturename}%
%\makeglossary%
%\makenomenclature

%------------------------------------------------------------
%Layout
%------------------------------------------------------------
\setlength{\topmargin}{-0.9cm}
\setlength{\headheight}{12pt}
\setlength{\headsep}{2.3cm}
\setlength{\topskip}{1ex}
\setlength{\oddsidemargin}{1.46cm}
\setlength{\evensidemargin}{1.46cm}
\setlength{\textwidth}{14.3cm}
\setlength{\textheight}{22cm}
\setlength{\footskip}{1.5cm}
\setlength{\marginparsep}{0pt}
\setlength{\marginparwidth}{0pt}

\setlength{\parindent}{3em} %Indent approx 6 character

%------------------------------------------------------------
%The form of page numbering
%------------------------------------------------------------
%\pagestyle{myheadings}
%\markright{}

%------------------------------------------------------------
%Define page numbering in the first chapter
%------------------------------------------------------------
\def\ps@chapterheading{%

  }

% it's me!
\usepackage{fancyhdr}

\setlength{\footskip}{20pt}
\setlength{\headsep}{20pt}
\renewcommand{\headrulewidth}{0.0pt}
	\fancyhf{}
	\fancyhead[L]{}
	\fancyhead[C]{}
	\fancyhead[R]{\thepage}
	\renewcommand{\headrulewidth}{0.0pt}
	\renewcommand{\footrulewidth}{0.0pt}
\pagestyle{fancy}

%------------------------------------------------------------
%Redefine chapter and sections
%------------------------------------------------------------
\setcounter{secnumdepth}{2}
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thechapter {\@Roman\c@chapter}
%\renewcommand \thesection {\@arabic\c@section.}
\renewcommand \thesection {\@arabic\c@chapter.\@arabic\c@section}
%\renewcommand\thesubsection {\@alph\c@subsection.}
\renewcommand\thesubsection {\@arabic\c@chapter.\@arabic\c@section.\@arabic\c@subsection}
%\renewcommand\thesubsubsection{\@roman\c@subsubsection.}
%\renewcommand\thesubsubsection{}
\renewcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}}
\renewcommand{\chapter}{\clearpage\thispagestyle{chapterheading}%
  \global\@topnum\z@ %Prevents figures from going at top of page
  \@afterindenttrue %Indent the 1st paragraph
  \secdef\@chapter\@schapter}
\renewcommand{\@makechapterhead}[1]{%
  {\parindent \z@ \centering \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \large\bfseries \@chapapp\space \thechapter
        \par\nobreak
        \vskip -0.2cm
    \fi
    \interlinepenalty\@M
    \large \bfseries #1\par\nobreak
    \vskip 20\p@
    }}
\renewcommand{\@makeschapterhead}[1]{%
  {\parindent \z@ \centering \normalfont
    \interlinepenalty\@M \large \bfseries #1\par\nobreak \vskip 20\p@ }}
%\renewcommand{\section}{\@startsection {section}{1}{\z@}%
%                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
%                                   {2.3ex \@plus.2ex}%
%                                   {\normalfont\normalsize\bfseries\centering}}
\renewcommand{\section}{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {0.5ex \@plus.2ex}%
                                   {\normalfont\normalsize\bfseries}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {0.5ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
%\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\parindent}%
%                                    {3.25ex \@plus1ex \@minus.2ex}%
%                                    {-1em}%
%                                    {\normalfont\normalsize\bfseries}}
\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\renewcommand{\paragraph}{\subparagraph}

\@addtoreset {equation}{chapter}
\renewcommand\theequation
  {\ifnum \c@chapter>\z@ \@arabic\c@chapter.\fi \@arabic\c@equation}
\renewcommand \thefigure
     {\ifnum \c@chapter>\z@ \@arabic\c@chapter.\fi \@arabic\c@figure}
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \@arabic\c@chapter.\fi \@arabic\c@table}

%------------------------------------------------------------
% Redefine caption names
%------------------------------------------------------------
\def\captionsbahasa{%
\def\prefacename{KATA PENGANTAR}%
\def\contentsname{DAFTAR ISI}%
\def\listfigurename{DAFTAR GAMBAR}%
\def\listtablename{DAFTAR TABEL}%
\def\listappendixname{DAFTAR LAMPIRAN}%
\def\nomenclaturename{DAFTAR SINGKATAN}%
\def\abstractname{ABSTRAK}%
\def\acknowledgmentname{HALAMAN PERSEMBAHAN}%
\def\approvalname{PENGESAHAN}%
\def\partname{BAGIAN}%
\def\chaptername{BAB}%
\def\appendixname{LAMPIRAN}%
\def\refname{DAFTAR PUSTAKA}%
\def\bibname{DAFTAR PUSTAKA}%
\def\indexname{Indeks}%
\def\figurename{Gambar}%
\def\tablename{Tabel}%
\def\pagename{Halaman}%
\def\bplagiasiname{PERNYATAAN BEBAS PLAGIASI}%
\def\spernyataanname{SURAT PERNYATAAN}%
}

%English
\def\captionsenglish{%
\def\prefacename{PREFACE}%
\def\contentsname{CONTENTS}%
\def\listfigurename{LIST OF FIGURES}%
\def\listtablename{LIST OF TABLES}%
\def\listappendixname{LIST OF APPENDICES}%
\def\nomenclaturename{NOMENCLATURE}%
\def\abstractname{\textit{ABSTRACT}}%
\def\partname{PART}%
\def\chaptername{CHAPTER}%
\def\appendixname{APPENDIX}%
\def\refname{REFERENCES}%
\def\bibname{REFERENCES}%
\def\indexname{Index}%
\def\figurename{Figure}%
\def\tablename{Table}%
\def\pagename{Page}%
}

%------------------------------------------------------------
% Define thesis's inputs (wrapper untuk settings.tex)
%------------------------------------------------------------
\gdef\@university{Universitas Syiah Kuala}
\gdef\@faculty{Fakultas Matematika dan Ilmu Pengetahuan Alam}
\gdef\@city{Banda Aceh}

% Wrapper commands - ini akan diisi oleh settings.tex
\newcommand{\setJudulCover}[1]{\gdef\@judulcover{#1}}
\newcommand{\setJudul}[1]{\gdef\@judul{#1}}
\newcommand{\setJudulInggris}[1]{\gdef\@judulinggris{#1}}
\newcommand{\setFullName}[1]{\gdef\@fullname{#1}}
\newcommand{\setIdNum}[1]{\gdef\@idnum{#1}}
\newcommand{\setExamDate}[1]{\gdef\@examdate{#1}}
\newcommand{\setApprovalDate}[1]{\gdef\@approvaldate{#1}}
\newcommand{\setValidationDate}[1]{\gdef\@validationdate{#1}}
\newcommand{\setDegree}[1]{\gdef\@degree{#1}}
\newcommand{\setYearSubmit}[1]{\gdef\@yearsubmit{#1}}
\newcommand{\setProgram}[1]{\gdef\@program{#1}}
\newcommand{\setDept}[1]{\gdef\@dept{#1}}
\newcommand{\setFirstSupervisor}[1]{\gdef\@firstsupervisor{#1}}
\newcommand{\setSecondSupervisor}[1]{\gdef\@secondsupervisor{#1}}
\newcommand{\setFirstNip}[1]{\gdef\@firstnip{#1}}
\newcommand{\setSecondNip}[1]{\gdef\@secondnip{#1}}
\newcommand{\setDekan}[1]{\gdef\@dekan{#1}}
\newcommand{\setDekanNip}[1]{\gdef\@dekannip{#1}}
\newcommand{\setKaprodi}[1]{\gdef\@kaprodi{#1}}
\newcommand{\setKaprodiNip}[1]{\gdef\@kaprodinip{#1}}
\newcommand{\setBulan}[1]{\gdef\@bulan{#1}}

% Default values
\gdef\@judulcover{}
\gdef\@judul{}
\gdef\@judulinggris{}
\gdef\@fullname{}
\gdef\@idnum{}
\gdef\@examdate{}
\gdef\@approvaldate{}
\gdef\@validationdate{}
\gdef\@degree{}
\gdef\@yearsubmit{}
\gdef\@program{}
\gdef\@dept{}
\gdef\@firstsupervisor{}
\gdef\@secondsupervisor{}
\gdef\@firstnip{}
\gdef\@secondnip{}
\gdef\@dekan{}
\gdef\@dekannip{}
\gdef\@kaprodi{}
\gdef\@kaprodinip{}
\gdef\@bulan{}

%------------------------------------------------------------
% Define cover in Indonesian (menggunakan variabel dari settings.tex)
%------------------------------------------------------------
\def\cover{%
  \newgeometry{a4paper,left=3.5cm,right=3cm,top=3cm,bottom=3cm}
  \disableboldchapterintoc
  \addcontentsline{toc}{chapter}{Halaman Judul}
  \thispagestyle{empty}%
  \pagenumbering{roman}
  \setcounter{page}{1}
  \begin{center}
    \begin{singlespace}
      \fontsize{18pt}{20pt}\selectfont\MakeUppercase{\Large{\bfseries\@judulcover}}\par\nobreak
      \vspace{2.5cm}
      \MakeUppercase{\large{\bfseries{TUGAS AKHIR}}}\\
      \vspace{1.0cm}
      \normalsize{Diajukan untuk melengkapi tugas-tugas dan\\ memenuhi syarat-syarat guna memperoleh gelar Sarjana Komputer\\}
      \vspace{1.0cm}
      {\normalfont
        Oleh:\\
        \vspace{1.0cm}
        \MakeUppercase{\normalfont\bfseries \underline{\@fullname}}\\
        \MakeUppercase{\normalfont\bfseries \@idnum}}\\
      \vspace{2cm}
      \vfill
      \includegraphics[width=0.3\textwidth]{image/usk_logo.png}
      \vfill
      \vspace{2cm}
      {\normalfont
        \MakeUppercase{\normalfont\bfseries Program Studi Sarjana \@dept}\\
        \MakeUppercase{\normalfont\bfseries Departemen \@dept}\\
        \MakeUppercase{\normalfont\bfseries\@faculty}\\
        \MakeUppercase{\normalfont\bfseries\@university, \@city}\\
        \MakeUppercase{\normalfont\bfseries \@bulan, \@yearsubmit}}\\
    \end{singlespace}
  \end{center}
}

%------------------------------------------------------------
% Approval Page (Style Proposal)
%------------------------------------------------------------
\def\approvalpage{%
  \clearpage\addcontentsline{toc}{chapter}{Halaman Pengesahan}
  \chapter*{\approvalname}%
  \thispagestyle{empty}
  \begin{center}
    \begin{doublespace}
      \vspace{0.5cm}
      \fontsize{12pt}{10pt}\selectfont\MakeUppercase{\large{\bfseries\@judul}}\par\nobreak

      \vspace{0.9cm}
      Oleh:
      \begin{singlespace}
        \begin{compactitem}
          \addtolength{\itemindent}{3cm}
          \setlength{\parsep}{0pt}
          \item[]{\makebox[2cm]{Nama\hfill} : \@fullname}
          \item[]{\makebox[2cm]{NPM\hfill} : \@idnum}
          \item[]{\makebox[2cm]{Departemen\hfill} : \@dept}
        \end{compactitem}
      \end{singlespace}

      \vspace{0.9cm}

      Menyetujui:\\
      \vspace{1.0cm}
      \onehalfspacing
      \begin{tabular}{lll}
        Pembimbing I & \hspace{2.5cm} & Pembimbing II \\
        \vspace{0.3cm} & \vspace{0.3cm} & \vspace{0.3cm}\\
        \underline{\@firstsupervisor}& &
        \underline{\@secondsupervisor} \\
        NIP. \@firstnip & &  NIP. \@secondnip
      \end{tabular}	

      \vspace{1.0cm}

      Mengetahui:\\
      Koordinator Program Studi S1 Informatika \\ 
      Universitas Syiah Kuala,\\
      \vspace{1.7cm}\\
      \underline{\@kaprodi} \\
      NIP. \@kaprodinip

    \end{doublespace}
  \end{center}
}

% %------------------------------------------------------------
% % Approval Page ALTERNATIVE (Style Skripsi - Full with Dekan)
% % Uncomment untuk sidang skripsi final
% %------------------------------------------------------------
% \def\approvalpage{%
%   \clearpage\addcontentsline{toc}{chapter}{Halaman Pengesahan}
%   \chapter*{\approvalname}%
%   \begin{center}
%     \begin{doublespace}
%       \vspace{0.5cm}
%       \fontsize{12pt}{10pt}\selectfont\MakeUppercase{\large{\bfseries\@judul}}\par\nobreak
%
%       \vspace{1cm}
%       \fontsize{12pt}{10pt}\selectfont\MakeUppercase{\large{\bfseries\@judulinggris}}\par\nobreak
%
%       \vspace{0.9cm}
%       Oleh:
%       \begin{singlespace}
%         \begin{compactitem}
%           \addtolength{\itemindent}{3cm}
%           \setlength{\parsep}{0pt}
%           \item[]{\makebox[2cm]{Nama\hfill} : \@fullname}
%           \item[]{\makebox[2cm]{NPM\hfill} : \@idnum}
%           \item[]{\makebox[2cm]{Departemen\hfill} : \@dept}
%         \end{compactitem}
%       \end{singlespace}
%       \vspace{0.9cm}
%
%       Menyetujui:\\
%       \vspace{1.0cm}
%       \onehalfspacing
%       \begin{tabular}{lll}
%         Pembimbing I & \hspace{2.5cm} & Pembimbing II \\
%         \vspace{0.3cm} & \vspace{0.3cm} & \vspace{0.3cm}\\
%         \underline{\@firstsupervisor}& &
%         \underline{\@secondsupervisor} \\
%         NIP. \@firstnip & &  NIP. \@secondnip
%       \end{tabular}	
%       \vspace{1.0cm}
%
%       Mengetahui:\\
%       \begin{tabular}{lll}
%         Dekan FMIPA & &
%         Koordinator Prodi S1 Informatika FMIPA \\ 
%         \vspace{-0.27cm} & \vspace{-0.27cm} & \vspace{-0.27cm}\\
%         Universitas Syiah Kuala, & & Universitas Syiah Kuala,\\
%         \vspace{0.3cm} & \vspace{0.3cm} & \vspace{0.3cm}\\
%         \underline{\@dekan} & & \underline{\@kaprodi}\\
%         NIP. \@dekannip & & NIP. \@kaprodinip
%       \end{tabular}
%     \end{doublespace}
%
%     \vspace{0.9cm}
%     Lulus Sidang Sarjana pada hari \@approvaldate \\
%     Tugas Akhir disahkan pada hari \@validationdate
%   \end{center}
% }

%------------------------------------------------------------
% Acknowlegment
%------------------------------------------------------------
\def\acknowledgment{%
  \chapter*{\acknowledgmentname}%
\null\vfill%Force the content of acknowledment to bottom page
\addcontentsline{toc}{chapter}{\acknowledgmentname}
}

\def\bplagiasi{%
  \chapter*{\bplagiasiname}%
  \vspace{0.2cm}
\addcontentsline{toc}{chapter}{Pernyataan Bebas Plagiasi}
}

\def\spernyataan{%
  \chapter*{\spernyataanname}%
  \vspace{0.2cm}
\addcontentsline{toc}{chapter}{Surat Pernyataan}
}

%------------------------------------------------------------
% Preface
%------------------------------------------------------------
\def\preface{%
  \chapter*{\prefacename}%
  \vspace{0.2cm}
\addcontentsline{toc}{chapter}{Kata Pengantar}
}

%------------------------------------------------------------
% Nomenclature
%------------------------------------------------------------
\def\singkatan{%
  \chapter*{\nomenclaturename}%
  \thispagestyle{plain}
  %\chapter*{\nomenclaturename}%
  %\printglossary
\addcontentsline{toc}{chapter}{\nomenclaturename}
}

%------------------------------------------------------------
% Redefine Abstract in Indonesian
%------------------------------------------------------------
\def\abstractind{%
  \clearpage\thispagestyle{chapterheading}%
  \chapter*{\abstractname}%
  {\parindent \z@ \centering \normalfont
  %{\large\textbf{\abstractname}} \\
  %\vspace{1.0cm}
    }
  \begin{singlespacing}%
  %\vskip 20\p@
  \addcontentsline{toc}{chapter}{Abstrak}
}
\def\endabstractind{\par\vfil\null\end{singlespacing}}

%------------------------------------------------------------
% Redefine Abstract in English
%------------------------------------------------------------
\def\abstracteng{%
  \clearpage\thispagestyle{chapterheading}\selectlanguage{english}
  \chapter*{\abstractname}%
  {\parindent \z@ \centering \normalfont
  %{\large\textbf{\abstractname}} \\
  %\vspace{1.0cm}
    }
  \begin{singlespacing}%
  %\vskip 20\p@
  \addcontentsline{toc}{chapter}{\textit{Abstract}}
}
\def\endabstracteng{\par\vfil\null\end{singlespacing}%
\selectlanguage{bahasa}}

% Tabularization
\newcommand{\itab}[1]{\hspace{0em}\rlap{#1}}
\newcommand{\tab}[1]{\hspace{.2\textwidth}\rlap{#1}}


\AtBeginDocument{%
\if@doublesp%
\doublespacing%
\else\if@singlesp%
\singlespacing%
\else
\onehalfspacing%
\fi%
\fi
}

\AtBeginDocument{%
\if@langindo%
\selectlanguage{bahasa}%
\else
\selectlanguage{english}%
\fi
}

\endinput
