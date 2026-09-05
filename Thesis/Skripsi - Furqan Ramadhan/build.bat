@echo off
REM ============================================================================
REM LaTeX Compilation Script for Windows
REM Supports: seminar-proposal, seminar-hasil, skripsi
REM Usage:
REM   build.bat           - Show help
REM   build.bat proposal  - Compile seminar-proposal.pdf
REM   build.bat hasil     - Compile seminar-hasil.pdf
REM   build.bat skripsi   - Compile skripsi.pdf
REM   build.bat all       - Compile all documents
REM   build.bat clean     - Remove all generated files
REM   build.bat mostlyclean - Remove intermediate files only
REM ============================================================================

setlocal enabledelayedexpansion

REM Document names mapping
set "PROPOSAL=seminar-proposal"
set "HASIL=seminar-hasil"
set "SKRIPSI=skripsi"

REM Check command line argument
if "%1"=="" goto help
if /i "%1"=="help" goto help
if /i "%1"=="proposal" goto build_proposal
if /i "%1"=="hasil" goto build_hasil
if /i "%1"=="skripsi" goto build_skripsi
if /i "%1"=="all" goto build_all
if /i "%1"=="clean" goto clean
if /i "%1"=="mostlyclean" goto mostlyclean

echo Unknown command: %1
goto help

REM ============================================================================
REM HELP
REM ============================================================================
:help
echo.
echo ============================================================================
echo  LaTeX Compilation Helper
echo ============================================================================
echo Available commands:
echo.
echo   build.bat proposal    - Compile seminar-proposal.pdf
echo   build.bat hasil       - Compile seminar-hasil.pdf
echo   build.bat skripsi     - Compile skripsi.pdf
echo   build.bat all         - Compile all documents
echo   build.bat clean       - Remove all generated files
echo   build.bat mostlyclean - Remove intermediate files only
echo.
echo ============================================================================
exit /b 0

REM ============================================================================
REM BUILD ALL DOCUMENTS
REM ============================================================================
:build_all
echo.
echo ============================================================================
echo  Building ALL documents...
echo ============================================================================
call :build_document %PROPOSAL%
call :build_document %HASIL%
call :build_document %SKRIPSI%
echo.
echo ============================================================================
echo  All documents compiled successfully!
echo ============================================================================
exit /b 0

REM ============================================================================
REM BUILD INDIVIDUAL DOCUMENTS
REM ============================================================================
:build_proposal
call :build_document %PROPOSAL%
exit /b %ERRORLEVEL%

:build_hasil
call :build_document %HASIL%
exit /b %ERRORLEVEL%

:build_skripsi
call :build_document %SKRIPSI%
exit /b %ERRORLEVEL%

REM ============================================================================
REM BUILD FUNCTION (Core compilation logic)
REM ============================================================================
:build_document
set "DOC=%~1"
echo.
echo ============================================================================
echo  Building: %DOC%.pdf
echo ============================================================================

REM Check if .tex file exists
if not exist "%DOC%.tex" (
    echo ERROR: File %DOC%.tex not found!
    exit /b 1
)

echo.
echo [STEP 1/4] First pdflatex compile...
pdflatex -interaction=nonstopmode --shell-escape "%DOC%.tex"
if errorlevel 1 (
    echo WARNING: First compile had errors, continuing...
)

echo.
echo [STEP 2/4] Running BibTeX...
bibtex "%DOC%"
if errorlevel 1 (
    echo WARNING: BibTeX had errors, continuing...
)

echo.
echo [STEP 3/4] Second pdflatex compile...
pdflatex -interaction=nonstopmode --shell-escape "%DOC%.tex"
if errorlevel 1 (
    echo WARNING: Second compile had errors, continuing...
)

echo.
echo [STEP 4/4] Third pdflatex compile...
pdflatex -interaction=nonstopmode --shell-escape "%DOC%.tex"
if errorlevel 1 (
    echo WARNING: Third compile had errors, continuing...
)

REM Check if PDF was generated
if exist "%DOC%.pdf" (
    echo.
    echo ============================================================================
    echo  SUCCESS! Generated: %DOC%.pdf
    echo ============================================================================
    dir /b "%DOC%.pdf" | findstr /r ".*"
    exit /b 0
) else (
    echo.
    echo ============================================================================
    echo  ERROR: PDF was not generated!
    echo ============================================================================
    exit /b 1
)

REM ============================================================================
REM CLEAN INTERMEDIATE FILES ONLY
REM ============================================================================
:mostlyclean
echo.
echo ============================================================================
echo  Cleaning intermediate files...
echo ============================================================================
del /Q *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg 2>nul
del /Q *.bcf *.run.xml *.synctex.gz *.fls *.fdb_latexmk 2>nul
if exist include\ del /Q include\*.aux 2>nul
if exist _minted-* rmdir /S /Q _minted-* 2>nul
echo.
echo Intermediate files cleaned!
echo ============================================================================
exit /b 0

REM ============================================================================
REM CLEAN ALL GENERATED FILES INCLUDING PDFs
REM ============================================================================
:clean
call :mostlyclean
echo.
echo Cleaning PDF files...
del /Q "%PROPOSAL%.pdf" "%HASIL%.pdf" "%SKRIPSI%.pdf" 2>nul
echo.
echo ============================================================================
echo  All generated files cleaned!
echo ============================================================================
exit /b 0

@REM basic commands
@REM REM Compile proposal
@REM build.bat proposal

@REM REM Compile seminar hasil
@REM build.bat hasil

@REM REM Compile skripsi
@REM build.bat skripsi

@REM REM Compile semua dokumen
@REM build.bat all

@REM REM Bersihkan file sementara (tetap ada PDF)
@REM build.bat mostlyclean

@REM REM Bersihkan semua (termasuk PDF)
@REM build.bat clean

@REM REM Tampilkan help
@REM build.bat
@REM build.bat help