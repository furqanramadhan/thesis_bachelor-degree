# Tools
RM = rm -rf
LATEX = pdflatex -interaction=nonstopmode --shell-escape
BIBTEX = bibtex

# Document names
DOCNAMES = seminar-proposal seminar-hasil skripsi

# Bibliography file
BIBFILE = daftar-pustaka.bib

# Default target
.DEFAULT_GOAL := help

# Help target
help:
	@echo "Available targets:"
	@echo "  make proposal    - Compile seminar-proposal.pdf"
	@echo "  make hasil       - Compile seminar-hasil.pdf"
	@echo "  make skripsi     - Compile skripsi.pdf"
	@echo "  make all         - Compile all documents"
	@echo "  make clean       - Remove all generated files"
	@echo "  make mostlyclean - Remove intermediate files only"

# Compile all documents
all: proposal hasil skripsi

# Individual document targets
proposal: seminar-proposal.pdf

hasil: seminar-hasil.pdf

skripsi: skripsi.pdf

# Explicit rules - ALWAYS continue even if LaTeX has errors
seminar-proposal.pdf: seminar-proposal.tex $(BIBFILE)
	
	@echo "STEP 1: First pdflatex compile"
	
	-$(LATEX) seminar-proposal.tex
	@echo ""
	
	@echo "STEP 2: Running BibTeX"
	
	-$(BIBTEX) seminar-proposal
	@echo ""
	
	@echo "STEP 3: Second pdflatex compile"
	
	-$(LATEX) seminar-proposal.tex
	@echo ""
	
	@echo "STEP 4: Third pdflatex compile"
	
	-$(LATEX) seminar-proposal.tex
	@echo ""
	@if [ -f seminar-proposal.pdf ]; then \
		echo "SUCCESS! Output: seminar-proposal.pdf"; \
		ls -lh seminar-proposal.pdf; \
	else \
		echo "ERROR: PDF was not generated!"; \
		exit 1; \
	fi

seminar-hasil.pdf: seminar-hasil.tex $(BIBFILE)
	
	@echo "STEP 1: First pdflatex compile"
	
	-$(LATEX) seminar-hasil.tex
	@echo ""
	
	@echo "STEP 2: Running BibTeX"
	
	-$(BIBTEX) seminar-hasil
	@echo ""
	
	@echo "STEP 3: Second pdflatex compile"
	
	-$(LATEX) seminar-hasil.tex
	@echo ""
	
	@echo "STEP 4: Third pdflatex compile"
	
	-$(LATEX) seminar-hasil.tex
	@echo ""
	@if [ -f seminar-hasil.pdf ]; then \
		echo "SUCCESS! Output: seminar-hasil.pdf"; \
		ls -lh seminar-hasil.pdf; \
	else \
		echo "ERROR: PDF was not generated!"; \
		exit 1; \
	fi

skripsi.pdf: skripsi.tex $(BIBFILE)
	
	@echo "STEP 1: First pdflatex compile"
	
	-$(LATEX) skripsi.tex
	@echo ""
	
	@echo "STEP 2: Running BibTeX"
	
	-$(BIBTEX) skripsi
	@echo ""
	
	@echo "STEP 3: Second pdflatex compile"
	
	-$(LATEX) skripsi.tex
	@echo ""
	
	@echo "STEP 4: Third pdflatex compile"
	
	-$(LATEX) skripsi.tex
	@echo ""
	@if [ -f skripsi.pdf ]; then \
		echo "SUCCESS! Output: skripsi.pdf"; \
		ls -lh skripsi.pdf; \
	else \
		echo "ERROR: PDF was not generated!"; \
		exit 1; \
	fi

# Clean intermediate files only, i want to remove .fuse hidden example name: .fuse_hidden0001dba700000020
mostlyclean:
	@echo "Cleaning intermediate files..."
	$(RM) *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg
	$(RM) *.bcf *.run.xml *.synctex.gz *.fls *.fdb_latexmk
	$(RM) include/*.aux
	$(RM) _minted-*
	$(RM) .fuse_hidden*
	@echo "Intermediate files cleaned!"

# Clean all generated files including PDFs
clean: mostlyclean
	@echo "Cleaning all generated files..."
	$(RM) $(addsuffix .pdf,$(DOCNAMES))
	@echo "Clean complete!"

# Phony targets
.PHONY: all clean mostlyclean help proposal hasil skripsi