# Tools
RM = rm -fr
LATEX = xelatex -shell-escape -interaction=nonstopmode -output-directory=$(TEXDIR)
BIBER = biber --input-directory=$(TEXDIR) --output-directory=$(TEXDIR)

# Directory tempat file .tex
TEXDIR = src/04-tugas_akhir

# Project-specific settings
DOCNAMES = thesis seminar_hasil seminar_proposal sidang_skripsi

# Default Targets
all: pdf
doc: pdf
pdf: $(DOCNAMES:=.pdf)

# Individual document targets
thesis: thesis.pdf
hasil: seminar_hasil.pdf
proposal: seminar_proposal.pdf
skripsi: sidang_skripsi.pdf

thesis.pdf: $(TEXDIR)/main.tex
	$(LATEX) $<
	$(BIBER) main
	$(LATEX) $<
	$(LATEX) $<
	mv $(TEXDIR)/main.pdf $@

%.pdf: $(TEXDIR)/%.tex
	$(LATEX) $<
	$(BIBER) $*
	$(LATEX) $<
	$(LATEX) $<
	mv $(TEXDIR)/$*.pdf $@

# Cleaning intermediate files
mostlyclean:
	$(RM) ./*.bbl

clean: mostlyclean
	$(RM) ./*.run.xml out/*.synctex.gz
	$(RM) ./*.dvi ./*.log ./*.aux ./*.blg ./*.toc ./*.lof ./*.lot ./*.out ./*.bcf ./*.run.xml ./*.synctex.gz ./*.pdf
	$(RM) $(addprefix ./_minted-,$(DOCNAMES))
	$(RM) $(TEXDIR)/*.aux $(TEXDIR)/*.log $(TEXDIR)/*.out $(TEXDIR)/*.toc $(TEXDIR)/*.lof $(TEXDIR)/*.lot
	$(RM) $(TEXDIR)/*.bcf $(TEXDIR)/*.run.xml $(TEXDIR)/*.synctex.gz $(TEXDIR)/*.bbl $(TEXDIR)/*.blg

# Add Phony targets to avoid filename conflicts
.PHONY: all clean doc mostlyclean pdf thesis hasil proposal skripsi

