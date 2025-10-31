# Tools
RM = rm -fr
LATEX = xelatex -shell-escape -interaction=nonstopmode -output-directory=$(TEXDIR)
BIBTEX = bibtex

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

# Individual document targets
%.pdf: $(TEXDIR)/%.tex
	$(LATEX) $<
	$(BIBTEX) $(TEXDIR)/$*
	$(LATEX) $<
	$(LATEX) $<
	mv $(TEXDIR)/$*.pdf $@

# Same for your thesis.pdf target
thesis.pdf: $(TEXDIR)/main.tex
	$(LATEX) $<
	$(BIBTEX) $(TEXDIR)/main
	$(LATEX) $<
	$(LATEX) $<
	mv $(TEXDIR)/main.pdf $@

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

