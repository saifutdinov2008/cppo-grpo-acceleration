# Build the PDF report.
#
#   make            # report.pdf via latexmk (preferred)
#   make pdflatex   # fall back to a manual pdflatex/bibtex cycle
#   make clean
#
# Requires a TeX distribution with algorithm2e, booktabs and listings.
.PHONY: all pdflatex clean

all: report.pdf

report.pdf: report.tex references.bib generated_tables.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex

pdflatex:
	pdflatex -interaction=nonstopmode -halt-on-error report.tex
	bibtex report || true
	pdflatex -interaction=nonstopmode -halt-on-error report.tex
	pdflatex -interaction=nonstopmode -halt-on-error report.tex

clean:
	latexmk -C 2>/dev/null || true
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk
