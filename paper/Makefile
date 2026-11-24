PDF = main.pdf
TEX = pdflatex -interaction=nonstopmode -halt-on-error

all: $(PDF)

$(PDF): main.tex references.bib $(wildcard sections/*.tex) $(wildcard figures/*.pdf)
	$(TEX) main.tex
	bibtex main
	$(TEX) main.tex
	$(TEX) main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.lof main.lot

.PHONY: all clean
