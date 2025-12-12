# Use pdf output
$pdf_mode = 1;

# Put *all* build artifacts in ./build
$out_dir = 'build';
$aux_dir = 'build';

# Nice pdflatex command (optional)
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 -file-line-error %O %S';