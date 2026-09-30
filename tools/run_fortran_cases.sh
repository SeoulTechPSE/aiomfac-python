#!/bin/sh
# Regenerate the reference outputs/dumps of the stage-3 validation cases with the instrumented Fortran build.
# usage: tools/run_fortran_cases.sh <instrumented_executable> <Auxiliary_dir> <inputs_dir> <out_dir>
#   (generate the inputs first with:  python tools/make_cases.py <inputs_dir>)
set -e
EXE=$(realpath "$1"); AUX=$(realpath "$2"); IN=$(realpath "$3"); OUT=$(realpath -m "$4")
WORK=$(mktemp -d); mkdir -p "$WORK/Inputfiles" "$WORK/Outputfiles" "$OUT/outputs" "$OUT/debug_terms"
cp -r "$AUX" "$WORK/Auxiliary"; cp "$EXE" "$WORK/aiomfac.out"; cd "$WORK"
for f in "$IN"/input_*.txt; do
  n=$(basename "$f" .txt | sed 's/input_//')
  cp "$f" Inputfiles/; rm -f Outputfiles/debug_terms.txt
  ./aiomfac.out "./Inputfiles/input_$n.txt" > "run_$n.log" 2>&1
  mv Outputfiles/debug_terms.txt "$OUT/debug_terms/debug_terms_$n.txt"
  cp "Outputfiles/AIOMFAC_output_$n.txt" "$OUT/outputs/"
  echo "case $n: $(grep -h 'final error indicator' "run_$n.log" | tail -1 | sed 's/.*indicator/indicator/')"
done
