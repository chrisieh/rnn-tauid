INPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/MC16A_PFO
OUTPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/MC16A_PFO_flat

python flat.py --truth $INPUT_PREFIX/*Gammatautau* $OUTPUT_PREFIX/Gammatautau
