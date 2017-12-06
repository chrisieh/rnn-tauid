INPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_MC16A
OUTPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/StreamTauIDDev_MC16A_flat

#python flat.py --truth $INPUT_PREFIX/*Gammatautau* $OUTPUT_PREFIX/Gammatautau
python flat.py $INPUT_PREFIX/*DIJETS*JZ1W* $OUTPUT_PREFIX/JZ1W.root
python flat.py $INPUT_PREFIX/*DIJETS*JZ2W* $OUTPUT_PREFIX/JZ2W.root
python flat.py $INPUT_PREFIX/*DIJETS*JZ3W* $OUTPUT_PREFIX/JZ3W.root
python flat.py $INPUT_PREFIX/*DIJETS*JZ4W* $OUTPUT_PREFIX/JZ4W.root
python flat.py $INPUT_PREFIX/*DIJETS*JZ5W* $OUTPUT_PREFIX/JZ5W.root
python flat.py $INPUT_PREFIX/*DIJETS*JZ6W* $OUTPUT_PREFIX/JZ6W.root
