INPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/StreamTriggerIDDev/v01
OUTPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/StreamTriggerIDDev_flat/v01

python flat.py --truth $INPUT_PREFIX/*Gammatautau* $OUTPUT_PREFIX/Gammatautau
python flat.py $INPUT_PREFIX/*DIJETS*JZ1W* $OUTPUT_PREFIX/JZ1W
python flat.py $INPUT_PREFIX/*DIJETS*JZ2W* $OUTPUT_PREFIX/JZ2W
python flat.py $INPUT_PREFIX/*DIJETS*JZ3W* $OUTPUT_PREFIX/JZ3W
python flat.py $INPUT_PREFIX/*DIJETS*JZ4W* $OUTPUT_PREFIX/JZ4W
python flat.py $INPUT_PREFIX/*DIJETS*JZ5W* $OUTPUT_PREFIX/JZ5W
python flat.py $INPUT_PREFIX/*DIJETS*JZ6W* $OUTPUT_PREFIX/JZ6W
