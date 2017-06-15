INPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/trigger_upgrade_v1
OUTPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/trigger_upgrade_flat_v1

# python flat.py --truth $INPUT_PREFIX/*Ztautau_new* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*120M180* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*180M250* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*250M400* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*400M600* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*600M800* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*800M1000* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*1000M1250* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*1250M1500* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*1500M1750* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*1750M2000* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*2000M2250* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*2250M2500* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*2500M2750* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*2750M3000* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*3000M3500* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*3500M4000* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*4000M4500* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*4500M5000* $OUTPUT_PREFIX/Ztautau
# python flat.py --truth $INPUT_PREFIX/*5000M* $OUTPUT_PREFIX/Ztautau

python flat.py $INPUT_PREFIX/*Dijets*JZ0W* $OUTPUT_PREFIX/JZ0W
# python flat.py $INPUT_PREFIX/*Dijets*JZ1W* $OUTPUT_PREFIX/JZ1W
# python flat.py $INPUT_PREFIX/*Dijets*JZ2W* $OUTPUT_PREFIX/JZ2W
# python flat.py $INPUT_PREFIX/*Dijets*JZ3W* $OUTPUT_PREFIX/JZ3W
# python flat.py $INPUT_PREFIX/*Dijets*JZ4W* $OUTPUT_PREFIX/JZ4W
# python flat.py $INPUT_PREFIX/*Dijets*JZ5W* $OUTPUT_PREFIX/JZ5W
# python flat.py $INPUT_PREFIX/*Dijets*JZ6W* $OUTPUT_PREFIX/JZ6W
# python flat.py $INPUT_PREFIX/*Dijets*JZ7W* $OUTPUT_PREFIX/JZ7W
