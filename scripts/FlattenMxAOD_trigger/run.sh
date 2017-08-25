######  Select resources
#PBS -N flatten_trigger
#PBS -l file=100g
#PBS -q long

###### Mail options. For details see man qsub
#PBS -m n
#PBS -M christopher.deutsch@uni-bonn.de

######  Output/Error Files
###(stderr and stdout are merged together to stdout)
#PBS -j oe
##PBS -o /lustre/user/cdeutsch/log
##PBS -e /lustre/user/cdeutsch/log

source /etc/profile
echo
echo "Environment variables..."
echo " User name: $USER"
echo " User home: $HOME"
echo " Queue name: $PBS_O_QUEUE"
echo " Job name: $PBS_JOBNAME"
echo " Job-id: $PBS_JOBID"
echo " Task-id: $PBS_ARRAYID"
echo " Work dir: $PBS_O_WORKDIR"
echo " Submit host: $PBS_O_HOST"
echo " Worker node: $HOSTNAME"
echo " Temp dir: $TMPDIR"
echo " parameters passed: $*"
echo

######
# NOTE: Each computing node has 200G temporary space mounted on /scratch
# NOTE: The path to the /scratch/job_id directory, where the program runs, is saved in
# NOTE: the system wide shell TMPDIR variable. This directory is automatically created when job starts and
# NOTE: removed after job completion. Please avoid creation of any directories/files
# NOTE: in the /scratch directory, as the space under /scratch is automaticaly cleaned by PBS daemon.
#

cd $PBS_O_WORKDIR
echo -e "Here is the program run dir: `pwd` \n"nn

setupATLAS
lsetup root

INPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/trigger_upgrade_v1
OUTPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/trigger_upgrade_flat_v2

python flat.py --truth $INPUT_PREFIX/*Ztautau_new* $OUTPUT_PREFIX/Ztautau
python flat.py --truth $INPUT_PREFIX/*120M180* $OUTPUT_PREFIX/120M180
python flat.py --truth $INPUT_PREFIX/*180M250* $OUTPUT_PREFIX/180M250
python flat.py --truth $INPUT_PREFIX/*250M400* $OUTPUT_PREFIX/250M400
python flat.py --truth $INPUT_PREFIX/*400M600* $OUTPUT_PREFIX/400M600
python flat.py --truth $INPUT_PREFIX/*600M800* $OUTPUT_PREFIX/600M800
python flat.py --truth $INPUT_PREFIX/*800M1000* $OUTPUT_PREFIX/800M1000
python flat.py --truth $INPUT_PREFIX/*1000M1250* $OUTPUT_PREFIX/1000M1250
python flat.py --truth $INPUT_PREFIX/*1250M1500* $OUTPUT_PREFIX/1250M1500
python flat.py --truth $INPUT_PREFIX/*1500M1750* $OUTPUT_PREFIX/1500M1750
python flat.py --truth $INPUT_PREFIX/*1750M2000* $OUTPUT_PREFIX/1750M2000
python flat.py --truth $INPUT_PREFIX/*2000M2250* $OUTPUT_PREFIX/2000M2250
python flat.py --truth $INPUT_PREFIX/*2250M2500* $OUTPUT_PREFIX/2250M2500
python flat.py --truth $INPUT_PREFIX/*2500M2750* $OUTPUT_PREFIX/2500M2750
python flat.py --truth $INPUT_PREFIX/*2750M3000* $OUTPUT_PREFIX/2750M3000
python flat.py --truth $INPUT_PREFIX/*3000M3500* $OUTPUT_PREFIX/3000M3500
python flat.py --truth $INPUT_PREFIX/*3500M4000* $OUTPUT_PREFIX/3500M4000
python flat.py --truth $INPUT_PREFIX/*4000M4500* $OUTPUT_PREFIX/4000M4500
python flat.py --truth $INPUT_PREFIX/*4500M5000* $OUTPUT_PREFIX/4500M5000
python flat.py --truth $INPUT_PREFIX/*5000M* $OUTPUT_PREFIX/5000M

python flat.py $INPUT_PREFIX/*Dijets*JZ0W* $OUTPUT_PREFIX/JZ0W
python flat.py $INPUT_PREFIX/*Dijets*JZ1W* $OUTPUT_PREFIX/JZ1W
python flat.py $INPUT_PREFIX/*Dijets*JZ2W* $OUTPUT_PREFIX/JZ2W
python flat.py $INPUT_PREFIX/*Dijets*JZ3W* $OUTPUT_PREFIX/JZ3W
python flat.py $INPUT_PREFIX/*Dijets*JZ4W* $OUTPUT_PREFIX/JZ4W
python flat.py $INPUT_PREFIX/*Dijets*JZ5W* $OUTPUT_PREFIX/JZ5W
python flat.py $INPUT_PREFIX/*Dijets*JZ6W* $OUTPUT_PREFIX/JZ6W
python flat.py $INPUT_PREFIX/*Dijets*JZ7W* $OUTPUT_PREFIX/JZ7W
