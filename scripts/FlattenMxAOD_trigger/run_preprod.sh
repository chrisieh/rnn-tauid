######  Select resources
#PBS -N flatten_sample
#PBS -l file=100g
#PBS -q medium
#PBS -t 0-6

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

INPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/StreamTriggerIDDev/v02
OUTPUT_PREFIX=/lustre/atlas/group/higgs/cdeutsch/StreamTriggerIDDev_flat/v02

if [[ $PBS_ARRAYID -eq 0 ]]; then
    python flat.py --truth $INPUT_PREFIX/*Gammatautau* $OUTPUT_PREFIX/Gammatautau
fi
if [[ $PBS_ARRAYID -eq 1 ]]; then
    python flat.py $INPUT_PREFIX/*DIJETS*JZ1W* $OUTPUT_PREFIX/JZ1W
fi
if [[ $PBS_ARRAYID -eq 2 ]]; then
    python flat.py $INPUT_PREFIX/*DIJETS*JZ2W* $OUTPUT_PREFIX/JZ2W
fi
if [[ $PBS_ARRAYID -eq 3 ]]; then
    python flat.py $INPUT_PREFIX/*DIJETS*JZ3W* $OUTPUT_PREFIX/JZ3W
fi
if [[ $PBS_ARRAYID -eq 4 ]]; then
    python flat.py $INPUT_PREFIX/*DIJETS*JZ4W* $OUTPUT_PREFIX/JZ4W
fi
if [[ $PBS_ARRAYID -eq 5 ]]; then
    python flat.py $INPUT_PREFIX/*DIJETS*JZ5W* $OUTPUT_PREFIX/JZ5W
fi
if [[ $PBS_ARRAYID -eq 6 ]]; then
    python flat.py $INPUT_PREFIX/*DIJETS*JZ6W* $OUTPUT_PREFIX/JZ6W
fi
