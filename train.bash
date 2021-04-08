srun -p HA_3D --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=expose --kill-on-bad-exit=1
python -u main.py --exp-cfg data/conf_debug.yaml --output-dir OUTPUT_FOLDER --model-type simple --batch_size=8


GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./tools/expose.sh
--exp-cfg data/conf_debug.yaml --output-dir OUTPUT_FOLDER --model-type simple --batch_size=8

GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh HA_3D expose 8 ./tools/expose.sh
--exp-cfg data/conf_debug.yaml --output-dir OUTPUT_FOLDER --model-type simple --batch_size=8

