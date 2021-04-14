srun -p HA_3D --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=expose --kill-on-bad-exit=1
python -u main.py --exp-cfg data/conf_debug.yaml --output-dir OUTPUT_FOLDER --model-type simple --batch_size=8



GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh HA_3D expose 8 ./tools/expose.sh
--exp-cfg data/conf_mine.yaml --output-dir OUTPUT_FOLDER --model-type simple --batch_size=16

srun -p HA_3D --gres=gpu:8 -n1 --ntasks-per-node=1 --job-name=data_process --kill-on-bad-exit=1
python generate_vertex --exp-cfg data/conf_debug.yaml  --exp-opts datasets.body.batch_size 1
 --show=False --output-folder OUTPUT_FOLDER --save-params False --save-vis False --save-mesh False --model-type simple