



export PYTHONPATH=./:../:../../

CUDA_VISIBLE_DEVICES=2 python experiments/run_pretrain.py -c coco/ours_base -i demo_run -d bashrun --retrain --newconfig  --saveall