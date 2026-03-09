#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=eval_fastclip
#SBATCH --partition=gpu
#SBATCH --output=%x_%j.log

source ~/.bashrc
conda activate fastclip

master_addr=127.0.0.1
export MASTER_ADDR=$master_addr
export MASTER_PORT=12805

export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

python src/training/main.py \
    --pretrained 'openai' \
    --imagenet-val '/home/jovyan/two-tower-retrieval-datavol-1/data/imageNet/val' \
    --zeroshot-frequency 1 \
    --batch-size 512 \
    --epochs 0 \
    --workers 6 \
    --model ViT-B-32 \
    --name eval_pretrained_clip_vit_b32_image_net \
    --seed 2024