## traing of 3-bit DeiT-T
python3 train.py -c./configs/ours_imagenet_recipe.attn_q.yml --model deit_tiny_distilled_patch16_224 \
your_path/dataset/imagenet-1k/imagenet \
--dataset 'torch/imagenet' \
--epochs 300 \
--batch-size 140 \
--weight-decay 0.05 \
--warmup-lr 1.0e-6 \
--lr 5.47e-4 \
--warmup-epochs 5 \
--mixup 0.0 --cutmix 0.0 \
--aq-enable \
--aq-mode lsq \
--aq-per-channel \
--aq_clip_learnable \
--aq-bitw 3 \
--wq-enable \
--wq-per-channel \
--wq-bitw 3 \
--wq-mode statsq \
--model_type deit \
--quantized \
--pretrained \
--pretrained_initialized \
--use-kd --teacher deit_tiny_distilled_patch16_224 \
--kd_hard_and_soft 1 \
--qk_reparam \
--qk_reparam_type 0 \
--teacher_pretrained \
--output ./outputs/w3a3_deit_t_qkreparam/ \
--visible_gpu '0,1,2,4' \
--world_size '4' \
--tcp_port '36969'

## Finetune 3-bit DeiT-T with CGA 
python3 cga.py -c./configs/ours_imagenet_recipe.attn_q.yml --model deit_tiny_distilled_patch16_224 \
your_path/dataset/imagenet-1k/imagenet \
--dataset 'torch/imagenet' \
--epochs 300 \
--batch-size 140 \
--weight-decay 0.05 \
--warmup-lr 1.0e-6 \
--lr 5.47e-4 \
--warmup-epochs 5 \
--mixup 0.0 --cutmix 0.0 \
--aq-enable \
--aq-mode lsq \
--aq-per-channel \
--aq_clip_learnable \
--aq-bitw 3 \
--wq-enable \
--wq-per-channel \
--wq-bitw 3 \
--wq-mode statsq \
--model_type deit \
--quantized \
--pretrained \
--pretrained_initialized \
--use-kd --teacher deit_tiny_distilled_patch16_224 \
--kd_hard_and_soft 1 \
--qk_reparam \
--qk_reparam_type 1 \
--boundaryRange 0.005 \
--freeze_for_n_epochs 30 \
--teacher_pretrained \
--resume put the model you wish to finetune here \
--output ./outputs/w3a3_deit_t_qkreparam_cga_0005/ \
--visible_gpu '4,5,6,7' \
--world_size '4' \
--tcp_port '36969'

