python3 eval.py -c./configs/swin_t_imagenet.attn_q.yml --model swin_t \
your_path/dataset/imagenet-1k/imagenet \
--dataset 'torch/imagenet' \
--epochs 300 \
--batch-size 128 \
--weight-decay 0.0 \
--warmup-lr 1.0e-6 \
--lr 2.0e-4 \
--warmup-epochs 0 \
--aq-enable \
--aq-mode lsq \
--aq-per-channel \
--aq_clip_learnable \
--aq-bitw 3 \
--wq-enable \
--wq-per-channel \
--wq-bitw 3 \
--wq-mode statsq \
--model_type swin \
--teacher_type swin \
--quantized \
--pretrained \
--pretrained_initialized \
--use-kd --teacher swin_t \
--kd_hard_and_soft 1 \
--qk_reparam \
--qk_reparam_type 0 \
--teacher_pretrained \
--resume your_path/model_saved/swin_t/w3a3/w3a3_swin_t_qkr_cga.pth.tar \
--output ./outputs/w3a3_swin_t_qkreparam/ \
--visible_gpu '0,1,2,4' \
--world_size '4' \
--tcp_port '12345'