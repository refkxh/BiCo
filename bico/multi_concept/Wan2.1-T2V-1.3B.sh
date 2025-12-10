export NCCL_P2P_DISABLE=1

# For images training
accelerate launch --config_file bico/multi_concept/training_config.yaml \
  bico/train.py \
  --dataset_base_path data/images \
  --dataset_metadata_path data/images/akita_prompts.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 8 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.concept_adapter_dict." \
  --output_path "./models/train/akita_pmptaug60_rp8_ep5_img" \
  --trainable_models "concept_adapter_dict" 



# For videos training

## video training stage 1
accelerate launch --config_file bico/multi_concept/training_config.yaml \
  bico/train.py \
  --dataset_base_path data/videos \
  --dataset_metadata_path data/videos/play_game_1_spatial_prompts.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 8 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.concept_adapter_dict." \
  --output_path "./models/train/play_game_1_pmptaug60_rp8_ep5_imgframes" \
  --trainable_models "concept_adapter_dict" 

## video training stage 2
accelerate launch --config_file bico/multi_concept/training_config.yaml \
  bico/train.py \
  --dataset_base_path data/videos \
  --dataset_metadata_path data/videos/play_game_1_prompts.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 8 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.concept_adapter_dict." \
  --output_path "./models/train/play_game_1_pmptaug60_rp8_ep5_video_moe" \
  --trainable_models "concept_adapter_dict" \
  --concept_adapter_load_path "models/train/play_game_1_pmptaug60_rp8_ep5_imgframes/epoch-4.safetensors" \
  --concept_adapter_moe



