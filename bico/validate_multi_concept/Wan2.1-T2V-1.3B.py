import os

import torch

from diffsynth import load_state_dict, save_video
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def add_suffix_for_state_dict_keys(state_dict, i):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("global_concept_adapter."):
            new_key = (
                f"global_concept_adapter_{i}." + key[len("global_concept_adapter.") :]
            )
        elif key.startswith("per_block_concept_adapters."):
            new_key = (
                f"per_block_concept_adapters_{i}."
                + key[len("per_block_concept_adapters.") :]
            )
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


adapters_state_dict_paths = [
    "models/akita_img_epoch-4.safetensors",
    "models/play_game_video_1_epoch-4.safetensors"
]

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="diffusion_pytorch_model*.safetensors",
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
        ),
        ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B",
            origin_file_pattern="Wan2.1_VAE.pth",
        ),
    ],
    init_concept_adapters=True,
    num_concept_adapters=len(adapters_state_dict_paths),
    init_concept_adapters_moe=[False, True],
)

state_dict = {}
for i, path in enumerate(adapters_state_dict_paths):
    state_dict_i = load_state_dict(path)
    if len(adapters_state_dict_paths) > 1:
        state_dict_i = add_suffix_for_state_dict_keys(state_dict_i, i)
    state_dict.update(state_dict_i)

missing_keys, unexpected_keys = pipe.concept_adapter_dict.load_state_dict(
    state_dict, strict=False
)
print("missing_keys:", missing_keys)
print("unexpected_keys:", unexpected_keys)
pipe.enable_vram_management()

video = pipe(
     prompt=[
        "# #",
        "A happy Akita dog with its tongue out,",
        "in a red plaid shirt and black headphones raises its paws excitedly while holding a gaming controller, deeply engaged in a game in a cozy living room setting."
    ],
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=1,
)
save_video(video, "test_results/akita_play_game_1_1.0.mp4", fps=15, quality=5)
exit()

