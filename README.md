# BiCo: Composing Concepts from Images and Videos via Concept-prompt Binding

[![project_page](https://img.shields.io/badge/Project-Page-green)](https://refkxh.github.io/BiCo_Webpage/) &nbsp; [![arxiv](https://img.shields.io/badge/arXiv-2512.09824-b31b1b.svg)](https://arxiv.org/abs/2512.09824/) &nbsp; [![license](https://img.shields.io/github/license/modelscope/DiffSynth-Studio.svg)](https://github.com/modelscope/DiffSynth-Studio/blob/master/LICENSE) &nbsp; [![stars](https://img.shields.io/github/stars/refkxh/BiCo.svg?style=social&label=Stars)](https://github.com/refkxh/BiCo)

Official implementation of the paper **Composing Concepts from Images and Videos via Concept-prompt Binding** [[Link](https://arxiv.org/abs/2512.09824/)].

## 👁️ Introduction

<div align="center"><img src="images/overview.jpg" width="1000"/></div>

We introduce **Bind & Compose (BiCo)**, a **one-shot** method that enables **flexible visual concept composition** by binding visual concepts with the corresponding prompt tokens and composing the target prompt with bound tokens from various sources.

## 🔥 News

- 10 Dec. 2025: Initial release of BiCo 🎉🎉🎉.

## 🛠️ Installation

We recommend installing BiCo in a conda environment with Python 3.12.

```bash
git clone https://github.com/refkxh/BiCo.git  
cd BiCo
pip install -e .
```

## 🔑 Quick Inference

This section provides step-by-step instructions for running multi-concept inference using pretrained BiCo models.

### Step 1: Download Pretrained Models

Download the pretrained concept adapter models from our [OneDrive link (TBD)](TBD).

### Step 2: Place Models in the Correct Directory

After downloading, place all model files (`.safetensors` files) into the `BiCo/models/` directory:

```bash
BiCo/
├── models/
│   ├── akita_img_epoch-4.safetensors
│   ├── play_game_video_1_epoch-4.safetensors
│   └── ... (other model files)
```

### Step 3: Modify the Inference Script

Open `bico/validate_multi_concept/Wan2.1-T2V-1.3B.py` and follow these steps:

#### 3.1 Configure Model Paths and MoE Settings

In the `adapters_state_dict_paths` list, specify the paths to the models you want to use:

```python
adapters_state_dict_paths = [
    "models/akita_img_epoch-4.safetensors",        # First concept: Akita dog
    "models/play_game_video_1_epoch-4.safetensors" # Second concept: Playing game
]
```

**Important Notes:**

- The order of models in this list determines which concept adapter corresponds to which position in the prompt.
- You can use any number of models by adding more paths to the list.

**Configure MoE Settings:**

You must also configure `init_concept_adapters_moe` in the `WanVideoPipeline.from_pretrained()` call. This parameter must match the order of models in `adapters_state_dict_paths`:

- **`False`** for image-trained models (models trained on images)
- **`True`** for video-trained models (models trained on videos)

```python
pipe = WanVideoPipeline.from_pretrained(
    # ... other parameters ...
    init_concept_adapters=True,
    num_concept_adapters=len(adapters_state_dict_paths),
    init_concept_adapters_moe=[False, True],  # Must match the order in adapters_state_dict_paths
    # False for akita_img_epoch-4.safetensors (image-trained)
    # True for play_game_video_1_epoch-4.safetensors (video-trained)
)
```

**Example:**

- If you have 2 models: first is image-trained (`False`), second is video-trained (`True`) → `[False, True]`
- If you have 3 models: all image-trained → `[False, False, False]`
- If you have 3 models: first two are video-trained, third is image-trained → `[True, True, False]`

#### 3.2 Configure the Prompt with Placeholders

In the `video = pipe()` function, you need to use `#` placeholders to indicate where each concept adapter should be activated. The number of `#` placeholders must match the number of models in `adapters_state_dict_paths`.

**Example: Combining "Akita" and "Play Game" concepts**

```python
video = pipe(
    prompt=[
        "# #",  # Two placeholders for two models (separated by space)
        "A happy Akita dog with its tongue out,",  # Description for first concept (Akita)
        "in a red plaid shirt and black headphones raises its paws excitedly while holding a gaming controller, deeply engaged in a game in a cozy living room setting."  # Description for second concept (Play game)
    ],
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    seed=1,
)
```

**Note**: You can add creative prompts directly in the placeholder string to enhance generation. For example, `"# # in a kitchen."` 

**Prompt Structure Explanation:**

- **First element** (`"# #"`): Contains placeholders - one `#` per model, separated by spaces. For 2 models, use `"# #"`; for 3 models, use `"# # #"`, etc. You can also add additional prompt text in this element for creative generation.
- **Subsequent elements**: Each element after the placeholders corresponds to a concept description. The order matches the order in `adapters_state_dict_paths`:
  - First description → First model (akita_img_epoch-4.safetensors)
  - Second description → Second model (play_game_video_1_epoch-4.safetensors)

**Finding the Right Prompts:**

- Each pretrained model comes with a prompt description (provided in the OneDrive link).
- You can select and combine prompts from different models to create your desired composition.
- The prompts should describe the specific concept that the model was trained on.

#### 3.3 Configure Output Path

The generated video will be saved to `DiffSynth-Studio/test_results/`:

```python
save_video(video, "test_results/akita_play_game_1_1.0.mp4", fps=15, quality=5)
```

Make sure the `test_results/` directory exists, or it will be created automatically.

### Step 4: Run Inference

Execute the inference script:

```bash
cd BiCo
python bico/validate_multi_concept/Wan2.1-T2V-1.3B.py
```

### Tips for Multi-Concept Composition

1. **Model Order**: The order of models in `adapters_state_dict_paths` must match the order of descriptions in the prompt list and the order in `init_concept_adapters_moe`.
2. **MoE Configuration**: Always set `init_concept_adapters_moe` correctly: `False` for image-trained models, `True` for video-trained models. The list order must match `adapters_state_dict_paths`.
3. **Placeholder Count**: Always use the same number of `#` placeholders as the number of models.
4. **Prompt Selection**: Choose prompts that best describe each concept. You can find recommended prompts for each model in the OneDrive documentation.
5. **Creative Prompts in Placeholders**: You can add additional creative prompts directly in the placeholder string (e.g., `"# # in a futuristic cyberpunk city"`) to enhance generation without affecting concept binding.
6. **Experiment**: Try different combinations of concepts and prompts to achieve your desired results!

## 💻 Training Your Own BiCo Models

This section provides step-by-step instructions for training your own BiCo concept adapters on custom images or videos.

### Data Preparation

#### For Video Data

**Step 1: Trim Longer Videos to 81 Frames**

First, use `data/cutto_81frames_imageio.py` to trim your video that is longer than 81 frames to exactly 81 frames:

```python
# Edit data/cutto_81frames_imageio.py
input_video_file = 'data/videos/your_video_original.mp4'
output_video_file = 'data/videos/your_video.mp4'

# Run the script
python data/cutto_81frames_imageio.py
```

This will create a trimmed video with exactly 81 frames.

**Step 2: Extract Video Frames**

Next, use `data/extract_frames.py` to extract all frames from the trimmed video:

```python
# Edit data/extract_frames.py
args.video_file = "data/videos/your_video.mp4"
args.output_dir = "data/videos/your_video"  # Frames will be saved here

# Run the script
python data/extract_frames.py --video_file data/videos/your_video.mp4 --output_dir data/videos/your_video
```

This will extract all frames and save them as JPG images in the specified output directory.

**Step 3: Prompt Generation and Augmentation**

Now, use `bico/prompt_aug.py` for prompt generation and augmentation. The process involves three steps:

1. **Extract concepts from video**: Run `extract_concept_video()`
2. **Generate per-frame prompts** (for spatial information): Run `generate_prompts_video_per_frame()`
3. **Generate video-level prompts** (for temporal information): Run `generate_prompts_video()`

**Configure the paths in `bico/prompt_aug.py`:**

```python
def extract_concept_video():
    dataset_path = "data"
    video_dirs = ["videos"]  # Modify to your video directory


def generate_prompts_video_per_frame():
    seed = 100
    max_num_frames = 81
    dataset_path = "data"
    video_dirs = ["videos"]  # Modify to your video directory


def generate_prompts_video():
    seed = 100
    num_prompts = 60
    dataset_path = "data"
    video_dirs = ["videos"]  # Modify to your video directory
```

**Run the prompt generation:**

```python
# In bico/prompt_aug.py, uncomment the following lines
if __name__ == "__main__":
    extract_concept_video()              # Step 1: Extract concepts
    generate_prompts_video_per_frame()  # Step 2: Generate per-frame prompts (spatial)
    generate_prompts_video()            # Step 3: Generate video prompts (temporal)

# Run the script
python bico/prompt_aug.py
```

This will generate:
- `your_video.json`: Concept extraction results
- `your_video_spatial_prompts.csv`: Per-frame prompts for spatial information (used in Stage 1 training)
- `your_video_prompts.csv`: Video-level prompts for temporal information (used in Stage 2 training)

#### For Image Data

**Prompt Generation and Augmentation**

For images, use `bico/prompt_aug.py` with the following functions:

1. **Extract concepts from images**: Run `extract_concept_img()`
2. **Generate image prompts**: Run `generate_prompts_img()`

**Configure the paths in `bico/prompt_aug.py`:**

```python
def extract_concept_img():
    dataset_path = "data"
    img_dirs = ["images"]  # Modify to your image directory


def generate_prompts_img():
    seed = 100
    num_prompts = 60
    dataset_path = "data"
    img_dirs = ["images"]  # Modify to your image directory
```

**Run the prompt generation:**

```python
# In bico/prompt_aug.py, uncomment the following lines
if __name__ == "__main__":
    extract_concept_img()    # Step 1: Extract concepts
    generate_prompts_img()   # Step 2: Generate prompts

# Run the script
python bico/prompt_aug.py
```

This will generate:

- `your_image.json`: Concept extraction results
- `your_image_prompts.csv`: Augmented prompts for training

### Training

#### Training on Images

Edit `bico/multi_concept/Wan2.1-T2V-1.3B.sh` and modify the training command:

```bash
export NCCL_P2P_DISABLE=1

# For images training
accelerate launch --config_file bico/multi_concept/training_config.yaml \
  bico/train.py \
  --dataset_base_path data/images \
  --dataset_metadata_path data/images/your_image_prompts.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 8 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.concept_adapter_dict." \
  --output_path "./models/train/your_concept_name" \
  --trainable_models "concept_adapter_dict"
```

**Key Parameters:**

- `--dataset_base_path`: Base path to your image directory
- `--dataset_metadata_path`: Path to the CSV file generated by `generate_prompts_img()`
- `--height` / `--width`: Image dimensions
- `--dataset_repeat`: Number of times to repeat the dataset per epoch
- `--num_epochs`: Number of training epochs
- `--output_path`: Where to save the trained model checkpoints

#### Training on Videos (Two-Stage Training)

Video training requires two stages: first training on video frames (spatial information), then training on the full video (temporal information).

**Stage 1: Train on Video Frames (Spatial Information)**

This stage learns spatial features from individual frames:

```bash
# Stage 1: Train on video frames
accelerate launch --config_file bico/multi_concept/training_config.yaml \
  bico/train.py \
  --dataset_base_path data/videos \
  --dataset_metadata_path data/videos/your_video_spatial_prompts.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 8 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.concept_adapter_dict." \
  --output_path "./models/train/your_video_stage1" \
  --trainable_models "concept_adapter_dict"
```

**Stage 2: Train on Full Video (Temporal Information)**

This stage loads the Stage 1 checkpoint and learns temporal information from the full video:

```bash
# Stage 2: Train on full video (load Stage 1 checkpoint)
accelerate launch --config_file bico/multi_concept/training_config.yaml \
  bico/train.py \
  --dataset_base_path data/videos \
  --dataset_metadata_path data/videos/your_video_prompts.csv \
  --height 480 \
  --width 832 \
  --dataset_repeat 8 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.concept_adapter_dict." \
  --output_path "./models/train/your_video_stage2" \
  --trainable_models "concept_adapter_dict" \
  --concept_adapter_load_path "models/train/your_video_stage1/epoch-4.safetensors" \
  --concept_adapter_moe
```

**Important Notes:**

- `--concept_adapter_load_path`: Path to the Stage 1 checkpoint
- `--concept_adapter_moe`: Enable MoE (Mixture of Experts) mode for video training
- `--dataset_metadata_path` in Stage 1 uses `*_spatial_prompts.csv` (per-frame prompts)
- `--dataset_metadata_path` in Stage 2 uses `*_prompts.csv` (video-level prompts)

## 📜 Citation

If you find BiCo useful for your research, please consider citing:

```
@misc{kong2025composingconceptsimagesvideos,
      title={Composing Concepts from Images and Videos via Concept-prompt Binding}, 
      author={Xianghao Kong and Zeyu Zhang and Yuwei Guo and Zhuoran Zhao and Songchun Zhang and Anyi Rao},
      year={2025},
      eprint={2512.09824},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.09824}, 
}
```

## 🙏 Acknowledgements

This project is built upon [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). We sincerely thank the authors for their open-source contributions.

## 📄 License

This project is licensed under the Apache 2.0 License. We claim no rights over the your generated contents, granting you the freedom to use them while ensuring that your usage complies with the provisions of this license. You are fully accountable for your use of the models, which must not involve sharing any content that violates applicable laws, causes harm to individuals or groups, disseminates personal information intended for harm, spreads misinformation, or targets vulnerable populations. See the [LICENSE](./LICENSE) file for details.
