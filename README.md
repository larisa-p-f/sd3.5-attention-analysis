# sd3.5-attention-analysis
This project extends [Wooyeol Baek’s `attention-map-diffusers`](https://github.com/wooyeolbaek/attention-map-diffusers) to collect and analyse cross-attention maps from **Stable Diffusion 3.5 (SD3.5)** generations. 

It consists of two main scripts in the base folder, and a dependency folder `attention_map_diffusers` containing the patched pipeline code.

---
## Sample Results
Below is an analysis of the prompt: "A dusty clock on top of a table"
<img src="assets/seed42-512x512-idx0.png" width="256">

1. Spatial Attention (Heatmaps)
We can visualise exactly where the model focuses for specific words. Notice how the attention for "clock" is localised to the animal, while "table" focuses on the surrounding structure.
Generated ImageWord: "Cat"Word: "Table"<img src="assets/generated_image.png" width="256"><img src="assets/heatmap_cat.png" width="256"><img src="assets/heatmap_table.png" width="256">2. Temporal Attention (Curves)

By analyzing attention across timesteps (X-axis), we can see when the model constructs different concepts.<div align="center"><img src="assets/words_mean_block12.png" width="600"><p><em>Mean attention per word over diffusion steps (Block 12)</em></p></div>

## Repository Structure

```
.
├── attention_map_diffusers/        # Submodule adapted from wooyeolbaek/attention-map-diffusers
│   ├── __init__.py
│   ├── modules.py
│   ├── README.md
│   ├── setup.py
│   └── utils.py
├── sd_35_stepavg_dump.py           # Script to generate and dump attention maps
├── analyse_stepavg.py              # Script to analyse dumped attention maps
```

---

## Components

### 1. `attention_map_diffusers/`
This folder comes directly from [wooyeolbaek/attention-map-diffusers](https://github.com/wooyeolbaek/attention-map-diffusers) and is required for hooking into SD3.5 pipelines. 
Key contents:
- `__init__.py` – initialises the package and exposes main functions (e.g. `init_pipeline`, `attn_maps` dictionary).
- `modules.py` – implements attention patching and hooks inside the Diffusers library.
- `utils.py` – helper utilities for managing hooks, attention reshaping, etc.
- `setup.py` – installation metadata if you want to install it as a package.
- `README.md` – upstream documentation from the original repository.

You don’t need to modify these files; they provide the low-level functionality used by the scripts.

---

### 2. `sd_35_stepavg_dump.py`
Hooks into **Stable Diffusion 3.5’s** joint cross-attention layers, records the attention data during image generation, and saves aggregated outputs for later analysis.

What it does:
- Monkeypatches the pipeline with `init_pipeline` from `attention_map_diffusers`.
- Captures **conditional attention maps** (classifier-free guidance cond branch).
- Aggregates across steps and blocks into compact NumPy arrays (`.npz`) and metadata (`.json`).
- Optionally saves generated images with embedded metadata.

Saved files:
- `stepavg_token_maps_v1.npz` (arrays of aggregated attention)
- `metadata.json` (prompts, tokens, seeds, block info, etc.)
- Generated images (if enabled), stored under `images/`

Usage:
```bash
python sd_35_stepavg_dump.py   --prompt "A furry cat beneath a table"   --steps 50   --guidance_scale 7.5   --seed 42   --height 512   --width 512   --out out_dir
```

---

### 3. `analyse_stepavg.py`
Loads the outputs of `sd_35_stepavg_dump.py` (`.npz` + `.json`) and produces visualisations and tables.

Outputs include:
- **Word heatmaps**: `analysis/*/heatmaps/words/`
- **Phrase heatmaps**: `analysis/*/heatmaps/phrases/` (if phrases are specified)
- **Curves**: per-word attention curves across timesteps per block
- **Tables**: CSVs of word/phrase scores (AUC and step-averaged values)

Usage:
```bash
python analyse_stepavg.py   --root out_dir   --topk 12   --phrases "in front of|on top of"   --weights attention
```

---

## Workflow

1. **Generate attention data** 
   Run `sd_35_stepavg_dump.py` with your prompt(s). 
   This creates an output directory with arrays, metadata, and optionally images.

2. **Analyse results** 
   Run `analyse_stepavg.py` on that directory to reconstruct heatmaps, curves, and tables.

3. **Explore** 
   Look inside the `analysis/` subfolders for plots and CSVs.

---

## Requirements
- Python 3.9+
- [🤗 Diffusers](https://github.com/huggingface/diffusers)
- PyTorch (with CUDA if using GPU)
- NumPy, Matplotlib, Pillow

Install the dependencies with:
```bash
pip install torch diffusers numpy matplotlib pillow
```

---

## Notes
- Only the **conditional branch** of classifier-free guidance is stored.
- Attention maps are resized to a canonical resolution for comparability across layers.
- Token decoding in metadata currently uses **CLIP-L tokens** only (77 slots), which is sufficient for short prompts.
- `analyse_stepavg.py` also reconstructs **T5 tokens** using the pipeline’s tokenizer for richer analysis.
