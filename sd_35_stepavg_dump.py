"""
This script hooks into SD3.5’s joint cross-attention layers and saves
the attention data for later analysis. It reuses the attention patching + 
hook system from Wooyeol Baek’s repository

    https://github.com/wooyeolbaek/attention-map-diffusers

Specifically, I import:
- `init_pipeline` to monkeypatch the Stable Diffusion 3.5 pipeline, and
- the global dictionary `attn_maps` which is populated by the registered
  hooks during generation.

What it does:

- Monkeypatches the SD3.5 pipeline with `init_pipeline`
- Collects cond-only attention maps into a global dict `attn_maps`
- Aggregates those maps into compact arrays (NPZ) + metadata (JSON)
- Optionally saves generated images with embedded metadata

What gets saved:

Arrays (NPZ)
- `sum_token_map` (B, H*, W*, T):  
  Average image to text attention per token, summed over all steps & blocks.  
  (Divide by N later to get the mean).  
- `N` (int): number of (step × block) updates included in the sum.  
- `per_block_step_token_mean` (B, Bk, S, T):  
  For each block & step, the average token attention across the whole image.  
  Tells you how much a word is attended over time.  
- `per_block_step_token_max` (B, Bk, S, T):  
  Like above, but takes the strongest pixel instead of the mean.  
  Useful for spotting sharp attention spikes.

Metadata (JSON)
- Prompt(s), negative prompt(s), seed, steps, guidance, model info, etc 
- Tokens
- Block names and sizes  
- Canonical spatial size used for resizing attention maps  
- T_from_model: true token length from the attention tensor (how 
many tokens the model actually attended to)
- special_mask: per batch, per token mask of specials to exclude later

Images saved into `out_dir/images/` with metadata embedded

Usage example:

python sd_35_stepavg_dump.py \
  --prompt "A furry cat beneath a table" \
  --steps 50 \
  --guidance_scale 7.5 \
  --seed 42 \
  --height 512 \
  --width 512 \
  --out OUTDIR

Notes:

- Only the *cond* half of classifier-free guidance is kept.
- Doesn’t save full per step or per block heatmaps.
- All attention maps are resized to the same size so layers can be compared.
- SD3.5 uses three text encoders (CLIP-L, CLIP-G, T5). For simplicity,
  this script only decodes CLIP-L tokens in metadata. That covers the
  first 77 slots, which is fine for short prompts.

"""
from __future__ import annotations

import os
import io
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, PngImagePlugin

# reuse init_pipeline(pipeline) and a global dict `attn_maps` filled by hooks from the repo
from attention_map_diffusers import init_pipeline as _init_pipeline, attn_maps as _HOOK_BUCKET

def init_sd35_hooks(pipe) -> None:
    """
    Patch SD3.5 with custom attention hooks.

    This just wraps `init_pipeline` from the attention-map-diffusers
    repo, so we can capture cross attention maps during generation.
    """
    _init_pipeline(pipe)

# configuration object that controls how script saves outputs
@dataclass
class DumpConfig:
    """
    Settings for saving outputs.

    - canonical_size: force all maps to this size (H,W), else pick max
    - half_precision: store arrays as float16 to save space
    - save_image: whether to save generated images
    - image_format: "png" or "jpg"
    - embed_image_metadata: include metadata inside image file
    """
    canonical_size: Optional[Tuple[int, int]] = None  # specify a target resolution (H*, W*) for resizing all attention maps, if None, the script picks the largest size found among blocks
    half_precision: bool = True  # store float16 to save space
    save_image: bool = True
    image_format: str = "png"  # png or jpg
    embed_image_metadata: bool = True

def run_and_save_stepavg_maps(
    pipe,
    prompts: List[str],
    negative_prompts: Optional[List[str]] = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.5,
    seed: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    out_dir: str = "out_stepavg",
    cfg: DumpConfig = DumpConfig(),
):
    """
    Run SD3.5, grab attention maps, and save them for later analysis.
    - Generates image(s) with the given prompt, seed, steps, etc.
    - Hooks attention layers to collect cond-only maps
    - Aggregates maps into arrays
    - Saves arrays (NPZ) + metadata (JSON)
    - Optionally saves generated image(s) with metadata baked in

    Returns:
    - List of PIL images from the generation
    """
    # set up output dirs
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # if given a list of seeds, build one generator per image (overkill but flexible)
    generator = None
    if seed is not None:
        if isinstance(seed, int):
            generator = torch.Generator(device=pipe._execution_device)
            generator.manual_seed(seed)
        else:
            # list of seeds per image
            generator = [torch.Generator(device=pipe._execution_device).manual_seed(int(s)) for s in seed]

    pipe_kwargs = dict(
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    if height is not None: pipe_kwargs["height"] = height
    if width is not None:  pipe_kwargs["width"] = width

    if negative_prompts is None:
        negative_prompts = [""] * len(prompts)  # if not provided, fills with ""

    _HOOK_BUCKET.clear()  # clear out global hook bucket before running otherwise leftover maps from previous run pollutes results

    # run the pipeline
    images = pipe(
        prompts,
        negative_prompt=negative_prompts if any(negative_prompts) else None,
        **pipe_kwargs,
    ).images

    # save images
    if cfg.save_image:
        _save_images_with_meta(
            images,
            img_dir,
            image_format=cfg.image_format,
            embed=cfg.embed_image_metadata,
            meta=dict(
                prompt=prompts,
                negative_prompt=negative_prompts,
                steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                height=height if height is not None else images[0].height,
                width=width if width is not None else images[0].width,
                model_id=getattr(pipe, "_internal_dict", {}).get("_name_or_path", None)
                        or getattr(pipe, "model_id", None),
                scheduler=pipe.scheduler.__class__.__name__ if hasattr(pipe, "scheduler") else None,
                timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ),
        )

    # build arrays from hooks, _build_arrays_and_metadata defined below
    arrays, meta = _build_arrays_and_metadata(
        hook_bucket=_HOOK_BUCKET,
        pipe=pipe,
        prompts=prompts,
        H_out=images[0].height,
        W_out=images[0].width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        canonical_size=cfg.canonical_size,
        half=cfg.half_precision,
    )

    _save_npz_and_meta(out_dir, arrays, meta)  # save npz and metadata

    _HOOK_BUCKET.clear()  # free memory

    return images


# build arrays

@torch.no_grad()
def _build_arrays_and_metadata(
    hook_bucket: Dict[int, Dict[str, torch.Tensor]],
    pipe,
    prompts: List[str],
    H_out: int,
    W_out: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: Optional[int],
    canonical_size: Optional[Tuple[int, int]],
    half: bool,
):
    """
    Take raw attention maps from hooks and build compact arrays + metadata.

    - Collect attention maps across steps and blocks
    - Average over heads, resize to a common spatial size
    - Sum over all steps/blocks -> `sum_token_map`
    - Compute per-block/step token stats -> mean + max
    - Gather metadata: tokens, prompt info, block names, sizes, etc.

    Returns:
    - arrays: dict of NumPy arrays (sum_token_map, per_block_step_token_mean, ...)
    - meta: dict of run metadata (prompts, seed, steps, model info, tokens, ...)
    """
    # if nothing is in hook_bucket, it raises an error
    if not hook_bucket:
        raise RuntimeError("No attention maps captured. Did you call init_sd35_hooks(pipe) before generation?")

    timesteps = sorted(hook_bucket.keys())  # all captured timesteps
    layer_names = sorted(next(iter(hook_bucket.values())).keys(), key=_layer_sort_key)   # all captured blocks
    S = len(timesteps)
    Bk = len(layer_names)  # number of blocks

    # infer batch and token dims and gather spatial sizes per block from the first timestep
    example = hook_bucket[timesteps[0]][layer_names[0]]  # (B_total, Hheads, H, W, T)
    B_total, Hheads, H0, W0, T = example.shape

    # if classifier free guidance duplicated batch, keep the cond half (which is the 2nd half) otherwise keep everything
    if B_total % 2 == 0 and guidance_scale is not None and guidance_scale > 1.0:
        B = B_total // 2
        cond_slice = slice(B, B_total)   # second half = cond
    else:
        B = B_total
        cond_slice = slice(0, B_total)

    # resize everything to the largest block’s size to compare maps from different blocks
    if canonical_size is None:
        # pick the size of the largest map seen
        maxH, maxW = 0, 0
        for ln in layer_names:
            tens = hook_bucket[timesteps[0]][ln]
            maxH = max(maxH, tens.shape[2])
            maxW = max(maxW, tens.shape[3])
        canonical_size = (maxH, maxW)
    Hc, Wc = canonical_size

    # allocate arrays for heatmaps
    dtype_out = torch.float16 if half else torch.float32
    device = example.device

    sum_token_map = torch.zeros((B, Hc, Wc, T), dtype=dtype_out, device=device)
    N_updates = 0

    # allocate arrays for time series stats
    per_block_step_token_mean = torch.zeros((B, Bk, S, T), dtype=dtype_out, device=device)
    per_block_step_token_max  = torch.zeros((B, Bk, S, T), dtype=dtype_out, device=device)

    # track each block's spatial size
    block_sizes = {}
    for ln in layer_names:
        tens = hook_bucket[timesteps[0]][ln]
        block_sizes[ln] = [int(tens.shape[2]), int(tens.shape[3])]

    eps = 1e-8

    for si, ts in enumerate(timesteps):
        layers = hook_bucket[ts]
        for bi, ln in enumerate(layer_names):
            attn = layers[ln]
            attn = attn[cond_slice]

            # head mean over the heads
            attn = attn.mean(dim=1)

            # resize to canonical size
            Bcur, Hcur, Wcur, Tcur = attn.shape
            assert Tcur == T
            # reshape
            attn_bt = attn.permute(0, 3, 1, 2).reshape(Bcur * Tcur, 1, Hcur, Wcur)
            attn_bt_rs = F.interpolate(attn_bt, size=(Hc, Wc), mode="bilinear", align_corners=False)
            attn_rs = attn_bt_rs.reshape(Bcur, Tcur, Hc, Wc).permute(0, 2, 3, 1)
            sum_token_map += attn_rs.to(dtype_out)
            N_updates += 1

            # renorm over tokens per pixel
            attn_ren = attn / (attn.sum(dim=-1, keepdim=True) + eps)
            # spatial stats
            mean_tok = attn_ren.mean(dim=(1, 2))
            max_tok  = attn_ren.amax(dim=(1, 2))
            per_block_step_token_mean[:, bi, si, :] = mean_tok.to(dtype_out)
            per_block_step_token_max[:,  bi, si, :] = max_tok.to(dtype_out)

    # detach tensors then move to cpu then convert to numpy
    sum_token_map_np = sum_token_map.detach().cpu().numpy()
    per_block_step_token_mean_np = per_block_step_token_mean.detach().cpu().numpy()
    per_block_step_token_max_np  = per_block_step_token_max.detach().cpu().numpy()

    arrays = dict(
        sum_token_map=sum_token_map_np,
        N=np.int32(N_updates),
        per_block_step_token_mean=per_block_step_token_mean_np,
        per_block_step_token_max=per_block_step_token_max_np,
    )

    # metadata
    # SD3.5 uses three tokenizers, for simplicity i use the clip tokenizer
    tok = getattr(pipe, "tokenizer", None)
    attn_tokens = []
    special_masks = []
    if tok is not None:
        tok_out = tok(prompts)
        input_ids = tok_out["input_ids"]
        if input_ids and isinstance(input_ids[0], list):  # handles batching
            id_batches = input_ids
        else:
            id_batches = [input_ids]
        for ids in id_batches:  # if multiple prompts
            toks = tok.convert_ids_to_tokens(ids)  # list of decoded tokens for each prompt in the batch
            attn_tokens.append(toks)
            special_masks.append(_heuristic_special_mask(toks))
    else:
        attn_tokens = [["<UNK>"] * T for _ in range(B)]  # if no tokenizer is found, fill with <UNK> placeholders and mark no tokens as special
        special_masks = [[False] * T for _ in range(B)]  # mark which tokens are special (padding, BOS/EOS, etc)

    meta: Dict[str, object] = dict(
        prompt=prompts,
        negative_prompt=getattr(pipe, "_last_negative_prompt", None),
        seed=seed,
        steps=num_inference_steps,
        guidance_scale=guidance_scale,
        model_id=getattr(pipe, "_internal_dict", {}).get("_name_or_path", None)
                 or getattr(pipe, "model_id", None),
        scheduler=pipe.scheduler.__class__.__name__ if hasattr(pipe, "scheduler") else None,
        image_size=[H_out, W_out],
        canonical_size=[Hc, Wc],
        batch_size=B,
        num_steps=S,
        num_blocks=Bk,
        blocks=layer_names,
        block_latent_sizes=block_sizes,
        tokens=attn_tokens,  # per-batch list of tokens
        T_from_model=T,
        special_mask=special_masks,  # per-batch, per-token
        timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )

    return arrays, meta

def _layer_sort_key(name: str) -> Tuple[int, str]:
    # sort like transformer_blocks.{idx}.attn ...
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "transformer_blocks" and i + 1 < len(parts):
            try:
                return (int(parts[i + 1]), name)
            except ValueError:
                return (1_000_000, name)
    return (1_000_000, name)

# builds a mask that flags which tokens are special
def _heuristic_special_mask(tokens: List[str]) -> List[bool]:
    """
    Mark which tokens are special.

    Returns a list of booleans aligned with the token list.
    True = special token, False = normal word piece.
    """
    specials = {
        "<|startoftext|>", "<|endoftext|>", "<s>", "</s>", "<pad>", "[BOS]", "[EOS]", "[PAD]",
        "<BOS>", "<EOS>", "<SEP>", "</w>", "<unk>", "<UNK>"
    }
    mask = []
    for t in tokens:
        is_spec = (t in specials) or t.startswith("Ġ<|") or t.endswith("|>")
        mask.append(bool(is_spec))
    return mask

def _save_npz_and_meta(out_dir: str, arrays: Dict[str, np.ndarray], meta: Dict[str, object]):
    # NPZ
    npz_path = os.path.join(out_dir, "stepavg_token_maps_v1.npz")
    np.savez_compressed(npz_path, **arrays)

    # JSON
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _save_images_with_meta(
    images: List[Image.Image],
    img_dir: str,
    image_format: str = "png",
    embed: bool = True,
    meta: Optional[Dict[str, object]] = None,
):
    assert image_format in {"png", "jpg"}
    # saves metadata in image, needs this if moving images around in directory and image gets separated from json

    # build per image metadata
    def _to_pnginfo(meta: Dict[str, object]) -> PngImagePlugin.PngInfo:
        info = PngImagePlugin.PngInfo()
        for k, v in meta.items():
            try:
                info.add_text(str(k), json.dumps(v))
            except Exception:
                info.add_text(str(k), str(v))
        return info

    # save each image
    for idx, im in enumerate(images):
        fname = f"seed{meta.get('seed','na')}-{im.width}x{im.height}-idx{idx}.{image_format}"
        fpath = os.path.join(img_dir, fname)
        if image_format == "png" and embed and meta is not None:
            info = _to_pnginfo(meta)
            im.save(fpath, format="PNG", pnginfo=info)
        else:
            im.save(fpath, format="JPEG" if image_format == "jpg" else None, quality=95)


# command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SD3.5 cond-only attention dump (step-averaged)")
    parser.add_argument("--prompt", type=str, default="A furry cat beneath a table", help="The prompt or prompts to guide the image generation")
    parser.add_argument("--neg", type=str, default="", help="The prompt or prompts not to guide the image generation. The default is an empty string")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024, help="Image height (pixels)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (pixels)")
    parser.add_argument("--out", type=str, default="out_sd35_stepavg")
    args = parser.parse_args()

    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16).to("cuda")
    init_sd35_hooks(pipe)

    run_and_save_stepavg_maps(
        pipe,
        [args.prompt],
        negative_prompts=[args.neg],
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        height=args.height,
        width=args.width,
        out_dir=args.out,
    )