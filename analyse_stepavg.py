"""
Analyse SD3.5 step-averaged attention dumps (cond-only)

Supports both CLIP and T5 token spans for analysis.

This script takes the saved outputs (`stepavg_token_maps_v1.npz` +
`metadata.json`) and rebuilds word heatmaps (and phrases if specified).
It also makes some block/step plots and tables to see where
different words got the most attention.

Inputs:

OUTDIR/
  stepavg_token_maps_v1.npz
  metadata.json

Outputs:
  heatmaps/words/*.png (visual heatmap showing where the image attends to each word in the prompt)
  heatmaps/phrases/*.png (visual heatmap for multi-word phrases (if --phrases))
  curves/words_mean_block{b}.png (line plots of word attention over timesteps in block b, top-K words only)
  curves/phrases_mean_block{b}.png (line plots of phrase attention over timesteps in block b (if --phrases))
  tables/words_auc.csv (table of per word attention scores in each block, ranked by area under curve strength across timesteps)
  tables/words_stepavg.csv (table of per word average attention strength across timesteps in each block)
  tables/phrases_auc.csv (aggregated over phrases)
  tables/phrases_stepavg.csv (aggregated over phrases)

Usage example:

python analyse_stepavg.py \
        --root my_outputs \
        --topk 12 \
        --phrases "in front of|on top of" \
        --weights attention

Notes:

- Block rankings use area under curve and step average.
"""
from __future__ import annotations

import os
import json
import csv
import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusion3Pipeline
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def _ensure_dirs(*paths: str) -> None:
    """ Make sure directories exist before saving."""
    for p in paths:
        os.makedirs(p, exist_ok=True)

def _load_npz_meta(root: str):
    """Load the NPZ arrays and the JSON metadata together."""
    npz_path = os.path.join(root, "stepavg_token_maps_v1.npz")
    meta_path = os.path.join(root, "metadata.json")
    f = np.load(npz_path)
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return f, meta

def _token_clean(piece: str) -> str:
    """Remove junk markers like </w>, Ġ, or ▁."""
    p = piece
    # drop end word markers
    if p.endswith("</w>"):
        p = p[:-4]
    # drop start word markers
    while p and (p[0] in ("Ġ", "▁")):
        p = p[1:]
    return p

def _build_word_mapping(tokens: list[str], special_mask: list[bool]) -> tuple[list[str], list[int]]:
    """
    Combine subtokens into words while ignoring special tokens.
    - Special tokens are skipped and close any open word.
    - Ġ or ▁ mark the beginning of a new word.
    - </w> ends the current word.
    - Otherwise, tokens are added onto the current word.

    Returns:
    words : list[str]
        List of finished words.
    word_ids : list[int]
        Which word each token belongs to (-1 if none).
    """
    words: list[str] = []  # final words
    word_ids: list[int] = [-1] * len(tokens)  # -1 = not part of a word

    # buffer that stores the current word being built
    cur_pieces: list[str] = []
    cur_indices: list[int] = []

    def flush():
        """Close the current word and save it into `words`."""
        if not cur_pieces:
            return
        # join all pieces to form one word
        words.append("".join(cur_pieces))
        wid = len(words) - 1

        # update the word_ids for all tokens that contributed
        for i in cur_indices:
            word_ids[i] = wid

        # reset buffers
        cur_pieces.clear()
        cur_indices.clear()

    # go through each token and its special_mask flag
    for i, (tok, is_spec) in enumerate(zip(tokens, special_mask)):
        if is_spec:

            # special tokens are ignored and close any open word
            flush()
            continue

        start_mark = tok.startswith("Ġ") or tok.startswith("▁")
        end_mark = tok.endswith("</w>")
        core = _token_clean(tok)  # remove markers

        if start_mark:  # close previous word
            flush()

        # add to the current word
        cur_pieces.append(core)
        cur_indices.append(i)

        if end_mark:  # close the word
            flush()

    flush()  # close any word still in progress
    return words, word_ids

def _per_pixel_normalise(stack: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    For each pixel, make the word maps sum to 1. Convert raw values into probabilities over words.
    """
    # input stack has shape (H, W, K)
    # for each pixel (h, w), there are K channels, one per word
    z = stack.sum(axis=-1, keepdims=True)  # sum the total attention mass across all words at each pixel
    return stack/(z+eps)  # divide each word’s value by the total at that pixel

def _save_heatmap_png(path: str, arr_hw: np.ndarray, title: str | None = None):
    plt.figure(figsize=(4, 4))  # create new figure 4x4 inch in size
    plt.imshow(arr_hw, origin="upper")  # origin="upper" means the origin is the top left corner
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def _auc_trapz(y: np.ndarray) -> float:
    "Measures how much attention a word received across all timesteps."
    # y is a 1d array of length S containing attention values across steps
    x = np.arange(len(y), dtype=float)  # x is the timestep indices
    return float(np.trapz(y, x))  # returns the area under the curve

def reconstruct_word_heatmaps(sum_token_map: np.ndarray, N: int, tokens: List[str], special_mask: List[bool]):
    """
    Take token-level maps and merge them into word-level maps.
    - Average the token maps (divide by N)
    - Merge sub-tokens into words
    - Add up token maps into word maps
    - Renormalise per pixel
    """
    H, W, _ = sum_token_map.shape  # unpack spatial and token dimensions
    avg_tok = (sum_token_map / float(N)).astype(np.float32)  # sum map -> average map

    L = len(tokens)  # number of decoded tokens in the prompt

    # build word mapping for tokoens to words and ignore special tokens
    words, word_ids = _build_word_mapping(tokens, special_mask)

    # alocate stack of per word heatmaps
    K = len(words)
    word_stack = np.zeros((H, W, K), dtype=np.float32)

    # for each token up to L add its heatmap to the corresponding slot
    for t in range(L):
        wid = word_ids[t]
        if wid >= 0:
            word_stack[:, :, wid] += avg_tok[:, :, t]

    # renormalise so that at each pixel the sum over all words is 1
    word_stack = _per_pixel_normalise(word_stack)
    return words, word_stack

def build_phrases(word_stack: np.ndarray, words: List[str], phrases: List[List[str]], weights: str = "attention"):
    """
    Merge word maps into phrase maps.
    - phrases, where each phrase is given as a list of words (e.g. [["in", "front", "of"], ["on", "top", "of"]])
    - weights="attention": scale by total mass
    - weights="uniform": treat all words equally
    """
    H, W, _ = word_stack.shape  # unpack dimensions
    P = len(phrases)  # number of phrases to construct
    phrase_stack = np.zeros((H, W, P), dtype=np.float32)  # allocate ouput array
    phrase_names = [" ".join(p) for p in phrases]  # join the word list w spaces so it builds readable names

    # calcilate total attention mass for each word across the image
    mass = word_stack.sum(axis=(0, 1))  # the shape is (K,)

    # loop over each phrase
    for pi, group in enumerate(phrases):

        # find the indices of words that appear in this phrase
        idxs = [i for i, w in enumerate(words) if w in group]
        if not idxs:
            continue  # skip if none of the phrases were present
        
        # decide the weights for each word
        if weights == "uniform":  # each word will contribute equally
            alpha = np.ones(len(idxs), dtype=np.float32) / float(len(idxs))
        
        else:  # scale word contributions by their global attention mass
            m = mass[idxs]
            s = float(m.sum()) if float(m.sum()) > 0 else 1.0
            alpha = (m / s).astype(np.float32)

        # weighted sum of word maps to build phrase maps
        for a, wid in zip(alpha, idxs):
            phrase_stack[:, :, pi] += a * word_stack[:, :, wid]

    return phrase_names, phrase_stack

def curves_from_means(per_block_step_token_mean: np.ndarray, tokens: List[str], special_mask: List[bool]):
    _, Bk, S, _ = per_block_step_token_mean.shape  # unpack in put shape
    X = per_block_step_token_mean[0]  # (Bk, S, T) (B=1 assumed)

    L = len(tokens)  # number of decoded tokens, excluding the pad tail

    # build mapping from tokens to words
    words, word_ids = _build_word_mapping(tokens, special_mask)
    K = len(words)

    word_means = np.zeros((Bk, S, K), dtype=np.float32)  # allocate array for word lvevl means

    # loop over tokens and aff their contributions into the correct word
    for t in range(L):
        wid = word_ids[t]
        if wid >= 0:  # skip any special tokens
            word_means[:, :, wid] += X[:, :, t]

    return words, word_means

def compute_block_scores(word_curves: np.ndarray, method: str = "auc") -> np.ndarray:
    """
    Compute either AUC or step average for each word.
        Scoring method:
        - "auc": total accumulated attention.
        - "mean": simple average across timesteps (average attention level).
    """
    Bk, _, K = word_curves.shape
    scores = np.zeros((Bk, K), dtype=np.float32)
    if method == "auc":
        for b in range(Bk):
            for k in range(K):
                scores[b, k] = _auc_trapz(word_curves[b, :, k])
    else:
        scores = word_curves.mean(axis=1)
    return scores

def analyse_branch(name: str, token_stack: np.ndarray, N: int,
                   tokens: List[str], special_mask: List[bool],
                   per_block_means: np.ndarray,
                   out_root: str, phrases: List[List[str]],
                   weights: str, topk: int):
    """
    Run full analysis for one encoder (CLIP or T5).
    """
    d_heat = os.path.join(out_root, "heatmaps", "words")
    d_phrs = os.path.join(out_root, "heatmaps", "phrases")
    d_curv = os.path.join(out_root, "curves")
    d_tabs = os.path.join(out_root, "tables")
    _ensure_dirs(d_heat, d_phrs, d_curv, d_tabs)

    # word heatmaps
    words, word_stack = reconstruct_word_heatmaps(token_stack, N, tokens, special_mask)
    mass = word_stack.sum(axis=(0, 1))
    top_idx = np.argsort(-mass)[: max(1, min(topk, len(words)))]

    for i in top_idx:
        path = os.path.join(d_heat, f"{i:03d}-{words[i]}.png")
        _save_heatmap_png(path, word_stack[:, :, i], title=words[i])

    # phrases
    if phrases:
        phrase_names, phrase_stack = build_phrases(word_stack, words, phrases, weights=weights)
        for i, pname in enumerate(phrase_names):
            path = os.path.join(d_phrs, f"{i:03d}-{pname}.png")
            _save_heatmap_png(path, phrase_stack[:, :, i], title=pname)

    # curves
    _, word_means = curves_from_means(per_block_means, tokens, special_mask)
    Bk, _, K = word_means.shape

    for b in range(Bk):
        plt.figure(figsize=(5, 3))  # more compact
        for i in top_idx:
            plt.plot(word_means[b, :, i], label=words[i])
        plt.xlabel("Timestep")
        plt.ylabel("Mean attention")
        plt.title(f"Block {b}")
        plt.legend(
        fontsize=7,
        ncol=1,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)  # to the right of plot
    )
        plt.tight_layout()
        plt.savefig(os.path.join(d_curv, f"words_mean_block{b}.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # tables
    auc_scores = compute_block_scores(word_means, method="auc")
    avg_scores = compute_block_scores(word_means, method="mean")
    with open(os.path.join(d_tabs, "words_auc.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["block"] + words)
        for b in range(Bk):
            w.writerow([b] + [f"{auc_scores[b,i]:.6f}" for i in range(K)])
    with open(os.path.join(d_tabs, "words_stepavg.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["block"] + words)
        for b in range(Bk):
            w.writerow([b] + [f"{avg_scores[b,i]:.6f}" for i in range(K)])

def main():
    ap = argparse.ArgumentParser(description="Analyse SD3.5 step-averaged attention NPZ")
    ap.add_argument("--root", required=True, help="OUTDIR with NPZ + metadata.json")
    ap.add_argument("--topk", type=int, default=12, help="Top-K words to export/plot (by attention mass)")
    ap.add_argument("--phrases", type=str, default="", help="Bar-separated phrases, words space-separated. e.g. 'in front of|on top of'")
    ap.add_argument("--weights", choices=["attention", "uniform"], default="attention", help="Phrase weighting scheme")
    args = ap.parse_args()

    f, meta = _load_npz_meta(args.root)
    out_root = os.path.join(args.root, "analysis")
    _ensure_dirs(out_root)

    # reconstruct word heatmaps
    sum_token_map = f["sum_token_map"][0]
    N = int(f["N"]) or 1
    per_block_all = f["per_block_step_token_mean"][0]

    # prepare phrases
    phrases: List[List[str]] = []
    if args.phrases.strip():
        for part in args.phrases.split("|"):
            words_in = [w.strip() for w in part.split() if w.strip()]
            if words_in: phrases.append(words_in)
    
    # split into clip / t5 spans
    clip_tok_stack = sum_token_map[:, :, :77]
    t5_tok_stack = sum_token_map[:, :, 77:333]
    per_block_clip = per_block_all[:, :, :77][None]   # add batch dim
    per_block_t5 = per_block_all[:, :, 77:333][None]

    # tokenizer
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
    tok_t5 = pipe.tokenizer_3

    # clip tokens are already in metadata
    tokens_clip = meta["tokens"][0]
    special_mask_clip = meta["special_mask"][0]

    # t5 tokens
    prompt = meta["prompt"][0]
    tok_out_t5 = tok_t5(prompt, return_tensors="pt")
    ids_t5 = tok_out_t5["input_ids"][0].tolist()
    tokens_t5 = tok_t5.convert_ids_to_tokens(ids_t5)
    special_mask_t5 = [tok in ["<pad>", "<s>", "</s>"] for tok in tokens_t5]

    # run both analyses
    analyse_branch("clip", clip_tok_stack, N, tokens_clip, special_mask_clip,
                   per_block_clip, os.path.join(out_root, "clip"),
                   phrases, args.weights, args.topk)

    analyse_branch("t5", t5_tok_stack, N, tokens_t5, special_mask_t5,
                   per_block_t5, os.path.join(out_root, "t5"),
                   phrases, args.weights, args.topk)

    print("Done. Outputs in:", out_root)


if __name__ == "__main__":
    main()
