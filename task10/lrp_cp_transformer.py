import math
import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# Baseline: Gradient × Input
# -----------------------------
def gradient_x_input(model, inputs):
    # device-safe, grads on embeddings
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    embeds = model.get_input_embeddings()(inputs["input_ids"])
    embeds = embeds.clone().detach().requires_grad_(True)
    outputs = model(inputs_embeds=embeds, attention_mask=inputs.get("attention_mask"))
    # use predicted class
    target_idx = int(outputs.logits.argmax(dim=-1).item())
    outputs.logits[:, target_idx].backward()
    if embeds.grad is None:
        raise RuntimeError("No grads on embeddings — ensure inputs_embeds was used.")
    return embeds.grad * embeds, target_idx


def _ln_cp(R_out, x, ln: nn.LayerNorm):
    """
    Conservative propagation through LayerNorm:
    redistribute R_out along features ∝ (x-μ)*γ/σ, then renormalize to conserve.
    x: (1, T, d), R_out: (1, T, d) or (1, d) at a single token (broadcast ok)
    """
    with torch.no_grad():
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        inv_std = torch.rsqrt(var + ln.eps)
        gamma = ln.weight.view(1, 1, -1).detach()
        weights = (x - mu) * (gamma * inv_std)
        denom = weights.abs().sum(dim=-1, keepdim=True) + EPS
        Rx = (weights / denom) * R_out.sum(dim=-1, keepdim=True)
        # exact conservation
        s_in, s_out = R_out.sum(), Rx.sum()
        if torch.isfinite(s_out) and s_out.abs() > 0:
            Rx = Rx * (s_in / s_out)
        return Rx


def lrp_cp_lastlayer_roberta(model, inputs, target_idx=None, layer_idx=-1):
    """
    Conservative Propagation tailored for RoBERTa/BERT encoders — last layer only.
    Returns per-token scalar relevance (numpy array, shape [T]).
    """
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)
        logits = out.logits
        hs_all = out.hidden_states  # list length L+1
        atts_all = out.attentions  # list length L
        probs = logits.softmax(-1)[0]
    if target_idx is None:
        target_idx = int(torch.argmax(probs).item())

    # hidden states
    hs_last = hs_all[-1]  # (1, T, d)
    hs_prev = hs_all[-2] if len(hs_all) >= 2 else hs_all[-1]
    # attentions for chosen layer (default last)
    A = atts_all[layer_idx][0]  # (H, T, T)
    A = A.mean(dim=0)  # (T, T) head-mean

    # relevance at logit -> CLS vector (conservative, proportional to |CLS|)
    R_logit = logits[0, target_idx].detach()
    cls_vec = hs_last[:, 0, :]  # (1, d)
    contrib = cls_vec.abs() + EPS
    R_cls_vec = (contrib / contrib.sum()) * R_logit  # (1, d)

    # Optional: CP through (last) LayerNorm on hs_last at CLS position
    # pick the last LayerNorm module in the model as proxy
    ln_list = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
    if ln_list:
        # expand to (1,1,d) to reuse _ln_cp, then squeeze back
        R_cls_vec = _ln_cp(
            R_cls_vec.unsqueeze(1), hs_last[:, 0, :].unsqueeze(1), ln_list[-1]
        ).squeeze(1)

    # distribute CLS relevance to tokens via attention probabilities (row 0)
    a_cls = A[0]  # (T,)
    R_tok = a_cls * R_cls_vec.abs().sum()  # (T,) raw mass
    R_tok = (R_tok / (R_tok.sum() + EPS)) * float(R_logit)  # conserve

    # Optional: CP through previous LayerNorm to apportion featurewise, then collapse
    if ln_list:
        x = hs_prev
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        inv_std = torch.rsqrt(var + ln_list[-1].eps)
        gamma = ln_list[-1].weight.view(1, 1, -1).detach()
        weights = (x - mu) * (gamma * inv_std)  # (1,T,d)
        denom = weights.abs().sum(dim=-1) + EPS  # (1,T)
        R_feat = (weights / denom.unsqueeze(-1)) * R_tok.view(1, -1, 1)
        R_tok = R_feat.sum(dim=-1).squeeze(0)  # (T,)

    return R_tok.detach().cpu().numpy(), int(target_idx)
