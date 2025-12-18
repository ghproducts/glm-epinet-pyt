from transformers import AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# This file contains feature functions for EpinetWrapper
# The epinet needs access to some base features friomm the model - these are several I have used
def _pool_layer(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    h:    [B, L, H]
    mask: [B, L] or None
    returns: [B, H] pooled over sequence dimension
    """
    if mask is None:
        return h.mean(dim=1)  # [B, H]
    m = mask.unsqueeze(-1).to(h.dtype)          # [B, L, 1]
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-6)




def NT_multi_feature_fn(m: "AutoModelForSequenceClassification", batch):
    inputs = batch.data if hasattr(batch, "data") else batch
    out = m(**inputs, output_hidden_states=True)
    mu = out.logits

    hs = out.hidden_states[1:]  # skip embeddings at index 0
    mask = inputs.get("attention_mask", None)

    pooled = []
    if mask is None:
        for h in hs:
            pooled.append(h.mean(dim=1))  # [B, H]
    else:
        msk = mask.unsqueeze(-1).to(hs[0].dtype)  # [B, L, 1]
        denom = msk.sum(dim=1).clamp_min(1e-6)    # [B, 1]
        for h in hs:
            pooled.append((h * msk).sum(dim=1) / denom)  # [B, H]

    feats = torch.cat(pooled, dim=-1)  # [B, num_layers * H]
    return mu, feats



def NT_tokens_feature_fn(model, batch):
    """
    - Base logits: from the model head (mu)
    - Hidden for MLP epinet: pooled last-layer embeddings
    - Extras for conv prior: one-hot tokens as float [B, L, V] + attention mask
    """
    out = model(**batch, output_hidden_states=True)
    mu = out.logits                               # [B, C]

    input_ids = batch["input_ids"]                # [B, L] (int64)
    mask = batch.get("attention_mask", None)      # [B, L] or None

    #Pooled embeddings for the MLP epinet path
    emb = out.hidden_states[-1]                   # [B, L, H] (float)
    if mask is None:
        hidden = emb.mean(1)                      # [B, H]
    else:
        m = mask.unsqueeze(-1).to(emb.dtype)      # [B, L, 1]
        hidden = (emb * m).sum(1) / m.sum(1).clamp_min(1e-6)

    # One-hot tokens as float for the conv prior path
    vocab_size = model.get_input_embeddings().num_embeddings
    tok_onehot = F.one_hot(input_ids, num_classes=vocab_size)   # [B, L, V], int
    tok_feats = tok_onehot.to(dtype=emb.dtype, device=emb.device)  # match AMP dtype/device

    extras = {
        "tokens": tok_feats,                      # [B, L, V] float – for conv prior
        "attention_mask": mask,                   # [B, L] or None
    }
    return mu, hidden, extras


def NT_first_last_feature_fn(m: AutoModelForSequenceClassification, batch):
    """
    Use pooled embeddings from the first and last hidden layers (not counting the embedding layer).
    Hidden states layout: [0] = embeddings, [1..L] = hidden layers.
    We pool layer 1 and layer L, then concatenate: [B, 2H].
    """
    inputs = batch.data if hasattr(batch, "data") else batch
    out = m(**inputs, output_hidden_states=True)
    mu = out.logits                                    # [B, C]

    hs = out.hidden_states
    first = hs[1]                                      # [B, L, H]
    last = hs[-1]                                      # [B, L, H]

    mask = inputs.get("attention_mask", None)

    pooled_first = _pool_layer(first, mask)            # [B, H]
    pooled_last  = _pool_layer(last,  mask)            # [B, H]

    feats = torch.cat([pooled_first, pooled_last], dim=-1)  # [B, 2H]
    return mu, feats



# in use
def NT_feature_fn(m: AutoModelForSequenceClassification, batch):
    inputs = batch.data if hasattr(batch, "data") else batch

    out = m(**inputs, output_hidden_states=True)
    mu = out.logits                                   # [B, C]
    last = out.hidden_states[-1]                      # [B, L, H]
    mask = inputs.get("attention_mask", None)

    if mask is None:
        pooled = last.mean(dim=1)                     # [B, H]
    else:
        msk = mask.unsqueeze(-1).to(last.dtype)       # [B, L, 1]
        pooled = (last * msk).sum(dim=1) / msk.sum(dim=1).clamp_min(1e-6)
    return mu, pooled


def hyenaDNA_feature_fn(m, batch):
    """
    HyenaDNA analogue of NT_feature_fn:
    - uses last hidden state from the backbone
    - mean-pools (masked if attention_mask is present)
    - returns (mu, pooled) where mu are classifier logits
    """
    inputs = batch.data if hasattr(batch, "data") else batch

    out = m(
        input_ids=inputs.get("input_ids"),
        inputs_embeds=inputs.get("inputs_embeds"),
        labels=inputs.get("labels"),
        output_hidden_states=True,
    )

    mu = out.logits                 # [B, C]
    last = out.hidden_states[-1]    # [B, L, H] (final LN output from HyenaLMBackbone)

    # You can still use attention_mask locally for pooling if your dataset provides it
    mask = inputs.get("attention_mask", None)

    if mask is None:
        pooled = last.mean(dim=1)   # [B, H]
    else:
        msk = mask.unsqueeze(-1).to(last.dtype)   # [B, L, 1]
        pooled = (last * msk).sum(dim=1) / msk.sum(dim=1).clamp_min(1e-6)

    return mu, pooled




