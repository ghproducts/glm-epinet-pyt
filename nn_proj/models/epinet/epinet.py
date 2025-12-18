from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple, Type, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from datasets import load_dataset, ClassLabel, Dataset, DatasetDict
from nn_proj.common.utils import compute_uncertainty

# functions defining the epinet and components 
# https://arxiv.org/abs/2107.08924
# https://github.com/google-deepmind/enn


@dataclass
class EpinetConfig:
    # standard epinet
    num_classes: int
    index_dim: int = 30                  # Dz
    hidden_sizes: Iterable[int] = (50, ) # epinet MLP widths
    prior_scale: float = 1             # weight on frozen prior head
    stop_grad_features: bool = True      # detach hidden before epinet
    concat_index: bool = True

    # cov epinet defaults
    fixed_conv_channels: Tuple[int, int, int] = (4, 8, 8)
    fixed_conv_kernels:  Tuple[int, int, int] = (3, 3, 3)
    fixed_conv_strides:  Tuple[int, int, int] = (2, 2, 1)
    fixed_conv_dropout: float = 0.0      
    conv_prior_scale:   float = 1.0      # paper found 1.0 works on ImageNet
    include_inputs: bool = False         # set to True if using conv prior
    vocab_size: int = 0 
    conv_embed_dim: int = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# Epinet components
# ---------------------------------------------------------------------

class GaussianIndexer(nn.Module):
    """z ~ N(0, I) with shape [Dz], shared across batch."""
    def __init__(self, index_dim: int):
        super().__init__()
        self.index_dim = index_dim
    @torch.no_grad()
    def forward(self, device=None, dtype=None) -> torch.Tensor:
        return torch.randn(self.index_dim, device=device, dtype=dtype or torch.float32)


def _mlp(in_dim: int, hidden: Iterable[int], out_dim: int) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class ProjectedMLP(nn.Module):
    def __init__(self, num_classes: int, index_dim: int, hidden: Iterable[int], concat_index: bool = True, trainable: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.index_dim = index_dim
        self.hidden = tuple(hidden)
        self.concat_index = concat_index
        self.trainable = trainable
        self.core: Optional[nn.Sequential] = None  # lazy init

    def _build(self, in_dim: int):
        # _mlp(in_dim, hidden, out_dim) must have NO activation on the final layer
        self.core = _mlp(in_dim, self.hidden, self.num_classes * self.index_dim)
        if not self.trainable:
            for p in self.core.parameters():                       
                p.requires_grad = False                            

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Expect x: [B, Din], z: [Dz]  (shared across batch)
        if x.dim() != 2 or z.dim() != 1:
            raise ValueError("Expected x:[B,Din], z:[Dz].")
        if z.shape[0] != self.index_dim:
            raise ValueError(f"z dim {z.shape[0]} != index_dim {self.index_dim}")

        if self.concat_index:
            B = x.shape[0]
            z_cat = z.unsqueeze(0).expand(B, -1)   # [B, Dz]
            h = torch.cat([z_cat, x], dim=-1)      # [B, Din+Dz]
        else:
            h = x

        if self.core is None:
            self._build(h.shape[-1])
            self.core.to(h.device)

        out = self.core(h)                         # [B, C*Dz]
        B = x.shape[0]
        m = out.view(B, self.num_classes, self.index_dim)  # [B, C, Dz]
        return torch.einsum('bcd,d->bc', m, z) 



class FixedConv1DPriorEnsemble(nn.Module):
    """
    an ensemble of Dz small conv nets on the raw input sequence.
    Returns: sum_i p_i(x) * z_i, shape [B, C].
    All weights are frozen (fixed prior).
    """
    def __init__(self, cfg: EpinetConfig, vocab_size: int, embed_dim: int = 8):
        super().__init__()
        self.cfg = cfg
        self.C = cfg.num_classes
        self.Dz = cfg.index_dim

        # Frozen token embedding so we can convolve over channels
        self.embed = nn.Embedding(vocab_size, embed_dim)
        for p in self.embed.parameters():
            p.requires_grad = False

        ch1, ch2, ch3 = cfg.fixed_conv_channels
        k1, k2, k3 = cfg.fixed_conv_kernels
        s1, s2, s3 = cfg.fixed_conv_strides

        # Build Dz independent tiny CNNs
        self.nets = nn.ModuleList()
        for _ in range(self.Dz):
            net = nn.Sequential(
                nn.Conv1d(embed_dim, ch1, kernel_size=k1, stride=s1),
                nn.ReLU(),
                nn.Conv1d(ch1, ch2, kernel_size=k2, stride=s2),
                nn.ReLU(),
                nn.Conv1d(ch2, ch3, kernel_size=k3, stride=s3),
                nn.ReLU(),
            )
            head = nn.Linear(ch3, self.C)

            # freeze everything
            for p in net.parameters():
                p.requires_grad = False
            for p in head.parameters():
                p.requires_grad = False

            self.nets.append(nn.ModuleDict({"net": net, "head": head}))

        self.dropout = nn.Dropout(cfg.fixed_conv_dropout)

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L], z: [Dz]
        B, L = input_ids.shape
        x = self.embed(input_ids)                # [B, L, E]
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1) # zero-out pads

        x = x.transpose(1, 2).contiguous()       # [B, E, L]
        x = self.dropout(x)

        out = 0.0
        for i in range(self.Dz):
            h = self.nets[i]["net"](x)           # [B, ch3, L’]
            pooled = h.mean(dim=-1)              # [B, ch3]  (global average pool)
            pi = self.nets[i]["head"](pooled)    # [B, C]
            out = out + pi * z[i]
        return out


# ---------------------------------------------------------------------
# Epinet definitinos
# ---------------------------------------------------------------------

class MLPEpinetWithPrior(nn.Module):
    """
    Basic epinet with mlp and matching mlp prior

    epinet(hidden, inputs, z) = train([hidden,(+flat inputs)], z) + prior_scale * prior(...)
    - hidden: [B, D_hidden]
    - inputs: Tensor or None (only used if include_inputs=True)
    - z:      [B, Dz]
    """
    def __init__(self, cfg: EpinetConfig):
        super().__init__()
        self.cfg = cfg
        C, Dz = cfg.num_classes, cfg.index_dim
        self.train_head = ProjectedMLP(C, Dz, cfg.hidden_sizes, trainable=True, concat_index=cfg.concat_index)
        self.prior_head = ProjectedMLP(C, Dz, cfg.hidden_sizes, trainable=False, concat_index=cfg.concat_index)
        self.indexer = GaussianIndexer(Dz)

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return x.view(B, -1)

    def forward(self,
                hidden: torch.Tensor,
                inputs: Optional[torch.Tensor],
                z: torch.Tensor) -> torch.Tensor:
        if hidden.dim() != 2:
            raise ValueError("hidden must be [B, D].")
        if self.cfg.include_inputs and inputs is not None:
            epi_in = torch.cat([hidden, self._flatten(inputs).to(hidden)], dim=-1)
        else:
            epi_in = hidden
        train = self.train_head(epi_in, z)
        prior = self.prior_head(epi_in, z)
        return train + self.cfg.prior_scale * prior



class MLPEpinetWithConvPrior(nn.Module):
    def __init__(self, cfg: EpinetConfig):
        super().__init__()
        self.cfg = cfg
        C, Dz = cfg.num_classes, cfg.index_dim

        self.train_head = ProjectedMLP(C, Dz, cfg.hidden_sizes, trainable=True,  concat_index=cfg.concat_index)
        self.prior_head = ProjectedMLP(C, Dz, cfg.hidden_sizes, trainable=False, concat_index=cfg.concat_index)
        self.conv_prior = FixedConv1DPriorEnsemble(cfg, vocab_size=cfg.vocab_size, embed_dim=cfg.conv_embed_dim)
        self.indexer = GaussianIndexer(Dz)

    def forward(self, hidden: torch.Tensor, inputs: Any, z: torch.Tensor) -> torch.Tensor:
        # hidden: [B, D]
        train = self.train_head(hidden, z)
        prior = self.prior_head(hidden, z)

        # inputs can be a dict from HF: {"input_ids":..., "attention_mask":...}
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids", None)
            attention_mask = inputs.get("attention_mask", None)
        else:
            input_ids, attention_mask = None, None

        if input_ids is None:
            raise ValueError("Conv prior requires inputs to provide input_ids (HF-style dict).")

        convp = self.conv_prior(input_ids, attention_mask, z)
        return train + self.cfg.prior_scale * prior + self.cfg.conv_prior_scale * convp


# ---------------------------------------------------------------------
# epinet wrapper
# ---------------------------------------------------------------------

class EpinetWrapper(nn.Module):
    """
    Combine a regular model with an epinet.

      - base_model: nn.Module
      - feature_fn: (base_model, batch) -> (mu:[B,C], hidden:[B,D])  # tensors
      - cfg: EpinetConfig

    Call:
      logits = wrapper(batch, n_index_samples=K, return_all=False/True)

    Notes:
      - We infer B/device for z *from the returned `hidden`* (no assumptions on `batch` type).
      - If cfg.include_inputs=True, raw `batch` must be a Tensor (else raw inputs are ignored).
    """
    def __init__(self,
                 base_model: nn.Module,
                 feature_fn: Callable[[nn.Module, Any], Tuple[torch.Tensor, torch.Tensor]],
                 cfg: EpinetConfig,
                 epinet: Optional[Type[nn.Module]] = None
                 ):
        super().__init__()
        self.base = base_model
        self.feature_fn = feature_fn
        self.cfg = cfg
        self.epinet = epinet(cfg) if epinet is not None else MLPEpinetWithPrior(cfg) #adjust this to change up the epinet


    def forward(self,
                batch: Any,
                n_index_samples: int = 1,
                return_all: bool = False,
                z: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1) Feature extraction (user-defined): must return tensors
        feat_out = self.feature_fn(self.base, batch)    # [B, C], [B, D]
        if len(feat_out) == 2:
            mu, hidden = feat_out
            extras = None
        else:
            mu, hidden, extras = feat_out[:3]
        hidden_for_epi = hidden.detach() if self.cfg.stop_grad_features else hidden

        # 2) Sample z using hidden's batch size & device
        B = hidden.shape[0]
        device = hidden.device
        z_dtype = hidden.dtype if hidden.is_floating_point() else next(self.parameters()).dtype

        if z is None:
            if n_index_samples == 1:
                z = self.epinet.indexer(device=device, dtype=z_dtype)             # [B, Dz]
            else:
                # [S, Dz]
                z = torch.stack(
                    [self.epinet.indexer(device=device, dtype=z_dtype) for _ in range(n_index_samples)],
                    dim=0
                )

        # 3) Inputs to epinet
        inputs_for_epinet: Optional[Any] = None
        if extras is not None:
            inputs_for_epinet = extras          
        elif self.cfg.include_inputs:
            inputs_for_epinet = batch

        # 4) Single vs multi z
        if z.dim() == 1:
            epi = self.epinet(hidden_for_epi, inputs_for_epinet, z)                  # [B, C]
            return mu + epi

        outs = []
        for s in range(z.shape[0]):
            epi_s = self.epinet(hidden_for_epi, inputs_for_epinet, z[s])             # [B, C]
            outs.append(mu + epi_s)
        stacked = torch.stack(outs, dim=0)                                           # [S, B, C]
        return stacked if return_all else stacked.mean(0)
    

# ---------------------------
# wrapper for huggingface compatibility
# ---------------------------
class HFEpinetSeqClassifier(nn.Module):
    def __init__(self, wrapper: EpinetWrapper, k_train=1, k_eval=8):
        super().__init__()
        self.wrapper = wrapper
        self.k_train = k_train
        self.k_eval = k_eval
    
    @property
    def cfg(self): 
        return self.wrapper.cfg
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        K = self.k_train if self.training else self.k_eval
        if self.training and K > 1 and labels is not None:
            logits_all = self.wrapper(batch, n_index_samples=K, return_all=True) # shape [K, B, C]
            K, B, C = logits_all.shape
            #average the per-sample CE
            loss = F.cross_entropy(
                logits_all.reshape(K * B, C), 
                labels.unsqueeze(0).expand(K, B).reshape(-1), 
                reduction="none"
            ).mean()
            logits = logits_all.mean(dim=0)  # [B, C]
        else:
            logits = self.wrapper(batch, n_index_samples=K)
            loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}
    

# ---------------------------------------------------------------------
# Prediction and uncertainty computation for epinet model
# ---------------------------------------------------------------------
@torch.no_grad()
def predict(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    collator: Any,
    k_samples: int = 16,
    batch_size: int = 32,
    outfile: Optional[str] = "val_uncertainty.csv",
    use_amp: bool = False,
    uncertainty_method: str = None,
) -> List[Dict[str, Any]]:
    """
    One base forward + K epinet head forwards per batch (return_all=True),
    computes uncertainty from the full stack, and optionally saves outputs.
    """
    # handle metadata columns first
    dataset = dataset.remove_columns(["sequence"])

    input_label_keys = {"input_ids", "attention_mask", "labels", "label"}
    metadata_cols = [
        c for c in dataset.column_names
        if c not in input_label_keys
    ]

    metas = [
        {col: dataset[col][i] for col in metadata_cols}
        for i in range(len(dataset))
    ]

    dataset = dataset.remove_columns(metadata_cols)

    # build ds
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    rows: List[Dict[str, Any]] = []

    idx = 0
    for batch in tqdm.tqdm(loader, desc="Predicting", total=len(loader)):
        labels_key = "labels" if "labels" in batch else ("label" if "label" in batch else None)
        labels = batch.pop(labels_key).to(DEVICE)               # [B]
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}

        # one base forward; K head forwards -> [K, B, C]
        #with autocast(enabled=use_amp):

        if uncertainty_method == "epinet":
            model.eval()
            logits_all = model.wrapper(inputs, n_index_samples=k_samples, return_all=True)
        elif uncertainty_method == "mc_dropout":
            model.train()
            K = k_samples
            #B = inputs["input_ids"].size(0)
            #big_inputs = {k: v.repeat_interleave(K, dim=0) for k, v in inputs.items()}  # [K*B, ...]a
            #out_big = model(**big_inputs).logits              # [B*K, C]
            #logits_all = out_big.view(B, K, -1).permute(1, 0, 2).contiguous()  # [K, B, C]
            logits_list = []
            for _ in range(k_samples):
                # each call uses batch size B, not K*B
                out = model(**inputs).logits  # [B, C]
                logits_list.append(out.unsqueeze(0))  # [1, B, C]
            logits_all = torch.cat(logits_list, dim=0)  #
        else: 
            model.eval()
            out = model(**inputs)
            logits_all = out.logits.unsqueeze(0)  #
        
        logits_all = logits_all.detach().cpu()

        #logits_all = model.wrapper(inputs, n_index_samples=k_samples, return_all=True)
        unc = compute_uncertainty(logits_all)           

        mean_probs = unc["mean_probs"].numpy()   # [B, C]
        B, C = mean_probs.shape

        for i in range(B):
            row = ({
                "labels": int(labels[i]),
                "pred": int(unc["predicted_class"][i]),
                "max_confidence": float(unc["max_confidence"][i]),
                "pred-pre_average": float(unc["predicted_class_logitmean"][i]),
                "max_confidence-pre_average": float(unc["max_confidence_logitmean"][i]),
                "U_total": float(unc["normalized_total_uncertainty"][i]),
                "U_epistemic": float(unc["normalized_epistemic_uncertainty"][i]),
                "U_aleatoric": float(unc["normalized_aleatoric_uncertainty"][i]),
                "vote_pct": float(unc["vote_percentage"][i]),
            })

            if True:
                for c in range(C):
                    row[f"prob_{c}"] = float(mean_probs[i, c])

            rows.append({**row, **metas[idx]})
            idx += 1


    if outfile is not None:
        outdir = os.path.dirname(outfile)
        if outdir:  
            os.makedirs(outdir, exist_ok=True)
        pd.DataFrame(rows).to_csv(outfile, index=False)
        print(f"[uncertainty] wrote {outfile} with {len(rows)} rows")


    return rows