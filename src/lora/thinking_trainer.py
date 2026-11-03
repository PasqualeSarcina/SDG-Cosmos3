"""Supervised scene plans, sharing the verified forward/DDP engine with legacy SFT."""

import math

import torch
from trl import SFTTrainer

from src.lora.token_classes import TokenClass as K
from src.lora.trainer import Trainer


class ThinkingTrainer(Trainer):
    """Weight plan text separately, preserving field compensation in the final.

    thinking_weight is per plan token; response_weight multiplies C/D/S after
    content normalization. D/S include generated reasoning control delimiters.
    Both numerator and denominator use the same weights. No section averaging.
    """

    def __init__(self, *args, thinking_weight=1.0, response_weight=0.5, thinking_protocol, **kwargs):
        for name, value in (("thinking_weight", thinking_weight), ("response_weight", response_weight)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0.")
            setattr(self, name, float(value))
        self.thinking_protocol = thinking_protocol
        super().__init__(*args, **kwargs)

    def _token_metadata(self, inputs):
        classes, labels = inputs["token_classes"], inputs["labels"]
        if classes.shape != labels.shape or classes.ndim != 2:
            raise ValueError("Labels and token classes must have identical [batch, sequence] shapes.")
        thinking = classes == K.THINKING
        if not thinking[:, 1:].any(dim=1).all():
            raise ValueError("Every supervised example must have thinking tokens after shifting.")
        if (labels[thinking] < 0).any() or (inputs["json_field_ids"][thinking] != -1).any():
            raise ValueError("Thinking tokens require real labels and field ID -1.")
        if "attention_mask" in inputs and (inputs["attention_mask"][thinking] == 0).any():
            raise ValueError("Thinking tokens cannot be padding.")
        # Reuse all original validations and field counts, excluding T only from
        # this intermediate calculation. The real labels are restored below and
        # passed intact to the ordinary or fused forward in Trainer.compute_loss.
        final_inputs = {**inputs, "labels": labels.masked_fill(thinking, -100),
                        "token_classes": classes.masked_fill(thinking, K.EXCLUDED)}
        _, _, raw, stats = super()._token_metadata(final_inputs)
        labels, classes = labels[:, 1:], classes[:, 1:]
        thinking = thinking[:, 1:]
        raw[(classes == K.DECISION) | (classes == K.STRUCTURE)] *= self.response_weight
        raw[thinking] = self.thinking_weight
        stats = torch.cat((stats, thinking.sum().double().reshape(1)))
        # Recover the exact final-response boundary from the validated protocol,
        # without another collator column or treating all post-T structure as final.
        close = (labels == self.thinking_protocol["close_id"]) & (classes == K.DECISION)
        if not (close.sum(dim=1) == 1).all():
            raise ValueError("Each example needs exactly one supervised thinking closure.")
        close_positions = close.long().argmax(dim=1)
        separator = self.thinking_protocol["answer_separator_ids"]
        for row, position in enumerate(close_positions.tolist()):
            if labels[row, position + 1:position + 1 + len(separator)].tolist() != separator:
                raise ValueError("Thinking/final separator IDs differ from preflight.")
        positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
        if (thinking & (positions >= close_positions.unsqueeze(1))).any():
            raise ValueError("Thinking tokens must precede the closing marker.")
        self._answer_mask = ((positions > (close_positions + len(separator)).unsqueeze(1))
                             & (classes != K.EXCLUDED))
        if ((classes == K.CONTENT) & ~self._answer_mask).any() or not self._answer_mask.any(dim=1).all():
            raise ValueError("Final content must follow the thinking boundary.")
        return labels, classes, raw, stats

    def _normalization(self, stats):
        if stats.shape != (5,) or not torch.isfinite(stats).all() or (stats < 0).any() or stats[4] <= 0:
            raise ValueError("Invalid supervised-thinking batch statistics [C, R, D, S, T].")
        k, answer_mass = super()._normalization(stats[:4])
        denominator = self.response_weight * answer_mass + self.thinking_weight * stats[4]
        if not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError("Invalid supervised-thinking denominator.")
        return self.response_weight * k, denominator

    def _record_loss_metrics(self, ce, classes, raw, k):
        ce, raw = ce.detach().double(), raw.detach().double()
        masks = [classes == kind for kind in (K.CONTENT, K.DECISION, K.STRUCTURE, K.THINKING)]
        weights = raw.clone()
        weights[masks[0]] *= k.to(weights)
        answer = self._answer_mask.to(ce.device)
        # Sums, not means: preserve the batch-specific field scale across logs.
        values = [(ce[m] * weights[m]).sum() for m in masks]
        values += [ce[m].sum() for m in masks]
        values += [m.sum() for m in masks]
        values += [weights.sum(), (ce[answer] * weights[answer]).sum(), weights[answer].sum(),
                   (ce[masks[0]] * raw[masks[0]]).sum(), raw[masks[0]].sum()]
        mode = "train" if self.model.training else "eval"
        self._cosmos_metrics[mode].append(torch.stack(values))

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        records = self._cosmos_metrics[mode]
        if records:
            totals = self._global_stats(torch.stack(records).sum(dim=0))
            wc, wd, ws, wt, cc, cd, cs, ct, nc, nd, ns, nt, mass, answer_sum, answer_mass, rc, r = totals
            metrics = {
                "weighted_loss": (wc + wd + ws + wt) / mass,
                "loss_content": rc / r, "ce_content": cc / nc,
                "ce_decision": cd / nd.clamp_min(1), "ce_structure": cs / ns.clamp_min(1),
                "ce_thinking": ct / nt, "loss_answer": answer_sum / answer_mass,
                "contribution_content": wc / mass, "contribution_decision": wd / mass,
                "contribution_structure": ws / mass, "contribution_thinking": wt / mass,
                "thinking_weight_fraction": self.thinking_weight * nt / mass,
                "thinking_tokens": nt, "loss_thinking_weight": self.thinking_weight,
                "loss_response_weight": self.response_weight,
                "loss_field_length_exponent": self.field_length_exponent,
                "loss_structure_weight": self.structure_weight,
            }
            prefix = "eval_" if mode == "eval" else ""
            logs = {**logs, **{prefix + key: float(value) for key, value in metrics.items()}}
            records.clear()
        # Skip the legacy metric aggregation, sharing only SFT's logging.
        SFTTrainer.log(self, logs, start_time)
