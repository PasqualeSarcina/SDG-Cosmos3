"""Position-weighted causal loss specified in docs/lora_training_plan.md."""

import inspect
import math
import os
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from trl import SFTConfig, SFTTrainer

from src.lora.token_classes import TokenClass


class Trainer(SFTTrainer):
    """Direct-batch SFT on complete, pretokenized examples with a custom collator.

    The collator must return unshifted tensors shaped like ``labels``:
    ``token_classes`` (TokenClass IDs) and ``json_field_ids`` (an occurrence ID
    for C, -1 elsewhere). The constructor requires both loss hyperparameters:

    * field_length_exponent: finite, nonnegative exponent in the raw content
      weight N_f**(-field_length_exponent), where N_f is a field occurrence's
      token count after shifting. Content weights are normalized to mean 1
      across the direct batch on all GPUs. At equal mean cross-entropy, a field
      contributes as N_f**(1-field_length_exponent): 0 gives equal weight per
      token, 0.5 compensates by square root, and 1 gives equal total weight per
      field.
    * structure_weight: finite, strictly positive weight for each fixed-structure token,
      relative to a mean content-token weight of 1. It is not a fixed fraction
      of the total loss. Decision tokens have weight 1; excluded tokens have 0.

    Neither parameter has a default. To reproduce the plan's initial settings,
    pass field_length_exponent=0.5 and structure_weight=0.05 explicitly.
    IDs are local to each example: different subjects' descriptions need distinct
    IDs. E labels must be -100, including
    padding even if its token ID equals EOS. Field lengths are counted after
    shifting, on complete examples. Parsing XML/JSON belongs to preprocessing.
    Statistics are shared across DDP ranks for each direct batch. Packing and
    pre-shifted labels are unsupported; targets are never truncated.

    Standard and custom loss metrics are reported to Weights & Biases.
    WANDB_PROJECT defaults to "sdg-cosmos3" if unset. Configure the run name
    and logging frequency with SFTConfig.run_name and SFTConfig.logging_steps.
    """

    args: SFTConfig

    _METADATA_KEYS = frozenset({"token_classes", "json_field_ids"})

    def __init__(
            self,
            model,
            args: SFTConfig,
            data_collator,
            field_length_exponent: float,
            structure_weight: float,
            **kwargs
    ):
        self.field_length_exponent = float(field_length_exponent)
        self.structure_weight = float(structure_weight)
        if not math.isfinite(self.field_length_exponent) or self.field_length_exponent < 0:
            raise ValueError("field_length_exponent must be finite and >= 0.")
        if not math.isfinite(self.structure_weight) or self.structure_weight <= 0:
            raise ValueError("structure_weight must be finite and > 0.")
        # Check for kwargs not compatible with the custom cosmos loss
        if kwargs.get("compute_loss_func") is not None:
            raise ValueError("Trainer implements compute_loss; do not pass compute_loss_func.")
        if args.packing or args.eval_packing or args.padding_free:
            raise ValueError("Cosmos loss requires packing=False and padding_free=False.")
        if args.loss_type != "nll" or args.label_smoothing_factor:
            raise ValueError("Cosmos loss requires loss_type='nll' and no label smoothing.")
        if args.gradient_accumulation_steps != 1:
            raise ValueError("Cosmos loss requires gradient_accumulation_steps=1.")
        # Default SFT preparation/collation can discard or truncate annotations.
        args.dataset_kwargs = {**(args.dataset_kwargs or {}), "skip_prepare_dataset": True}
        args.remove_unused_columns = False
        args.report_to = ["wandb"]
        os.environ.setdefault("WANDB_PROJECT", "sdg-cosmos3")
        super().__init__(model=model, args=args, data_collator=data_collator, **kwargs)
        if self.args.n_gpu > 1:
            raise ValueError("Use one process per GPU (DDP), rather than DataParallel.")
        if self.is_deepspeed_enabled:
            raise ValueError("DeepSpeed loss scaling must be verified before using Cosmos loss.")
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and any(getattr(pc, name, 1) > 1 for name in ("tp_size", "cp_size", "sp_size")):
            raise ValueError("Cosmos loss currently supports data parallelism only.")
        if self.aux_loss_enabled:
            raise ValueError("The plan specifies language loss only; disable router_aux_loss_coef.")
        # The loss computes its own weighted denominator for the direct batch.
        # Do not ask Transformers to count labels or pass loss kwargs to forward.
        self.model_accepts_loss_kwargs = False
        unwrapped = self.accelerator.unwrap_model(self.model)
        base = unwrapped.get_base_model() if hasattr(unwrapped, "get_base_model") else unwrapped
        self._supports_fused_lm_head = "fused_lm_head" in inspect.signature(base.forward, follow_wrapped=False).parameters
        self._cosmos_metrics = {"train": [], "eval": []}

    def _token_metadata(self, inputs: dict[str, torch.Tensor]):
        """Prepare per-token annotations and raw weights for the Cosmos loss.

        Validate the collator's labels, token classes and field occurrence IDs,
        then align them with next-token predictions by dropping position zero.
        Count each field's remaining content tokens within each example and
        assign raw weights: N_f**(-field_length_exponent) for content, 1 for
        decisions, structure_weight for structure and 0 for excluded positions.

        Args:
            inputs (dict[str, torch.Tensor]): Batch produced by the data collator, of shape [batch_size, sequence_length]

        Returns:
            tuple: Shifted targets, token categories, weights and batch statistics.

                * ``labels``: Target token IDs from position 1 onward;
                  -100 marks positions excluded from the loss.
                * ``token_classes``: Categories aligned with the target tokens.
                * ``unnormalized_token_weights``: Per-token weights before
                  CONTENT weights are normalized to mean 1.
                * ``local_stats``: Float64 tensor of shape [4] containing
                  CONTENT count, unnormalized CONTENT weight sum,
                  DECISION count and STRUCTURE count.

        """
        # labels: target token IDs; -100 marks positions excluded from the loss.
        # token_classes: category of each token: EXCLUDED, CONTENT, DECISION, or STRUCTURE, as listed in TokenClass
        # json_field_ids: IDs of the JSON fields for CONTENT positions; -1 elsewhere.
        # All three tensors have shape [batch_size, sequence_length].
        for key in ("labels", "token_classes", "json_field_ids"):
            if key not in inputs:
                raise ValueError(f"Missing required collator tensor: {key}.")
        if "shift_labels" in inputs:
            raise ValueError("Pass unshifted labels and annotations; shift_labels is unsupported.")
        labels, token_classes, json_field_ids = (inputs[key] for key in ("labels", "token_classes", "json_field_ids"))
        if not all(torch.is_tensor(value) and value.dtype == torch.long for value in (labels, token_classes, json_field_ids)):
            raise ValueError("labels, token_classes and json_field_ids must be torch.long tensors.")
        if labels.ndim != 2 or labels.shape[0] == 0 or labels.shape[1] < 2:
            raise ValueError("Expected nonempty [batch, sequence] labels with at least two positions.")
        if token_classes.shape != labels.shape or json_field_ids.shape != labels.shape:
            raise ValueError("Labels and loss annotations must have identical shapes.")
        if token_classes.device != labels.device or json_field_ids.device != labels.device:
            raise ValueError("Labels and loss annotations must be on the same device.")
        if self.args.max_length is not None and labels.shape[1] > self.args.max_length:
            raise ValueError("Example exceeds max_length; audit the data instead of truncating the target.")
        if ((token_classes < TokenClass.EXCLUDED) | (token_classes > TokenClass.STRUCTURE)).any():
            raise ValueError("Unknown loss token class.")
        excluded, content = token_classes == TokenClass.EXCLUDED, token_classes == TokenClass.CONTENT
        if not torch.equal(labels == -100, excluded):
            raise ValueError("Exactly E positions must have label -100.")
        if ((labels < 0) & ~excluded).any():
            raise ValueError("Supervised labels must be nonnegative token IDs.")
        if (json_field_ids[content] < 0).any() or (json_field_ids[~content] != -1).any():
            raise ValueError("Only C positions need a nonnegative field occurrence ID; use -1 elsewhere.")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            if attention_mask.shape != labels.shape or ((attention_mask == 0) & ~excluded).any():
                raise ValueError("Padding must have class E and label -100.")
        # Dropping position 0 because there is not a previous token to compare with
        labels, token_classes, json_field_ids = labels[:, 1:], token_classes[:, 1:], json_field_ids[:, 1:]
        content = token_classes == TokenClass.CONTENT
        if not content.any(dim=1).all():
            raise ValueError("Every complete example must have C tokens after the causal shift.")
        # Initialization of the token weight matrix (before normalization)
        unnormalized_token_weights = torch.zeros_like(labels, dtype=torch.float32)
        # Iterate over the batch
        for row in range(labels.shape[0]):
            # Binary mask of the tokens belonging to class CONTENT
            mask = content[row]
            # Find unique field IDs among CONTENT tokens and compute:
            # - inverse_indices: index of each original value in the unique IDs tensor.
            # - counts: number of occurrences of each unique field ID.
            _, inverse_indices, counts = torch.unique(json_field_ids[row, mask], return_inverse=True, return_counts=True)
            # Assign to every CONTENT token a weight based on its field length
            # Uses the inverse_indices to retrieve the length of the field from counts
            unnormalized_token_weights[row, mask] = counts[inverse_indices].float().pow(-self.field_length_exponent)
        unnormalized_token_weights[token_classes == TokenClass.DECISION] = 1.0
        unnormalized_token_weights[token_classes == TokenClass.STRUCTURE] = self.structure_weight
        # Collect local batch statistics for loss normalization across GPUs:
        # [CONTENT token count, sum of raw CONTENT weights,
        #  DECISION token count, STRUCTURE token count].
        # These determine the CONTENT scaling factor and the weighted loss denominator.
        stats = torch.stack((
            content.sum(), unnormalized_token_weights[content].double().sum(),
            (token_classes == TokenClass.DECISION).sum(), (token_classes == TokenClass.STRUCTURE).sum(),
        )).double()
        return labels, token_classes, unnormalized_token_weights, stats

    def _global_stats(self, stats):
        return self.accelerator.reduce(stats, reduction="sum")

    def _normalization(self, stats):
        """Compute the CONTENT weight scaling factor and weighted loss denominator.

        Scale CONTENT weights so their mean over the global batch is 1.
        Their total weight then equals the CONTENT token count. DECISION
        tokens retain weight 1, STRUCTURE tokens retain structure_weight,
        and EXCLUDED tokens contribute nothing to the denominator.

        Args:
            stats (torch.Tensor): Tensor of shape [4], summed across GPUs,
                containing [CONTENT token count, unnormalized CONTENT weight
                sum, DECISION token count, STRUCTURE token count].

        Returns:
            tuple: Two scalar tensors used to normalize the loss.

                * ``content_scale``: CONTENT token count divided by the sum
                  of unnormalized CONTENT weights. Multiply CONTENT weights
                  by this factor to make their mean equal to 1.
                * ``denominator``: Sum of final token weights, computed as
                  CONTENT count + DECISION count + structure_weight times
                  STRUCTURE count.

        Raises:
            ValueError: If statistics have the wrong shape, contain nonfinite
                or negative values, CONTENT count or weight sum is zero,
                or the denominator is not finite and strictly positive.
        """
        if stats.shape != (4,) or not torch.isfinite(stats).all() or (stats < 0).any():
            raise ValueError("Invalid effective-batch statistics [C, R, D, S].")
        content, unnormalized_content_weight_sum, decisions, structure = stats.unbind()
        denominator = content + decisions + self.structure_weight * structure
        if content <= 0 or unnormalized_content_weight_sum <= 0 or not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError("The batch needs content and a finite, strictly positive denominator.")
        return content / unnormalized_content_weight_sum, denominator

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels, classes, unnormalized_token_weights, local_stats = self._token_metadata(inputs)
        stats = self._global_stats(local_stats)
        # num_items_in_batch belongs to the base API, but a plain token count
        # cannot normalize our position-weighted loss.
        k, denominator = self._normalization(stats)
        weights = unnormalized_token_weights.clone()
        weights[classes == TokenClass.CONTENT] *= k.to(weights)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Final loss weights must be finite and nonnegative.")
        if (weights[classes != TokenClass.EXCLUDED] <= 0).any():
            raise ValueError("Only E positions may have zero weight.")

        model_inputs = {key: value for key, value in inputs.items()
                        if key not in self._METADATA_KEYS and key != "labels"}
        model_inputs["use_cache"] = False
        if self._supports_fused_lm_head and labels.device.type == "cuda" and not return_outputs:
            # TRL's Triton head requires CUDA and already shifts once.
            outputs = model(**model_inputs, labels=inputs["labels"], fused_lm_head=True)
            per_token = -outputs.log_probs.float()
            if per_token.shape != labels.shape or not torch.equal(outputs.label_mask.to(labels.device), labels != -100):
                raise ValueError("Fused loss positions do not align with the shifted annotations.")
        else:
            outputs = model(**model_inputs)
            logits = outputs["logits"] if isinstance(outputs, Mapping) else outputs.logits
            if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[:2] != inputs["labels"].shape:
                raise ValueError("Expected full [B, L, vocabulary] logits; enable logits in Unsloth.")
            per_token = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                labels.to(logits.device).reshape(-1), ignore_index=-100, reduction="none",
            ).reshape(labels.shape)
        numerator = (per_token * weights.to(per_token.device)).sum()
        loss = numerator / denominator.to(numerator)
        self._record_loss_metrics(per_token, classes.to(per_token.device), unnormalized_token_weights.to(per_token.device), k)
        if model.training:
            # DDP averages gradients; our denominator already includes all ranks.
            loss = loss * self.accelerator.num_processes
        else:
            # Evaluation gathers/averages losses across ranks, not gradients.
            # Each rank must report the complete global-batch loss, rather than
            # its local numerator divided by a global denominator.
            loss = self._global_stats(numerator.detach()) / denominator.to(numerator)
        return (loss, outputs) if return_outputs else loss

    def _record_loss_metrics(self, ce, classes, raw, k):
        content, decisions, structure = (classes == kind for kind in (
            TokenClass.CONTENT, TokenClass.DECISION, TokenClass.STRUCTURE,
        ))
        ce = ce.detach().double()
        local = torch.stack((
            (ce[content] * raw[content]).sum(), ce[content].sum(),
            ce[decisions].sum(), ce[structure].sum(), content.sum(),
            raw[content].double().sum(), decisions.sum(), structure.sum(),
        ))
        # Aggregate token sums across direct batches of different lengths.
        local = torch.cat((local, (local[0] * k.to(local)).reshape(1)))
        mode = "train" if self.model.training else "eval"
        self._cosmos_metrics[mode].append(local)

    def log(self, logs, start_time=None):
        mode = "train" if self.model.training else "eval"
        records = self._cosmos_metrics[mode]
        if records:
            totals = self._global_stats(torch.stack(records).sum(dim=0))
            raw_ce, ce_c, ce_d, ce_s, count_c, mass, count_d, count_s, weighted_c = totals
            denominator = count_c + count_d + self.structure_weight * count_s
            metrics = {
                "loss_content": raw_ce / mass,
                "ce_content": ce_c / count_c,
                "ce_decision": ce_d / count_d.clamp_min(1),
                "ce_structure": ce_s / count_s.clamp_min(1),
                "contribution_content": weighted_c / denominator,
                "contribution_decision": ce_d / denominator,
                "contribution_structure": self.structure_weight * ce_s / denominator,
                "weighted_loss": (weighted_c + ce_d + self.structure_weight * ce_s) / denominator,
                "loss_field_length_exponent": self.field_length_exponent,
                "loss_structure_weight": self.structure_weight,
            }
            prefix = "eval_" if mode == "eval" else ""
            logs = {**logs, **{prefix + key: float(value) for key, value in metrics.items()}}
            records.clear()
        super().log(logs, start_time)
