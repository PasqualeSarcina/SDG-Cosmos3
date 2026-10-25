"""Right padding for complete, unshifted Cosmos training examples."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch

from src.lora.token_classes import TokenClass


class PaddingTokenizer(Protocol):
    """Tokenizer interface needed for collation."""

    @property
    def pad_token_id(self) -> int | None: ...


@dataclass
class CosmosDataCollator:
    """Pad labels/metadata independently of token IDs, including PAD == EOS.

    Use the same max_length as preprocessing and SFTConfig. Right padding keeps
    the causal shift independent of the lengths of other examples in the batch.
    Inputs must be unpadded features produced by prepare_caption/load_caption_dataset;
    per-example invariants are established during preprocessing, not rechecked here.
    The collator never computes weights, shifts labels or truncates examples.
    """

    tokenizer: PaddingTokenizer
    max_length: int
    pad_to_multiple_of: int | None = None
    _pad_token_id: int = field(init=False, repr=False)

    def __post_init__(self):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Set tokenizer.pad_token_id before constructing the collator.")
        self._pad_token_id = pad_token_id
        if self.pad_to_multiple_of is not None and (
            isinstance(self.pad_to_multiple_of, bool)
            or not isinstance(self.pad_to_multiple_of, int) or self.pad_to_multiple_of <= 0
        ):
            raise ValueError("pad_to_multiple_of must be a positive integer.")

    def __call__(self, features: Sequence[Mapping]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch.")
        keys = ("input_ids", "attention_mask", "labels", "token_classes", "json_field_ids")
        examples = [
            {key: torch.as_tensor(feature[key], dtype=torch.long, device="cpu") for key in keys}
            for feature in features
        ]
        length = max(example["input_ids"].numel() for example in examples)
        if self.pad_to_multiple_of:
            length = ((length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        if length > self.max_length:
            raise ValueError("Padding to the requested multiple exceeds max_length; no truncation applied.")
        fills = (self._pad_token_id, 0, -100, int(TokenClass.EXCLUDED), -1)
        batch = {key: torch.full((len(examples), length), fill, dtype=torch.long) for key, fill in zip(keys, fills)}
        for row, example in enumerate(examples):
            size = example["input_ids"].numel()
            for key in keys:
                batch[key][row, :size] = example[key]
        return batch
