"""Qwen document embeddings shared by the RAG ingestion pipeline."""

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024


def load_encoder(device="auto"):
    """Load the shared Qwen document encoder only when embeddings are needed."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA non disponibile. Usa --device cpu.")
    model = SentenceTransformer(MODEL_NAME, device=device)
    get_dimension = getattr(model, "get_embedding_dimension", None)
    if get_dimension is None:
        get_dimension = model.get_sentence_embedding_dimension
    if get_dimension() != EMBEDDING_DIMENSION:
        raise ValueError(f"Il modello deve produrre {EMBEDDING_DIMENSION} dimensioni.")
    return model


def encode_documents(model, texts, batch_size=32, show_progress_bar=False):
    """Encode documents without query instructions or silent truncation."""
    import numpy as np

    for index, text in enumerate(texts):
        tokens = model.tokenizer(text, truncation=False, return_attention_mask=False,
                                 return_token_type_ids=False)["input_ids"]
        if len(tokens) > model.max_seq_length:
            raise ValueError(
                f"Documento {index + 1}: {len(tokens)} token, "
                f"limite del modello {model.max_seq_length}."
            )
    embeddings = np.asarray(model.encode(
        texts, batch_size=batch_size, prompt="", convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=show_progress_bar,
    ), dtype=np.float32)
    if embeddings.shape != (len(texts), EMBEDDING_DIMENSION):
        raise ValueError(f"Forma degli embedding non valida: {embeddings.shape}.")
    finite_rows = np.isfinite(embeddings).all(axis=1)
    if not finite_rows.all():
        indices = (np.flatnonzero(~finite_rows)[:10] + 1).tolist()
        raise ValueError(f"Embedding con NaN/Inf per i documenti: {indices}.")
    # Model-side normalization may run in bf16/fp16. Normalize again after the
    # float32 conversion; float64 accumulation avoids norm overflow/underflow.
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1, keepdims=True)
    if (norms == 0).any():
        indices = (np.flatnonzero(norms[:, 0] == 0)[:10] + 1).tolist()
        raise ValueError(f"Embedding nulli per i documenti: {indices}.")
    return (embeddings / norms).astype(np.float32)
