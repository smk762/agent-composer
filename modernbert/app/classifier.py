import asyncio
import json
import logging
import os
from typing import List, Optional

import torch
import torch.nn.functional as F

from .model import ModelManager, ModelState
from .schemas import ClassifyRequest, ClassifyResponse, EmbedRequest, EmbedResponse

logger = logging.getLogger(__name__)


class Classifier:
    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager
        self.max_length: int = int(os.environ.get("MODERNBERT_MAX_LENGTH", "512"))
        self.top_k: int = int(os.environ.get("MODERNBERT_TOP_K", "10"))
        self.threshold: float = float(os.environ.get("MODERNBERT_THRESHOLD", "0.05"))
        self.pool: str = os.environ.get("MODERNBERT_POOL", "mean")  # mean | cls
        self.labels: List[str] = []
        self._label_embeddings: Optional[torch.Tensor] = None

        self._init_labels()

    # ----------------------------------------------------------------- labels

    def _init_labels(self) -> None:
        raw = os.environ.get("MODERNBERT_LABELS", "")
        if raw:
            self.labels = [l.strip() for l in raw.split(",") if l.strip()]
        path = os.environ.get("MODERNBERT_LABELS_FILE", "/data/labels.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                self.labels = data if isinstance(data, list) else data.get("labels", self.labels)
            except Exception as exc:
                logger.warning("Could not read labels file %s: %s", path, exc)

    async def update_labels(self, labels: List[str]) -> None:
        self.labels = labels
        self._label_embeddings = None
        if self.manager.state in (ModelState.CPU, ModelState.GPU):
            await self._embed_labels()

    async def _embed_labels(self) -> None:
        if not self.labels:
            return
        loop = asyncio.get_event_loop()
        embs = await loop.run_in_executor(None, lambda: self._encode_sync(self.labels))
        self._label_embeddings = F.normalize(embs, dim=-1)
        logger.debug("Label embeddings ready (%d labels)", len(self.labels))

    # ----------------------------------------------------------------- encode

    def _pool_hidden(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            return hidden[:, 0]
        m = mask.unsqueeze(-1).float()
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def _encode_sync(self, texts: List[str]) -> torch.Tensor:
        m = self.manager
        device = m.current_device()
        inputs = m.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = m.model(**inputs)
        return self._pool_hidden(out.last_hidden_state, inputs["attention_mask"])

    # ----------------------------------------------------------------- classify

    async def classify(self, req: ClassifyRequest) -> ClassifyResponse:
        top_k = req.top_k or self.top_k
        threshold = req.threshold if req.threshold is not None else self.threshold

        turns = (req.context or [])[-5:]
        text = (" [SEP] ".join(turns) + " [SEP] " + req.text) if turns else req.text

        if self.manager.task == "classify":
            return await self._supervised(text, top_k, threshold)
        return await self._zero_shot(text, top_k, threshold)

    async def _zero_shot(self, text: str, top_k: int, threshold: float) -> ClassifyResponse:
        if not self.labels:
            return ClassifyResponse(labels=[], scores=[], raw={}, task="zero_shot",
                                    model=self.manager.model_id, model_state=self.manager.state)

        await self.manager.ensure_ready()
        if self._label_embeddings is None:
            await self._embed_labels()

        loop = asyncio.get_event_loop()
        text_emb = await loop.run_in_executor(
            None, lambda: F.normalize(self._encode_sync([text]), dim=-1)
        )
        label_embs = self._label_embeddings.to(text_emb.device)
        scores = (text_emb @ label_embs.T).squeeze(0).cpu().tolist()

        ranked = sorted(zip(self.labels, scores), key=lambda x: -x[1])
        top = [(l, s) for l, s in ranked if s >= threshold][:top_k]

        return ClassifyResponse(
            labels=[l for l, _ in top],
            scores=[round(s, 4) for _, s in top],
            raw={l: round(s, 4) for l, s in ranked},
            task="zero_shot",
            model=self.manager.model_id,
            model_state=self.manager.state,
        )

    async def _supervised(self, text: str, top_k: int, threshold: float) -> ClassifyResponse:
        num_labels = len(self.labels) or 2
        await self.manager.ensure_ready(num_labels=num_labels)

        loop = asyncio.get_event_loop()

        def _run():
            device = self.manager.current_device()
            inputs = self.manager.tokenizer(
                text, truncation=True, max_length=self.max_length, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = self.manager.model(**inputs).logits
            # binary → softmax; multi-label → sigmoid
            if num_labels == 2:
                return torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
            return torch.sigmoid(logits).squeeze(0).cpu().tolist()

        probs = await loop.run_in_executor(None, _run)
        named = list(zip(self.labels or [f"class_{i}" for i in range(len(probs))], probs))
        ranked = sorted(named, key=lambda x: -x[1])
        top = [(l, s) for l, s in ranked if s >= threshold][:top_k]

        return ClassifyResponse(
            labels=[l for l, _ in top],
            scores=[round(s, 4) for _, s in top],
            raw={l: round(s, 4) for l, s in ranked},
            task="classify",
            model=self.manager.model_id,
            model_state=self.manager.state,
        )

    # ----------------------------------------------------------------- embed

    async def embed(self, req: EmbedRequest) -> EmbedResponse:
        await self.manager.ensure_ready()
        loop = asyncio.get_event_loop()
        embs = await loop.run_in_executor(None, lambda: self._encode_sync(req.texts))
        vecs = embs.cpu().tolist()
        return EmbedResponse(embeddings=vecs, model=self.manager.model_id, dim=embs.shape[-1])
