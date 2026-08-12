from __future__ import annotations

import contextlib
from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


def choose_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, mps, cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HuggingFaceImageClassifier:
    """Adapter that makes a Hugging Face image classifier usable by LIME."""

    def __init__(
        self,
        model_id_or_path: str,
        *,
        device: str = "auto",
        inference_batch_size: int = 16,
        hf_token: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id_or_path = model_id_or_path
        self.device = choose_device(device)
        self.inference_batch_size = max(1, int(inference_batch_size))

        shared: dict[str, object] = {"trust_remote_code": trust_remote_code}
        if hf_token:
            shared["token"] = hf_token

        self.processor = AutoImageProcessor.from_pretrained(model_id_or_path, **shared)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id_or_path,
            low_cpu_mem_usage=True,
            **shared,
        )
        self.model.eval().to(self.device)

        raw_mapping = getattr(self.model.config, "id2label", {}) or {}
        self.id2label = {
            int(key): str(value)
            for key, value in raw_mapping.items()
            if str(key).lstrip("-").isdigit()
        }

    def label(self, class_id: int) -> str:
        return self.id2label.get(int(class_id), str(int(class_id)))

    def _autocast(self):
        if self.device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        return contextlib.nullcontext()

    def _predict_chunk(self, images: Sequence[np.ndarray]) -> np.ndarray:
        cleaned = [np.clip(np.asarray(image), 0, 255).astype(np.uint8, copy=False) for image in images]
        encoded = self.processor(images=cleaned, return_tensors="pt")
        encoded = {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }
        with torch.inference_mode(), self._autocast():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1)
        return probabilities.detach().cpu().numpy()

    def predict_proba(self, images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        if isinstance(images, np.ndarray) and images.ndim == 3:
            items: list[np.ndarray] = [images]
        else:
            items = list(images)
        if not items:
            labels = int(getattr(self.model.config, "num_labels", 0))
            return np.empty((0, labels), dtype=np.float32)

        batch_size = min(self.inference_batch_size, len(items))
        outputs: list[np.ndarray] = []
        cursor = 0
        while cursor < len(items):
            end = min(cursor + batch_size, len(items))
            try:
                outputs.append(self._predict_chunk(items[cursor:end]))
                cursor = end
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or batch_size <= 1:
                    raise
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                self.inference_batch_size = min(self.inference_batch_size, batch_size)
        return np.concatenate(outputs, axis=0)

    def predict_one(self, image: np.ndarray) -> tuple[int, str, float, np.ndarray]:
        probabilities = self.predict_proba(image)[0]
        class_id = int(np.argmax(probabilities))
        return class_id, self.label(class_id), float(probabilities[class_id]), probabilities
