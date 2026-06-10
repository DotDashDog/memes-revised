"""CLIP / OpenCLIP embedding backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from experiments.wp01.config import MODELS, ClipModelSpec


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ClipBackend:
    spec: ClipModelSpec
    model: torch.nn.Module
    preprocess: object
    tokenizer: object | None
    device: torch.device

    @classmethod
    def load(cls, key: str) -> "ClipBackend":
        spec = MODELS[key]
        device = _device()
        if spec.family == "openai_clip":
            from transformers import CLIPModel, CLIPProcessor

            processor = CLIPProcessor.from_pretrained(spec.model_id)
            model = CLIPModel.from_pretrained(spec.model_id).to(device).eval()
            return cls(spec=spec, model=model, preprocess=processor, tokenizer=None, device=device)

        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            spec.model_id,
            pretrained=spec.pretrained,
        )
        tokenizer = open_clip.get_tokenizer(spec.model_id)
        model = model.to(device).eval()
        return cls(spec=spec, model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)

    @torch.inference_mode()
    def encode_images(self, images_bgr: Sequence[np.ndarray], batch_size: int = 32) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(images_bgr), batch_size):
            batch = images_bgr[start : start + batch_size]
            pil = [Image.fromarray(img[:, :, ::-1]) for img in batch]
            if self.spec.family == "openai_clip":
                inputs = self.preprocess(images=pil, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self.model.get_image_features(**inputs)
                if not isinstance(feats, torch.Tensor):
                    vision_out = self.model.vision_model(**inputs)
                    feats = self.model.visual_projection(vision_out.pooler_output)
            else:
                tensors = torch.stack([self.preprocess(im) for im in pil]).to(self.device)
                feats = self.model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.append(feats.detach().cpu().numpy())
        return np.vstack(vectors)

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            if self.spec.family == "openai_clip":
                inputs = self.preprocess(text=batch, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self.model.get_text_features(**inputs)
                if not isinstance(feats, torch.Tensor):
                    text_out = self.model.text_model(**inputs)
                    feats = self.model.text_projection(text_out.pooler_output)
            else:
                tokens = self.tokenizer(batch).to(self.device)
                feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.append(feats.detach().cpu().numpy())
        return np.vstack(vectors)
