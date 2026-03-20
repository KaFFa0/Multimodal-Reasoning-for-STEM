import torch
import gc
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig
from typing import List, Dict, Any

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def compute_exact_match(pred: str, ref: str) -> bool:
    return pred.strip() == ref.strip()

def compute_bleu(pred: str, ref: str) -> float:
    pred_tokens = pred.split()
    ref_tokens = [ref.split()]
    return sentence_bleu(ref_tokens, pred_tokens,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=SmoothingFunction().method1)

def compute_levenshtein_similarity(pred: str, ref: str) -> float:
    distance = Levenshtein.distance(pred, ref)
    max_len = max(len(pred), len(ref))
    if max_len == 0:
        return 1.0
    return 1 - distance / max_len

def evaluate_predictions(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Вычисление всех трех метрик для предиктов и таргетов"""
    em = np.mean([compute_exact_match(p, r) for p, r in zip(predictions, references)])
    bleu = np.mean([compute_bleu(p, r) for p, r in zip(predictions, references)])
    lev = np.mean([compute_levenshtein_similarity(p, r) for p, r in zip(predictions, references)])
    return {"exact_match": float(em), "bleu": float(bleu), "levenshtein_sim": float(lev)}

def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Преобразование примера датасета в формат с сообщениями"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Convert this handwritten formula into LaTeX"}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["latex"]}]
        }
    ]
    return {
        "messages": messages,
        "image": example["image"]
    }

class VLDataCollator:
    """формируем батчи данных (изображение + текст)"""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        texts = [
            self.processor.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            for item in batch
        ]
        images = [item["image"] for item in batch]

        batch_enc = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        batch_enc["labels"] = batch_enc["input_ids"].clone()
        return batch_enc

def get_quant_config(compute_dtype=torch.bfloat16):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

def get_lora_config(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=None):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

def load_base_model(model_name: str, quant_config: BitsAndBytesConfig, device_map="auto"):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quant_config,
    )
    return model

def load_processor(model_name: str):
    return AutoProcessor.from_pretrained(model_name)