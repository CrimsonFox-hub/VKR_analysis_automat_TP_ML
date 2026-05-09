import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList
)
from typing import Optional
from .config import LLM_MODEL_NAME, GENERATION_CONFIG
from .logger import get_logger

log = get_logger(__name__)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: list[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class LLM:
    def __init__(
        self,
        model_name: Optional[str] = None,
        force_cpu: bool = True,
        max_memory: Optional[dict] = None
    ):
        self.model_name = model_name or LLM_MODEL_NAME
        self.device = "cpu"
        self.generation_config = GENERATION_CONFIG.copy()

        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Стоп-токены
        self.stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            self.tokenizer.convert_tokens_to_ids("</s>"),
        ]
        self.stop_token_ids = [tid for tid in self.stop_token_ids if tid is not None]

        # Параметры загрузки модели
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
        }
        if max_memory:
            model_kwargs["max_memory"] = max_memory

        log.info(f"Загрузка ЛЛМ {self.device}: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self.model.eval()
        log.info("Успешно загружено")

    def generate(self, prompt: str) -> str:
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)

            stopping_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids)]) if self.stop_token_ids else None

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria
                )

            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            log.exception("Ошибка генерации")
            return f"[Ошибка генерации: {str(e)}]"

    def generate_stream(self, prompt: str):
        from transformers import TextStreamer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            **self.generation_config,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        with torch.no_grad():
            self.model.generate(**generation_kwargs)
            
            
_llm_instance: Optional[LLM] = None

def get_llm() -> LLM:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLM()
    return _llm_instance