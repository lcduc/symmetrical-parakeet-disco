import logging
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class VLMClient:
    """Legacy GPU OCR tier wrapping Qwen2-VL-2B-Instruct (4-bit quantized) via transformers.

    Intended to be constructed once and reused (see OCRRouterService); `_load`
    is safe to call from concurrent threads (`asyncio.to_thread` workers).
    """

    MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
    TRANSCRIPTION_PROMPT = "Transcribe all text visible in this image, verbatim and in reading order."

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._processor = None
        self._load_lock = Lock()

    def _load(self) -> None:
        """Lazily construct the quantized model and processor on first use."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            logger.info("Loading VLM model %s (device=%s)", self.MODEL_NAME, self.device)
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2VLForConditionalGeneration,
            )

            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self._processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.MODEL_NAME,
                quantization_config=quantization_config,
                device_map=self.device,
            )

    def extract_text(self, file_path: str | Path) -> str:
        """Transcribe all visible text in the image at `file_path`."""
        self._load()

        from PIL import Image

        image = Image.open(file_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.TRANSCRIPTION_PROMPT},
                ],
            }
        ]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text_prompt], images=[image], return_tensors="pt").to(
            self._model.device
        )
        generated_ids = self._model.generate(**inputs, max_new_tokens=2048)
        trimmed_ids = [
            output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated_ids)
        ]
        decoded = self._processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0].strip()
