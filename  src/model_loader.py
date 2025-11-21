"""
Qwen/Qwen-Image modelini yuklash va inference qilish moduli
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import logging

# Logging sozlash
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model va cache sozlamalari
MODEL_NAME = "Qwen/Qwen-Image"
CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/runpod-volume/qwen_image")

# Global o'zgaruvchilar
tokenizer = None
model = None


def load_model():
    """
    Modelni yuklash funksiyasi.
    Birinchi ishga tushirishda HuggingFace'dan yuklab, volume'ga saqlaydi.
    Keyingi martalar uchun volume'dan yuklaydi.

    Returns:
        tuple: (tokenizer, model)
    """
    global tokenizer, model

    try:
        logger.info(f"Model yuklanmoqda: {MODEL_NAME}")
        logger.info(f"Cache direktori: {CACHE_DIR}")

        # Tokenizer yuklash
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        logger.info("✓ Tokenizer yuklandi")

        # Model yuklash
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=CACHE_DIR
        )

        # Modelni eval rejimga o'tkazish
        model.eval()

        logger.info("✓ Model yuklandi va GPUga joylashtirildi")
        logger.info(f"Device: {next(model.parameters()).device}")

        return tokenizer, model

    except Exception as e:
        logger.error(f"Model yuklashda xatolik: {str(e)}")
        raise


def inference(prompt, image_path=None, max_new_tokens=512, temperature=0.7):
    """
    Matn generatsiya qilish funksiyasi

    Args:
        prompt (str): Kiritish prompti
        image_path (str, optional): Rasm fayli yo'li
        max_new_tokens (int): Maksimal generatsiya qilinadigan tokenlar soni
        temperature (float): Generatsiya harorati

    Returns:
        str: Generatsiya qilingan matn
    """
    global tokenizer, model

    if tokenizer is None or model is None:
        raise RuntimeError("Model yuklanmagan. Avval load_model() ni chaqiring.")

    try:
        logger.info(f"Inference boshlanmoqda. Prompt: {prompt[:50]}...")

        # Input tayyorlash
        if image_path:
            # Rasm bilan ishlash
            image = Image.open(image_path)
            inputs = tokenizer(
                prompt,
                images=image,
                return_tensors="pt"
            )
        else:
            # Faqat matn
            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            )

        # Inputlarni GPU'ga ko'chirish
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generatsiya
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50
            )

        # Natijani decode qilish
        generated_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        logger.info("✓ Generatsiya tugadi")
        return generated_text

    except Exception as e:
        logger.error(f"Inference xatoligi: {str(e)}")
        raise


def get_model_info():
    """
    Model haqida ma'lumot qaytarish

    Returns:
        dict: Model ma'lumotlari
    """
    global model

    if model is None:
        return {"error": "Model yuklanmagan"}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "model_name": MODEL_NAME,
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "cache_dir": CACHE_DIR
    }