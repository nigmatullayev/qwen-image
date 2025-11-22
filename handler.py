# """
# RunPod Serverless Handler
# Qwen/Qwen-Image model uchun API endpoint
# """
# import runpod
# import logging
# import traceback
# from src.model_loader import load_model, inference, get_model_info
#
# # Logging sozlash
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
# # Modelni yuklash (container ishga tushganda)
# logger.info("=" * 50)
# logger.info("RunPod Serverless Worker boshlandi")
# logger.info("=" * 50)
#
# try:
#     tokenizer, model = load_model()
#     logger.info("✓ Model muvaffaqiyatli yuklandi")
#     model_info = get_model_info()
#     logger.info(f"Model info: {model_info}")
# except Exception as e:
#     logger.error(f"Model yuklashda xatolik: {str(e)}")
#     logger.error(traceback.format_exc())
#     raise
#
#
# def handler(event):
#     """
#     RunPod handler funksiyasi
#
#     Args:
#         event (dict): RunPod event ob'ekti
#
#     Returns:
#         dict: API javobi
#     """
#     try:
#         logger.info(f"Yangi so'rov qabul qilindi: {event}")
#
#         # Input ma'lumotlarini olish
#         input_data = event.get("input", {})
#
#         # Prompt tekshirish
#         prompt = input_data.get("prompt")
#         if not prompt:
#             return {
#                 "error": "prompt parametri majburiy",
#                 "status": "error"
#             }
#
#         # Ixtiyoriy parametrlar
#         max_new_tokens = input_data.get("max_new_tokens", 512)
#         temperature = input_data.get("temperature", 0.7)
#         image_path = input_data.get("image_path", None)
#
#         # Parametrlarni validatsiya qilish
#         if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
#             max_new_tokens = 512
#
#         if not isinstance(temperature, (int, float)) or temperature < 0:
#             temperature = 0.7
#
#         logger.info(f"Parametrlar - max_tokens: {max_new_tokens}, temp: {temperature}")
#
#         # Inference jarayoni
#         result = inference(
#             prompt=prompt,
#             image_path=image_path,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature
#         )
#
#         # Natijani qaytarish
#         response = {
#             "output": {
#                 "generated_text": result,
#                 "model": "Qwen/Qwen-Image",
#                 "prompt": prompt,
#                 "parameters": {
#                     "max_new_tokens": max_new_tokens,
#                     "temperature": temperature
#                 }
#             },
#             "status": "success"
#         }
#
#         logger.info("✓ So'rov muvaffaqiyatli bajarildi")
#         return response
#
#     except Exception as e:
#         error_msg = f"Handler xatoligi: {str(e)}"
#         logger.error(error_msg)
#         logger.error(traceback.format_exc())
#
#         return {
#             "error": error_msg,
#             "status": "error",
#             "traceback": traceback.format_exc()
#         }
#
#
# # Health check endpoint (ixtiyoriy)
# def health_check(event):
#     """
#     Health check funksiyasi
#     """
#     try:
#         model_info = get_model_info()
#         return {
#             "status": "healthy",
#             "model_info": model_info
#         }
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e)
#         }
#
#
# # RunPod serverless'ni boshlash
# if __name__ == "__main__":
#     logger.info("RunPod serverless handler ishga tushirildi")
#     runpod.serverless.start(
#         {
#             "handler": handler,
#             "return_aggregate_stream": True
#         }
#     )

"""
RunPod Serverless Handler - Qwen/Qwen-Image (Standalone)
Barcha kod bir faylda
"""
import os
import torch
import runpod
import logging
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image

# Logging sozlash
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model sozlamalari
MODEL_NAME = "Qwen/Qwen-Image"
CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/runpod-volume/qwen_image")

# Global o'zgaruvchilar
tokenizer = None
model = None


# ============================================
# MODEL LOADER FUNCTIONS
# ============================================

def load_model():
    """Modelni yuklash funksiyasi"""
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
    """Matn generatsiya qilish funksiyasi"""
    global tokenizer, model

    if tokenizer is None or model is None:
        raise RuntimeError("Model yuklanmagan. Avval load_model() ni chaqiring.")

    try:
        logger.info(f"Inference boshlanmoqda. Prompt: {prompt[:50]}...")

        # Input tayyorlash
        if image_path:
            image = Image.open(image_path)
            inputs = tokenizer(prompt, images=image, return_tensors="pt")
        else:
            inputs = tokenizer(prompt, return_tensors="pt")

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
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info("✓ Generatsiya tugadi")
        return generated_text

    except Exception as e:
        logger.error(f"Inference xatoligi: {str(e)}")
        raise


def get_model_info():
    """Model haqida ma'lumot"""
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


# ============================================
# RUNPOD HANDLER
# ============================================

# Modelni yuklash (container ishga tushganda)
logger.info("=" * 50)
logger.info("RunPod Serverless Worker boshlandi")
logger.info("=" * 50)

try:
    tokenizer, model = load_model()
    logger.info("✓ Model muvaffaqiyatli yuklandi")
    model_info = get_model_info()
    logger.info(f"Model info: {model_info}")
except Exception as e:
    logger.error(f"Model yuklashda xatolik: {str(e)}")
    logger.error(traceback.format_exc())
    raise


def handler(event):
    """RunPod handler funksiyasi"""
    try:
        logger.info(f"Yangi so'rov qabul qilindi: {event}")

        # Input ma'lumotlarini olish
        input_data = event.get("input", {})

        # Prompt tekshirish
        prompt = input_data.get("prompt")
        if not prompt:
            return {
                "error": "prompt parametri majburiy",
                "status": "error"
            }

        # Ixtiyoriy parametrlar
        max_new_tokens = input_data.get("max_new_tokens", 512)
        temperature = input_data.get("temperature", 0.7)
        image_path = input_data.get("image_path", None)

        # Parametrlarni validatsiya qilish
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            max_new_tokens = 512

        if not isinstance(temperature, (int, float)) or temperature < 0:
            temperature = 0.7

        logger.info(f"Parametrlar - max_tokens: {max_new_tokens}, temp: {temperature}")

        # Inference jarayoni
        result = inference(
            prompt=prompt,
            image_path=image_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # Natijani qaytarish
        response = {
            "output": {
                "generated_text": result,
                "model": MODEL_NAME,
                "prompt": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature
                }
            },
            "status": "success"
        }

        logger.info("✓ So'rov muvaffaqiyatli bajarildi")
        return response

    except Exception as e:
        error_msg = f"Handler xatoligi: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())

        return {
            "error": error_msg,
            "status": "error",
            "traceback": traceback.format_exc()
        }


# Health check endpoint
def health_check(event):
    """Health check funksiyasi"""
    try:
        model_info = get_model_info()
        return {
            "status": "healthy",
            "model_info": model_info
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# RunPod serverless'ni boshlash
if __name__ == "__main__":
    logger.info("RunPod serverless handler ishga tushirildi")
    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True
        }
    )