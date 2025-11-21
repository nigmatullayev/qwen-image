"""
RunPod Serverless Handler
Qwen/Qwen-Image model uchun API endpoint
"""
import runpod
import logging
import traceback
from src.model_loader import load_model, inference, get_model_info

# Logging sozlash
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """
    RunPod handler funksiyasi

    Args:
        event (dict): RunPod event ob'ekti

    Returns:
        dict: API javobi
    """
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
                "model": "Qwen/Qwen-Image",
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


# Health check endpoint (ixtiyoriy)
def health_check(event):
    """
    Health check funksiyasi
    """
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