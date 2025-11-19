import torch
import runpod
from diffusers import AutoPipelineForText2Image
import base64
from io import BytesIO

# Load model (only once thanks to network volume)
model_id = "Qwen/Qwen-Image"
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    use_safetensors=True
).to("cuda")


def handler(event):
    input_data = event.get("input", {})
    prompt = input_data.get("prompt", "")
    negative = input_data.get("negative_prompt", "")
    width = input_data.get("width", 1024)
    height = input_data.get("height", 1024)
    steps = input_data.get("num_inference_steps", 50)
    cfg = input_data.get("true_cfg_scale", 4.0)
    seed = input_data.get("seed", None)

    if seed is not None:
        torch.manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=cfg
    ).images[0]

    # Encode to Base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "image": image_b64,
        "seed": seed
    }


runpod.serverless.start({"handler": handler})
