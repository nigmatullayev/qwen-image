# 🚀 Qwen-Image RunPod Serverless Deployment

Bu loyiha **Qwen/Qwen-Image** modelini RunPod Serverless platformasida deploy qilish uchun mo'ljallangan.

## 📋 Talablar

- RunPod akkaunti
- GitHub repository
- RunPod Network Volume (30GB tavsiya)
- GPU: T4 / A10G / L40

## 📁 Loyiha tuzilmasi

```
repo-root/
│
├── handler.py              # Asosiy RunPod handler
├── requirements.txt        # Python bog'liqliklar
├── runpod.toml            # RunPod konfiguratsiya
├── README.md              # Dokumentatsiya
├── .gitignore             # Git ignore fayllar
├── test_api.py            # API test skripti
│
└── src/
    ├── __init__.py        # Package init
    └── model_loader.py    # Model yuklash va inference
```

## 🔧 O'rnatish

### 1. GitHub Repository yarating

```bash
git init
git add .
git commit -m "Initial commit: Qwen-Image RunPod deployment"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. RunPod Network Volume yarating

1. RunPod → Storage → Create Network Volume
2. Nom: `qwen-image-cache`
3. Hajm: 30GB
4. Mount path: `/runpod-volume`

### 3. RunPod Serverless Endpoint yarating

1. **RunPod Dashboard** → **Serverless** → **Create Endpoint**

2. **Endpoint Settings:**
   - Name: `qwen-image-api`
   - GPU: T4 / A10G / L40
   - Workers: Auto-scale

3. **Container Setup:**
   - Source: **GitHub Repository**
   - Repository URL: `https://github.com/<username>/<repo>`
   - Branch: `main`

4. **Network Volume:**
   - Attach: `qwen-image-cache`
   - Mount path: `/runpod-volume`

5. **Environment Variables:**
   ```
   TRANSFORMERS_CACHE=/runpod-volume/qwen_image
   HF_HOME=/runpod-volume/qwen_image
   HF_TOKEN=<your-huggingface-token>  # Agar kerak bo'lsa
   ```

6. **Deploy** tugmasini bosing

## 📡 API Ishlatish

### cURL misoli

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run \
  -H "Authorization: Bearer <YOUR_RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a cute cat in cyberpunk city",
      "max_new_tokens": 512,
      "temperature": 0.7
    }
  }'
```

### Python misoli

```python
import requests

ENDPOINT_ID = "your-endpoint-id"
API_KEY = "your-runpod-api-key"

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "input": {
        "prompt": "a beautiful sunset over mountains",
        "max_new_tokens": 300,
        "temperature": 0.8
    }
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### JavaScript misoli

```javascript
const ENDPOINT_ID = 'your-endpoint-id';
const API_KEY = 'your-runpod-api-key';

const response = await fetch(`https://api.runpod.ai/v2/${ENDPOINT_ID}/run`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    input: {
      prompt: 'a futuristic cityscape',
      max_new_tokens: 400,
      temperature: 0.7
    }
  })
});

const result = await response.json();
console.log(result);
```

## 📥 Request Formati

```json
{
  "input": {
    "prompt": "your prompt here",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "image_path": null
  }
}
```

### Parametrlar

- `prompt` (string, majburiy): Matn prompti
- `max_new_tokens` (int, ixtiyoriy): Maksimal generatsiya uzunligi (default: 512)
- `temperature` (float, ixtiyoriy): Generatsiya harorati (default: 0.7)
- `image_path` (string, ixtiyoriy): Rasm fayli yo'li

## 📤 Response Formati

```json
{
  "output": {
    "generated_text": "...",
    "model": "Qwen/Qwen-Image",
    "prompt": "your prompt",
    "parameters": {
      "max_new_tokens": 512,
      "temperature": 0.7
    }
  },
  "status": "success"
}
```

## 🧪 Test qilish

Test skriptini ishlatish:

```bash
# Environment variables o'rnatish
export RUNPOD_ENDPOINT_ID="your-endpoint-id"
export RUNPOD_API_KEY="your-api-key"

# Test skriptini ishga tushirish
python test_api.py
```

## 🔒 Xavfsizlik

- ✅ HuggingFace tokenlar faqat Environment Variables'da
- ✅ API keylari GitHub'ga yuklanmaydi
- ✅ HTTPS protokoli ishlatiladi
- ✅ RunPod authentication

## ⚡ Optimallashtirish

Loyiha quyidagi optimallashtirish usullarini ishlatadi:

- `torch.float16` - xotira iste'molini kamaytiradi
- `device_map="auto"` - avtomatik GPU distribution
- `torch.no_grad()` - gradient hisoblashni o'chiradi
- `model.eval()` - inference rejimi
- Network Volume - model cache'ini saqlash

## 🐛 Debugging

### Loglarni ko'rish

RunPod Dashboard → Endpoint → Logs

### Keng tarqalgan muammolar

1. **Model yuklanmayapti:**
   - Network Volume to'g'ri ulangan ekanligini tekshiring
   - Environment variables to'g'ri sozlanganligini tekshiring

2. **GPU xotirasi yetmayapti:**
   - Katta GPU tanlang (A10G / L40)
   - `max_new_tokens` ni kamaytiring

3. **Timeout xatoliklari:**
   - Worker timeout'ni oshiring
   - Model cache'ini to'g'ri sozlang

## 📊 Monitoring

RunPod Dashboard'da:
- Request statistikasi
- GPU ishlatilishi
- Xatoliklar loglari
- Response time

## 🔄 Yangilash

```bash
# Kodni yangilang
git add .
git commit -m "Update: new features"
git push

# RunPod avtomatik yangilanadi
```

## 📞 Yordam

Muammolar yuzaga kelsa:
- [RunPod Documentation](https://docs.runpod.io)
- [HuggingFace Qwen Model](https://huggingface.co/Qwen/Qwen-Image)
- GitHub Issues

## 📝 License

MIT License

---

**Yaratilgan sana:** 2025
**Model:** Qwen/Qwen-Image
