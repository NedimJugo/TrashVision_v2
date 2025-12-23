# 📡 TrashVision - API Dokumentacija

Kompletna dokumentacija REST API-ja za TrashVision Agent sistem.

**Base URL**: `http://localhost:8000`

---

## 📋 Sadržaj

- [Pregled](#-pregled)
- [Autentikacija](#-autentikacija)
- [Endpointi](#-endpointi)
  - [Upload Slika](#1-upload-slike)
  - [Provjera Statusa](#2-provjera-statusa-slike)
  - [Direktna Predikcija](#3-direktna-predikcija-legacy)
  - [User Feedback](#4-user-feedback)
  - [Learning Statistika](#5-learning-statistika)
  - [System Status](#6-system-status)
- [Status Kodovi](#-status-kodovi)
- [Rate Limiting](#-rate-limiting)
- [Primjeri Klijenta](#-primjeri-klijenta)
- [Swagger UI](#-swagger-ui)

---

## 🌐 Pregled

TrashVision API omogućava:
- ✅ Upload slika za klasifikaciju
- ✅ Provjeru statusa procesuiranja
- ✅ Direktnu sinhrononu klasifikaciju
- ✅ Korisnički feedback za učenje
- ✅ Monitoring sistema i statistiku

**API Verzija**: `2.0.0`  
**Format**: JSON  
**Charset**: UTF-8  

---

## 🔑 Autentikacija

Trenutno API **nema autentikaciju** (otvoreni endpoint za development).

**Produkcijska implementacija** bi koristila JWT tokens:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/api/images/upload
```

---

## 📡 Endpointi

### 1. Upload Slike

Uploaduje sliku i stavlja je u queue za automatsku klasifikaciju od strane Classification Agent-a.

**Endpoint**: `POST /api/images/upload`

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file` (required): Image file (JPEG, PNG, WebP)

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/images/upload" \
  -F "file=@/path/to/image.jpg"
```

**Python Example**:
```python
import requests

with open('bottle.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/images/upload',
        files={'file': f}
    )
    
data = response.json()
print(data['image_id'])  # Track this ID
```

**JavaScript Example**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/api/images/upload', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(data.image_id);
```

**Response** (200 OK):
```json
{
  "success": true,
  "image_id": 123,
  "filename": "bottle.jpg",
  "status": "queued",
  "message": "Image queued for classification"
}
```

**Error Response** (400 Bad Request):
```json
{
  "detail": "File must be an image"
}
```

**Error Response** (500 Internal Server Error):
```json
{
  "detail": "Failed to save image: disk full"
}
```

**Status Lifecycle**:
```
queued → classified (confidence > 70%)
queued → pending_review (confidence < 70%)
```

---

### 2. Provjera Statusa Slike

Provjerava trenutni status i rezultate klasifikacije za uploadovanu sliku.

**Endpoint**: `GET /api/images/{image_id}`

**Parameters**:
- `image_id` (path, required): ID slike dobijen iz upload response-a

**cURL Example**:
```bash
curl "http://localhost:8000/api/images/123"
```

**Python Example**:
```python
import requests

response = requests.get('http://localhost:8000/api/images/123')
data = response.json()

if data['status'] == 'classified':
    print(f"Category: {data['prediction']['class']}")
    print(f"Confidence: {data['prediction']['confidence']}")
```

**Response** (200 OK) - **Status: queued**:
```json
{
  "image_id": 123,
  "filename": "bottle.jpg",
  "status": "queued",
  "processed_at": null,
  "needs_review": false,
  "prediction": null
}
```

**Response** (200 OK) - **Status: classified**:
```json
{
  "image_id": 123,
  "filename": "bottle.jpg",
  "status": "classified",
  "processed_at": "2025-12-23T14:30:15.123456",
  "needs_review": false,
  "prediction": {
    "class": "plastic",
    "confidence": 0.95,
    "top3": [
      {"class": "plastic", "confidence": 0.95},
      {"class": "metal", "confidence": 0.03},
      {"class": "glass", "confidence": 0.01}
    ]
  }
}
```

**Response** (200 OK) - **Status: pending_review**:
```json
{
  "image_id": 123,
  "filename": "bottle.jpg",
  "status": "pending_review",
  "processed_at": "2025-12-23T14:30:15.123456",
  "needs_review": true,
  "prediction": {
    "class": "plastic",
    "confidence": 0.62,
    "top3": [
      {"class": "plastic", "confidence": 0.62},
      {"class": "glass", "confidence": 0.30},
      {"class": "metal", "confidence": 0.05}
    ]
  }
}
```

**Error Response** (404 Not Found):
```json
{
  "detail": "Image not found"
}
```

---

### 3. Direktna Predikcija (Legacy)

**Sinhronona** klasifikacija koja ne ide kroz agent queue. Koristi se za kompatibilnost sa starim frontendom.

**Endpoint**: `POST /predict`

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file` (required): Image file

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@bottle.jpg"
```

**Python Example**:
```python
import requests

with open('can.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    
data = response.json()
print(f"Category: {data['predictions'][0]['class']}")
print(f"Disposal: {data['predictions'][0]['disposal']}")
```

**Response** (200 OK):
```json
{
  "success": true,
  "predictions": [
    {
      "class": "plastic",
      "name": "Plastika",
      "confidence": 0.95,
      "disposal": "Žuti kontejner za plastiku",
      "recyclable": true,
      "emoji": "♻️",
      "color": "yellow"
    },
    {
      "class": "metal",
      "name": "Metal",
      "confidence": 0.03
    },
    {
      "class": "glass",
      "name": "Staklo",
      "confidence": 0.01
    }
  ]
}
```

**Response Fields**:
- `class`: Kategorija (cardboard, glass, metal, paper, plastic, trash)
- `name`: Prikazano ime na lokalnom jeziku
- `confidence`: Preciznost (0.0 - 1.0)
- `disposal`: Instrukcije za odlaganje
- `recyclable`: Da li je reciklažno (true/false)
- `emoji`: Emoji za prikaz
- `color`: Boja kontejnera

**Kategorije**:

| Class | Name | Emoji | Disposal | Recyclable |
|-------|------|-------|----------|------------|
| `cardboard` | Karton | 📦 | Plavi kontejner | ✅ |
| `glass` | Staklo | 🍾 | Zeleni kontejner | ✅ |
| `metal` | Metal | 🥫 | Žuti kontejner | ✅ |
| `paper` | Papir | 📄 | Plavi kontejner | ✅ |
| `plastic` | Plastika | 🧴 | Žuti kontejner | ✅ |
| `trash` | Ostalo | 🗑️ | Crni kontejner | ❌ |

**Error Response** (400 Bad Request):
```json
{
  "detail": "File must be an image"
}
```

**Error Response** (500 Internal Server Error):
```json
{
  "detail": "Model not loaded"
}
```

---

### 4. User Feedback

Omogućava korisnicima da potvrde ili isprave predikciju. Koristise za **continuous learning**.

**Endpoint**: `POST /feedback`

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file` (required): Image file
  - `predicted_class` (optional): Originalna predikcija modela
  - `actual_class` (required): Ispravna kategorija (korisnikova korekcija)
  - `confidence` (optional): Confidence score originalne predikcije

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/feedback" \
  -F "file=@bottle.jpg" \
  -F "predicted_class=metal" \
  -F "actual_class=plastic" \
  -F "confidence=0.68"
```

**Python Example**:
```python
import requests

with open('bottle.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/feedback',
        files={'file': f},
        data={
            'predicted_class': 'metal',
            'actual_class': 'plastic',
            'confidence': 0.68
        }
    )
    
data = response.json()
print(f"Progress: {data['progress_percentage']}%")
print(f"Should retrain: {data['should_retrain']}")
```

**JavaScript Example**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('predicted_class', 'metal');
formData.append('actual_class', 'plastic');
formData.append('confidence', 0.68);

const response = await fetch('http://localhost:8000/feedback', {
  method: 'POST',
  body: formData
});

const data = await response.json();
console.log(`New samples: ${data.new_samples_count}/${data.threshold}`);
```

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Hvala na feedbacku!",
  "should_retrain": false,
  "new_samples_count": 5,
  "threshold": 10,
  "progress_percentage": 50.0
}
```

**Response** (200 OK) - **Threshold dostignut**:
```json
{
  "success": true,
  "message": "Hvala na feedbacku!",
  "should_retrain": true,
  "new_samples_count": 10,
  "threshold": 10,
  "progress_percentage": 100.0
}
```

**Behaviour**:
1. Slika se čuva u `data/new_samples/{actual_class}/`
2. Brojač novih uzoraka se inkrementira
3. Kada se dosegne threshold (default: 10), **Learning Agent** automatski retrenira model

**Error Response** (400 Bad Request):
```json
{
  "detail": "Invalid category: invalid_class"
}
```

---

### 5. Learning Statistika

Prikazuje trenutno stanje learning sistema i progress prema retraining-u.

**Endpoint**: `GET /api/learning/stats`

**cURL Example**:
```bash
curl "http://localhost:8000/api/learning/stats"
```

**Python Example**:
```python
import requests

response = requests.get('http://localhost:8000/api/learning/stats')
stats = response.json()

print(f"Progress: {stats['progress_percentage']:.1f}%")
print(f"Samples: {stats['new_samples_count']}/{stats['threshold']}")
```

**Response** (200 OK):
```json
{
  "new_samples_count": 5,
  "threshold": 10,
  "progress_percentage": 50.0,
  "auto_retrain_enabled": true,
  "last_retrain_at": "2025-12-23T10:00:00.123456",
  "retrain_count": 3
}
```

**Response Fields**:
- `new_samples_count`: Broj novih uzoraka od zadnjeg retraining-a
- `threshold`: Potreban broj uzoraka za automatski retraining
- `progress_percentage`: Procenat prema threshold-u (0-100)
- `auto_retrain_enabled`: Da li je automatski retraining uključen
- `last_retrain_at`: Timestamp zadnjeg retraining-a (ISO 8601)
- `retrain_count`: Ukupan broj izvršenih retraining-a

**UI Example**:
```
Progress bar: [=====-----] 50%
Samples collected: 5/10
Last retraining: 2 hours ago
```

---

### 6. System Status

Vraća zdravlje i status cijelog sistema (agenti, baza, model).

**Endpoint**: `GET /status`

**cURL Example**:
```bash
curl "http://localhost:8000/status"
```

**Python Example**:
```python
import requests

response = requests.get('http://localhost:8000/status')
status = response.json()

if not status['model_loaded']:
    print("⚠️ Model nije učitan!")

print(f"Classification: {status['classification_agent']['total_processed']} processed")
```

**Response** (200 OK):
```json
{
  "classification_agent": {
    "is_running": true,
    "total_processed": 150,
    "last_run": "2025-12-23T14:30:00.123456",
    "run_count": 450
  },
  "learning_agent": {
    "is_running": true,
    "last_check": "2025-12-23T14:29:00.123456",
    "check_count": 15
  },
  "database_connected": true,
  "model_loaded": true
}
```

**Response Fields**:

**classification_agent**:
- `is_running`: Da li je agent aktivan
- `total_processed`: Ukupan broj procesuiranih slika
- `last_run`: Zadnji put kada je agent radio
- `run_count`: Broj pokretanja agenta (tick count)

**learning_agent**:
- `is_running`: Da li je agent aktivan
- `last_check`: Zadnji put kada je agent provjerio threshold
- `check_count`: Broj provjera

**System**:
- `database_connected`: Da li je baza dostupna
- `model_loaded`: Da li je YOLO model učitan u memoriju

**Health Check Logic**:
```python
if not status['database_connected']:
    # Critical error
    
if not status['model_loaded']:
    # Critical error
    
if not status['classification_agent']['is_running']:
    # Warning: agent down
```

---

## 📊 Status Kodovi

| Kod | Naziv | Značenje |
|-----|-------|----------|
| **200** | OK | Uspješan request |
| **400** | Bad Request | Loš format ili validacija |
| **404** | Not Found | Resurs ne postoji |
| **422** | Unprocessable Entity | Validacioni error (FastAPI) |
| **500** | Internal Server Error | Server greška |

**Common Error Scenarios**:

| Scenario | Status | Detail |
|----------|--------|--------|
| Invalid image format | 400 | "File must be an image" |
| Image not found | 404 | "Image not found" |
| Invalid category | 400 | "Invalid category: xyz" |
| Model not loaded | 500 | "Model not loaded" |
| Database error | 500 | "Database connection failed" |

---

## ⏱️ Rate Limiting

Trenutno **nema rate limiting-a** (development).

**Produkcijska preporuka**:
```
- 10 requests/minute za /api/images/upload
- 100 requests/minute za GET endpointe
- Unlimited za /status (health checks)
```

**Implementacija sa slowapi**:
```python
from slowapi import Limiter

@app.post("/api/images/upload")
@limiter.limit("10/minute")
async def upload_image(file: UploadFile):
    pass
```

**Response Headers**:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1640269200
```

**Error Response** (429 Too Many Requests):
```json
{
  "detail": "Rate limit exceeded. Try again in 30 seconds."
}
```

---

## 💻 Primjeri Klijenta

### Python Client

```python
import requests
from pathlib import Path

class TrashVisionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def upload_image(self, image_path: str):
        """Upload sliku za klasifikaciju"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/api/images/upload",
                files={'file': f}
            )
        return response.json()
    
    def get_status(self, image_id: int):
        """Provjeri status slike"""
        response = requests.get(
            f"{self.base_url}/api/images/{image_id}"
        )
        return response.json()
    
    def predict_sync(self, image_path: str):
        """Direktna predikcija"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/predict",
                files={'file': f}
            )
        return response.json()
    
    def submit_feedback(self, image_path: str, predicted: str, actual: str):
        """Pošalji feedback"""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/feedback",
                files={'file': f},
                data={
                    'predicted_class': predicted,
                    'actual_class': actual
                }
            )
        return response.json()
    
    def get_learning_stats(self):
        """Learning statistika"""
        response = requests.get(f"{self.base_url}/api/learning/stats")
        return response.json()
    
    def health_check(self):
        """System status"""
        response = requests.get(f"{self.base_url}/status")
        return response.json()

# Usage
client = TrashVisionClient()

# Upload
result = client.upload_image("bottle.jpg")
image_id = result['image_id']

# Poll status
import time
while True:
    status = client.get_status(image_id)
    if status['status'] == 'classified':
        print(f"Result: {status['prediction']['class']}")
        break
    time.sleep(1)

# Feedback
client.submit_feedback("bottle.jpg", "metal", "plastic")
```

### JavaScript/TypeScript Client

```typescript
class TrashVisionClient {
  constructor(private baseUrl: string = 'http://localhost:8000') {}

  async uploadImage(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${this.baseUrl}/api/images/upload`, {
      method: 'POST',
      body: formData
    });
    
    return response.json();
  }

  async getStatus(imageId: number): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/images/${imageId}`);
    return response.json();
  }

  async predictSync(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      body: formData
    });
    
    return response.json();
  }

  async submitFeedback(
    file: File, 
    predicted: string, 
    actual: string
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('predicted_class', predicted);
    formData.append('actual_class', actual);
    
    const response = await fetch(`${this.baseUrl}/feedback`, {
      method: 'POST',
      body: formData
    });
    
    return response.json();
  }

  async getLearningStats(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/learning/stats`);
    return response.json();
  }

  async healthCheck(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/status`);
    return response.json();
  }
}

// Usage
const client = new TrashVisionClient();

// Upload
const fileInput = document.getElementById('file') as HTMLInputElement;
const file = fileInput.files[0];
const result = await client.uploadImage(file);

// Poll status
const imageId = result.image_id;
while (true) {
  const status = await client.getStatus(imageId);
  if (status.status === 'classified') {
    console.log('Category:', status.prediction.class);
    break;
  }
  await new Promise(resolve => setTimeout(resolve, 1000));
}
```

### cURL Batch Script

```bash
#!/bin/bash

# Upload multiple images
for image in images/*.jpg; do
  echo "Uploading $image..."
  curl -s -X POST "http://localhost:8000/api/images/upload" \
    -F "file=@$image" | jq '.image_id'
done

# Get learning stats
echo "Learning progress:"
curl -s "http://localhost:8000/api/learning/stats" | jq '.progress_percentage'
```

---

## 📖 Swagger UI

FastAPI automatski generiše **interaktivnu dokumentaciju**:

### Swagger UI
**URL**: http://localhost:8000/docs

Features:
- ✅ Isprobaj sve endpointe direktno iz browser-a
- ✅ Vidi request/response sheme
- ✅ Test upload-a slika
- ✅ Export OpenAPI spec

### ReDoc
**URL**: http://localhost:8000/redoc

Features:
- ✅ Čitljivija dokumentacija
- ✅ Searchable
- ✅ Better for reading (ne za testiranje)

### OpenAPI JSON
**URL**: http://localhost:8000/openapi.json

Download OpenAPI 3.0 specifikaciju za:
- Postman import
- API client generatore
- Automated testing tools

---

## 🔄 API Workflow Examples

### Example 1: Basic Upload & Classification

```python
import requests
import time

# 1. Upload
with open('bottle.jpg', 'rb') as f:
    resp = requests.post('http://localhost:8000/api/images/upload', files={'file': f})
    image_id = resp.json()['image_id']

# 2. Wait for classification (max 10s)
for _ in range(10):
    resp = requests.get(f'http://localhost:8000/api/images/{image_id}')
    data = resp.json()
    
    if data['status'] == 'classified':
        print(f"✅ Category: {data['prediction']['class']}")
        print(f"   Confidence: {data['prediction']['confidence']}")
        break
    
    time.sleep(1)
```

### Example 2: Feedback Loop

```python
# 1. Predict
with open('can.jpg', 'rb') as f:
    resp = requests.post('http://localhost:8000/predict', files={'file': f})
    prediction = resp.json()['predictions'][0]

# 2. User corrects
predicted_class = prediction['class']  # "plastic"
actual_class = "metal"  # User correction

# 3. Submit feedback
with open('can.jpg', 'rb') as f:
    resp = requests.post(
        'http://localhost:8000/feedback',
        files={'file': f},
        data={
            'predicted_class': predicted_class,
            'actual_class': actual_class,
            'confidence': prediction['confidence']
        }
    )
    
feedback = resp.json()
print(f"Progress: {feedback['progress_percentage']}%")

if feedback['should_retrain']:
    print("🎓 Model will retrain soon!")
```

### Example 3: Monitoring Dashboard

```python
import requests
import time

while True:
    # System status
    status = requests.get('http://localhost:8000/status').json()
    
    # Learning stats
    learning = requests.get('http://localhost:8000/api/learning/stats').json()
    
    print(f"""
    ========================================
    📊 TrashVision Dashboard
    ========================================
    Classification Agent: {"✅" if status['classification_agent']['is_running'] else "❌"}
      - Processed: {status['classification_agent']['total_processed']}
      - Runs: {status['classification_agent']['run_count']}
    
    Learning Agent: {"✅" if status['learning_agent']['is_running'] else "❌"}
      - Progress: {learning['progress_percentage']:.1f}%
      - Samples: {learning['new_samples_count']}/{learning['threshold']}
      - Retrains: {learning['retrain_count']}
    
    System:
      - DB: {"✅" if status['database_connected'] else "❌"}
      - Model: {"✅" if status['model_loaded'] else "❌"}
    ========================================
    """)
    
    time.sleep(5)
```

---

## 🆕 Advanced API Features (NOVO!)

### Cost-Aware Classification

Nema poseban endpoint - **automatski integrisano** u postojeće endpointe!

Kada je `use_optimizer=True` u `ClassificationService`, svi `/api/images/upload` i `/predict` endpointi automatski koriste cost-aware decision making.

**Interna logika:**
```python
# Umjesto:
decision = max(probabilities)  # Klasični pristup

# Sad:
decision = DecisionOptimizer.optimize_decision(probabilities)
# ↑ Uzima u obzir error cost matrix!
```

**Response isti**, ali odluka je pametnija:
```json
{
  "predicted_class": "metal",
  "confidence": 0.55,
  "_cost_reasoning": "Expected cost: 0.45 (optimized vs paper: 1.65)",
  "_is_fallback": false
}
```

### Error Cost Matrix Query

**Endpoint**: `GET /api/cost-matrix`

Query error cost matrix.

**cURL Example:**
```bash
curl "http://localhost:8000/api/cost-matrix"
```

**Response** (200 OK):
```json
{
  "matrix": {
    "metal_to_paper": 3.0,
    "paper_to_metal": 1.0,
    "battery_to_plastic": 5.0,
    "paper_to_cardboard": 0.3
  },
  "legend": {
    "0.0": "Correct classification",
    "0.3-1.0": "Minor/normal error",
    "2.0-3.0": "High cost (contamination)",
    "5.0": "CRITICAL (hazard)"
  }
}
```

**Specific Cost Query:**

**Endpoint**: `GET /api/cost-matrix/{true_category}/{predicted_category}`

```bash
curl "http://localhost:8000/api/cost-matrix/metal/paper"
```

**Response**:
```json
{
  "true_category": "metal",
  "predicted_category": "paper",
  "cost": 3.0,
  "severity": "HIGH",
  "reason": "Metal contaminates paper recycling stream"
}
```

### Simulation Status

**Endpoint**: `GET /api/simulation/status`

Get current simulation state (ako je simulation pokrenut).

**Response**:
```json
{
  "is_running": true,
  "simulation_time_sec": 45.2,
  "total_processed": 15,
  "correct_sorts": 12,
  "incorrect_sorts": 2,
  "uncertain_sorts": 1,
  "accuracy_percent": 80.0,
  "total_cost": 3.50,
  "average_cost_per_item": 0.23,
  "items_on_belt": 5,
  "robot_state": "picking",
  "bins": {
    "plastic": {"items": 5, "weight_kg": 2.5, "contamination": 0},
    "metal": {"items": 4, "weight_kg": 4.8, "contamination": 1},
    "paper": {"items": 3, "weight_kg": 0.9, "contamination": 0}
  }
}
```

### Decision Explanation

**Endpoint**: `POST /api/explain-decision`

Explain why a specific decision was made.

**Request:**
```json
{
  "probabilities": {
    "metal": 0.55,
    "paper": 0.40,
    "glass": 0.05
  }
}
```

**Response:**
```json
{
  "classic_decision": {
    "category": "metal",
    "confidence": 0.55,
    "reasoning": "Highest probability"
  },
  "cost_aware_decision": {
    "category": "metal",
    "confidence": 0.55,
    "expected_cost": 0.45,
    "reasoning": "Minimizes expected cost",
    "alternatives": {
      "paper": {
        "expected_cost": 1.65,
        "reasoning": "High cost if true category is metal (3.0)"
      },
      "glass": {
        "expected_cost": 0.95,
        "reasoning": "Low probability makes this risky"
      }
    }
  },
  "recommendation": "metal",
  "cost_savings": "59% lower cost vs worst case"
}
```

---

## 📞 Podrška

Za probleme sa API-jem:
- **GitHub Issues**: https://github.com/your-username/trashvision/issues
- **Email**: your.email@example.com
- **Swagger UI**: http://localhost:8000/docs (za testiranje)

---

## 📜 Changelog

### Version 2.1.0 (Current) - ADVANCED FEATURES
- ✅ **Cost-Aware Decision Making** - Optimizer integrisano u classification
- ✅ **Error Cost Matrix** - 10x10 matrica troškova greške
- ✅ **Decision Optimizer** - Expected cost minimization
- ✅ **Confidence Thresholds** - 3 nivoa sa fallback strategijama
- ✅ **Sorting Simulation** - Robotic sorting sa cost tracking
- ✅ Novi `/api/cost-matrix` endpointi
- ✅ `/api/simulation/status` za simulation monitoring
- ✅ `/api/explain-decision` za decision reasoning

### Version 2.0.0 (Legacy)
- ✅ Novi `/api/images/upload` sa queue sistemom
- ✅ Agent-based processing
- ✅ `/api/learning/stats` endpoint
- ✅ `/status` system health check

### Version 1.0.0
- ✅ `/predict` direktna predikcija
- ✅ `/feedback` korisnički feedback

---

**Autor**: Nedim  
**Verzija**: 2.1.0  
**Datum**: 2025-12-23
