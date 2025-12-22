# 🗑️ TrashVision - AI Agent za Klasifikaciju Otpada

**TrashVision** je inteligentni AI agent sistem koji koristi YOLO v8 za automatsku klasifikaciju otpada u realnom vremenu. Sistem je implementiran sa Domain-Driven Design (DDD) arhitekturom i autonomnim agentima za klasifikaciju i kontinuirano učenje.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123+-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 📋 Sadržaj

- [Ključne Karakteristike](#-ključne-karakteristike)
- [Arhitektura](#-arhitektura)
- [Instalacija](#-instalacija)
- [Pokretanje](#-pokretanje)
- [API Dokumentacija](#-api-dokumentacija)
- [Struktura Projekta](#-struktura-projekta)
- [Kategorije Otpada](#-kategorije-otpada)
- [Autonomni Agenti](#-autonomni-agenti)
- [Development](#-development)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Ključne Karakteristike

### 🤖 Autonomni AI Agenti
- **Classification Agent**: Automatski klasifikuje uploadovane slike svakih 2 sekunde
- **Learning Agent**: Periodično retrenira model sa novim uzorcima (svakih 60 sekundi)

### 🎯 Klasifikacija Otpada
- Podržava 6 kategorija: Karton, Staklo, Metal, Papir, Plastika, Trash (ostalo)
- YOLO v8 model sa ~95%+ tačnošću
- Confidence score i top-3 predikcije

### 🔄 Kontinuirano Učenje
- User feedback automatski dodaje uzorke u learning dataset
- Auto-retraining kada se sakupi dovoljno novih uzoraka (default: 10 uzoraka)
- Verzionisanje modela sa metrikama

### 📊 Monitoring i Statistika
- Real-time status svih agenata
- Broj procesuiranih slika
- Progress bar za retraining
- Queue status

### 🌐 RESTful API
- FastAPI sa automatskom Swagger dokumentacijom
- Upload slika za klasifikaciju
- Feedback sistem za korekcije
- Status endpointi

---

## 🏗️ Arhitektura

### Domain-Driven Design (DDD)

Projekat je organizovan po DDD principima:

```
AiAgents/TrashAgent/
├── Domain/              # Business logika i entiteti
│   ├── entities.py      # WasteImage, SystemSettings
│   ├── enums.py         # WasteCategory, ImageStatus
│   └── value_objects.py # RecyclingInfo
│
├── Application/         # Use case sloj
│   ├── Services/        # Business services
│   └── Agents/          # Agent runners
│
├── Infrastructure/      # Tehnički detalji
│   ├── database.py      # SQLAlchemy
│   ├── file_storage.py  # Disk operacije
│   ├── yolo_classifier.py # YOLO inference
│   └── waste_classifier.py # Abstrakcija
│
└── Web/                 # API layer
    ├── main.py          # FastAPI app
    ├── controllers/     # (deprecated)
    └── workers/         # Background agent workers
```

### Agent Arhitektura

Svaki agent implementira **Sense → Think → Act** ciklus:

```python
class SoftwareAgent(Generic[TPercept, TAction, TResult]):
    async def sense() -> Optional[TPercept]  # Opazi okolinu
    async def think(percept) -> TAction      # Donesi odluku
    async def act(action) -> TResult         # Izvrši akciju
```

**Classification Agent**:
- Sense: Čita sledeću sliku iz queue-a
- Think: Klasifikuje sliku sa YOLO modelom
- Act: Sačuva rezultat i ažurira status

**Learning Agent**:
- Sense: Proveri da li ima dovoljno novih uzoraka
- Think: Odluči da li pokrenuti retraining
- Act: Retrenira model i sačuva novu verziju

---

## 🚀 Instalacija

### Predusloviovi

- Python 3.11 ili noviji
- CUDA 12.4+ (opciono, za GPU podršku)
- 8GB+ RAM
- 2GB+ disk prostora za model

### Korak 1: Kloniranje repozitorijuma

```bash
git clone https://github.com/your-username/trashvision.git
cd trashvision
```

### Korak 2: Kreiranje virtualnog okruženja

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Korak 3: Instalacija zavisnosti

```bash
pip install -r requirements.txt
```

**Napomena**: Ako želite CPU verziju PyTorch-a, izbacite `--extra-index-url` liniju iz `requirements.txt`.

### Korak 4: Model Weights

**Model se nalazi u repozitorijumu** - ekstraktujte `trashvision_model_weights.zip`:

```bash
# Windows (PowerShell)
Expand-Archive -Path trashvision_model_weights.zip -DestinationPath models\trashvision_v1\ -Force

# Linux/Mac
unzip trashvision_model_weights.zip -d models/trashvision_v1/
```

**Šta će biti ekstraktovano:**
```
models/trashvision_v1/
└── weights/
    ├── best.pt   (2.85 MB) - Najbolji model za inference
    └── last.pt   (2.85 MB) - Posljednji checkpoint
```

**Napomena**: Aplikacija koristi `best.pt` za klasifikaciju.

---

## 🎮 Pokretanje

### Osnovni Start

```bash
python run_agent.py
```

Aplikacija će startovati na `http://localhost:8000`.

### Output

```
============================================================
🚀 Starting TrashVision Agent...
============================================================
📂 Initializing infrastructure...
✅ Database initialized
📥 Loading YOLO model: models/trashvision_v1/weights/best.pt
✅ Model loaded successfully
🤖 Starting Classification Agent...
✅ Classification worker started
🎓 Starting Learning Agent...
✅ Learning worker started

============================================================
✅ TRASHVISION AGENT READY!
============================================================
📍 API: http://localhost:8000
📚 Docs: http://localhost:8000/docs
🤖 Classification Agent: Running (every 2s)
🎓 Learning Agent: Running (every 60s)
============================================================
```

### Pristup Dokumentaciji

- **Frontend**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 📡 API Dokumentacija

### 1. Upload Slike

**Endpoint**: `POST /api/images/upload`

Uploaduje sliku i stavlja je u queue za klasifikaciju.

```bash
curl -X POST "http://localhost:8000/api/images/upload" \
  -F "file=@slika.jpg"
```

**Response**:
```json
{
  "success": true,
  "image_id": 123,
  "filename": "slika.jpg",
  "status": "queued",
  "message": "Image queued for classification"
}
```

### 2. Provjera Statusa

**Endpoint**: `GET /api/images/{image_id}`

Provjerava status klasifikacije.

```bash
curl "http://localhost:8000/api/images/123"
```

**Response**:
```json
{
  "image_id": 123,
  "filename": "slika.jpg",
  "status": "classified",
  "processed_at": "2025-12-23T14:30:00",
  "needs_review": false,
  "prediction": {
    "class": "plastic",
    "confidence": 0.95,
    "top3": [
      {"class": "plastic", "confidence": 0.95},
      {"class": "metal", "confidence": 0.03},
      {"class": "cardboard", "confidence": 0.01}
    ]
  }
}
```

### 3. Direktna Predikcija (Legacy)

**Endpoint**: `POST /predict`

Direktna, sinhronona predikcija (ne ide kroz agent queue).

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@slika.jpg"
```

**Response**:
```json
{
  "success": true,
  "predictions": [
    {
      "class": "plastic",
      "name": "Plastika",
      "confidence": 0.95,
      "disposal": "Žuti kontejner",
      "recyclable": true,
      "emoji": "♻️",
      "color": "yellow"
    }
  ]
}
```

### 4. User Feedback

**Endpoint**: `POST /feedback`

Korisnički feedback sa korekcijom predikcije.

```bash
curl -X POST "http://localhost:8000/feedback" \
  -F "file=@slika.jpg" \
  -F "predicted_class=plastic" \
  -F "actual_class=metal" \
  -F "confidence=0.95"
```

**Response**:
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

### 5. Learning Statistika

**Endpoint**: `GET /api/learning/stats`

Vraća statistiku learning-a.

```bash
curl "http://localhost:8000/api/learning/stats"
```

**Response**:
```json
{
  "new_samples_count": 5,
  "threshold": 10,
  "progress_percentage": 50.0,
  "auto_retrain_enabled": true,
  "last_retrain_at": "2025-12-23T10:00:00",
  "retrain_count": 3
}
```

### 6. System Status

**Endpoint**: `GET /status`

Vraća status cijelog sistema.

```bash
curl "http://localhost:8000/status"
```

**Response**:
```json
{
  "classification_agent": {
    "is_running": true,
    "total_processed": 150,
    "last_run": "2025-12-23T14:30:00",
    "run_count": 450
  },
  "learning_agent": {
    "is_running": true,
    "last_check": "2025-12-23T14:29:00",
    "check_count": 15
  },
  "database_connected": true,
  "model_loaded": true
}
```

---

## 🗂️ Struktura Projekta

```
trashvision/
│
├── AiAgents/                       # Core agent framework
│   ├── Core/                       # Bazne klase za agente
│   │   ├── software_agent.py       # Generic agent base
│   │   ├── perception_source.py    # Sensor interface
│   │   ├── actuator.py             # Action executor
│   │   ├── policy.py               # Decision strategy
│   │   └── learning_component.py   # Learning logic
│   │
│   └── TrashAgent/                 # Trash classification agent
│       ├── Domain/                 # Domain layer (DDD)
│       │   ├── entities.py         # WasteImage, SystemSettings
│       │   ├── enums.py            # Categories, statuses
│       │   └── value_objects.py    # RecyclingInfo
│       │
│       ├── Application/            # Use case layer
│       │   ├── Services/           # Business services
│       │   │   ├── queue_service.py
│       │   │   ├── classification_service.py
│       │   │   ├── review_service.py
│       │   │   └── training_service.py
│       │   └── Agents/             # Agent runners
│       │       ├── classification_runner.py
│       │       └── learning_runner.py
│       │
│       ├── Infrastructure/         # Technical layer
│       │   ├── database.py         # SQLAlchemy + models
│       │   ├── file_storage.py     # File operations
│       │   ├── waste_classifier.py # Classifier interface
│       │   └── yolo_classifier.py  # YOLO implementation
│       │
│       └── Web/                    # API layer
│           ├── main.py             # FastAPI app + routes
│           ├── dto/                # Response DTOs
│           └── workers/            # Background workers
│               ├── classification_worker.py
│               └── learning_worker.py
│
├── app/frontend/                   # Frontend (HTML/JS)
│   └── index.html
│
├── data/                           # Data storage
│   ├── uploads/                    # Uploaded images
│   └── new_samples/                # Learning dataset
│       ├── cardboard/
│       ├── glass/
│       ├── metal/
│       ├── paper/
│       ├── plastic/
│       └── trash/
│
├── models/                         # Trained models
│   └── trashvision_v1/
│       ├── weights/
│       │   ├── best.pt             # Best model
│       │   └── last.pt             # Last epoch
│       ├── args.yaml               # Training config
│       └── results.csv             # Training metrics
│
├── trashvision.db                  # SQLite database
├── requirements.txt                # Python dependencies
├── run_agent.py                    # Main launcher
└── README.md                       # This file
```

---

## 🗑️ Kategorije Otpada

Sistem podržava 6 kategorija:

| Kategorija | Klasa | Emoji | Kontejner | Reciklažno |
|-----------|-------|-------|-----------|------------|
| Karton | `cardboard` | 📦 | Plavi | ✅ Da |
| Staklo | `glass` | 🍾 | Zeleni | ✅ Da |
| Metal | `metal` | 🥫 | Žuti | ✅ Da |
| Papir | `paper` | 📄 | Plavi | ✅ Da |
| Plastika | `plastic` | 🧴 | Žuti | ✅ Da |
| Ostalo | `trash` | 🗑️ | Crni | ❌ Ne |

### Recycling Info

Svaka kategorija ima detaljne informacije:

```python
RecyclingInfo(
    is_recyclable=True,
    container_color="yellow",
    disposal_instruction="Ubacite u žuti kontejner",
    environmental_impact="Plastika se razgrađuje 450+ godina",
    fun_fact="1 tona reciklirane plastike = 700kg nafte"
)
```

---

## 🤖 Autonomni Agenti

### Classification Agent

**Svrha**: Automatski procesuira uploadovane slike.

**Tick Rate**: Svake 2 sekunde

**Workflow**:
1. **Sense**: Pročita najstariju sliku sa statusom `QUEUED` iz baze
2. **Think**: 
   - Klasifikuje sliku sa YOLO modelom
   - Računa confidence score
   - Odlučuje da li treba manual review (confidence < 70%)
3. **Act**: 
   - Sačuva predikciju u bazu
   - Ažurira status (`CLASSIFIED` ili `PENDING_REVIEW`)
   - Loguje rezultat

**Konfiguracija**:
```python
CLASSIFICATION_INTERVAL_SECONDS = 2
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.70
```

### Learning Agent

**Svrha**: Automatski retrenira model sa novim uzorcima.

**Tick Rate**: Svake 60 sekundi (provjerava threshold)

**Workflow**:
1. **Sense**: Provjeri broj novih uzoraka u `data/new_samples/`
2. **Think**: 
   - Odluči da li je dostignut threshold (default: 10 uzoraka)
   - Pripremi dataset za training
3. **Act**: 
   - Retrenira YOLO model (5 epoha, fine-tuning)
   - Sačuva novu verziju modela
   - Resetuje brojač novih uzoraka
   - Loguje metriku (accuracy, precision, recall)

**Konfiguracija**:
```python
LEARNING_CHECK_INTERVAL_SECONDS = 60
RETRAIN_THRESHOLD = 10  # Broj novih uzoraka
AUTO_RETRAIN_ENABLED = True
TRAINING_EPOCHS = 5
```

---

## 🛠️ Development

### Potrebni Alati

```bash
pip install black mypy pytest
```

### Code Formatting

```bash
black AiAgents/
```

### Type Checking

```bash
mypy AiAgents/ --ignore-missing-imports
```

### Testing

```bash
pytest tests/
```

### Development Mode (Auto-reload)

```python
# U run_agent.py, promjeni:
uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
```

---

## 🐛 Troubleshooting

### Problem: Model se ne učitava

**Simptom**:
```
⚠️  Model not found: models/trashvision_v1/weights/best.pt
```

**Rješenje**:
1. Provjerite da li postoji fajl `models/trashvision_v1/weights/best.pt`
2. Skinite pretreniran model ili trenirajte svoj
3. Provjerite putanju u [main.py](AiAgents/TrashAgent/Web/main.py#L99)

### Problem: CUDA nije dostupna

**Simptom**:
```
WARNING: CUDA not available, using CPU
```

**Rješenje**:
1. Instalirajte CUDA toolkit 12.4+
2. Instalirajte PyTorch sa CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
3. Provjerite: `python -c "import torch; print(torch.cuda.is_available())"`

### Problem: Database greška

**Simptom**:
```
sqlalchemy.exc.OperationalError: no such table
```

**Rješenje**:
```bash
# Obrišite bazu i ponovno je kreirajte
rm trashvision.db
python run_agent.py
```

### Problem: Port 8000 zauzet

**Simptom**:
```
ERROR: [Errno 10048] address already in use
```

**Rješenje**:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

---

## 📊 Performance

### Brzina Inference

- **GPU (RTX 3060)**: ~5ms po slici
- **CPU (Intel i7)**: ~150ms po slici

### Memory Usage

- **Model (YOLO v8n)**: ~6MB
- **Runtime**: ~200-300MB RAM
- **Database**: ~1MB po 1000 slika

### Throughput

- **Classification Agent**: ~30 slika/minut (sa 2s tick rate)
- **Training**: ~2-5 minuta za 100 slika (5 epoha)

---

## 🤝 Contributing

Contributor-i su dobrodošli! Molimo vas:

1. Forkujte repo
2. Kreirajte feature branch (`git checkout -b feature/AmazingFeature`)
3. Commitujte promjene (`git commit -m 'Add AmazingFeature'`)
4. Pushajte branch (`git push origin feature/AmazingFeature`)
5. Otvorite Pull Request

---

## 📜 Licenca

MIT License - slobodno koristite i modifikujte.

---

## 👨‍💻 Autor

**Nedim**  
GitHub: [@your-username](https://github.com/your-username)

---

## 🙏 Acknowledgments

- **YOLO v8** - Ultralytics za odličan object detection framework
- **FastAPI** - Za brz i moderan web framework
- **SQLAlchemy** - Za ORM koji olakšava rad sa bazom

---

## 📧 Kontakt

Za pitanja i sugestije:
- Email: your.email@example.com
- Issues: [GitHub Issues](https://github.com/your-username/trashvision/issues)

---

## 🤖 LLM u Razvoju

Ovaj projekat je razvijen uz asistenciju LLM-a kroz iterativni proces:

- **💬 Diskusija ideje**: Claude AI za brainstorming, evaluaciju i izbor koncepta
- **🏗️ Arhitektura**: Claude AI za DDD dizajn i Clean Architecture specifikaciju
- **💻 Implementacija**: GitHub Copilot za code generation (~80% koda)
- **🔍 Code Review**: GPT-4 i Copilot Chat za arhitekturni review i bug detection
- **📚 Dokumentacija**: Claude AI za generisanje README, ARCHITECTURE, API docs (~95%)
- **🔧 Refactoring**: GitHub Copilot za iterativne ispravke

**Multi-LLM Workflow**:
```
Claude AI (Concept) → Copilot (Code) → GPT-4 (Review) → Copilot (Fix) → Repeat
```

**Detaljno**: [LLM Usage Documentation](docs/LLM_USAGE.md)

**Conversation Log**: https://claude.ai/share/71369185-f519-48b4-978e-6d5c92f2f3be

---

**Happy Recycling! ♻️🌍**
