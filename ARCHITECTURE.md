# 🏗️ TrashVision - Arhitekturna Dokumentacija

Ovaj dokument opisuje tehničku arhitekturu TrashVision projekta, dizajn odluke, i obrazloženje implementiranih pattern-a.

---

## 📋 Sadržaj

- [Pregled Arhitekture](#-pregled-arhitekture)
- [Domain-Driven Design (DDD)](#-domain-driven-design-ddd)
- [Agent Arhitektura](#-agent-arhitektura)
- [Slojevi Sistema](#-slojevi-sistema)
- [Data Flow](#-data-flow)
- [Dependency Management](#-dependency-management)
- [Design Patterns](#-design-patterns)
- [Database Schema](#-database-schema)
- [File Storage](#-file-storage)
- [API Layer](#-api-layer)
- [Background Workers](#-background-workers)
- [Scalability & Performance](#-scalability--performance)
- [Security Considerations](#-security-considerations)
- [Future Improvements](#-future-improvements)

---

## 🎯 Pregled Arhitekture

TrashVision koristi **Clean Architecture** i **Domain-Driven Design (DDD)** principe za organizaciju koda. Sistem je podijeljen na jasne slojeve sa definisanim zavisnostima.

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Web Layer                            │
│  (FastAPI, REST API, HTTP requests/responses)                │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                    Application Layer                         │
│  (Use Cases, Services, Agent Runners, Business Logic)        │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                      Domain Layer                            │
│  (Entities, Value Objects, Enums, Business Rules)            │
└──────────────────▲──────────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────────┐
│                  Infrastructure Layer                        │
│  (Database, File Storage, YOLO Model, External Services)     │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Rule

**Pravilo zavisnosti**: Unutrašnji slojevi NE znaju za spoljašnje.

```
Domain ← Application ← Web
   ↑          ↑
   └─Infrastructure─┘
```

- **Domain**: Nema zavisnosti na ostale slojeve
- **Application**: Zavisi od Domain-a
- **Infrastructure**: Zavisi od Domain-a (implementira interface-e)
- **Web**: Zavisi od Application i Infrastructure

---

## 🏛️ Domain-Driven Design (DDD)

### Ubiquitous Language

Koristimo **jedinstveni jezik** kroz cijeli projekat:

| Termin | Značenje |
|--------|----------|
| **WasteImage** | Slika otpada koja se klasifikuje |
| **Percept** | Opažanje koje agent dobija (slika iz queue-a) |
| **Classification** | Proces kategorizacije otpada |
| **Prediction** | Rezultat ML modela |
| **Review** | Manual provjera low-confidence predikcija |
| **Learning** | Proces kontinuiranog poboljšanja modela |
| **Retraining** | Treniranje modela sa novim uzorcima |

### Bounded Contexts

Projekat ima **jedan bounded context**: **Waste Classification**

Ako bi se proširivao, mogli bi se dodati:
- **User Management** (autentikacija, dozvole)
- **Analytics** (statistika, reporti)
- **Billing** (plaćanje za API calls)

### Aggregates

**WasteImage** je glavni aggregate root:

```python
class WasteImage:
    """
    Aggregate Root: Waste Image
    
    Enkapsulira:
    - Metadata slike (filename, size, dimensions)
    - Status lifecycle (queued → classified → reviewed)
    - Predikciju (category, confidence)
    """
    id: Optional[int]
    filepath: str
    filename: str
    status: ImageStatus  # Enum
    
    # Prediction data
    predicted_category: Optional[WasteCategory]
    confidence: Optional[float]
    
    # Timestamps
    uploaded_at: datetime
    processed_at: Optional[datetime]
```

**Invariant Rules**:
- Slika mora imati filepath i filename
- Status transition: `QUEUED → CLASSIFIED → REVIEWED`
- Confidence mora biti između 0.0 i 1.0
- `processed_at` se postavlja samo kada je `status = CLASSIFIED`

---

## 🤖 Agent Arhitektura

### Generic Agent Framework

Bazirana na **Sense-Think-Act** paradigmi autonomnih agenata.

```python
class SoftwareAgent(Generic[TPercept, TAction, TResult]):
    """
    Generic parametri:
    - TPercept: Tip percepta (što agent opaža)
    - TAction: Tip akcije (odluka)
    - TResult: Tip rezultata (outcome)
    """
    
    @abstractmethod
    async def sense() -> Optional[TPercept]:
        """Opazi okolinu - čitaj iz queue-a, senzora, etc."""
        pass
    
    @abstractmethod
    async def think(percept: TPercept) -> TAction:
        """Donesi odluku - ML inference, rule engine, etc."""
        pass
    
    @abstractmethod
    async def act(action: TAction) -> TResult:
        """Izvrši akciju - save to DB, send notification, etc."""
        pass
```

### Classification Agent

```python
ClassificationAgent(
    SoftwareAgent[
        WasteImage,           # TPercept (slika iz queue-a)
        ClassificationAction, # TAction (predict + status)
        ClassificationResult  # TResult (saved prediction)
    ]
)
```

**Workflow**:

1. **Sense**: 
   ```python
   image = await queue_service.get_next_queued()
   # SELECT * FROM waste_images WHERE status = 'queued' LIMIT 1
   ```

2. **Think**:
   ```python
   prediction = await classifier.predict(image.filepath)
   action = ClassificationAction(
       image_id=image.id,
       category=prediction["class"],
       confidence=prediction["confidence"],
       status=CLASSIFIED if confidence > 0.7 else PENDING_REVIEW
   )
   ```

3. **Act**:
   ```python
   result = await classification_service.save_prediction(action)
   # UPDATE waste_images SET status = ..., predicted_category = ...
   ```

### Learning Agent

```python
LearningAgent(
    SoftwareAgent[
        LearningOpportunity,  # TPercept (broj novih uzoraka)
        TrainingAction,       # TAction (train/skip)
        TrainingResult        # TResult (metrics)
    ]
)
```

**Workflow**:

1. **Sense**:
   ```python
   sample_count = await file_storage.count_new_samples()
   opportunity = LearningOpportunity(
       sample_count=sample_count,
       threshold=settings.retrain_threshold,
       should_train=(sample_count >= threshold)
   )
   ```

2. **Think**:
   ```python
   if opportunity.should_train:
       action = TrainingAction(
           dataset_path="data/new_samples",
           epochs=5,
           model_base="models/trashvision_v1/weights/best.pt"
       )
   else:
       action = TrainingAction.skip()
   ```

3. **Act**:
   ```python
   result = await trainer.train(action)
   # Runs YOLO training, saves new model
   # Resets sample counter
   ```

---

## 📦 Slojevi Sistema

### 1. Domain Layer

**Lokacija**: `AiAgents/TrashAgent/Domain/`

**Svrha**: Čista business logika, nezavisna od tehnologije.

**Komponente**:

#### Entities (`entities.py`)

```python
@dataclass
class WasteImage:
    """
    Entity: Slika otpada
    Ima identity (ID) i lifecycle
    """
    id: Optional[int] = None
    filepath: str
    filename: str
    status: ImageStatus = ImageStatus.QUEUED
    
    # Methods
    def mark_as_classified(self, category, confidence):
        self.status = ImageStatus.CLASSIFIED
        self.predicted_category = category
        self.confidence = confidence
        self.processed_at = datetime.utcnow()

@dataclass
class SystemSettings:
    """
    Entity: Globalne postavke sistema
    Singleton (samo jedna instanca)
    """
    retrain_threshold: int = 10
    auto_retrain_enabled: bool = True
    new_samples_count: int = 0
    retrain_count: int = 0
    
    def should_trigger_retraining(self) -> bool:
        return (self.auto_retrain_enabled and 
                self.new_samples_count >= self.retrain_threshold)
```

#### Enums (`enums.py`)

```python
class WasteCategory(str, Enum):
    """Kategorije otpada"""
    CARDBOARD = "cardboard"
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    PLASTIC = "plastic"
    TRASH = "trash"
    
    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES[self]
    
    @property
    def emoji(self) -> str:
        return EMOJIS[self]

class ImageStatus(str, Enum):
    """Lifecycle slike"""
    QUEUED = "queued"           # Uploaded, čeka klasifikaciju
    CLASSIFIED = "classified"   # Klasifikovana, visok confidence
    PENDING_REVIEW = "pending_review"  # Niska preciznost, treba review
    REVIEWED = "reviewed"       # Manual review završen
    ERROR = "error"             # Greška tokom procesiranja
```

#### Value Objects (`value_objects.py`)

```python
@dataclass(frozen=True)
class RecyclingInfo:
    """
    Value Object: Informacije o recikliranju
    Immutable, nema identity
    """
    is_recyclable: bool
    container_color: str
    disposal_instruction: str
    environmental_impact: str
    fun_fact: str
    
    @staticmethod
    def get_for_category(category: WasteCategory) -> 'RecyclingInfo':
        return RECYCLING_INFO_MAP[category]
```

### 2. Application Layer

**Lokacija**: `AiAgents/TrashAgent/Application/`

**Svrha**: Koordinira domain objekte za use case-ove.

#### Services (`Application/Services/`)

**QueueService** - Upravljanje queue-om slika

```python
class QueueService:
    async def enqueue(self, image: WasteImage) -> WasteImage:
        """Dodaj sliku u queue (save to DB)"""
        
    async def get_next_queued(self) -> Optional[WasteImage]:
        """Uzmi sledeću sliku sa statusom QUEUED"""
        
    async def get_by_id(self, image_id: int) -> Optional[WasteImage]:
        """Pronađi sliku po ID-u"""
```

**ClassificationService** - Upravljanje predikcijama

```python
class ClassificationService:
    async def classify_image(self, image: WasteImage) -> Classification:
        """Klasifikuj sliku i sačuvaj rezultat"""
        
    async def save_prediction(self, image_id, category, confidence):
        """Sačuvaj predikciju"""
```

**ReviewService** - Manual review

```python
class ReviewService:
    async def submit_review(self, image_id, corrected_category):
        """Korisnik koriguje predikciju"""
        
    async def get_pending_reviews(self) -> List[WasteImage]:
        """Lista slika koje čekaju review"""
```

**TrainingService** - Model training

```python
class TrainingService:
    async def check_training_opportunity(self) -> bool:
        """Da li je vrijeme za retraining?"""
        
    async def train_model(self, dataset_path: str, epochs: int):
        """Treniraj model"""
```

#### Agent Runners (`Application/Agents/`)

**ClassificationAgentRunner** - Izvršava classification agent

```python
class ClassificationAgentRunner:
    def __init__(self, 
                 perception_source: QueuePerceptionSource,
                 actuator: ClassificationActuator,
                 policy: ClassificationPolicy):
        self.agent = SoftwareAgent(...)
    
    async def run_once(self):
        """Jedan tick agenta"""
        await self.agent.step_async()
```

**LearningAgentRunner** - Izvršava learning agent

```python
class LearningAgentRunner:
    def __init__(self,
                 perception_source: LearningPerceptionSource,
                 actuator: TrainingActuator,
                 policy: LearningPolicy):
        self.agent = SoftwareAgent(...)
    
    async def run_once(self):
        """Jedan tick agenta"""
        await self.agent.step_async()
```

### 3. Infrastructure Layer

**Lokacija**: `AiAgents/TrashAgent/Infrastructure/`

**Svrha**: Implementacija tehničkih detalja (DB, file system, ML model).

#### Database (`database.py`)

```python
# SQLAlchemy models
class WasteImageModel(Base):
    __tablename__ = "waste_images"
    id = Column(Integer, primary_key=True)
    filepath = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    status = Column(String, default="queued")
    predicted_category = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

# Helper class
class DatabaseHelper:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def get_session(self):
        return self.SessionLocal()
```

#### File Storage (`file_storage.py`)

```python
class FileStorage:
    def __init__(self, base_dir: str):
        self.uploads_dir = Path(base_dir) / "uploads"
        self.learning_dir = Path(base_dir) / "new_samples"
    
    async def save_uploaded_image(self, file_data: bytes, filename: str) -> str:
        """Sačuvaj uploadovanu sliku"""
        
    async def copy_to_learning_set(self, filepath: str, category: str):
        """Kopiraj sliku u learning dataset"""
        
    async def count_new_samples(self) -> int:
        """Broji nove uzorke za sve kategorije"""
```

#### YOLO Classifier (`yolo_classifier.py`)

```python
class YoloWasteClassifier:
    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[YOLO] = None
        self.model_path: Optional[str] = None
    
    async def load_model(self, model_path: str) -> bool:
        """Učitaj YOLO model"""
        self.model = YOLO(str(model_path))
        
    async def predict(self, image_path: str) -> Dict[str, Any]:
        """Predikcija za sliku"""
        results = self.model(image_path)
        return {
            "class": results[0].probs.top1,
            "confidence": float(results[0].probs.top1conf),
            "top3": results[0].probs.top3
        }
    
    def is_loaded(self) -> bool:
        return self.model is not None
```

#### YOLO Trainer (`yolo_classifier.py`)

```python
class YoloTrainer:
    def __init__(self, base_model_path: str):
        self.base_model = YOLO(base_model_path)
    
    async def train(self, dataset_path: str, epochs: int = 5) -> Dict:
        """
        Fine-tune model sa novim podacima
        Returns: Training metrics
        """
        results = self.base_model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=224,
            patience=50,
            save=True,
            project="models",
            name=f"trashvision_v{version}"
        )
        return {
            "accuracy": results.top1,
            "loss": results.box_loss
        }
```

### 4. Web Layer

**Lokacija**: `AiAgents/TrashAgent/Web/`

**Svrha**: HTTP API, request handling, response formatting.

#### Main App (`main.py`)

```python
app = FastAPI(
    title="TrashVision Agent API",
    version="2.0.0",
    lifespan=lifespan  # Async startup/shutdown
)

@app.post("/api/images/upload")
async def upload_image(file: UploadFile):
    """Upload endpoint"""
    # 1. Validate file
    # 2. Save to storage
    # 3. Create WasteImage entity
    # 4. Enqueue via QueueService
    # 5. Return response
```

#### Background Workers (`Web/workers/`)

```python
class ClassificationWorker:
    """
    Periodički poziva ClassificationAgentRunner
    """
    def __init__(self, runner, interval_seconds: int = 2):
        self.runner = runner
        self.interval = interval_seconds
        self._task = None
    
    def start(self):
        self._task = asyncio.create_task(self._run_loop())
    
    async def _run_loop(self):
        while True:
            await self.runner.run_once()
            await asyncio.sleep(self.interval)
```

---

## 🔄 Data Flow

### Upload → Classify → Review Flow

```
┌─────────┐
│  User   │
└────┬────┘
     │ POST /api/images/upload
     ▼
┌─────────────────┐
│   FastAPI       │
│   Endpoint      │
└────┬────────────┘
     │ 1. Validate
     │ 2. Save to uploads/
     │ 3. Create WasteImage entity
     ▼
┌─────────────────┐
│  QueueService   │
│  .enqueue()     │
└────┬────────────┘
     │ INSERT INTO waste_images (status='queued')
     ▼
┌─────────────────┐
│   Database      │
│   (SQLite)      │
└────┬────────────┘
     │
     │ [Classification Agent - Every 2s]
     │
     ▼
┌──────────────────────┐
│ ClassificationAgent  │
│ .step_async()        │
└────┬─────────────────┘
     │
     ├─► SENSE: SELECT * FROM waste_images WHERE status='queued' LIMIT 1
     │
     ├─► THINK: classifier.predict(image_path)
     │           → category, confidence
     │
     └─► ACT: UPDATE waste_images 
               SET status = 'classified' | 'pending_review',
                   predicted_category = ...,
                   confidence = ...

     ┌────────────────────┐
     │ If confidence < 0.7 │
     │ status = 'pending_review'
     └─────┬──────────────┘
           │
           ▼
     ┌────────────────┐
     │ Manual Review   │ (User corrects via /feedback)
     └────────────────┘
```

### Learning / Retraining Flow

```
┌─────────┐
│  User   │
└────┬────┘
     │ POST /feedback (corrected category)
     ▼
┌─────────────────┐
│  FileStorage    │
│  .copy_to_      │
│  learning_set() │
└────┬────────────┘
     │ Copy image to data/new_samples/{category}/
     │ Increment SystemSettings.new_samples_count
     ▼
┌─────────────────┐
│   Database      │
│ UPDATE settings │
│ SET new_samples_count += 1
└────┬────────────┘
     │
     │ [Learning Agent - Every 60s]
     │
     ▼
┌──────────────────────┐
│   LearningAgent      │
│   .step_async()      │
└────┬─────────────────┘
     │
     ├─► SENSE: Check new_samples_count >= threshold (10)
     │
     ├─► THINK: If yes → TrainingAction.train()
     │           If no  → TrainingAction.skip()
     │
     └─► ACT: If train:
               1. YoloTrainer.train(data/new_samples, epochs=5)
               2. Save new model: models/trashvision_v{N}/weights/best.pt
               3. Reset new_samples_count = 0
               4. Increment retrain_count
               5. Update last_retrain_at
```

---

## 🔗 Dependency Management

### Dependency Injection (Simplified)

Projekat koristi **manual DI** (bez framework-a kao Dependency Injector):

```python
# Initialization u lifespan event-u
async def lifespan(app: FastAPI):
    # 1. Infrastructure
    db = DatabaseHelper("sqlite:///trashvision.db")
    classifier = YoloWasteClassifier()
    trainer = YoloTrainer("models/trashvision_v1/weights/best.pt")
    file_storage = FileStorage("data")
    
    # 2. Application Services
    queue_service = QueueService(db)
    classification_service = ClassificationService(db, classifier)
    training_service = TrainingService(trainer, file_storage, db)
    
    # 3. Agent Runners
    classification_runner = ClassificationAgentRunner(
        perception_source=QueuePerceptionSource(queue_service),
        actuator=ClassificationActuator(classification_service),
        policy=ClassificationPolicy(classifier)
    )
    
    learning_runner = LearningAgentRunner(
        perception_source=LearningPerceptionSource(training_service),
        actuator=TrainingActuator(trainer),
        policy=LearningPolicy()
    )
    
    # 4. Background Workers
    classification_worker = ClassificationWorker(classification_runner, interval=2)
    learning_worker = LearningWorker(learning_runner, interval=60)
    
    # 5. Store u global state
    app_state.db = db
    app_state.classifier = classifier
    # ...
    
    # 6. Start workers
    classification_worker.start()
    learning_worker.start()
    
    yield
    
    # Cleanup
    classification_worker.stop()
    learning_worker.stop()
```

### Interface Abstractions

```python
# Domain-level interface (u Domain layeru)
class WasteClassifier(ABC):
    @abstractmethod
    async def predict(self, image_path: str) -> Dict:
        pass

# Infrastructure implementation
class YoloWasteClassifier(WasteClassifier):
    async def predict(self, image_path: str) -> Dict:
        results = self.model(image_path)
        return {
            "class": results[0].probs.top1,
            "confidence": float(results[0].probs.top1conf)
        }
```

---

## 🎨 Design Patterns

### 1. Repository Pattern

```python
# Application/Services/queue_service.py
class QueueService:
    """
    Repository za WasteImage entitete
    Sakriva SQL detalje
    """
    def __init__(self, db: DatabaseHelper):
        self.db = db
    
    async def get_by_id(self, image_id: int) -> Optional[WasteImage]:
        session = self.db.get_session()
        db_image = session.query(WasteImageModel).get(image_id)
        if db_image:
            return self._to_entity(db_image)
        return None
    
    def _to_entity(self, model: WasteImageModel) -> WasteImage:
        """Mapira DB model → Domain entity"""
        return WasteImage(
            id=model.id,
            filepath=model.filepath,
            # ...
        )
```

### 2. Strategy Pattern

```python
# Core/policy.py
class Policy(ABC, Generic[TPercept, TAction]):
    """
    Strategy za donošenje odluka u Think fazi
    """
    @abstractmethod
    async def decide(self, percept: TPercept) -> TAction:
        pass

# TrashAgent implementacija
class ClassificationPolicy(Policy[WasteImage, ClassificationAction]):
    async def decide(self, image: WasteImage) -> ClassificationAction:
        prediction = await self.classifier.predict(image.filepath)
        
        status = (ImageStatus.CLASSIFIED 
                  if prediction["confidence"] > 0.7 
                  else ImageStatus.PENDING_REVIEW)
        
        return ClassificationAction(
            image_id=image.id,
            category=prediction["class"],
            confidence=prediction["confidence"],
            status=status
        )
```

### 3. Template Method Pattern

```python
class SoftwareAgent(ABC, Generic[TPercept, TAction, TResult]):
    """
    Template method: step_async() definiše algoritam
    Podklase implementiraju sense(), think(), act()
    """
    async def step_async(self) -> Optional[TResult]:
        # 1. Sense
        percept = await self.sense()
        if percept is None:
            return None
        
        # 2. Think
        action = await self.think(percept)
        
        # 3. Act
        result = await self.act(action)
        
        return result
```

### 4. Adapter Pattern

```python
# Infrastructure adaptira YOLO API → Domain interface
class YoloWasteClassifier(WasteClassifier):
    """
    Adapter: YOLO ultralytics API → naš interface
    """
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)  # External library
    
    async def predict(self, image_path: str) -> Dict:
        # Adapt ultralytics output → naš format
        results = self.model(image_path)
        return {
            "class": results[0].probs.top1,      # YOLO field
            "confidence": float(results[0].probs.top1conf)
        }
```

### 5. Observer Pattern (Implicit)

Background workers su tip observer-a koji periodično "posmatraju" queue:

```python
class ClassificationWorker:
    """
    Observer koji periodično provjerava queue
    """
    async def _run_loop(self):
        while self._is_running:
            # Check for new queued images
            await self.runner.run_once()
            await asyncio.sleep(self.interval)
```

---

## 🗄️ Database Schema

### ERD

```
┌──────────────────────────────────┐
│         waste_images             │
├──────────────────────────────────┤
│ id (PK)              INTEGER     │
│ filepath             STRING      │
│ filename             STRING      │
│ file_size_bytes      INTEGER     │
│ width                INTEGER     │
│ height               INTEGER     │
│ status               STRING      │
│ predicted_category   STRING      │
│ confidence           FLOAT       │
│ uploaded_at          DATETIME    │
│ processed_at         DATETIME    │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│       system_settings            │
├──────────────────────────────────┤
│ id (PK)              INTEGER     │
│ retrain_threshold    INTEGER     │
│ auto_retrain_enabled BOOLEAN     │
│ new_samples_count    INTEGER     │
│ last_retrain_at      DATETIME    │
│ retrain_count        INTEGER     │
└──────────────────────────────────┘
```

### Indexes

```sql
CREATE INDEX idx_status ON waste_images(status);
CREATE INDEX idx_uploaded_at ON waste_images(uploaded_at);
CREATE INDEX idx_processed_at ON waste_images(processed_at);
```

### Queries

**Get next queued image**:
```sql
SELECT * FROM waste_images 
WHERE status = 'queued' 
ORDER BY uploaded_at ASC 
LIMIT 1;
```

**Count pending reviews**:
```sql
SELECT COUNT(*) FROM waste_images 
WHERE status = 'pending_review';
```

**Get recent classifications**:
```sql
SELECT * FROM waste_images 
WHERE status = 'classified' 
ORDER BY processed_at DESC 
LIMIT 10;
```

---

## 📁 File Storage

### Directory Structure

```
data/
├── uploads/                      # Uploaded images
│   ├── {uuid}_{filename}.jpg
│   └── ...
│
└── new_samples/                  # Learning dataset
    ├── cardboard/
    │   ├── sample_001.jpg
    │   └── ...
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

### File Naming Convention

**Uploads**: `{uuid}_{original_filename}`
- Example: `a3f2d9e1-bottle.jpg`
- Reason: Prevent name collisions

**Learning samples**: `{category}_{timestamp}_{filename}`
- Example: `plastic_20251223_143000_bottle.jpg`
- Reason: Easy sorting, debugging

### Storage Operations

```python
class FileStorage:
    async def save_uploaded_image(self, file_data: bytes, filename: str):
        unique_name = f"{uuid.uuid4()}_{filename}"
        path = self.uploads_dir / unique_name
        await asyncio.to_thread(path.write_bytes, file_data)
        return str(path)
    
    async def copy_to_learning_set(self, filepath: str, category: str):
        src = Path(filepath)
        dst = self.learning_dir / category / f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{src.name}"
        await asyncio.to_thread(shutil.copy2, src, dst)
```

---

## 🌐 API Layer

### REST API Design

**Princips**: RESTful, resource-oriented

```
GET    /api/images           - List images
POST   /api/images/upload    - Upload new image
GET    /api/images/{id}      - Get image status
DELETE /api/images/{id}      - Delete image

POST   /feedback             - Submit user feedback

GET    /api/learning/stats   - Learning statistics
POST   /api/learning/train   - Manual trigger training

GET    /status               - System health
```

### Response Format

**Success Response**:
```json
{
  "success": true,
  "data": { ... },
  "message": "Optional message"
}
```

**Error Response**:
```json
{
  "success": false,
  "error": "Error description",
  "detail": "Detailed technical info"
}
```

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: whitelist specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## ⚙️ Background Workers

### Worker Implementation

```python
class BackgroundWorker:
    """Base class za background workers"""
    
    def __init__(self, runner, interval_seconds: int):
        self.runner = runner
        self.interval = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._is_running = False
    
    def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._task = asyncio.create_task(self._run_loop())
    
    def stop(self):
        self._is_running = False
        if self._task:
            self._task.cancel()
    
    async def _run_loop(self):
        while self._is_running:
            try:
                await self.runner.run_once()
            except Exception as e:
                print(f"❌ Worker error: {e}")
            
            await asyncio.sleep(self.interval)
```

### Graceful Shutdown

```python
async def lifespan(app: FastAPI):
    # Startup
    workers = start_workers()
    
    yield
    
    # Shutdown
    print("🛑 Shutting down workers...")
    for worker in workers:
        worker.stop()
    
    # Wait for tasks to complete
    await asyncio.sleep(1)
    print("✅ Shutdown complete")
```

---

## 📈 Scalability & Performance

### Current Bottlenecks

1. **SQLite**: Single-threaded, file-based
2. **Single Process**: Nema horizontal scaling
3. **Blocking I/O**: File operations blokiraju event loop

### Scaling Strategies

#### 1. Database Migration

**SQLite → PostgreSQL**

```python
# Before
DATABASE_URL = "sqlite:///trashvision.db"

# After
DATABASE_URL = "postgresql://user:pass@host:5432/trashvision"
```

Benefits:
- Concurrent writes
- Better performance
- ACID guarantees

#### 2. Message Queue

**Queue Service → RabbitMQ/Redis**

```python
# Before: SQLite queue
await queue_service.enqueue(image)

# After: Redis queue
await redis.lpush("classification_queue", image.to_json())
```

Benefits:
- Distributed workers
- Better throughput
- Fault tolerance

#### 3. Horizontal Scaling

**Multiple Worker Instances**

```yaml
# docker-compose.yml
services:
  api:
    image: trashvision:latest
    command: uvicorn main:app
    
  classification-worker-1:
    image: trashvision:latest
    command: python -m workers.classification
    
  classification-worker-2:
    image: trashvision:latest
    command: python -m workers.classification
```

#### 4. Caching

**Redis Cache za predikcije**

```python
# Check cache first
cached = await redis.get(f"prediction:{image_hash}")
if cached:
    return json.loads(cached)

# If miss, predict and cache
prediction = await classifier.predict(image_path)
await redis.setex(
    f"prediction:{image_hash}", 
    3600,  # 1 hour
    json.dumps(prediction)
)
```

#### 5. Async I/O

**aiofiles za non-blocking file operations**

```python
# Before
with open(path, 'rb') as f:
    data = f.read()

# After
async with aiofiles.open(path, 'rb') as f:
    data = await f.read()
```

---

## 🔒 Security Considerations

### Current Security

**Threats**:
- ✅ **SQL Injection**: Zaštićeno (SQLAlchemy ORM)
- ⚠️ **File Upload**: Basic validation (MIME type)
- ❌ **Authentication**: Nema (open API)
- ❌ **Rate Limiting**: Nema
- ⚠️ **File Path Traversal**: Partially mitigated

### Recommendations

#### 1. Authentication & Authorization

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Verify JWT token
    if not is_valid_token(token):
        raise HTTPException(status_code=401)
    return token

@app.post("/api/images/upload")
async def upload_image(file: UploadFile, token: str = Depends(verify_token)):
    # Protected endpoint
    pass
```

#### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/images/upload")
@limiter.limit("10/minute")
async def upload_image(file: UploadFile):
    pass
```

#### 3. File Validation

```python
from PIL import Image

async def validate_image(file_data: bytes):
    # Check file size
    if len(file_data) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(400, "File too large")
    
    # Verify it's a real image
    try:
        img = Image.open(io.BytesIO(file_data))
        img.verify()
    except:
        raise HTTPException(400, "Invalid image")
    
    # Check dimensions
    if img.width > 4096 or img.height > 4096:
        raise HTTPException(400, "Image too large")
```

#### 4. Input Sanitization

```python
import re

def sanitize_filename(filename: str) -> str:
    # Remove path traversal attempts
    filename = os.path.basename(filename)
    # Allow only alphanumeric, dots, underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    return filename
```

---

## 🚀 Future Improvements

### Short-term (1-3 mjeseca)

1. **Unit Tests**: pytest coverage za sve slojeve
2. **CI/CD**: GitHub Actions za automatic testing
3. **Docker**: Kontejnerizacija aplikacije
4. **Monitoring**: Prometheus + Grafana dashboard
5. **Logging**: Structured logging (JSON)

### Mid-term (3-6 mjeseci)

1. **PostgreSQL Migration**: Zamjena SQLite-a
2. **Redis Queue**: Distributed task queue
3. **API Authentication**: JWT tokens
4. **Frontend Rewrite**: React/Vue sa real-time updates
5. **Model Versioning**: MLflow za tracking

### Long-term (6-12 mjeseci)

1. **Multi-tenancy**: Support za više korisnika/organizacija
2. **Cloud Deployment**: AWS/GCP auto-scaling
3. **Mobile App**: React Native companion app
4. **Advanced ML**: Ensemble models, active learning
5. **Analytics Dashboard**: Detaljni reporti

---

## 📚 References

- **Domain-Driven Design**: Eric Evans - "Domain-Driven Design"
- **Clean Architecture**: Robert C. Martin - "Clean Architecture"
- **Software Agents**: Stuart Russell & Peter Norvig - "Artificial Intelligence: A Modern Approach"
- **FastAPI**: https://fastapi.tiangolo.com
- **YOLO**: https://docs.ultralytics.com

---

**Autor**: Nedim  
**Verzija**: 2.0  
**Datum**: 2025-12-23
