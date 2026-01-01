# TrashVision - Dokumentacija Projekta

**Predmet:** Umjetna inteligencija 2025/2026  
**Godina:** 4  
**Profesorica:** Nina Bijedić
**Asistentu:**Adil Joldić, Azra Smajić  
**Student:** Nedim Jugo  
**Datum:** Decembar 2025

---

# SADRŽAJ

1. [Opis Ideje i Problema](#1-opis-ideje-i-problema)
2. [Vrste Agenata - Izbor i Obrazloženje](#2-vrste-agenata---izbor-i-obrazloženje)
3. [Agent Ciklus - Sense → Think → Act → Learn](#3-agent-ciklus---sense--think--act--learn)
4. [Ideje za Proširenje](#4-ideje-za-proširenje)
5. [Diskusija Ideje sa LLM-om](#5-diskusija-ideje-sa-llm-om)
6. [Specifikacija Agenta](#6-specifikacija-agenta)
7. [Arhitektura Sistema - Clean Architecture](#7-arhitektura-sistema---clean-architecture)
8. [Tehnička Implementacija](#8-tehnička-implementacija)
9. [Struktura Projekta](#9-struktura-projekta)
10. [Review Implementacije sa LLM-om](#10-review-implementacije-sa-llm-om)
11. [Dataset i Izvori](#11-dataset-i-izvori)
12. [Korištenje LLM-a Tokom Razvoja](#12-korištenje-llm-a-tokom-razvoja)
13. [Zaključak](#13-zaključak)

---

# 1. OPIS IDEJE I PROBLEMA

## 1.1 Problem koji Agent Rješava

**TrashVision** je inteligentni AI agent sistem koji rješava problem automatske klasifikacije otpada u realnom vremenu. U modernom svijetu, pravilno sortiranje otpada je ključno za:

- **Zaštitu okoline** - Pogrešno sortirani otpad završava na deponijama umjesto na reciklaži
- **Ekonomsku efikasnost** - Reciklirani materijali imaju vrijednost, a kontaminacija skupo košta
- **Automatizaciju procesa** - Ljudski sortiranje je sporo, skupo i nepouzdano

### Zašto je ovo AGENT, a ne samo analitička aplikacija?

Ključna razlika između agenta i obične aplikacije je **iterativno ponašanje kroz vrijeme**:

| Obična Aplikacija | TrashVision Agent |
|-------------------|-------------------|
| Korisnik uploaduje → dobije rezultat | Agent KONTINUIRANO prati queue i procesira |
| Jednokratna operacija | Petlja: Sense→Think→Act svake 2 sekunde |
| Statički model | Model se UČI iz povratnih informacija |
| Pasivna | Agent AUTONOMNO odlučuje kada i šta procesirati |

**TrashVision nije "upload → result" sistem.** Sistem kontinuirano:
1. **Nadgleda** nove slike u queue-u (Sense)
2. **Odlučuje** kako klasifikovati na osnovu pravila i ML modela (Think)
3. **Izvršava** akcije - klasifikuje ili šalje na review (Act)
4. **Uči** iz korisničkih povratnih informacija i retrenira model (Learn)

## 1.2 Šta se Dešava Kroz Vrijeme?

Agent radi u kontinuiranoj petlji:

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT LIFECYCLE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  Sense   │───▶│  Think   │───▶│   Act    │             │
│   │(Queue)   │    │(ML+Rules)│    │(Persist) │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│        ▲                               │                    │
│        │                               │                    │
│        └───────────────────────────────┘                    │
│                   (svakih 2s)                               │
│                                                             │
│   ┌──────────────────────────────────────────┐             │
│   │            Learning Agent                 │             │
│   │  - Prati feedback (reviews)               │             │
│   │  - Retrenira model kad ima dovoljno       │             │
│   │  - Aktivira novu verziju                  │             │
│   └──────────────────────────────────────────┘             │
│                   (svakih 60s)                              │
└─────────────────────────────────────────────────────────────┘
```

---

# 2. VRSTE AGENATA - IZBOR I OBRAZLOŽENJE

## 2.1 Implementirane Vrste Agenata

TrashVision implementira **višestruke vrste agenata** u jednom sistemu:

### 1. Klasifikacioni Agent (Classification Agent)
**Opis:** Donosi odluke u zonama/pragovima na osnovu confidence score-a.

**Pravila:**
- **confidence ≥ 70%** → Automatska klasifikacija (CLASSIFIED status)
- **confidence < 70%** → Potreban human review (PENDING_REVIEW status)
- **confidence < 30%** → Low confidence fallback

**Zašto ovaj tip?** Klasifikacija otpada zahtijeva pouzdane odluke. Kada model nije siguran, bolje je tražiti ljudsku verifikaciju nego automatski napraviti grešku koja može kontaminirati reciklažni materijal.

### 2. Learning Agent (Continuous Learning)
**Opis:** Mijenja svoje ponašanje nakon novih iskustava.

**Mehanizam:**
- Prati koliko je novih uzoraka dobio kroz user feedback
- Kad pređe prag (default: 10 uzoraka) → triggeruje retraining
- Kreira novu verziju modela i aktivira je

**Zašto ovaj tip?** Distribucija otpada se mijenja kroz vrijeme (npr. više plastike ljeti). Agent mora učiti i prilagođavati se.

### 3. Ciljno-orijentisani Agent (Goal-Oriented)
**Opis:** Ima jasan cilj koji optimizuje - **minimizacija troška greške**.

**Error Cost Matrix:**
```
Primjer: Greška "metal → paper" košta 3.0 (opasno!)
         Greška "paper → metal" košta 1.0 (manje opasno)
```

**Decision Optimizer** minimizuje expected cost, a ne samo maksimizuje probability.

**Zašto ovaj tip?** Nije svaka greška jednako skupa. Metal u papiru može uništiti mašinu za reciklažu papira, dok papir u metalu prolazi kroz topljenje bez štete.

### 4. Multi-Agent Sistem
**Opis:** Više agenata sa različitim ulogama koji rade paralelno.

**Agenti u sistemu:**
1. **ClassificationAgent** - Klasifikuje slike (tick: 2s)
2. **LearningAgent** - Prati feedback i retrenira (tick: 60s)

**Zašto ovaj tip?** Razdvajanje odgovornosti. Klasifikacija mora biti brza (real-time), dok learning može biti spor (batch processing).

## 2.2 Zašto Baš Ove Vrste?

| Vrsta | Razlog Izbora |
|-------|---------------|
| Klasifikacioni | Problem klasifikacije otpada sa pragovima pouzdanosti |
| Learning | Potreba za adaptacijom na nove uzorke i promjene |
| Ciljno-orijentisani | Različiti troškovi grešaka u reciklaži |
| Multi-agent | Paralelno procesiranje sa različitim frekvencijama |

---

# 3. AGENT CIKLUS - SENSE → THINK → ACT → LEARN

## 3.1 Classification Agent Ciklus

### SENSE - Šta Agent Opaža?

```python
async def sense(self) -> Optional[WasteImage]:
    """
    SENSE: Preuzmi sljedeću sliku iz queue-a.
    
    Query: SELECT * FROM images WHERE status='queued' ORDER BY uploaded_at LIMIT 1
    """
    queue_size = await self._queue.get_queue_size(ImageStatus.QUEUED)
    
    if queue_size == 0:
        return None  # Nema posla - agent miruje
    
    image = await self._queue.dequeue_next(ImageStatus.QUEUED)
    image.mark_as_processing()  # Lock
    
    return image
```

**Percept:** `WasteImage` objekt sa svim meta-podacima (filepath, filename, status, uploaded_at)

### THINK - Kako Agent Zaključuje?

```python
async def think(self, image: WasteImage) -> ClassificationDecision:
    """
    THINK: Klasifikuj sliku i donesi odluku.
    
    1. Klasifikuj sliku preko YOLO modela
    2. Primijeni policy (confidence thresholds)
    3. Optimizuj odluku prema cost matrix
    """
    decision = await self._classifier.classify_image(image, self._settings)
    
    # Decision sadrži:
    # - predicted_category: WasteCategory
    # - confidence: float
    # - new_status: ImageStatus (CLASSIFIED ili PENDING_REVIEW)
    # - expected_cost: float
    
    return decision
```

**Mehanizam zaključivanja:**
1. **ML Model (YOLO v8)** - daje probability distribution
2. **Threshold Policy** - odlučuje status na osnovu confidence
3. **Decision Optimizer** - minimizuje expected cost

### ACT - Šta Agent Konkretno Radi?

```python
async def act(self, action: ClassificationDecision) -> ClassificationResult:
    """
    ACT: Sačuvaj predikciju i ažuriraj status.
    
    1. Sačuvaj Prediction u DB
    2. Ažuriraj Image status
    3. Vrati rezultat za Web sloj
    """
    prediction = await self._classifier.save_prediction(
        image=image,
        decision=action,
        model_version=self._model_version
    )
    
    await self._queue.update_status(
        image_id=image.id,
        new_status=action.new_status
    )
    
    return ClassificationResult(...)
```

**Akcije:**
- Kreiranje `Prediction` entiteta u bazi
- Promjena statusa slike: `QUEUED → CLASSIFIED` ili `QUEUED → PENDING_REVIEW`
- Emitovanje rezultata za real-time UI

## 3.2 Learning Agent Ciklus

### SENSE - Pročitaj System State

```python
async def sense(self) -> Optional[SystemSettings]:
    """Pročitaj broj novih uzoraka i threshold"""
    
    if not self._settings.auto_retrain_enabled:
        return None  # Učenje isključeno
    
    return self._settings  # new_samples_count, retrain_threshold
```

### THINK - Da Li Treba Retraining?

```python
async def think(self, settings: SystemSettings) -> TrainingDecision:
    """Odluči da li i kako retrenirati"""
    
    should_retrain = settings.should_trigger_retraining()
    # True ako: new_samples_count >= retrain_threshold
    
    mode = TrainingMode.INCREMENTAL if settings.new_samples_count < 500 else TrainingMode.FULL
    
    return TrainingDecision(
        should_retrain=should_retrain,
        mode=mode,
        new_samples_count=settings.new_samples_count
    )
```

### ACT - Pokreni Retraining

```python
async def act(self, action: TrainingDecision) -> LearningResult:
    """Pokreni retraining ako je potrebno"""
    
    if not action.should_retrain:
        return LearningResult(retrain_triggered=False)
    
    new_version = await self._trainer.retrain_model(
        mode=action.mode,
        epochs=self._settings.epochs
    )
    
    return LearningResult(
        retrain_triggered=True,
        new_model_version=new_version.version_number
    )
```

### LEARN - Reset Counters i Log

```python
async def learn(self, result: LearningResult) -> None:
    """Ažuriraj znanje nakon retraining-a"""
    
    if result.retrain_triggered:
        self._settings.reset_new_samples_counter()
        self._settings.increment_total_retrains()
        
        # Log metrike
        self._log_training_metrics(result)
```

---

# 4. IDEJE ZA PROŠIRENJE

## 4.1 Implementirana Proširenja

### ✅ Nesigurni Slučajevi
Agent prepoznaje kada "nije siguran":
- **Confidence < 70%** → Šalje na human review
- **Confidence < 30%** → Low confidence fallback sa specijalnim tretmanom
- **Top-2 razlika < 10%** → Nesigurna odluka, preferira review

### ✅ Cost-Aware Decision Making
Agent ne gleda samo max probability, već **minimizuje expected cost**:

```python
# Error Cost Matrix primjeri:
metal → paper = 3.0  # Visok trošak (može oštetiti mašinu)
paper → metal = 1.0  # Nizak trošak (papir će izgorjeti)
plastic → glass = 2.5  # Srednji trošak (kontaminacija)
```

### ✅ Objašnjenje Odluka
Svaka odluka ima `reasoning` polje:

```python
decision = OptimizedDecision(
    predicted_category=WasteCategory.METAL,
    confidence=0.85,
    reasoning="High confidence (85%), low expected cost (0.12)"
)
```

### ✅ Više Agenata sa Podjelom Odgovornosti
- **ClassificationAgent** - Brza klasifikacija (2s tick)
- **LearningAgent** - Sporo učenje (60s tick)

### ✅ Simulacija Okoline
**Sorting Simulation** za testiranje ponašanja:
- Conveyor Belt simulacija (10m, 0.2-0.3 m/s)
- Robotic Arm state machine (scan → pick → move → drop)
- Sorting Bins sa tracking-om kontaminacije
- Cost tracking i efikasnost

## 4.2 Moguća Buduća Proširenja

### Aktivno Učenje
Agent bi mogao tražiti dodatne informacije:
- Kada je nesiguran, pitati korisnika za specifične feature-e
- "Da li je ova boca staklo ili plastika?"

### Adaptivni Pragovi
Pragovi bi se mogli mijenjati kroz vrijeme:
- Ako ima puno grešaka na 70% → povećaj na 75%
- Ako su reviews potvrđivali → smanji na 65%

### A/B Testing za Modele
Paralelno pokretanje dva modela i poređenje performansi.

### Context-Aware Agent Proširenje
Agent bi mogao uzimati u obzir kontekst:
- **Vrijeme dana** - Više plastičnih boca ujutro (od kafića)
- **Lokacija** - Industrijska zona vs. stambena
- **Sezona** - Više kartonskih kutija oko praznika

### Real-time Video Stream
Umjesto pojedinačnih slika, procesirati video stream:
- Frame extraction svakih N ms
- Tracking objekata kroz frame-ove
- Batch processing za efikasnost

### Reinforcement Learning
Agent koji uči optimalne akcije kroz nagrade:
- Pozitivna nagrada: Tačna klasifikacija potvrđena review-om
- Negativna nagrada: Pogrešna klasifikacija
- Q-learning za optimizaciju decision policy-ja

### Distribuirani Multi-Agent Sistem
Više agenata na različitim lokacijama:
- Centralni Learning Agent koji agregira feedback
- Lokalni Classification Agenti optimizirani za svoju lokaciju
- Federated Learning za privatnost podataka

### Objašnjivi AI (Explainable AI - XAI)
Vizualno objašnjenje zašto je agent donio odluku:
- Grad-CAM heatmape koje pokazuju koje dijelove slike agent gleda
- Feature importance za svaku kategoriju
- Usporedba sa sličnim primjerima iz training seta

### Edge Deployment
Pokretanje agenta na edge uređajima:
- Optimizacija modela za Raspberry Pi / Jetson Nano
- Offline rad sa periodičnom sinhronizacijom
- Kompresija modela (quantization, pruning)

---

# 5. DISKUSIJA IDEJE SA LLM-OM

## 5.1 Inicijalna Diskusija

### Početna Ideja i Diskusija sa LLM-om (2. decembar 2025.)

**Ja:** Druga ideja je TrashVision, AI sistem zasnovan na računalnom vidu (Computer Vision), čiji je cilj prepoznavanje vrste i količine otpada na osnovu slike. Koristio bi unaprijed istrenirane model(e) klasifikacije slike (npr. YOLOv8, MobileNet ili ResNet), s mogućnošću dodatnog učenja na datasetima poput TrashNet-a. Nakon što korisnik učita sliku, agent bi detektovao tip otpada (plastika, papir, staklo, metal, organski) i eventualno procijenio zapreminu, uz prikaz vizualnog rezultata i preporuke za pravilno odlaganje. Ovo je moja ideja za AI agenta. Umjesto TrashNet hoću ovaj dataset: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data. Hoću da mi pomogneš sa strukturom projekta, izgradnjom, treniranjem - sve korak po korak uz provjere napretka.

**LLM:** Odličan izbor projekta! TrashVision je praktična primjena AI-ja s realnim utjecajem. Pomoći ću ti da ga izgradiš korak po korak.

**Pregled projekta TrashVision:**

Prije nego počnemo, hajde da razjasnimo arhitekturu:

**Komponente sistema:**
1. Model za klasifikaciju - YOLOv8 ili sličan za detekciju i klasifikaciju otpada
2. Backend API - Python (Flask/FastAPI) za procesiranje slika
3. Frontend - Web interfejs za upload slika i prikaz rezultata
4. Dataset - Garbage Classification v2 sa Kaggle-a

---

## 5.2 Diskusija o Agent Arhitekturi

### Razjašnjenje Agent vs Aplikacija Koncepta (22. decembar 2025.)

**Ja:** Pošalji mi analizu da li moj TrashVision ispunjava zahtjeve za AI agenta.

**LLM:** Hajde da detaljno analiziram da li vaš TrashVision sistem ispunjava sve zahtjeve iz dokumentacije.

**Glavni problemi:**

🔴 **KRITIČNO - Ovo nije agent**
Trenutno je to ML aplikacija koja radi: Korisnik uploada sliku → Model klasifikuje → Vraća rezultat

Fali:
- Agent arhitektura (Sense→Think→Act→Learn ciklus)
- Runner klase sa Tick/Step metodama
- Background worker koji kontinuirano radi

🔴 **Nema Clean Architecture**
Sve je pomiješano u api.py - fali razdvajanje na:
- Core (generičke abstrakcije)
- Domain (entiteti: Image, Prediction, Review)
- Application (servisi + agent runneri)
- Infrastructure (DB, ML)
- Web (tanak host)

🔴 **Web sloj sadrži biznis logiku**
CLASSES_INFO rječnik, pravila reciklaže, predikcija logika - sve u api.py. To treba biti u Domain/Application sloju.

---

## 5.3 Kritika i Poboljšanja

### LLM Kritika Inicijalnog Pristupa

**LLM:** Šta trebaš uraditi:
1. Kreirati Runner klase - ključno!
2. Refaktorisati u slojeve (Domain, Application, Infrastructure, Web)
3. Dodati background worker koji radi agent loop
4. Dokumentovati korištenje LLM-a (diskusija ideje, review, iteracije)

**Ključna razlika između agenta i obične aplikacije:**

| Obična Aplikacija | TrashVision Agent |
|-------------------|-------------------|
| Korisnik uploaduje → dobije rezultat | Agent KONTINUIRANO prati queue i procesira |
| Jednokratna operacija | Petlja: Sense→Think→Act svake 2 sekunde |
| Statički model | Model se UČI iz povratnih informacija |
| Pasivna | Agent AUTONOMNO odlučuje kada i šta procesirati |

**TrashVision nije "upload → result" sistem.** Sistem mora kontinuirano:
1. **Nadgledati** nove slike u queue-u (Sense)
2. **Odlučivati** kako klasifikovati na osnovu pravila i ML modela (Think)
3. **Izvršavati** akcije - klasifikuje ili šalje na review (Act)
4. **Učiti** iz korisničkih povratnih informacija i retrenirati model (Learn)

---

# 6. SPECIFIKACIJA AGENTA

## 6.1 Acceptance Kriteriji

### Classification Agent
- [x] Agent procesira slike iz queue-a svakih 2 sekunde
- [x] Confidence ≥ 70% → automatska klasifikacija
- [x] Confidence < 70% → human review
- [x] Decision optimizer minimizuje expected cost
- [x] Rezultat se emituje na Web sloj

### Learning Agent
- [x] Agent provjerava nove uzorke svakih 60 sekundi
- [x] Kad ima ≥ 10 novih uzoraka → triggeruje retraining
- [x] Kreira novu verziju modela
- [x] Aktivira novi model automatski
- [x] Reset counter nakon retraining-a

### Sistemski Zahtjevi
- [x] Web sloj je TANAK (samo transport)
- [x] Sva biznis logika u shared sloju
- [x] Tick/Step je atomaran i idempotentan
- [x] No-work izlaz bez greške

## 6.2 Definicija Percepata, Akcija i Rezultata

| Agent | Percept | Action | Result |
|-------|---------|--------|--------|
| Classification | WasteImage | ClassificationDecision | ClassificationResult |
| Learning | SystemSettings | TrainingDecision | LearningResult |

## 6.3 System Settings

```python
@dataclass
class SystemSettings:
    # Thresholds
    min_confidence_threshold: float = 0.70
    review_threshold: float = 0.50
    
    # Retraining
    auto_retrain_enabled: bool = True
    retrain_threshold: int = 10
    new_samples_count: int = 0
    
    # Training config
    incremental_epochs: int = 5
    full_epochs: int = 20
```

---

# 7. ARHITEKTURA SISTEMA - CLEAN ARCHITECTURE

## 7.1 Slojevi i Odgovornosti

TrashVision koristi **Clean Architecture** sa jasnom podjelom odgovornosti:

```
┌──────────────────────────────────────────────────────────────────┐
│                         WEB LAYER (Tanki!)                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │   Controllers  │  │    Workers     │  │   DTO/Mapping  │     │
│  │  (Endpoints)   │  │  (Background)  │  │                │     │
│  └────────────────┘  └────────────────┘  └────────────────┘     │
│                                                                  │
│  SAMO: routing, DI, background loop, delay, emit events          │
│  NE SMIJE: thresholds, decision rules, retrain logic             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│  ┌────────────────────────────┐  ┌────────────────────────────┐ │
│  │        Services            │  │         Runners            │ │
│  │  - QueueService            │  │  - ClassificationRunner    │ │
│  │  - ClassificationService   │  │  - LearningRunner          │ │
│  │  - ReviewService           │  │                            │ │
│  │  - TrainingService         │  │  SENSE → THINK → ACT       │ │
│  └────────────────────────────┘  └────────────────────────────┘ │
│                                                                  │
│  AGENT LOGIKA: policy, thresholds, decision, state machine       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                       DOMAIN LAYER                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  Entities  │  │   Enums    │  │   Value    │  │  Decision  │ │
│  │            │  │            │  │  Objects   │  │  Optimizer │ │
│  │ WasteImage │  │ ImageStatus│  │ Prediction │  │ Error Cost │ │
│  │ Review     │  │ Category   │  │ Decision   │  │ Matrix     │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │
│                                                                  │
│  BUSINESS RULES: status transitions, validation, costs           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  Database  │  │    File    │  │    YOLO    │  │ Simulation │ │
│  │  (SQLite)  │  │  Storage   │  │ Classifier │  │  (Sorting) │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │
│                                                                  │
│  TECHNICAL: DB access, file I/O, ML inference, external APIs     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                        CORE LAYER                                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    SoftwareAgent<T>                        │ │
│  │  - sense() → TPercept                                      │ │
│  │  - think(percept) → TAction                                │ │
│  │  - act(action) → TResult                                   │ │
│  │  - step_async() → Optional[TResult]                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐ │
│  │  IPolicy   │  │ IActuator  │  │ IPerception│  │ ILearning  │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘ │
│                                                                  │
│  FRAMEWORK: generičke abstrakcije, bez domain znanja             │
└──────────────────────────────────────────────────────────────────┘
```

## 7.2 Šta Web Sloj SMIJE Raditi

```python
# Web worker - SAMO loop i delay
async def _run_loop(self):
    while self._is_running:
        try:
            result = await self.runner.step_async()  # Pozovi runner
            
            if result:
                await self._result_callback(result)  # Emit event
                
        except Exception as e:
            print(f"Error: {e}")
        
        await asyncio.sleep(self.tick_interval)  # Delay
```

## 7.3 Šta Web Sloj NE SMIJE Raditi

```python
# ❌ LOŠE - biznis logika u Web sloju
if result.confidence > 0.7:  # Threshold u Web-u!
    status = "classified"
else:
    status = "pending_review"

# ❌ LOŠE - retrain odluka u Web-u
if new_samples >= 10:  # Retrain rule u Web-u!
    await trainer.retrain()
```

---

# 8. TEHNIČKA IMPLEMENTACIJA

## 8.1 Core Framework (AiAgents.Core)

### SoftwareAgent Bazna Klasa

```python
class SoftwareAgent(ABC, Generic[TPercept, TAction, TResult]):
    """
    Bazna klasa za sve software agente.
    Implementira osnovni Sense → Think → Act ciklus.
    """
    
    @abstractmethod
    async def sense(self) -> Optional[TPercept]:
        """SENSE: Opazi okolinu"""
        pass
    
    @abstractmethod
    async def think(self, percept: TPercept) -> TAction:
        """THINK: Donesi odluku"""
        pass
    
    @abstractmethod
    async def act(self, action: TAction) -> TResult:
        """ACT: Izvrši akciju"""
        pass
    
    async def step_async(self) -> Optional[TResult]:
        """Jedan tick agent ciklusa"""
        percept = await self.sense()
        if percept is None:
            return None  # No-work exit
        
        action = await self.think(percept)
        result = await self.act(action)
        
        return result
```

### Learning Agent Ekstenzija

```python
class LearningAgent(SoftwareAgent[TPercept, TAction, TResult]):
    """Agent sa sposobnošću učenja"""
    
    @abstractmethod
    async def learn(self, result: TResult) -> None:
        """LEARN: Ažuriraj znanje"""
        pass
    
    async def step_async(self) -> Optional[TResult]:
        result = await super().step_async()
        
        if result:
            await self.learn(result)  # Learn nakon Act
        
        return result
```

## 8.2 Domain Entiteti

### WasteImage

```python
@dataclass
class WasteImage:
    """Slika otpada koju agent procesira"""
    id: Optional[int] = None
    filepath: str = ""
    filename: str = ""
    status: ImageStatus = ImageStatus.QUEUED
    uploaded_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    
    def mark_as_processing(self):
        self.status = ImageStatus.PROCESSING
    
    def mark_as_classified(self):
        self.status = ImageStatus.CLASSIFIED
        self.processed_at = datetime.now()
```

### WasteCategory Enum

```python
class WasteCategory(str, Enum):
    """10 kategorija otpada"""
    BATTERY = "battery"
    BIOLOGICAL = "biological"
    CARDBOARD = "cardboard"
    CLOTHES = "clothes"
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    PLASTIC = "plastic"
    SHOES = "shoes"
    TRASH = "trash"
```

## 8.3 Decision Optimizer (Cost-Aware)

```python
class DecisionOptimizer:
    """Optimizira odluke korištenjem error cost matrix"""
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.70,
        review_threshold: float = 0.50,
        max_acceptable_cost: float = 1.0
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.review_threshold = review_threshold
        self.cost_matrix = ErrorCostMatrix()
    
    def optimize_decision(self, prediction_result: Dict) -> OptimizedDecision:
        """
        Optimizuj odluku:
        1. Provjeri confidence thresholds
        2. Izračunaj expected costs za sve kategorije
        3. Izaberi kategoriju sa minimalnim expected cost
        """
        original_confidence = prediction_result["confidence"]
        
        if original_confidence < self.review_threshold:
            return self._apply_fallback_low_confidence(...)
        
        prob_dist = self._build_probability_distribution(...)
        all_costs = self.cost_matrix.get_all_expected_costs(prob_dist)
        
        best_category = min(all_costs, key=all_costs.get)
        
        return OptimizedDecision(
            predicted_category=best_category,
            expected_cost=all_costs[best_category],
            reasoning="Minimized expected cost"
        )
```

## 8.4 Tick/Step Pravila (Svih 8 Pravila)

### Pravilo #1: Step/Tick = Jedna Iteracija Agentičkog Ciklusa
Svaki Step() mora biti prepoznatljiv kao:
- **Sense:** pročitaj jedno "stanje svijeta" (DB/queue/model state)
- **Think:** donesi odluku (policy/pravila)
- **Act:** izvrši akciju (promijeni svijet)
- **Learn (ako postoji):** ažuriraj znanje / brojač / metriku

> ⚠️ Ako funkcionalnosti nisu jasno razdvojene na Sense/Think/Act/Learn u tick-u → nije agent, nego helper metoda.

### Pravilo #2: Tick/Step Radi "Malo", Ne "Sve"
Tick/step mora biti kratak i atomaran.

- ✅ **Dobar primjer:** Scoring agent obradi JEDNU sliku
- ✅ **Dobar primjer:** Retrain agent uradi JEDNU provjeru
- ❌ **Loš primjer:** U tick-u obradi 200 slika + retrain + emit + cleanup (MONOLIT!)

### Pravilo #3: Tick/Step Mora Imati "No-Work" Izlaz Bez Štete
Kad nema posla, tick/step ne smije praviti štetu, ni trošiti resurse.

```python
# ✅ DOBRO
if queue_size == 0:
    return None  # Mirovanje bez exception-a

# ❌ LOŠE
if queue_size == 0:
    raise Exception("No images!")  # Baca exception kad nema posla
```

### Pravilo #4: Tick/Step Ne Smije Sadržavati "Host" Stvari
Tick/step je logika agenta (shared). Host (Web/Console) je samo orkestracija.

**Tick/step NE SMIJE implementirati:**
- ❌ `asyncio.sleep()` / `Task.Delay(...)`
- ❌ SignalR/WebSocket emit
- ❌ HTTP logiku, CORS, routing
- ❌ DTO mapping za API

**Tick/step MOŽE implementirati:**
- ✅ DB upis/čitanje preko servisa
- ✅ Odluku: status, thresholds, retrain rule
- ✅ Kreiranje domain objekata (Prediction, Review...)

### Pravilo #5: Tick Mora Biti Idempotentan Koliko God Može
Ako se tick/step ponovi (zbog restarta, exception-a), ne smije urušiti sistem.

**Implementacija:**
- Poruka se "dequeue-a" na siguran način (ne procesira se duplo)
- Upis prediction-a/statusa je konzistentan
- Status: `QUEUED → PROCESSING → DONE` (ili transakcija)

### Pravilo #6: Tick/Step Može Imati Jasno Definisan "Rezultat"
Tick/step može vraćati DTO rezultat koji host može koristiti za UI/log/realtime.

```python
# Preporuka
async def step_async(self) -> Optional[ClassificationResult]:
    # ... Sense → Think → Act ...
    
    if no_work:
        return None  # Nema posla
    
    return ClassificationResult(
        image_id=image.id,
        predicted_category=decision.category,
        confidence=decision.confidence
    )
```

**Host onda:**
- Logira rezultat
- Emituje WebSocket/SignalR event
- Odluči koliko da čeka

### Pravilo #7: Ako Akcija Traje Dugo, Tick Mora Biti State-Machine
Ako tick/step može trajati "dugo" (npr. trening modela 5-60 minuta), onda:

**Opcija A: State Machine**
```
CheckThreshold → StartTraining → ActivateModel → ResetCounter
```

**Opcija B: Async Job**
Trening je zaseban servis/job, tick samo prati status.

- ❌ **Loš primjer:** tick/step blokira 30-60 sekundi i drži worker "zaleđen"

### Pravilo #8: Granice Odgovornosti - Razdvajanje na Manje Agente

**Classification Agent Tick:**
- Sense: uzmi 1 queued sliku
- Think: klasifikuj (ML model + thresholds)
- Act: upiši prediction + postavi status

**Learning Agent Tick:**
- Sense: pročitaj settings + sample counter
- Think: ShouldRetrain?
- Act: treniraj + kreiraj ModelVersion + activate
- Learn: reset counter, log metrike

### Šablon za Async Step (Python)

```python
async def step_async(self) -> Optional[ClassificationResult]:
    # SENSE
    image = await self._queue.dequeue_next(ImageStatus.QUEUED)
    if image is None:
        return None  # No-work exit
    
    # THINK
    decision = await self._classifier.classify_image(image)
    
    # ACT
    await self._queue.save_prediction_and_update_status(
        image, decision
    )
    
    # (LEARN optional - u Learning Agent-u)
    
    return ClassificationResult(
        image_id=image.id,
        predicted_category=decision.category.value,
        confidence=decision.confidence
    )
```

### Dvije Najčešće Greške

1. **Stavljanje thresholds i retrain pravila u Web layer**
   → Minus jer Web mora biti tanak host

2. **Tick/step koji u sebi radi delay i realtime emit**
   → Minus jer tick/step mora biti dijeljiva "jezgra" (shared)

---

# 9. STRUKTURA PROJEKTA

```
trashvision/
│
├── AiAgents/
│   ├── __init__.py
│   │
│   ├── Core/                           # FRAMEWORK LAYER
│   │   ├── __init__.py
│   │   ├── software_agent.py           # SoftwareAgent<T> bazna klasa
│   │   ├── learning_component.py       # ILearningComponent
│   │   ├── perception_source.py        # IPerceptionSource
│   │   ├── policy.py                   # IPolicy
│   │   └── actuator.py                 # IActuator
│   │
│   └── TrashAgent/                     # AGENT IMPLEMENTATION
│       ├── __init__.py
│       │
│       ├── Domain/                     # DOMAIN LAYER
│       │   ├── __init__.py
│       │   ├── entities.py             # WasteImage, Prediction, Review
│       │   ├── enums.py                # ImageStatus, WasteCategory
│       │   ├── value_objects.py        # ClassificationDecision
│       │   ├── error_costs.py          # ErrorCostMatrix
│       │   └── decision_optimizer.py   # DecisionOptimizer
│       │
│       ├── Application/                # APPLICATION LAYER
│       │   ├── __init__.py
│       │   │
│       │   ├── Services/               # Use-case servisi
│       │   │   ├── __init__.py
│       │   │   ├── queue_service.py
│       │   │   ├── classification_service.py
│       │   │   ├── review_service.py
│       │   │   └── training_service.py
│       │   │
│       │   └── Agents/                 # Agent runneri
│       │       ├── __init__.py
│       │       ├── classification_runner.py  # Sense→Think→Act
│       │       └── learning_runner.py        # Sense→Think→Act→Learn
│       │
│       ├── Infrastructure/             # INFRASTRUCTURE LAYER
│       │   ├── __init__.py
│       │   ├── database.py             # SQLAlchemy/SQLite
│       │   ├── file_storage.py         # File operations
│       │   ├── waste_classifier.py     # ML abstraction
│       │   ├── yolo_classifier.py      # YOLO v8 inference
│       │   └── sorting_simulation.py   # Sorting simulation
│       │
│       └── Web/                        # WEB LAYER (Tanki!)
│           ├── __init__.py
│           ├── main.py                 # FastAPI app
│           │
│           ├── controllers/            # API endpoints
│           │   ├── __init__.py
│           │   ├── prediction_controller.py
│           │   └── learning_controller.py
│           │
│           ├── dto/                    # Data Transfer Objects
│           │   ├── __init__.py
│           │   └── responses.py
│           │
│           └── workers/                # Background workers
│               ├── __init__.py
│               ├── classification_worker.py  # Loop + delay
│               └── learning_worker.py        # Loop + delay
│
├── app/
│   └── frontend/
│       ├── index.html                  # Main UI
│       └── simulation.html             # Sorting simulation UI
│
├── data/
│   ├── uploads/                        # Uploaded images
│   └── new_samples/                    # Feedback samples
│
├── models/
│   └── trashvision_v1/
│       └── weights/
│           ├── best.pt                 # Best model weights
│           └── last.pt                 # Last checkpoint
│
├── docs/
│   └── LLM_USAGE.md                    # LLM usage documentation
│
├── run_agent.py                        # Entry point
├── simulation_demo.py                  # Simulation demo
├── requirements.txt                    # Dependencies
├── README.md                           # Project documentation
├── API.md                              # API documentation
└── ARCHITECTURE.md                     # Architecture documentation
```

---

# 9.1 KAKO POKRENUTI SISTEM

## Preduvjeti

Prije pokretanja sistema, provjerite da imate:

1. **Python 3.9+** instaliran
2. **Virtualno okruženje** aktivirano
3. **Sve dependencies** instalirane:
```bash
pip install -r requirements.txt
```
4. **Trenirani model** na lokaciji `models/trashvision_v1/weights/best.pt`

## Pokretanje Backend Servera (Agent Sistema)

### Korak 1: Aktivirajte virtualno okruženje

**Windows (PowerShell):**
```powershell
cd trashvision
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
cd trashvision
venv\Scripts\activate
```

**Linux/Mac:**
```bash
cd trashvision
source venv/bin/activate
```

### Korak 2: Pokrenite agent sistem

```bash
python run_agent.py
```

### Očekivani Output

Kada se sistem uspješno pokrene, vidjet ćete:

```
============================================================
🚀 TRASHVISION AGENT STARTING...
============================================================
📦 1/6: Initializing database...
✅ Database ready!
🤖 2/6: Loading ML model...
📥 Loading YOLO model: models\trashvision_v1\weights\best.pt
✅ YOLO model loaded successfully
💾 3/6: Initializing file storage...
⚙️ 4/6: Creating services...
🤖 5/6: Creating agent runners...
🔄 6/6: Starting background workers...
✅ ClassificationWorker started (tick every 2s)
✅ LearningWorker started (check every 60s)
============================================================
✅ TRASHVISION AGENT READY!
============================================================
📍 API: http://localhost:8000
📚 Docs: http://localhost:8000/docs
🤖 Classification Agent: Running (every 2s)
🎓 Learning Agent: Running (every 60s)
============================================================
```

### Korak 3: Testirajte API

- **Health Check:** http://localhost:8000/
- **API Dokumentacija:** http://localhost:8000/docs
- **Status Agenata:** http://localhost:8000/status

## Pokretanje Frontenda

### Opcija A: Direktno otvaranje HTML fajla

Jednostavno otvorite `app/frontend/index.html` u web pregledniku (Chrome, Firefox, Edge).

### Opcija B: Korištenje Simple HTTP Server

```bash
cd app/frontend
python -m http.server 8080
```
Zatim otvorite: http://localhost:8080

---

# 9.2 OPIS FRONTEND FAJLOVA

## index.html - Glavni Korisnički Interfejs

**Lokacija:** `app/frontend/index.html`

**Namjena:** Glavna web aplikacija za interakciju sa TrashVision sistemom.

**Funkcionalnosti:**

| Feature | Opis |
|---------|------|
| 📸 **Upload Slika** | Drag & drop ili click za upload pojedinačne slike |
| 📦 **Batch Upload** | Upload više slika odjednom sa CSV exportom rezultata |
| 📹 **Live Camera** | Real-time klasifikacija putem web kamere (svake 3 sekunde) |
| 🔄 **Auto-Classify** | Automatska klasifikacija odmah nakon upload-a |
| 💬 **Feedback** | Mogućnost ispravljanja pogrešnih predikcija |
| 📊 **Analitika** | Dashboard sa statistikama klasifikacija |
| 📜 **Historija** | Pregled prethodnih 12 klasifikacija |
| 🌙/☀️ **Dark/Light Mode** | Prebacivanje između tamne i svijetle teme |
| 💡 **Fun Facts** | Edukativne činjenice o reciklaži |

**Korištenje:**
1. Otvorite `index.html` u pregledniku
2. Provjerite da backend radi na `http://localhost:8000`
3. Upload-ujte sliku ili pokrenite kameru
4. Pogledajte rezultat klasifikacije sa preporukama za odlaganje

---

## simulation.html - Simulacija Sortiranja

**Lokacija:** `app/frontend/simulation.html`

**Namjena:** Vizualna simulacija industrijskog sistema za sortiranje otpada.

**Funkcionalnosti:**

| Feature | Opis |
|---------|------|
| 🏭 **Conveyor Belt** | Simulacija transportne trake (10m, 0.2-0.3 m/s) |
| 🤖 **Robotic Arm** | State machine robotizirane ruke (scan → pick → move → drop) |
| 📦 **Sorting Bins** | Kontejneri za svaku kategoriju otpada |
| 📊 **Real-time Stats** | Praćenje efikasnosti, kontaminacije i troškova |
| ⚠️ **Error Tracking** | Praćenje grešaka sortiranja po kategorijama |

**State Machine Robotske Ruke:**
```
IDLE → SCANNING → PICKING → MOVING → DROPPING → IDLE
```

**Korištenje:**
1. Otvorite `simulation.html` u pregledniku
2. Kliknite "Start Simulation"
3. Pratite kako objekti prolaze kroz sistem
4. Analizirajte statistike sortiranja

---

## Razlika Između Fajlova

| Aspekt | index.html | simulation.html |
|--------|------------|-----------------|
| **Namjena** | Produkcijska klasifikacija | Edukativna simulacija |
| **Koristi ML Model** | ✅ Da (YOLO v8) | ❌ Ne (mock podaci) |
| **User Input** | Upload slika, kamera | Samo start/stop |
| **Output** | Predikcija + preporuke | Vizualna animacija |
| **Backend zahtjev** | localhost:8000 | Nije potreban |
| **Ciljna publika** | Krajnji korisnici | Demonstracija koncepta |

---

# 10. REVIEW IMPLEMENTACIJE SA LLM-OM

## 10.1 Code Review Pitanja

Tokom razvoja, koristili smo LLM za review implementacije. Primjeri pitanja:

1. **"Da li Web sloj sadrži biznis logiku?"**
2. **"Da li je Sense/Think/Act/Learn jasno razdvojeno?"**
3. **"Koje su 3 najveće greške i kako ih popraviti?"**
4. **"Gdje je coupling prejak, gdje treba refaktor?"**

---

### Code Review Sesija (22. decembar 2025.)

**Ja:** Hajde pogledaj sve implementacije meni ovo kao da ne radi dobro.

**LLM:** Provjerio sam kod i našao nekoliko problema! Evo fikseva:

**🔴 Problem 1: Live Webcam ne radi jer video nije definisan nakon što se modal rebuild-uje**

U funkciji `startLiveClassification()`, pozivas `video.srcObject = currentStream;` DVA PUTA - jednom prije rebuild-a modala, i jednom poslije. Ali nakon rebuild-a, stari video element više ne postoji!

**✅ FIX:**
```javascript
async function startLiveClassification() {
    try {
        currentStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: 640, height: 480 }
        });
        // Prvo prikaži modal
        document.getElementById('cameraModal').style.display = 'flex';
        // ZATIM rebuild sadržaj
        // NA KRAJU attach stream (sad video postoji!)
        const video = document.getElementById('cameraVideo');
        video.srcObject = currentStream;
    } catch (error) {
        alert('Greška pri pristupu kameri: ' + error.message);
    }
}
```

**🔴 Problem 2: Batch Upload ne hvata drag-and-drop**
U kodu, drop event samo poziva `handleFile()` za JEDAN fajl.

**🔴 Problem 3: Analytics crta chart u pogrešnim CSS bojama**
Canvas ne razumije CSS varijable!

**🔴 Problem 4: updateAnalytics() baca grešku kad je history prazan**

---

### Iteracija: Feedback Endpoint 500 Error

**Ja:** feedback ne radi - INFO: 127.0.0.1:55177 - "POST /feedback HTTP/1.1" 500 Internal Server Error

**LLM:** Vidim problem! `/feedback` endpoint ima bug - pokušava da čuva file bytes direktno, ali `copy_to_learning_set` očekuje filepath.

Problem je i što `SystemSettings.increment_samples()` ne snima u bazu! 

**✅ FIX - Popravljam /feedback endpoint:**
```python
@app.post("/feedback")
async def feedback(
    file: UploadFile = File(...),
    predicted_class: str = Form(...),
    actual_class: str = Form(...),
    confidence: float = Form(...)
):
    # Validacija kategorije
    try:
        actual_category = WasteCategory(actual_class.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {actual_class}")
    
    # Snimi fajl privremeno
    temp_path = f"data/temp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Kopiraj u learning dataset
    await app_state.file_storage.copy_to_learning_set(temp_path, actual_category)
    
    # Inkrementiraj counter I SNIMI U BAZU
    app_state.settings.increment_new_samples()
    await app_state.queue_service.save_settings(app_state.settings)
```

---

## 10.2 Iteracije i Popravke

### Iteracija 1: Windows Multiprocessing Greška

**Problem:** `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.`

**Rješenje:** Dodavanje `if __name__ == '__main__':` bloka i `workers=0` parametra u train.py

### Iteracija 2: Import Greške - Relativni Importi

**Problem:** `ImportError: attempted relative import with no known parent package`

**Rješenje:** Kreiranje `run_agent.py` launcher skripte u root direktoriju:
```python
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import uvicorn
if __name__ == "__main__":
    uvicorn.run("AiAgents.TrashAgent.Web.main:app", host="0.0.0.0", port=8000, reload=True)
```

### Iteracija 3: QueueService Mock Problem

**Problem:** Agent queue ne radi - frontend traži `/api/images/717499` → 404 Not Found, ali agent NIKADA ne procesira queue.

**Rješenje:** Implementacija pravog DB CRUD-a u QueueService:
```python
async def enqueue(self, image: WasteImage) -> WasteImage:
    """Snimi sliku u bazu i vrati sa ID-om"""
    with self._get_session() as session:
        db_image = ImageModel(
            filepath=image.filepath,
            filename=image.filename,
            status=image.status.value
        )
        session.add(db_image)
        session.commit()
        session.refresh(db_image)
        image.id = db_image.id
        return image
```

### Iteracija 4: YOLO Classifier Async Problem

**Problem:** `YoloWasteClassifier.__init__` pokušava pozvati `asyncio.run()` unutar već postojećeg event loop-a.

**Rješenje:** Razdvajanje inicijalizacije od učitavanja modela:
```python
class YoloWasteClassifier:
    def __init__(self, model_path: str = None):
        self._model_path = model_path
        self._model = None  # Lazy loading
    
    async def load_model(self):
        """Učitaj model async - pozovi nakon __init__"""
        if self._model_path and Path(self._model_path).exists():
            self._model = YOLO(self._model_path)
```

---

# 11. DATASET I IZVORI

## 11.1 Dataset

**Naziv dataseta:** Garbage Classification v2

**Link:** https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data

**Opis:** Dataset sadrži slike otpada kategorizovane u 10 klasa:
1. Battery (Baterije) - 944 slika
2. Biological (Organski otpad) - 997 slika
3. Cardboard (Karton) - 1,825 slika
4. Clothes (Odjeća) - 5,327 slika
5. Glass (Staklo) - 3,061 slika
6. Metal - 1,020 slika
7. Paper (Papir) - 1,680 slika
8. Plastic (Plastika) - 1,984 slika
9. Shoes (Obuća) - 1,977 slika
10. Trash (Mješoviti otpad) - 946 slika

**Broj slika:** 19,761 ukupno

**Podjela:**
- Training set: 70% (13,827 slika)
- Validation set: 15% (2,966 slika)
- Test set: 15% (2,968 slika)

## 11.2 Obrada Dataseta

- Slike su pretprocesirane na veličinu 224x224 piksela
- YOLO v8 classification format
- Augmentacija korištena tokom treniranja:
  - HSV Hue: 0.015
  - HSV Saturation: 0.7
  - HSV Value: 0.4
  - Rotacija: ±10°
  - Translacija: 0.1
  - Skaliranje: 0.5
  - Horizontal flip: 50%

## 11.3 Rezultati Treniranja

**Model:** YOLOv8n-cls (nano classification)

**Trajanje:** 1.58 sati (50 epoha) na NVIDIA GTX 1650

**Finalni Rezultati:**
- **Top-1 Accuracy:** 90.4%
- **Top-5 Accuracy:** 99.5%
- **Loss:** 0.424

**Tačnost po klasama:**
| Klasa | Precision | Recall | F1-Score | Tačnost |
|-------|-----------|--------|----------|---------|
| Clothes | 0.99 | 0.98 | 0.99 | 98.5% |
| Shoes | 0.96 | 0.95 | 0.95 | 95.3% |
| Biological | 0.95 | 0.95 | 0.95 | 94.7% |
| Battery | 0.93 | 0.93 | 0.93 | 93.0% |
| Cardboard | 0.92 | 0.90 | 0.91 | 89.8% |
| Glass | 0.85 | 0.89 | 0.87 | 89.3% |
| Paper | 0.89 | 0.88 | 0.89 | 88.5% |
| Metal | 0.87 | 0.82 | 0.85 | 82.4% |
| Plastic | 0.83 | 0.81 | 0.82 | 80.5% |
| Trash | 0.78 | 0.68 | 0.73 | 68.3% |

**Najčešće konfuzije:**
- Glass ↔ Plastic (54 greške) - Oba su prozirna/sjajna
- Cardboard → Paper (19 grešaka) - Slični materijali
- Trash → Paper (16 grešaka) - Generički otpad često sadrži papir

---

# 12. KORIŠTENJE LLM-A TOKOM RAZVOJA

## 12.1 Diskusija Ideje Projekta

### Inicijalni Razgovor (2. decembar 2025.)

**Ja:** Druga ideja je TrashVision, AI sistem zasnovan na računalnom vidu (Computer Vision), čiji je cilj prepoznavanje vrste i količine otpada na osnovu slike. Koristio bi unaprijed istrenirane model(e) klasifikacije slike (npr. YOLOv8, MobileNet ili ResNet), s mogućnošću dodatnog učenja na datasetima poput TrashNet-a. Nakon što korisnik učita sliku, agent bi detektovao tip otpada (plastika, papir, staklo, metal, organski) i eventualno procijenio zapreminu, uz prikaz vizualnog rezultata i preporuke za pravilno odlaganje. Hoću da koristim ovaj dataset: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2/data. Hoću da mi pomogneš sa strukturom projekta, izgradnjom, treniranjem svim ali korak po korak uz provjere napretka.

**LLM:** Odličan izbor projekta! TrashVision je praktična primjena AI-ja s realnim utjecajem. Pomoći ću ti da ga izgradiš korak po korak.

**Pregled projekta TrashVision:**

Prije nego počnemo, hajde da razjasnimo arhitekturu:

**Komponente sistema:**
1. Model za klasifikaciju - YOLOv8 ili sličan za detekciju i klasifikaciju otpada
2. Backend API - Python (Flask/FastAPI) za procesiranje slika
3. Frontend - Web interfejs za upload slika i prikaz rezultata
4. Dataset - Garbage Classification v2 sa Kaggle-a

**Korak 1: Priprema razvojnog okruženja**

Počnimo od početka. Prvo trebamo:
1. Kreirati strukturu projekta
2. Instalirati potrebne biblioteke
3. Preuzeti i analizirati dataset

**Akcije za Korak 1:**
```bash
# Kreiraj direktorijume
mkdir trashvision
cd trashvision
mkdir data models notebooks src app
```

**Struktura projekta:**
```
trashvision/
├── data/           # Dataset
├── models/         # Trenirani modeli
├── notebooks/      # Jupyter notebooks za eksperimente
├── src/            # Izvorni kod
├── app/            # Web aplikacija
├── requirements.txt
└── README.md
```

---

## 12.2 Izrada Specifikacije

### Korak-po-Korak Setup (2. decembar 2025.)

**Ja:** Haj opet detaljnije korak po korak.

**LLM:** Odlično! Idemo polako i detaljno.

**KORAK 1.1: Instalacija Pythona i provjera okruženja**

Prvo, provjeri šta imaš:

Otvori Command Prompt (CMD) ili PowerShell i upiši:
```bash
python --version
```

**KORAK 1.2: Kreiranje projekta**
```bash
cd Desktop
mkdir trashvision
cd trashvision
```

**KORAK 1.3: Kreiranje strukture foldera**
```bash
mkdir data
mkdir data\raw
mkdir data\processed
mkdir models
mkdir notebooks
mkdir src
mkdir app
mkdir app\backend
mkdir app\frontend
```

**KORAK 1.4: Kreiranje requirements.txt**
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
scikit-learn>=1.3.0
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
kaggle>=1.5.0
jupyter>=1.0.0
```

---

### Analiza Dataseta (2. decembar 2025.)

**Ja:** [Output iz explore_dataset.py]
✅ Dataset pronađen!
📁 Putanja: data\raw\garbage-dataset
📊 Broj kategorija: 10
📦 battery - 944 slika
📦 biological - 997 slika
📦 cardboard - 1825 slika
📦 clothes - 5327 slika
📦 glass - 3061 slika
📦 metal - 1020 slika
📦 paper - 1680 slika
📦 plastic - 1984 slika
📦 shoes - 1977 slika
📦 trash - 946 slika
📷 UKUPNO SLIKA: 19761

**LLM:** Izvrsno! 🎉 Imamo odličan pregled podataka:

📊 **Analiza dataseta:**
- ✅ 19,761 slika ukupno
- ✅ 10 kategorija otpada
- ✅ Dataset je nebalansiran (clothes ima 5327, a battery samo 944)
- ✅ Različite dimenzije slika (treba normalizacija)

**KORAK 3: Priprema podataka za treniranje**

Sada trebamo:
1. Podijeliti podatke na train/validation/test (70%/15%/15%)
2. Balansirati klase (opciono)
3. Kreirati strukturu za YOLO format

---

## 12.3 Pomoć pri Implementaciji

### Treniranje Modela (3. decembar 2025.)

**Ja:** [Output iz train.py - treniranje završeno]
Epoch 50/50: loss=0.424, top1_acc=0.904, top5_acc=0.995
✅ Treniranje završeno uspješno!
Model sačuvan u: models/trashvision_v1/weights/best.pt

**LLM:** 🎉 BRAVO! Treniranje uspješno završeno!

Odlični rezultati! Pogledajmo šta smo postigli:

📊 **Finalni Rezultati:**
- ✅ Top-1 Accuracy: 90.4% (90.4% slika tačno klasifikovano)
- ✅ Top-5 Accuracy: 99.5% (99.5% slika ima tačnu klasu u top 5)
- ✅ Loss: 0.424 (počelo 1.01 → palo na 0.42)
- ⏱️ Trajanje: 1.58 sati (50 epoha)
- 💾 Model sačuvan: models/trashvision_v1/weights/best.pt

---

### Implementacija Agent Arhitekture (22. decembar 2025.)

**Ja:** Hajde daj korak po korak [za Clean Architecture refaktoring].

**LLM:** Odlično! Idemo korak po korak. 🚀

**KORAK 1: CORE SLOJ (Framework apstrakcije)**

Ovo je framework sloj - generičke apstrakcije koje ne znaju ništa o "trash" ili "waste". Mogu se koristiti za bilo koji tip agenta.

**📁 Fajl 1/5: software_agent.py (Bazna klasa)**
```python
class SoftwareAgent(ABC, Generic[TPercept, TAction, TResult]):
    """
    Bazna klasa za sve software agente.
    Implementira osnovni Sense → Think → Act ciklus.
    """
    
    @abstractmethod
    async def sense(self) -> Optional[TPercept]:
        """SENSE: Opazi okolinu"""
        pass
    
    @abstractmethod
    async def think(self, percept: TPercept) -> TAction:
        """THINK: Donesi odluku"""
        pass
    
    @abstractmethod
    async def act(self, action: TAction) -> TResult:
        """ACT: Izvrši akciju"""
        pass
    
    async def step_async(self) -> Optional[TResult]:
        """Jedan tick agent ciklusa"""
        percept = await self.sense()
        if percept is None:
            return None  # No-work exit
        
        action = await self.think(percept)
        result = await self.act(action)
        
        return result
```

---

### Classification Agent Runner (22. decembar 2025.)

**LLM:** Fajl 6/7: classification_runner.py (GLAVNI AGENT!)

```python
class ClassificationAgentRunner(SoftwareAgent[WasteImage, ClassificationDecision, ClassificationResult]):
    """
    Classification Agent - Sense→Think→Act ciklus za klasifikaciju otpada.
    
    SENSE: Preuzmi sljedeću sliku iz queue-a
    THINK: Klasifikuj sliku + primijeni policy (confidence thresholds)
    ACT: Sačuvaj predikciju i ažuriraj status slike
    """
    
    async def sense(self) -> Optional[WasteImage]:
        """SENSE: Preuzmi sljedeću sliku iz queue-a"""
        queue_size = await self._queue.get_queue_size(ImageStatus.QUEUED)
        
        if queue_size == 0:
            return None  # No-work exit
        
        image = await self._queue.dequeue_next(ImageStatus.QUEUED)
        self._context.current_image = image
        return image
    
    async def think(self, image: WasteImage) -> ClassificationDecision:
        """THINK: Klasifikuj sliku preko YOLO modela + primijeni policy"""
        decision = await self._classifier.classify_image(image, self._settings)
        
        # Decision sadrži:
        # - predicted_category: WasteCategory
        # - confidence: float
        # - new_status: ImageStatus (CLASSIFIED ili PENDING_REVIEW)
        
        return decision
    
    async def act(self, decision: ClassificationDecision) -> ClassificationResult:
        """ACT: Sačuvaj predikciju i ažuriraj status"""
        image = self._context.current_image
        
        await self._queue.save_prediction_and_update_status(image, decision)
        
        return ClassificationResult(
            image_id=image.id,
            predicted_category=decision.category.value,
            confidence=decision.confidence,
            status=decision.new_status.value
        )
```

---

## 12.4 Pregled i Poboljšanja

### Frontend Debugging (22. decembar 2025.)

**Ja:** zasto ne radi historija, zasto nemam kameru kao live kao sto sam imao da prepoznaje, feedback ne radi i zelim da izgleda barem priblizno kao prije

**LLM:** Razumijem! Hajde da vratim SVE stare funkcionalnosti + dodam kameru! 🎥

Daću ti KOMPLETAN frontend koji ima:
- ✅ Live Camera mode (kao prije)
- ✅ History sa slikama
- ✅ Feedback koji radi
- ✅ Isti dizajn kao prije

---

### Final Frontend Features (22. decembar 2025.)

**Ja:** a gdje su mi boje kao prije i gdje mi je dark/light mode i gdje mi je analitika i gdje mi je batch recognition, daj da je sve na jednom jeziku recimo bosanski

**LLM:** Haaa razumijem! Hoćeš SVE kao što je bilo! 💪

Hajde da napravim POTPUNI FRONTEND sa:
- ✅ Dark/Light mode toggle
- ✅ Analitika (grafikoni, statistike)
- ✅ Batch recognition (upload više slika odjednom)
- ✅ SVE na BOSANSKOM jeziku
- ✅ Tvoje boje i dizajn
- ✅ Fun facts
- ✅ Live camera
- ✅ Auto-classify

**✅ ŠTA IMA (SVE NA BOSANSKOM):**

🎨 **Dizajn:**
- Dark/Light mode toggle (🌙/☀️)
- Tvoje boje i layout
- Responsive design

📸 **Klasifikacija:**
- Upload jedne slike - automatska klasifikacija
- Batch upload - više slika odjednom (sa CSV download-om)
- Live Camera - svake 3 sekunde auto-klasifikuje

📊 **Analitika:**
- Statistika (Skenova, Reciklirano, CO₂)
- Najčešća kategorija
- Prosječna pouzdanost
- Distribucija kategorija (bar chart)

💬 **Feedback:**
- "Ispravi" button na rezultatu
- Category selector grid
- Learning progress tracking

🖼️ **Historija:**
- 12 zadnjih klasifikacija
- Thumbnail grid
- Čuva se u localStorage

---

### Finalna Verifikacija Sistema

**Ja:** [Backend output pokazuje da sve radi]
```
✅ ClassificationWorker started (tick every 2s)
✅ LearningWorker started (check every 60s)
🔮 YOLO Prediction: glass (81.72%) - 53.1ms
🔮 YOLO Prediction: clothes (99.91%) - 49.8ms
👁️ SENSE: New samples: 1/100
🧠 THINK: No retraining needed (1.0% to threshold)
```

**LLM:** Odlično! Agent radi, slika se enqueue-uje! 🎉

✅ **Sada imaš:**

**NOVE AGENT funkcionalnosti:**
- /api/images/upload - Upload u queue (agent procesira)
- /api/images/{id} - Provjeri status
- /api/learning/stats - Learning progress
- /status - Agent status

**STARE funkcionalnosti (za kompatibilnost):**
- /predict - Direktna predikcija (BEZ agent queue-a)
- /feedback - User feedback
- Sve što je radio stari frontend!

---

# 13. ZAKLJUČAK

## 13.1 Šta je Postignuto

TrashVision projekat uspješno demonstrira implementaciju **inteligentnog softverskog agenta** za klasifikaciju otpada. Ključna postignuća:

1. **Agent Arhitektura**
   - Jasno razdvojen Sense → Think → Act → Learn ciklus
   - Multi-agent sistem sa paralelnim radom
   - Clean Architecture sa tankim Web slojem

2. **Inteligentno Ponašanje**
   - Cost-aware decision making
   - Kontinuirano učenje iz feedback-a
   - Nesigurni slučajevi sa human review

3. **Tehnička Implementacija**
   - YOLO v8 model sa 90.4% tačnošću
   - FastAPI backend
   - Real-time UI sa statusom agenata

## 13.2 Statistike Projekta

| Metrika | Vrijednost |
|---------|------------|
| Ukupno linija koda | ~4,170 |
| Broj fajlova | ~30 |
| Model tačnost (Top-1) | 90.4% |
| Model tačnost (Top-5) | 99.5% |
| Broj kategorija | 10 |
| Dataset veličina | 19,761 slika |
| Vrijeme treniranja | 1.58h (50 epoha) |
| GPU korišten | NVIDIA GTX 1650 |

## 13.3 Naučene Lekcije

1. **Agent ≠ Aplikacija** - Iterativno ponašanje kroz vrijeme je ključna razlika
2. **Clean Architecture** - Web sloj mora biti tanak, biznis logika u Application sloju
3. **LLM Collaboration** - Kombinacija više modela daje najbolje rezultate
4. **Debugging** - Detaljno logovanje je ključno za praćenje agent ponašanja
5. **Async programiranje** - Pravilno rukovanje async/await je kritično za agent loop

## 13.4 Moguća Poboljšanja

1. Aktivno učenje sa upitima korisniku
2. Adaptivni pragovi na osnovu performansi
3. Proširenje na video stream
4. Reinforcement learning za optimizaciju policy-ja
5. Edge deployment na Raspberry Pi/Jetson Nano

---

# PRILOZI

## Prilog A: API Dokumentacija

### Pregled API-ja

**Base URL**: `http://localhost:8000`  
**API Verzija**: `2.0.0`  
**Format**: JSON  
**Charset**: UTF-8  

TrashVision API omogućava:
- ✅ Upload slika za klasifikaciju
- ✅ Provjeru statusa procesuiranja
- ✅ Direktnu sinhrononu klasifikaciju
- ✅ Korisnički feedback za učenje
- ✅ Monitoring sistema i statistiku

### Endpointi

#### 1. Upload Slike (Agent Queue)

**Endpoint**: `POST /api/images/upload`

Uploaduje sliku i stavlja je u queue za automatsku klasifikaciju od strane Classification Agent-a.

**Request**:
- Content-Type: `multipart/form-data`
- Body: `file` (required) - Image file (JPEG, PNG, WebP)

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/api/images/upload" \
  -F "file=@/path/to/image.jpg"
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

#### 2. Provjera Statusa Slike

**Endpoint**: `GET /api/images/{image_id}`

**Response** (200 OK) - Status: classified:
```json
{
  "image_id": 123,
  "filename": "bottle.jpg",
  "status": "classified",
  "processed_at": "2025-12-23T14:30:15.123456",
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

#### 3. Direktna Predikcija (Legacy)

**Endpoint**: `POST /predict`

Sinhronona klasifikacija koja ne ide kroz agent queue. Koristi se za kompatibilnost sa frontendom.

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
      "emoji": "♻️"
    }
  ]
}
```

**Kategorije i Odlaganje**:

| Kategorija | Emoji | Odlaganje | Reciklabilno |
|------------|-------|-----------|--------------|
| `battery` | 🔋 | Poseban kontejner za baterije | ✅ |
| `biological` | 🥬 | Braon/zelena kanta za kompost | ✅ |
| `cardboard` | 📦 | Plavi kontejner za papir | ✅ |
| `clothes` | 👕 | Kontejner za tekstil | ✅ |
| `glass` | 🍾 | Zeleni kontejner | ✅ |
| `metal` | 🥫 | Žuti kontejner | ✅ |
| `paper` | 📄 | Plavi kontejner | ✅ |
| `plastic` | 🧴 | Žuti kontejner | ✅ |
| `shoes` | 👟 | Kontejner za tekstil | ✅ |
| `trash` | 🗑️ | Crni kontejner | ❌ |

#### 4. User Feedback

**Endpoint**: `POST /feedback`

Omogućava korisnicima da potvrde ili isprave predikciju za continuous learning.

**Request**:
- `file` (required): Image file
- `predicted_class` (optional): Originalna predikcija
- `actual_class` (required): Ispravna kategorija
- `confidence` (optional): Confidence score

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

#### 5. Learning Statistika

**Endpoint**: `GET /api/learning/stats`

**Response** (200 OK):
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

#### 6. System Status

**Endpoint**: `GET /status`

**Response** (200 OK):
```json
{
  "classification_agent": {
    "is_running": true,
    "total_processed": 150,
    "last_run": "2025-12-23T14:30:00"
  },
  "learning_agent": {
    "is_running": true,
    "last_check": "2025-12-23T14:29:00"
  },
  "database_connected": true,
  "model_loaded": true
}
```

### Status Kodovi

| Kod | Naziv | Značenje |
|-----|-------|----------|
| **200** | OK | Uspješan request |
| **400** | Bad Request | Loš format ili validacija |
| **404** | Not Found | Resurs ne postoji |
| **500** | Internal Server Error | Server greška |

---

## Prilog B: Database Schema

### Tabele

#### waste_images
```sql
CREATE TABLE waste_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL,
    filename TEXT NOT NULL,
    status TEXT DEFAULT 'queued',
    predicted_category TEXT,
    confidence REAL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);
```

**Status Lifecycle**:
```
queued → processing → classified (confidence ≥ 70%)
queued → processing → pending_review (confidence < 70%)
pending_review → reviewed
```

#### system_settings
```sql
CREATE TABLE system_settings (
    id INTEGER PRIMARY KEY,
    retrain_threshold INTEGER DEFAULT 10,
    auto_retrain_enabled BOOLEAN DEFAULT TRUE,
    new_samples_count INTEGER DEFAULT 0,
    retrain_count INTEGER DEFAULT 0,
    last_retrain_at TIMESTAMP
);
```

---

## Prilog C: Rezime Korištenja LLM-a

### Kombinacija Različitih LLM Alata

Projekt je razvijen korištenjem **kombinacije različitih LLM alata** prema preporukama predmeta:

1. **GPT / Copilot Chat (šira slika)** - korišten za diskusiju ideje, arhitekturu, rizike, edge-caseove
2. **Claude AI / Cursor (implementacija)** - korišten za generisanje većih dijelova koda sa kontekstom projekta
3. **GitHub Copilot u IDE (sitne stvari)** - korišten za metode, DTO-e, mapping

**Optimalni tok koji je primijenjen:**
- Diskusija sa GPT/Copilot Chat → generisanje ideja, specifikacija
- Spec u Claude AI → implementacija po slojevima i modulima
- Review u GPT/Copilot Chat → refaktor, čišćenje, provjera agent ciklusa
- 2-3 iteracije između modela → vidljivo u Sekcijama 10 i 12

### Faze Razvoja sa LLM Asistencijom

| Faza | Period | Opis | LLM Alat |
|------|--------|------|----------|
| Diskusija Ideje | 2. dec 2025. | Inicijalna ideja, izbor dataseta, planiranje arhitekture | Claude AI (šira slika) |
| Specifikacija | 2-3. dec 2025. | Setup okruženja, analiza dataseta, konfiguracija treniranja | Claude AI |
| Implementacija | 3-22. dec 2025. | Model treniranje, backend API, frontend, agent arhitektura | GitHub Copilot (IDE) |
| Review/Debug | 22. dec 2025. | Clean Architecture refaktoring, debugging, finalne popravke | Claude AI (review) |

### Statistike

- **Korišteni LLM alati:** GitHub Copilot Chat (Claude Opus 4.5), Claude AI
- **Ukupno sesija:** ~20+ sati razvoja uz LLM asistenciju
- **Primarna namjena:** Korak-po-korak implementacija sa verifikacijom

### Ključni Doprinosi LLM-a

1. **Arhitektura** - Dizajn Clean Architecture sa slojevima
2. **Agent Pattern** - Implementacija Sense→Think→Act→Learn ciklusa
3. **Debugging** - Rješavanje import grešaka, async problema, frontend bugova
4. **Dokumentacija** - Generisanje API dokumentacije i komentara

---

**Kraj Dokumentacije**

*Dokument pripremljen za predmet Umjetna inteligencija 2025/2026*  
*Profesorica: Nina Bijedić*  
*Asistenti: Adil Joldić, Azra Smajić*

