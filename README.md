# 🏛 MCA eConsultation — AI-Powered Sentiment Analysis System

> **Ministry of Corporate Affairs | Problem Statement ID: 25035**
> AI-driven sentiment analysis of stakeholder feedback submitted through the eConsultation module.

---

## 📋 Overview

The MCA eConsultation platform receives stakeholder comments on proposed amendments to corporate legislation. When large volumes of feedback are submitted, manual analysis becomes impractical and risks overlooking critical observations.

This system automates the analysis using three AI models:

| Model | Purpose | Details |
|-------|---------|---------|
| **RoBERTa** | Sentiment Classification | 3-class (positive/neutral/negative) with confidence scores |
| **BART-large-CNN** | Abstractive Summarization | Chunk-based approach for long documents |
| **WordCloud** | Keyword Visualization | Domain-specific stopword filtering |

### What It Does

1. **Upload** a CSV file with stakeholder comments
2. **Classify** each comment as positive, neutral, or negative with confidence scores
3. **Flag** uncertain predictions (confidence < 60%) for human review
4. **Summarize** feedback per sentiment group using BART
5. **Generate** word clouds highlighting key terms
6. **Categorize** comments by topic (Penalties, Audit, Reporting, Compliance)
7. **Produce** downloadable CSV and PDF reports
8. **Track** analysis history per user

---

## 🚀 How to Run This Project

### Prerequisites

| Software | Version | Install (macOS) | Check |
|----------|---------|-----------------|-------|
| Java JDK | 17+ | `brew install openjdk@17` | `java -version` |
| Maven | 3.9+ | `brew install maven` | `mvn -version` |
| Python | 3.10+ | `brew install python@3.10` | `python3 --version` |
| PostgreSQL | 14+ | `brew install postgresql@14` | `psql --version` |

### Step 1: Set Up PostgreSQL

```bash
# Start PostgreSQL
brew services start postgresql@14

# Create the database user and database
psql -U $(whoami) -d postgres -c "CREATE ROLE postgres WITH LOGIN SUPERUSER PASSWORD '1234';"
psql -U postgres -c "CREATE DATABASE mca_db;"
```

> Tables (`users`, `analysis_history`) are auto-created by Hibernate on first run.

### Step 2: Set Up the Python AI Service

```bash
cd python-ai-service

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the AI service
python ai_server.py
```

**First run downloads AI models from HuggingFace (~2GB):**
- RoBERTa sentiment model (~500MB)
- BART-large-CNN summarization model (~1.6GB)

Wait until you see:
```
✓ Sentiment model loaded → mps
✓ Summarization model loaded → mps
AI Service ready on 0.0.0.0:5001
```

> **Optional:** Place your fine-tuned RoBERTa model in `python-ai-service/policy_sentiment_final/`. If not present, the system uses the CardiffNLP fallback model.

### Step 3: Build and Start Spring Boot

Open a **new terminal**:

```bash
cd mca-econsultation

# Set Java 17 (macOS Homebrew)
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export PATH="$JAVA_HOME/bin:$PATH"

# Build
mvn clean package -DskipTests

# Run
mvn spring-boot:run
```

Wait until you see:
```
Tomcat started on port 8080
Started MCAApplication in X.XX seconds
```

### Step 4: Open the Application

Go to **http://localhost:8080**

1. Browse the dashboard without logging in
2. Click **Sign Up** to create an account, or use **Sign in with Google**
3. Upload `sample_data.csv` (included in project root) or your own CSV
4. View the analysis dashboard with charts, summaries, and word cloud
5. Download CSV or PDF reports
6. Check **History** for past analyses

---

## 🐳 Run with Docker (Alternative)

```bash
cd mca-econsultation

# Build and start all 3 services
docker compose up --build

# Access at http://localhost:8080

# Stop
docker compose down
```

This starts PostgreSQL (5432), AI Service (5001), and Spring Boot (8080) together.

---

## ⚙️ Configuration

Edit `src/main/resources/application.properties`:

```properties
# Database
spring.datasource.url=jdbc:postgresql://localhost:5432/mca_db
spring.datasource.username=postgres
spring.datasource.password=1234

# Google OAuth2 (get from console.cloud.google.com)
spring.security.oauth2.client.registration.google.client-id=YOUR_CLIENT_ID
spring.security.oauth2.client.registration.google.client-secret=YOUR_SECRET

# AI Service URL
ai.service.url=http://localhost:5001

# Static resources (word cloud images)
spring.web.resources.static-locations=classpath:/static/,file:python-ai-service/static/
```

### Google OAuth2 Setup

1. Go to **https://console.cloud.google.com** → Create project
2. **APIs & Services → Credentials → Create OAuth client ID**
3. Configure consent screen (External)
4. Application type: **Web application**
5. Authorized JavaScript Origins: `http://localhost:8080`
6. Authorized Redirect URIs: `http://localhost:8080/login/oauth2/code/google`
7. Copy Client ID and Secret into `application.properties`

---

## 🏗 Architecture

```
┌─────────────────────┐    REST API    ┌────────────────────────┐
│    Spring Boot 3.x   │◄─────────────►│   Flask AI Service     │
│    (Port 8080)       │               │   (Port 5001)          │
│                      │               │                        │
│  • Spring Security   │               │  • RoBERTa (Sentiment) │
│  • Google OAuth2     │               │  • BART-CNN (Summary)  │
│  • CSV Parsing       │               │  • WordCloud (Viz)     │
│  • iText 7 (PDF)     │               │  • Explainability (XAI)│
│  • Thymeleaf (UI)    │               │  • Evaluation Pipeline │
│  • Chart.js          │               │                        │
└──────────┬───────────┘               │  Modules:              │
           │                           │  • config.py           │
      ┌────▼─────┐                     │  • preprocessing.py    │
      │PostgreSQL│                     │  • evaluation.py       │
      │ (mca_db) │                     │  • explainability.py   │
      └──────────┘                     └────────────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Spring Boot 3.2.5, Java 17 |
| AI Service | Python 3.10+, Flask 3.0 |
| Sentiment Model | RoBERTa-base (HuggingFace Transformers) |
| Summarization | facebook/bart-large-cnn |
| Database | PostgreSQL 15 |
| ORM | Spring Data JPA + Hibernate |
| Auth | Spring Security + BCrypt + Google OAuth2 |
| PDF | iText 7 |
| Frontend | Thymeleaf + Chart.js + ChartDataLabels |
| NLP Preprocessing | NLTK, regex |
| Evaluation | scikit-learn, matplotlib |
| Containerization | Docker + Docker Compose |

---

## 📁 Project Structure

```
mca-econsultation/
├── pom.xml                          # Maven build config
├── docker-compose.yml               # Docker orchestration
├── Dockerfile                       # Spring Boot container
├── sample_data.csv                  # Sample CSV for testing
├── README.md                        # This file
│
├── src/main/java/com/mca/econsult/
│   ├── MCAApplication.java          # Spring Boot entry point
│   ├── config/
│   │   └── SecurityConfig.java      # Security + OAuth2 config
│   ├── controller/
│   │   ├── AuthController.java      # Login / Signup / Register
│   │   ├── AnalysisController.java  # Upload / Dashboard / Download
│   │   └── HistoryController.java   # Analysis history
│   ├── service/
│   │   ├── UserService.java         # User auth + registration
│   │   ├── SentimentService.java    # Calls AI /predict endpoint
│   │   ├── SummaryService.java      # Calls AI /summarize endpoint
│   │   ├── WordCloudService.java    # Calls AI /wordcloud endpoint
│   │   └── ReportService.java       # PDF generation with iText 7
│   ├── model/
│   │   ├── User.java                # JPA entity → users table
│   │   └── AnalysisHistory.java     # JPA entity → analysis_history
│   └── repository/
│       ├── UserRepository.java      # Spring Data JPA
│       └── HistoryRepository.java   # Spring Data JPA
│
├── src/main/resources/
│   ├── application.properties       # App config (DB, AI URL, OAuth2)
│   ├── templates/                   # Thymeleaf HTML templates
│   │   ├── login.html               # Futuristic login page
│   │   ├── signup.html              # Registration page
│   │   ├── index.html               # Upload page (public)
│   │   ├── dashboard.html           # Analysis dashboard
│   │   └── history.html             # Past analyses
│   └── static/                      # Generated files (PDF, CSV)
│
├── python-ai-service/
│   ├── ai_server.py                 # Flask app — all AI endpoints
│   ├── config.py                    # Centralized hyperparameters
│   ├── preprocessing.py             # Text cleaning pipeline
│   ├── evaluation.py                # Model eval (accuracy, F1, CM)
│   ├── explainability.py            # Attention-based XAI
│   ├── requirements.txt             # Python dependencies
│   ├── Dockerfile                   # AI service container
│   └── policy_sentiment_final/      # Fine-tuned model (optional)
│
└── research_paper/
    ├── IEEE_Research_Paper.docx      # IEEE conference paper (Word)
    └── IEEE_Research_Paper.html      # IEEE paper (browser/PDF)
```

---

## 🔌 API Endpoints

### Spring Boot (Port 8080)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | Public | Upload page |
| GET | `/login` | Public | Login page |
| GET | `/signup` | Public | Registration page |
| POST | `/register` | Public | Create account |
| POST | `/upload` | Required | Upload CSV + run analysis |
| GET | `/dashboard` | Public | Analysis dashboard |
| GET | `/history` | Public | Past analyses |
| GET | `/download/csv` | Public | Download last_analysis.csv |
| GET | `/download/pdf` | Public | Download report.pdf |

### Python AI Service (Port 5001)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Sentiment classification (standardized response) |
| POST | `/predict_legacy` | Sentiment classification (flat response for Spring Boot) |
| POST | `/summarize` | Chunk-based abstractive summarization |
| POST | `/wordcloud` | Word cloud generation with domain filtering |
| POST | `/explain` | Attention-based explainability |
| POST | `/evaluate` | Model evaluation pipeline |
| GET | `/health` | System health + model status + GPU info |

---

## 📊 Dashboard Features

- **KPI Cards** — Total comments, positive/neutral/negative %, uncertain count
- **Pie Chart** — Sentiment distribution with percentage labels
- **Bar Chart** — Confidence distribution with count labels and axis titles
- **Word Cloud** — Domain-filtered keyword visualization
- **Summary Cards** — AI-generated summaries per sentiment group
- **Topics Tab** — Auto-categorized (Penalties, Audit, Reporting, Compliance, Other)
- **Needs Review Tab** — Uncertain predictions (confidence < 60%) for human review
- **All Comments Tab** — Searchable/filterable table with confidence bars
- **Downloads** — CSV export + PDF report with embedded word cloud

---

## 🔐 Authentication

- **Local accounts** — Username/email/password with BCrypt (strength 12)
- **Google OAuth2** — One-click Google sign-in (shows real name, not numeric ID)
- **Public access** — Home, dashboard, history, downloads viewable without login
- **Protected** — CSV upload requires authentication (redirects to login)

---

## 📄 CSV Format

Your CSV must have a column named `comment_text`:

```csv
comment_text
"The new policy is excellent and well thought out"
"Penalties are too harsh for small companies"
"No significant impact observed"
```

UTF-8 BOM characters are handled automatically.

---

## 🧪 AI Service Enhancements (v2.0)

| Feature | Description |
|---------|-------------|
| **Preprocessing** | URL/email/emoji removal, NLTK stopwords, 80+ domain terms |
| **Chunk Summarization** | Splits at sentence boundaries, re-summarizes combined chunks |
| **Explainability** | Attention-based token importance + confidence interpretation |
| **Evaluation** | Accuracy, precision, recall, F1, confusion matrix (JSON + PNG) |
| **GPU Acceleration** | Auto-detects CUDA / MPS / CPU, moves models to optimal device |
| **Logging** | Rotating file logs (5MB, 3 backups) + structured console output |
| **Validation** | Input validation, max 5000 texts/request, standardized responses |

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| 1,000 comments (full pipeline) | < 30 seconds |
| Sentiment throughput | ~340 samples/sec (GPU) |
| Model load time | ~8-10 seconds |
| Word cloud generation | ~200ms |
| Summarization (50 comments) | ~3 seconds |
| Max file upload size | 10MB |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8080 in use | `lsof -ti:8080 \| xargs kill -9` |
| Port 5001 in use | `lsof -ti:5001 \| xargs kill -9` |
| "comment_text column not found" | CSV must have `comment_text` header row |
| Word cloud not showing | Check `spring.web.resources.static-locations` path in properties |
| Java not found | `export JAVA_HOME=/opt/homebrew/opt/openjdk@17` |
| PostgreSQL refused | `brew services start postgresql@14` then `pg_isready` |
| AI models not loading | Check internet connection; first run downloads ~2GB from HuggingFace |
| Google OAuth error | Verify redirect URI is exactly `http://localhost:8080/login/oauth2/code/google` |

---

## 🗄 Database Queries

```bash
# Connect
psql -U postgres -d mca_db

# View all users
SELECT id, username, email, created_at FROM users;

# View analysis history
SELECT username, filename, timestamp,
       round(positive_pct::numeric,1) as pos,
       round(neutral_pct::numeric,1) as neu,
       round(negative_pct::numeric,1) as neg,
       uncertain_count
FROM analysis_history ORDER BY timestamp DESC;
```

---

## 👥 Team

| Role | Name |
|------|------|
| Developer | [Your Name] |
| Institution | [Your Institution] |
| Problem Statement | Ministry of Corporate Affairs |
| Problem ID | 25035 |
