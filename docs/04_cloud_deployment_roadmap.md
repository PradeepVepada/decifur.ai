# HybdRAG Cloud Deployment Roadmap
## From Local Prototype to Production Infrastructure

> **Current State:** Local deployment (Neo4j Desktop + Streamlit)  
> **Target State:** Fully managed cloud infrastructure  
> **Timeline:** 4-6 weeks (recommended)

---

## Executive Summary

This roadmap outlines the migration of HybdRAG from a local development environment to a production-ready cloud deployment. The strategy prioritizes cost efficiency during development while providing a clear path to scalable infrastructure.

**Key Decisions:**
- **Database:** Neo4j Desktop → Neo4j AuraDB
- **Compute:** Streamlit Cloud (free tier) → AWS/GCP (scaling)
- **Storage:** Local filesystem → Cloud object storage
- **CI/CD:** Manual → GitHub Actions

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Current Local Setup                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   User       │────►│  Streamlit   │────►│  RAG Engine  │   │
│   │  (Browser)   │     │  :8506       │     │              │   │
│   └──────────────┘     └──────────────┘     └──────┬───────┘   │
│                                                     │           │
│                               ┌─────────────────────┼─────────┐│
│                               │                     ▼         ││
│                               │  ┌──────────────────────────┐ ││
│                               │  │     Neo4j Desktop        │ ││
│                               │  │     127.0.0.1:7687       │ ││
│                               │  │     - 226 papers          │ ││
│                               │  │     - 7,597 chunks        │ ││
│                               │  │     - ~84k entities       │ ││
│                               │  └──────────────────────────┘ ││
│                               │                                ││
│   ┌──────────────┐           │  ┌──────────────────────────┐ ││
│   │   Mistral    │◄──────────┼──│  API Keys (.env)         │ ││
│   │   API        │           │  └──────────────────────────┘ ││
│   └──────────────┘           │                                ││
│                               └────────────────────────────────┘│
│                                                                 │
│   Limitations:                                                  │
│   • No remote access                                            │
│   • Single user                                                 │
│   • Manual restart required                                     │
│   • No auto-scaling                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        Target Cloud Architecture                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────┐     ┌──────────────────────────────────────────────────────┐│
│   │   Users      │────►│              Streamlit Cloud / AWS                    ││
│   │  (Global)    │     │              (Auto-scaling)                           ││
│   └──────────────┘     │                                                      ││
│                        │  ┌─────────────────────────────────────────────────┐ ││
│                        │  │  Streamlit App Container                        │ ││
│                        │  │  - RAG Engine                                   │ ││
│                        │  │  - PCC Memory Manager                           │ ││
│                        │  │  - Embedding Service (shared)                   │ ││
│                        │  └─────────────────────────────────────────────────┘ ││
│                        └──────────────────────────────────────────────────────┘│
│                                         │                                       │
│                                         ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐ │
│   │                        Neo4j AuraDB (Managed)                           │ │
│   │                                                                         │ │
│   │   • Vector Index (chunk_embeddings)                                     │ │
│   │   • Knowledge Graph (226 papers, 7,597 chunks)                         │ │
│   │   • Memory Episodes (PCC Long-term)                                     │ │
│   │   • Auto-backup & HA                                                    │ │
│   └─────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                       │
│                                         ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐ │
│   │                        External Services                                │ │
│   │                                                                         │ │
│   │   ┌────────────────┐  ┌────────────────┐  ┌─────────────────────────┐ │ │
│   │   │  Mistral API   │  │  OpenAI API    │  │  Cloud Storage          │ │ │
│   │   │  (Generation)  │  │  (Evaluation)  │  │  (PDFs, Exports)        │ │ │
│   │   └────────────────┘  └────────────────┘  └─────────────────────────┘ │ │
│   └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Database Migration (Week 1)

### 1.1 Neo4j AuraDB Setup

**Option A: Free Tier (Development/Testing)**
```
Neo4j AuraDB Free
├── 1 instance
├── 1 GB storage
├── No SLA
├── Good for: Testing, small datasets
└── Cost: $0/month
```

**Option B: Professional Tier (Production)**
```
Neo4j AuraDB Professional
├── Memory: 4-16 GB
├── Storage: Up to 100 GB
├── 99.9% SLA
├── Good for: Production with 200+ papers
└── Cost: $65-390/month
```

**Recommended for Devreotes Corpus:** AuraDB Professional (Memory: 4GB)

### 1.2 Data Export/Import Steps

```powershell
# Step 1: Export from local Neo4j
# In Neo4j Desktop, use: CALL apoc.export.cypher.all('hybdrag_backup', {})
# Or use neo4j-admin dump

.\.venv_gpu\Scripts\python.exe -c "
from neo4j import GraphDatabase
# Export logic here
"

# Step 2: Create AuraDB instance
# Go to https://console.neo4j.io
# Create new instance with credentials

# Step 3: Update .env
NEO4J_URI=neo4s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<generated-password>
```

### 1.3 Migration Checklist

- [ ] Create Neo4j AuraDB account
- [ ] Create new instance (Professional tier)
- [ ] Export local data (APOC or neo4j-admin)
- [ ] Import data to AuraDB
- [ ] Verify vector indexes created
- [ ] Test connection from local environment
- [ ] Update `.env` with new credentials
- [ ] Run full test suite

---

## Phase 2: Embedding Service Optimization (Week 2)

### 2.1 Challenge: GPU Embeddings in Cloud

The current setup uses GPU-accelerated SentenceTransformer embeddings (~4GB VRAM). Cloud containers typically don't have GPU access by default.

**Solutions:**

| Option | Pros | Cons | Cost |
|--------|------|------|------|
| **CPU Embeddings** | Simple, no extra cost | Slower (~3s vs ~0.5s) | $0 |
| **GPU Container (AWS)** | Fast inference | Complex setup | $0.50-2.00/hr |
| **Embedding API** | No local compute | API costs, latency | ~$0.0001/query |
| **Separate Service** | Shared GPU pool | More infrastructure | Variable |

**Recommendation:** Start with CPU embeddings for simplicity, upgrade to GPU service if latency is critical.

### 2.2 CPU Optimization

```python
# rag_engine.py modifications for CPU deployment
import os

# Detect deployment environment
IS_CLOUD = os.environ.get('DEPLOYMENT_ENV') == 'cloud'

if IS_CLOUD:
    # Use smaller model for CPU
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # 384-dim, faster
    EMBEDDING_DIM = 384
else:
    # Use full model for local GPU
    EMBEDDING_MODEL = 'all-mpnet-base-v2'  # 768-dim
    EMBEDDING_DIM = 768
```

---

## Phase 3: Application Deployment (Week 3)

### 3.1 Option A: Streamlit Cloud (Simplest)

**Prerequisites:**
- GitHub repository
- Streamlit Community Cloud account (free)

**Steps:**
1. Push code to GitHub
2. Connect Streamlit Cloud to repo
3. Set secrets in Streamlit Cloud UI:
   ```toml
   [secrets]
   NEO4J_URI = "neo4s://xxxxxxxx.databases.neo4j.io"
   NEO4J_USER = "neo4j"
   NEO4J_PASSWORD = "your-password"
   MISTRAL_API_KEY = "your-key"
   ```
4. Deploy

**Pros:** Free, automatic deployments, no DevOps  
**Cons:** Limited customization, no GPU, 1GB memory limit

### 3.2 Option B: Docker + Cloud Run (Scalable)

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only torch)
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK CMD curl -f http://localhost:8506/_stcore/health || exit 1

EXPOSE 8506

CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.port=8506", \
     "--server.address=0.0.0.0"]
```

**requirements-cloud.txt:**
```
# Core (same as requirements.txt but CPU torch)
mistralai
neo4j
python-dotenv

# Embeddings (CPU)
sentence-transformers

# Biomedical NER
scispacy
https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# UI
streamlit

# CPU-only torch (much smaller)
torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Deploy to Google Cloud Run:**
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/hybdrag

# Deploy
gcloud run deploy hybdrag \
  --image gcr.io/PROJECT_ID/hybdrag \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --port 8506 \
  --set-env-vars "NEO4J_URI=neo4s://...,NEO4J_USER=neo4j,NEO4J_PASSWORD=..."
```

### 3.3 Option C: AWS ECS/Fargate (Full Control)

```yaml
# task-definition.json
{
  "family": "hybdrag",
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [{
    "name": "streamlit",
    "image": "your-ecr-repo/hybdrag:latest",
    "portMappings": [{
      "containerPort": 8506,
      "hostPort": 8506
    }],
    "environment": [
      {"name": "NEO4J_URI", "value": "neo4s://..."},
      {"name": "NEO4J_USER", "value": "neo4j"},
      {"name": "NEO4J_PASSWORD", "value": "..."},
      {"name": "MISTRAL_API_KEY", "value": "..."}
    ]
  }]
}
```

---

## Phase 4: CI/CD Pipeline (Week 4)

### 4.1 GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy HybdRAG

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run tests
        env:
          NEO4J_URI: ${{ secrets.NEO4J_URI }}
          NEO4J_USER: ${{ secrets.NEO4J_USER }}
          NEO4J_PASSWORD: ${{ secrets.NEO4J_PASSWORD }}
          MISTRAL_API_KEY: ${{ secrets.MISTRAL_API_KEY }}
        run: python -m pytest tests/
  
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Cloud Run
        uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: hybdrag
          image: gcr.io/${{ secrets.GCP_PROJECT }}/hybdrag
          region: us-central1
```

---

## Phase 5: Monitoring & Observability (Week 5-6)

### 5.1 Health Checks

```python
# health.py
from fastapi import APIRouter
from neo4j import GraphDatabase

router = APIRouter()

@router.get("/health")
async def health_check():
    checks = {
        "status": "healthy",
        "neo4j": check_neo4j(),
        "mistral_api": check_mistral(),
        "embedding_service": check_embeddings()
    }
    return checks

def check_neo4j():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as s:
            s.run("RETURN 1")
        return "ok"
    except Exception as e:
        return f"error: {str(e)}"
```

### 5.2 Metrics to Track

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Query latency (P50) | <2s | >5s |
| Query latency (P95) | <5s | >10s |
| Error rate | <1% | >5% |
| Neo4j connection pool | <80% | >90% |
| API quota usage | <80% | >90% |

### 5.3 Logging Strategy

```python
# Structured logging
import structlog

logger = structlog.get_logger()

async def process_query(query: str):
    log = logger.bind(query=query, user_id=user_id)
    
    log.info("query_received")
    
    try:
        result = await engine.ask(query)
        log.info("query_completed", 
                 latency=result.latency,
                 chunks_retrieved=len(result.chunks))
    except Exception as e:
        log.error("query_failed", error=str(e))
        raise
```

---

## Cost Estimation

### Monthly Cost Breakdown (Production)

| Component | Service | Estimated Cost |
|-----------|---------|----------------|
| **Database** | Neo4j AuraDB Professional (4GB) | $65/month |
| **Compute** | Google Cloud Run (moderate traffic) | $20-50/month |
| **Storage** | Cloud Storage (PDFs, exports) | $5/month |
| **Mistral API** | ~10k queries/month | $10-30/month |
| **OpenAI API** | Evaluation only | $5-10/month |
| **Monitoring** | Basic Cloud Monitoring | $0 (free tier) |
| **CI/CD** | GitHub Actions | $0 (free tier) |
| **Total** | | **$105-160/month** |

### Cost Optimization Tips

1. **Use Neo4j AuraDB Free** for development ($0)
2. **Set Cloud Run minimum instances to 0** (scale to zero)
3. **Use Mistral-small** for most queries (cheaper)
4. **Cache embeddings** to reduce API calls
5. **Set API rate limits** to prevent abuse

---

## Security Checklist

- [ ] Enable HTTPS only (no HTTP)
- [ ] Set up CORS policies
- [ ] Rotate API keys regularly
- [ ] Enable Neo4j authentication
- [ ] Use environment variables for secrets (never hardcode)
- [ ] Set up IP allowlisting for Neo4j (if possible)
- [ ] Enable audit logging
- [ ] Set up DDoS protection (Cloudflare/Cloud Armor)

---

## Rollback Plan

If deployment fails:

1. **Immediate:** Revert to previous container image
2. **Database:** Restore from Neo4j AuraDB backup
3. **DNS:** Point domain back to previous deployment
4. **Communicate:** Notify users via status page

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Database Migration | AuraDB instance with migrated data |
| 2 | Embedding Optimization | CPU-compatible embedding service |
| 3 | Application Deployment | Streamlit running in cloud |
| 4 | CI/CD Pipeline | Automated testing and deployment |
| 5-6 | Monitoring | Health checks, alerts, logging |

---

## Next Steps

1. **Decision:** Choose deployment platform (Streamlit Cloud vs GCP vs AWS)
2. **Budget:** Confirm monthly cost budget with stakeholders
3. **Timeline:** Set go-live date
4. **Team:** Assign DevOps responsibilities

---

## Resources

- [Neo4j AuraDB Documentation](https://neo4j.com/docs/aura/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

---

*This roadmap should be reviewed and updated as deployment progresses.*
