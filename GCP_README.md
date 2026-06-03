# GCP Deployment Files - README

This directory contains all files needed to deploy the Librarian application to Google Cloud Platform.

## Deployment Files Created

### Core Deployment Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Container definition for Cloud Run deployment |
| `.dockerignore` | Excludes unnecessary files from Docker image |
| `.gcloudignore` | Excludes files from gcloud deployment |
| `cloudbuild.yaml` | Automated build configuration for Cloud Build |

### Deployment Scripts

| Script | Platform | Purpose |
|--------|----------|---------|
| `deploy.sh` | Unix/Linux/Mac | Automated deployment script (Bash) |
| `deploy.ps1` | Windows | Automated deployment script (PowerShell) |

### Cloud Adapters

| File | Purpose |
|------|---------|
| `storage_adapter.py` | Handles PDF and vector DB storage in Cloud Storage |
| `firestore_adapter.py` | Handles metadata storage in Firestore |
| `config.py` (updated) | Environment-aware configuration |

### Documentation

| File | Purpose |
|------|---------|
| `DEPLOYMENT.md` | Complete deployment guide with troubleshooting |
| `GCP_QUICKSTART.md` | 5-minute quick start guide |

## Quick Start

### Prerequisites
- Google Cloud account
- gcloud CLI installed
- OpenAI API key
- Local vector database built

### Deploy in 5 Minutes

```bash
# For Unix/Linux/Mac
./deploy.sh

# For Windows
.\deploy.ps1
```

Follow the prompts and your application will be live!

## Architecture

```
Production (GCP):
├── Cloud Run (API)
│   ├── FastAPI application
│   ├── ChromaDB (loaded from Cloud Storage)
│   └── Search functionality
├── Cloud Storage
│   ├── Bucket: PDFs
│   ├── Bucket: Vector Database
│   └── Bucket: Frontend (HTML)
├── Firestore
│   └── Projects metadata
└── Secret Manager
    └── OpenAI API key
```

## Configuration

### Environment Variables (Automatic)

When you deploy, these are set automatically:

| Variable | Value | Set By |
|----------|-------|--------|
| `ENVIRONMENT` | `production` | deploy script |
| `GCS_BUCKET_PDFS` | `project-id-librarian-pdfs` | deploy script |
| `GCS_BUCKET_VECTORS` | `project-id-librarian-vectors` | deploy script |
| `USE_CLOUD_STORAGE` | `true` | config.py |
| `USE_FIRESTORE` | `true` | config.py |
| `OPENAI_API_KEY` | (from Secret Manager) | deploy script |

### How It Works

The application automatically detects the environment:

- **Development** (`ENVIRONMENT=development`):
  - Uses local `Projects/` folder for PDFs
  - Uses local `vector_db/` folder
  - Uses local `metadata_db.json` file

- **Production** (`ENVIRONMENT=production`):
  - Uses Cloud Storage for PDFs
  - Uses Cloud Storage for vector database
  - Uses Firestore for metadata
  - API keys from Secret Manager

## Cost Breakdown

For **100 searches/month**:

| Service | Monthly Cost |
|---------|--------------|
| Cloud Run | $0 (free tier) |
| Cloud Storage (3 buckets, ~2GB) | $0.50-1.50 |
| Firestore | $0 (free tier) |
| Secret Manager | $0 (free tier) |
| **Total** | **~$0.50-2/month** |

## Files Modified for Cloud Support

### New Files
- ✅ `storage_adapter.py` - Cloud Storage integration
- ✅ `firestore_adapter.py` - Firestore integration
- ✅ `Dockerfile` - Container definition
- ✅ `deploy.sh` / `deploy.ps1` - Deployment automation

### Updated Files
- ✅ `config.py` - Added cloud environment variables
- ✅ `requirements.txt` - Added GCP dependencies

### Files to Update Manually (After First Deploy)
- ⚠️ `search_ui.html` - Update API URL to Cloud Run URL

## Next Steps

1. **Read the docs:**
   - Start with: [GCP_QUICKSTART.md](GCP_QUICKSTART.md)
   - Full guide: [DEPLOYMENT.md](DEPLOYMENT.md)

2. **Deploy:**
   ```bash
   ./deploy.sh  # or .\deploy.ps1 on Windows
   ```

3. **Test your deployment:**
   ```bash
   # Get your service URL
   gcloud run services describe librarian --region=us-central1 --format='value(status.url)'
   
   # Test it
   curl https://your-service-url/health
   ```

4. **Update frontend:**
   - Edit `search_ui.html` with your service URL
   - Re-upload to Cloud Storage

5. **Share your demo!**
   - Frontend URL: `https://storage.googleapis.com/YOUR_PROJECT-librarian-frontend/index.html`

## Troubleshooting

### Common Issues

**"Permission denied" during deployment:**
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

**"Bucket already exists":**
- This is normal on subsequent deployments
- The script will skip bucket creation and use existing ones

**Search returns 0 results:**
1. Check vector database was uploaded
2. Check logs: `gcloud run services logs read librarian --region=us-central1`
3. Verify Secret Manager has OpenAI key

**Frontend can't reach API:**
- Verify `search_ui.html` has correct API URL
- Check CORS settings in `api.py` (should allow all origins for public demo)

### Getting Help

View logs in real-time:
```bash
gcloud run services logs tail librarian --region=us-central1
```

Check service status:
```bash
gcloud run services describe librarian --region=us-central1
```

## Cleanup

To remove all deployed resources:

```bash
# Delete Cloud Run service
gcloud run services delete librarian --region=us-central1

# Delete storage buckets
PROJECT_ID=$(gcloud config get-value project)
gsutil -m rm -r gs://${PROJECT_ID}-librarian-pdfs
gsutil -m rm -r gs://${PROJECT_ID}-librarian-vectors
gsutil -m rm -r gs://${PROJECT_ID}-librarian-frontend

# Delete secret
gcloud secrets delete openai-api-key
```

## Support

For deployment issues:
1. Check [DEPLOYMENT.md](DEPLOYMENT.md) troubleshooting section
2. View service logs
3. Verify all prerequisites are met

---

**Ready to deploy? Run `./deploy.sh` and watch the magic happen! ✨**
