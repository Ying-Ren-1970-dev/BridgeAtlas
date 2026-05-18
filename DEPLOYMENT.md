# Librarian Deployment Guide - Google Cloud Platform

This guide explains how to deploy the Librarian search application to Google Cloud Platform using Cloud Run and Cloud Storage.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Users                                              │
│    │                                                │
│    ├─► Cloud Storage (Frontend)                    │
│    │   └─► search_ui.html                          │
│    │                                                │
│    └─► Cloud Run (API)                             │
│        └─► FastAPI Application                     │
│            ├─► Cloud Storage (PDFs)                │
│            ├─► Cloud Storage (Vector DB)           │
│            └─► Firestore (Metadata)                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Google Cloud Platform Account**
   - Create a GCP account at https://cloud.google.com/
   - Create a new project or select an existing one

2. **Google Cloud SDK (gcloud CLI)**
   - Install from: https://cloud.google.com/sdk/docs/install
   - Authenticate: `gcloud auth login`
   - Set project: `gcloud config set project YOUR_PROJECT_ID`

3. **OpenAI API Key**
   - Obtain from: https://platform.openai.com/api-keys
   - Required for semantic search functionality

4. **Local Development Environment**
   - Python 3.11+
   - All project dependencies installed
   - Vector database built (`python main.py build`)

## Deployment Steps

### Option 1: Automated Deployment (Recommended)

#### For Unix/Linux/Mac:

```bash
# Make the script executable
chmod +x deploy.sh

# Run the deployment script
./deploy.sh
```

#### For Windows:

```powershell
# Run the PowerShell deployment script
.\deploy.ps1
```

The automated script will:
1. Enable required GCP APIs
2. Create Cloud Storage buckets
3. Upload PDFs, vector database, and frontend
4. Create OpenAI API key secret
5. Build and deploy to Cloud Run
6. Output the service URL

### Option 2: Manual Deployment

#### 1. Enable Required APIs

```bash
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    firestore.googleapis.com \
    secretmanager.googleapis.com
```

#### 2. Create Cloud Storage Buckets

```bash
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"

# Create buckets
gsutil mb -p $PROJECT_ID -l $REGION gs://${PROJECT_ID}-librarian-pdfs
gsutil mb -p $PROJECT_ID -l $REGION gs://${PROJECT_ID}-librarian-vectors
gsutil mb -p $PROJECT_ID -l $REGION gs://${PROJECT_ID}-librarian-frontend
```

#### 3. Upload Data to Cloud Storage

```bash
# Upload PDFs
gsutil -m rsync -r Projects/ gs://${PROJECT_ID}-librarian-pdfs/

# Upload vector database
gsutil -m rsync -r vector_db/ gs://${PROJECT_ID}-librarian-vectors/

# Upload frontend
gsutil cp search_ui.html gs://${PROJECT_ID}-librarian-frontend/index.html

# Make frontend publicly accessible
gsutil iam ch allUsers:objectViewer gs://${PROJECT_ID}-librarian-frontend
gsutil web set -m index.html gs://${PROJECT_ID}-librarian-frontend
```

#### 4. Create OpenAI API Key Secret

```bash
# Store your OpenAI API key in Secret Manager
echo -n "your-openai-api-key-here" | \
    gcloud secrets create openai-api-key \
    --data-file=- \
    --replication-policy="automatic"
```

#### 5. Deploy to Cloud Run

```bash
gcloud run deploy librarian \
    --source . \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 300 \
    --set-env-vars ENVIRONMENT=production,GCS_BUCKET_PDFS=${PROJECT_ID}-librarian-pdfs,GCS_BUCKET_VECTORS=${PROJECT_ID}-librarian-vectors \
    --set-secrets OPENAI_API_KEY=openai-api-key:latest
```

#### 6. Update Frontend API URL

After deployment, get your Cloud Run service URL:

```bash
gcloud run services describe librarian \
    --platform managed \
    --region us-central1 \
    --format 'value(status.url)'
```

Update the API URL in `search_ui.html`:
- Find line with `const API_BASE_URL = 'http://localhost:8000';`
- Replace with your Cloud Run URL: `const API_BASE_URL = 'https://your-service-url';`

Re-upload the frontend:

```bash
gsutil cp search_ui.html gs://${PROJECT_ID}-librarian-frontend/index.html
```

## Cost Estimation

For **100 searches/month**:

| Service | Usage | Cost/Month |
|---------|-------|------------|
| Cloud Run | ~200 CPU-seconds | **$0** (free tier) |
| Cloud Storage | ~2 GB storage + minimal egress | **$0.50-1.50** |
| Firestore | Minimal reads/writes | **$0** (free tier) |
| Secret Manager | 1 secret, minimal access | **$0** (free tier) |
| **Total** | | **~$0.50-2/month** |

## Environment Variables

The application uses these environment variables in production:

| Variable | Description | Example |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `production` |
| `GCS_BUCKET_PDFS` | Bucket name for PDF files | `myproject-librarian-pdfs` |
| `GCS_BUCKET_VECTORS` | Bucket name for vector database | `myproject-librarian-vectors` |
| `GCP_PROJECT_ID` | GCP project ID | `my-gcp-project` |
| `OPENAI_API_KEY` | OpenAI API key (from Secret Manager) | `sk-...` |
| `USE_CLOUD_STORAGE` | Enable Cloud Storage | `true` |
| `USE_FIRESTORE` | Enable Firestore | `true` |

## Post-Deployment

### View Logs

```bash
gcloud run services logs read librarian --region=us-central1
```

### Test the Deployment

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe librarian --platform managed --region us-central1 --format 'value(status.url)')

# Test health endpoint
curl $SERVICE_URL/health

# Test search
curl -X POST $SERVICE_URL/search \
  -H "Content-Type: application/json" \
  -d '{"query": "girder", "k": 10}'
```

### Update the Deployment

After making code changes:

```bash
# Redeploy
./deploy.sh

# Or manually
gcloud run deploy librarian --source . --region us-central1
```

### Monitor Costs

```bash
# View billing
gcloud billing accounts list
gcloud billing projects describe YOUR_PROJECT_ID
```

## Security Considerations

### For Public Demo:

✅ **Current setup is fine for demo:**
- `--allow-unauthenticated` enables public access
- No sensitive data in search results
- Rate limiting handled by Cloud Run

### For Production with Authentication:

1. **Remove public access:**
```bash
gcloud run services remove-iam-policy-binding librarian \
    --member="allUsers" \
    --role="roles/run.invoker" \
    --region=us-central1
```

2. **Add Identity-Aware Proxy (IAP):**
```bash
gcloud iap web enable --resource-type=backend-services
```

3. **Add Cloud Armor for DDoS protection:**
```bash
gcloud compute security-policies create librarian-policy
gcloud compute security-policies rules create 1000 \
    --security-policy librarian-policy \
    --expression "origin.region_code == 'US'" \
    --action "allow"
```

## Troubleshooting

### Deployment Fails

```bash
# Check build logs
gcloud builds list --limit=5

# View specific build
gcloud builds describe BUILD_ID
```

### Service Not Responding

```bash
# Check service status
gcloud run services describe librarian --region=us-central1

# View recent logs
gcloud run services logs tail librarian --region=us-central1
```

### Cold Start Issues

To reduce cold starts, set a minimum instance:

```bash
gcloud run services update librarian \
    --min-instances 1 \
    --region us-central1
```

Note: This keeps one instance always running (~$8-15/month).

### Storage Access Issues

```bash
# Check bucket permissions
gsutil iam get gs://your-bucket-name

# Grant service account access
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
    --role="roles/storage.objectViewer"
```

## Cleanup

To delete all resources and avoid charges:

```bash
# Delete Cloud Run service
gcloud run services delete librarian --region=us-central1

# Delete storage buckets
gsutil -m rm -r gs://${PROJECT_ID}-librarian-pdfs
gsutil -m rm -r gs://${PROJECT_ID}-librarian-vectors
gsutil -m rm -r gs://${PROJECT_ID}-librarian-frontend

# Delete secret
gcloud secrets delete openai-api-key

# Delete Firestore data
gcloud firestore databases delete --database='(default)'
```

## Support

For issues or questions:
1. Check logs: `gcloud run services logs read librarian --region=us-central1`
2. Review GCP Status: https://status.cloud.google.com/
3. Check OpenAI Status: https://status.openai.com/

## Next Steps

- Set up monitoring with Cloud Monitoring
- Configure custom domain with Cloud Run
- Add caching with Cloud CDN
- Implement user authentication
- Set up CI/CD with Cloud Build triggers
