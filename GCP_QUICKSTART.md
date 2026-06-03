# Quick Start - Deploy to GCP in 5 Minutes

## Prerequisites Checklist

- [ ] GCP account created
- [ ] gcloud CLI installed and authenticated
- [ ] GCP project created
- [ ] OpenAI API key ready
- [ ] Local development working (`python main.py build` completed)

## 5-Minute Deployment

### Step 1: Set Your Project (30 seconds)

```bash
# Set your GCP project ID
gcloud config set project YOUR_PROJECT_ID

# Verify
gcloud config list project
```

### Step 2: Run Deployment Script (4 minutes)

**For Windows (PowerShell):**
```powershell
.\deploy.ps1
```

**For Mac/Linux:**
```bash
chmod +x deploy.sh
./deploy.sh
```

When prompted, enter:
- Your GCP Project ID
- Service name (press Enter for default: `librarian`)
- Region (press Enter for default: `us-central1`)

### Step 3: Update Frontend URL (30 seconds)

After deployment completes, you'll see:
```
Service URL: https://librarian-xxxxx.run.app
```

1. Open `search_ui.html`
2. Find line ~10: `const API_BASE_URL = 'http://localhost:8000';`
3. Replace with: `const API_BASE_URL = 'https://librarian-xxxxx.run.app';`
4. Save the file

Re-upload:
```bash
PROJECT_ID=$(gcloud config get-value project)
gsutil cp search_ui.html gs://${PROJECT_ID}-librarian-frontend/index.html
```

### Step 4: Test Your Deployment

Visit your frontend URL:
```
https://storage.googleapis.com/YOUR_PROJECT_ID-librarian-frontend/index.html
```

Try a search query like "girder" or "retaining wall"!

## What Was Deployed?

✅ **Cloud Run Service** - Your API running at `https://librarian-xxxxx.run.app`
✅ **3 Storage Buckets:**
  - PDFs: `gs://YOUR_PROJECT_ID-librarian-pdfs`
  - Vectors: `gs://YOUR_PROJECT_ID-librarian-vectors`  
  - Frontend: `gs://YOUR_PROJECT_ID-librarian-frontend`
✅ **Secret Manager** - OpenAI API key stored securely
✅ **Auto-scaling** - Scales to 0 when not in use (FREE!)

## Monthly Cost

**~$0.50-2/month** for 100 searches

Everything else is covered by GCP's generous free tier!

## Next Steps

📖 Read full deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)  
🔒 Add authentication: See DEPLOYMENT.md → Security section  
📊 Set up monitoring: Cloud Console → Cloud Run → Metrics  

---

**That's it! Your Librarian is now live! 🎉**
