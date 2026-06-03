#!/bin/bash
# Deployment script for Librarian to Google Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Librarian GCP Deployment Script      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found. Please install it first.${NC}"
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get configuration from user or use defaults
echo -e "${YELLOW}Enter your GCP Project ID:${NC}"
read -p "> " PROJECT_ID

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: Project ID is required${NC}"
    exit 1
fi

echo -e "${YELLOW}Enter service name (default: librarian):${NC}"
read -p "> " SERVICE_NAME
SERVICE_NAME=${SERVICE_NAME:-librarian}

echo -e "${YELLOW}Enter region (default: us-central1):${NC}"
read -p "> " REGION
REGION=${REGION:-us-central1}

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Project ID: $PROJECT_ID"
echo "  Service: $SERVICE_NAME"
echo "  Region: $REGION"
echo ""

# Set the project
echo -e "${YELLOW}Setting GCP project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}Enabling required APIs...${NC}"
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com \
    firestore.googleapis.com \
    secretmanager.googleapis.com

# Create storage buckets
echo -e "${YELLOW}Creating Cloud Storage buckets...${NC}"
BUCKET_PDFS="${PROJECT_ID}-librarian-pdfs"
BUCKET_VECTOR="${PROJECT_ID}-librarian-vectors"
BUCKET_FRONTEND="${PROJECT_ID}-librarian-frontend"

gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_PDFS/ 2>/dev/null || echo "  Bucket $BUCKET_PDFS already exists"
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_VECTOR/ 2>/dev/null || echo "  Bucket $BUCKET_VECTOR already exists"
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_FRONTEND/ 2>/dev/null || echo "  Bucket $BUCKET_FRONTEND already exists"

# Upload PDFs
echo -e "${YELLOW}Uploading PDFs to Cloud Storage...${NC}"
if [ -d "Projects" ] && [ "$(ls -A Projects/*.pdf 2>/dev/null)" ]; then
    gsutil -m rsync -r Projects/ gs://$BUCKET_PDFS/
    echo -e "${GREEN}  ✓ PDFs uploaded${NC}"
else
    echo -e "${YELLOW}  No PDFs found in Projects folder${NC}"
fi

# Upload vector database
echo -e "${YELLOW}Uploading vector database...${NC}"
if [ -d "vector_db" ] && [ "$(ls -A vector_db 2>/dev/null)" ]; then
    gsutil -m rsync -r vector_db/ gs://$BUCKET_VECTOR/
    echo -e "${GREEN}  ✓ Vector database uploaded${NC}"
else
    echo -e "${YELLOW}  No vector database found${NC}"
fi

# Upload frontend
echo -e "${YELLOW}Uploading frontend...${NC}"
if [ -f "search_ui.html" ]; then
    gsutil cp search_ui.html gs://$BUCKET_FRONTEND/index.html
    gsutil iam ch allUsers:objectViewer gs://$BUCKET_FRONTEND
    gsutil web set -m index.html gs://$BUCKET_FRONTEND
    echo -e "${GREEN}  ✓ Frontend uploaded${NC}"
    echo -e "  Frontend URL: https://storage.googleapis.com/$BUCKET_FRONTEND/index.html"
fi

# Create or update OpenAI API key secret
echo -e "${YELLOW}Setting up OpenAI API key secret...${NC}"
if [ -f "config.py" ]; then
    OPENAI_KEY=$(python3 -c "import config; print(config.OPENAI_API_KEY)" 2>/dev/null || echo "")
    if [ ! -z "$OPENAI_KEY" ]; then
        echo -n "$OPENAI_KEY" | gcloud secrets create openai-api-key \
            --data-file=- \
            --replication-policy="automatic" 2>/dev/null || \
        echo -n "$OPENAI_KEY" | gcloud secrets versions add openai-api-key --data-file=-
        echo -e "${GREEN}  ✓ OpenAI API key secret created${NC}"
    else
        echo -e "${YELLOW}  Warning: Could not read OpenAI API key from config.py${NC}"
        echo -e "${YELLOW}  Please create it manually or set it after deployment${NC}"
    fi
fi

# Build and deploy to Cloud Run
echo -e "${YELLOW}Building and deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 300 \
    --set-env-vars ENVIRONMENT=production,GCS_BUCKET_PDFS=$BUCKET_PDFS,GCS_BUCKET_VECTORS=$BUCKET_VECTOR \
    --set-secrets OPENAI_API_KEY=openai-api-key:latest

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       Deployment Successful! 🎉        ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Service URL:${NC} $SERVICE_URL"
echo -e "${GREEN}Frontend URL:${NC} https://storage.googleapis.com/$BUCKET_FRONTEND/index.html"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update the API URL in search_ui.html to: $SERVICE_URL"
echo "2. Re-upload the frontend: gsutil cp search_ui.html gs://$BUCKET_FRONTEND/index.html"
echo "3. Test the deployment: curl $SERVICE_URL/health"
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo "  gcloud run services logs read $SERVICE_NAME --region=$REGION"
echo ""
echo -e "${YELLOW}To update deployment:${NC}"
echo "  ./deploy.sh"
echo ""
