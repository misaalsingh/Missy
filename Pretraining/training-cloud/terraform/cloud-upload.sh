#!/bin/bash

set -e

INSTANCE_NAME="gpt-training-vm"
ZONE="us-west1-b"
PROJECT_ID="YOUR-PROJECT-ID"  # Update this!

echo "=========================================="
echo "Uploading code and data to cloud instance"
echo "=========================================="
echo ""

# Check if instance exists
echo "Checking instance status..."
gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Instance '$INSTANCE_NAME' not found in zone '$ZONE'"
    echo "Did you run 'terraform apply' first?"
    exit 1
fi

echo "✓ Instance found"
echo ""

# Upload Python files
echo "📤 Uploading Python files..."
gcloud compute scp --zone=$ZONE --project=$PROJECT_ID \
  ../train.py \
  ../model.py \
  $INSTANCE_NAME:/home/training/

echo "✓ Python files uploaded"
echo ""

# Create data directory on instance
echo "Creating data directory on instance..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID \
  --command="mkdir -p /home/training/data"

echo "✓ Data directory created"
echo ""

# Upload data (this is the big one!)
echo "📤 Uploading training data..."
echo "⏳ This will take 10-20 minutes for ~12GB of data..."
echo ""

gcloud compute scp --recurse --zone=$ZONE --project=$PROJECT_ID \
  ../../training-local/data/memmap_batches \
  $INSTANCE_NAME:/home/training/data/

echo ""
echo "✓ Data uploaded successfully!"
echo ""

# Verify upload
echo "Verifying upload..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID \
  --command="ls -lh /home/training/ && ls -lh /home/training/data/memmap_batches/"

echo ""
echo "=========================================="
echo "✅ Upload complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH into instance:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "2. Start training:"
echo "   cd /home/training"
echo "   ./start_training.sh"
echo ""
echo "3. Check status:"
echo "   ./check_training.sh"
echo ""
echo "4. Detach from tmux:"
echo "   Press Ctrl+B, then D"
echo ""
echo "5. Monitor from local machine:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command='tail -f /home/training/training.log'"
echo ""
echo "=========================================="