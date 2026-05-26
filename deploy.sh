#!/usr/bin/env bash
#
# Deploy the Azul webapp to AWS.
#
# Usage:
#   ./deploy.sh [--skip-frontend] [--dry-run]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REGION="us-west-2"
STACK_NAME="AzulStack"

# --- Parse arguments ---
SKIP_FRONTEND=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --skip-frontend) SKIP_FRONTEND=true ;;
        --dry-run)       DRY_RUN=true ;;
        -h|--help)
            echo "Usage: ./deploy.sh [--skip-frontend] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- Helpers ---
log() { echo -e "\033[1;34m==>\033[0m $*"; }
err() { echo -e "\033[1;31mERROR:\033[0m $*" >&2; }

# --- Check prerequisites ---
log "Checking prerequisites..."

if ! command -v cdk &>/dev/null; then
    err "CDK CLI not found. Install with: npm install -g aws-cdk"
    exit 1
fi

if ! command -v aws &>/dev/null; then
    err "AWS CLI not found."
    exit 1
fi

if ! aws sts get-caller-identity &>/dev/null; then
    err "AWS credentials not configured or expired."
    exit 1
fi

# --- Check Docker access ---
if ! docker info &>/dev/null 2>&1; then
    err "Docker is not accessible. Ensure Docker is running."
    exit 1
fi

# --- Step 1: Build frontend ---
if [ "$SKIP_FRONTEND" = false ]; then
    log "Building frontend..."
    if [ -d "webapp/node_modules" ]; then
        (cd webapp && npx tsc && npx vite build)
    else
        log "Installing frontend dependencies..."
        (cd webapp && npm ci && npx tsc && npx vite build)
    fi
    log "Frontend built successfully."
else
    log "Skipping frontend build."
    if [ ! -d "webapp/dist" ]; then
        err "webapp/dist does not exist. Run without --skip-frontend."
        exit 1
    fi
fi

# --- Step 2: CDK Deploy ---
log "Deploying CDK stack..."
cdk deploy --require-approval never --force
log "CDK deploy complete."

# --- Done ---
log "Deployment finished!"
echo ""
aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query 'Stacks[0].Outputs[].[Description, OutputValue]' --output table
