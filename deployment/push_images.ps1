# Docker Push Script for MEdi Chain AI
# Username: saya2572

$DOCKER_USER = "saya2572"
$TAG = "v1.0"

Write-Host "--- Starting Build and Push Process for $DOCKER_USER ---" -ForegroundColor Cyan

# 1. Build images using compose
Write-Host "Step 1: Building images..." -ForegroundColor Yellow
docker-compose -f deployment/docker-compose.yml build

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed. Please ensure Docker Desktop is running."
    exit $LASTEXITCODE
}

# 2. Tag images
Write-Host "Step 2: Tagging images..." -ForegroundColor Yellow
docker tag deployment-medi-ui:latest ${DOCKER_USER}/medi-chain-ui:${TAG}
docker tag deployment-medi-api:latest ${DOCKER_USER}/medi-chain-api:${TAG}


# 3. Push images
Write-Host "Step 3: Pushing to Docker Hub..." -ForegroundColor Yellow
docker push ${DOCKER_USER}/medi-chain-ui:${TAG}
docker push ${DOCKER_USER}/medi-chain-api:${TAG}

Write-Host "--- Process Complete ---" -ForegroundColor Green
Write-Host "View your images at: https://hub.docker.com/u/${DOCKER_USER}"
