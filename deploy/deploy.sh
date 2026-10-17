#!/usr/bin/env bash
# Build the image for arm64 (Graviton), push it to Amazon ECR and run it on AWS Lambda
# behind a public HTTPS Function URL. Safe to re-run: it updates what already exists.
#
#   ./deploy/deploy.sh
#
# Needs Docker with buildx and AWS CLI v2 (a 2025 or later release) signed in to your account.
# Settings can be overridden: APP=firstshot AWS_REGION=ap-south-1 MEMORY=3008 TIMEOUT=60
set -euo pipefail

APP="${APP:-firstshot}"
REGION="${AWS_REGION:-ap-south-1}"
MEMORY="${MEMORY:-3008}"
TIMEOUT="${TIMEOUT:-60}"
ROLE="$APP-lambda-role"
TAG="$(date +%Y%m%d-%H%M%S)"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
IMAGE="$REGISTRY/$APP:$TAG"
echo "Deploying $APP to account $ACCOUNT in $REGION"

# 1. Container registry
aws ecr describe-repositories --repository-names "$APP" --region "$REGION" >/dev/null 2>&1 ||
  aws ecr create-repository --repository-name "$APP" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr get-login-password --region "$REGION" |
  docker login --username AWS --password-stdin "$REGISTRY" >/dev/null

# 2. Build and push. Lambda rejects multi-platform image indexes, hence --provenance=false.
docker buildx build --platform linux/arm64 --provenance=false -t "$IMAGE" --push .

# 3. Execution role: CloudWatch Logs only
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]
  }' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "Waiting for the new IAM role to propagate"
  sleep 12
fi
ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)"

# 4. Logs kept for two weeks instead of forever
aws logs create-log-group --log-group-name "/aws/lambda/$APP" --region "$REGION" 2>/dev/null || true
aws logs put-retention-policy --log-group-name "/aws/lambda/$APP" --retention-in-days 14 --region "$REGION"

# 5. Function
if aws lambda get-function --function-name "$APP" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$APP" --image-uri "$IMAGE" --region "$REGION" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$APP" --region "$REGION"
  aws lambda update-function-configuration --function-name "$APP" --memory-size "$MEMORY" \
    --timeout "$TIMEOUT" --region "$REGION" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$APP" --region "$REGION"
else
  aws lambda create-function --function-name "$APP" --package-type Image --code ImageUri="$IMAGE" \
    --role "$ROLE_ARN" --architectures arm64 --memory-size "$MEMORY" --timeout "$TIMEOUT" \
    --region "$REGION" >/dev/null
  aws lambda wait function-active-v2 --function-name "$APP" --region "$REGION"
fi

# 6. Public HTTPS endpoint. Since October 2025 a public Function URL needs both permissions.
if ! aws lambda get-function-url-config --function-name "$APP" --region "$REGION" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$APP" --auth-type NONE --region "$REGION" >/dev/null
  aws lambda add-permission --function-name "$APP" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE \
    --region "$REGION" >/dev/null
  aws lambda add-permission --function-name "$APP" --statement-id public-invoke \
    --action lambda:InvokeFunction --principal '*' --invoked-via-function-url \
    --region "$REGION" >/dev/null
fi

URL="$(aws lambda get-function-url-config --function-name "$APP" --region "$REGION" --query FunctionUrl --output text)"
echo "Checking ${URL}api/health"
for _ in $(seq 1 10); do
  if curl -sf "${URL}api/health" >/dev/null; then break; fi
  sleep 3
done
echo "Live at $URL"
echo "API docs at ${URL}docs"
