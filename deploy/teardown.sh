#!/usr/bin/env bash
# Remove everything deploy.sh created, so nothing keeps billing.
#   ./deploy/teardown.sh
set -uo pipefail

APP="${APP:-firstshot}"
REGION="${AWS_REGION:-ap-south-1}"
ROLE="$APP-lambda-role"

aws lambda delete-function-url-config --function-name "$APP" --region "$REGION" 2>/dev/null
aws lambda delete-function --function-name "$APP" --region "$REGION" 2>/dev/null
aws ecr delete-repository --repository-name "$APP" --region "$REGION" --force >/dev/null 2>&1
aws logs delete-log-group --log-group-name "/aws/lambda/$APP" --region "$REGION" 2>/dev/null
aws iam detach-role-policy --role-name "$ROLE" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null
aws iam delete-role --role-name "$ROLE" 2>/dev/null
echo "Removed $APP from $REGION"
