#!/usr/bin/env bash
# Ingest an ARRAY of BOQ objects — one document (one embedding) per BOQ.
# Each BOQ is flattened to a clean readable summary to generate better embeddings,
# and numeric/relational fields are stored in the payload for vector DB filtering.
# Usage: ./ingest-boqs.sh [file] [customerId]
set -euo pipefail

DATA_FILE="${1:-rag-data/boqs.json}"
CUSTOMER_ID="${2:-cust_12345}"
HOST="${HOST:-http://localhost:3000}"

BODY=$(python3 - "$DATA_FILE" "$CUSTOMER_ID" <<'PY'
import json, sys, re

data_file, customer_id = sys.argv[1], sys.argv[2]
try:
    with open(data_file) as f:
        boqs = json.load(f)
except FileNotFoundError:
    print(json.dumps({"error": f"File not found: {data_file}"}))
    sys.exit(0)

if isinstance(boqs, dict):
    boqs = [boqs]

def strip_html(s): return re.sub(r"<[^>]+>", "", s or "").strip()
def to_num(x):
    try: return float(x)
    except (TypeError, ValueError): return 0

docs = []
for b in boqs:
    lines = [
        f"BOQ: {b.get('name', 'Unnamed')} (Code: {b.get('code')}).",
        f"Type: {b.get('type')}. Associated Project ID: {b.get('project_id')}.",
        f"Status: {b.get('status')}. Stage: {b.get('stage')}.",
        f"Area: {b.get('area')}. Height: {b.get('height')}.",
        f"Cost: {b.get('cost')}. Discounted Cost: {b.get('discountedCost')} (Discount: {b.get('overallDiscount')}).",
        f"Sales Cost: {b.get('salesCost')}. Sales Discounted Cost: {b.get('salesDiscountedCost')}.",
    ]
    
    if b.get('rejectionReason'):
        lines.append(f"Rejection Reason: {b.get('rejectionReason')}")
    
    if b.get('approveRemarks'):
        lines.append(f"Approval Remarks: {b.get('approveRemarks')}")

    docs.append({
        "id": b.get("id"),
        "text": "\n".join(lines),
        "metadata": {
            "entityType": "boq",
            "boqId": b.get("id"),
            "projectId": b.get("project_id"),
            "code": b.get("code"),
            "type": b.get("type"),
            "status": b.get("status"),
            "stage": b.get("stage"),
            "cost": to_num(b.get("cost")),
            "discountedCost": to_num(b.get("discountedCost"))
        },
    })

print(json.dumps({"collection": "collection_gemini", "customerId": customer_id, "documents": docs}))
PY
)

# Check if there was an error in python processing
if echo "$BODY" | grep -q '"error":'; then
    echo "Error processing file: $(echo "$BODY" | grep -o '"error": "[^"]*"' | cut -d'"' -f4)"
    exit 1
fi

echo "Ingesting $(python3 -c "import json;print(len(json.load(open('$DATA_FILE'))))") BOQs into collection_gemini ..."
curl -sS -X POST "$HOST/rag/openai/qdrant/ingest" -H "Content-Type: application/json" -d "$BODY"
echo
echo "Ensuring necessary payload indices exist (safe to re-run) ..."
curl -sS -X PATCH "$HOST/rag/qdrant/collection_gemini/add-index" \
  -H "Content-Type: application/json" -d '{ "field": "entityType", "schema": "keyword" }'
echo
curl -sS -X PATCH "$HOST/rag/qdrant/collection_gemini/add-index" \
  -H "Content-Type: application/json" -d '{ "field": "projectId", "schema": "keyword" }'
echo
