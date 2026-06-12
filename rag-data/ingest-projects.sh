#!/usr/bin/env bash
# Ingest an ARRAY of project objects — one document (one embedding) per project.
# Each project is flattened to a clean readable summary (better embeddings than raw JSON),
# and numeric/team fields are stored in the payload so the analytics tool can compute on them.
# Usage: ./ingest-projects.sh [file] [customerId]
set -euo pipefail

DATA_FILE="${1:-rag-data/projects.json}"
CUSTOMER_ID="${2:-cust_123}"
HOST="${HOST:-http://localhost:3000}"

BODY=$(python3 - "$DATA_FILE" "$CUSTOMER_ID" <<'PY'
import json, sys, re

data_file, customer_id = sys.argv[1], sys.argv[2]
with open(data_file) as f:
    projects = json.load(f)
if isinstance(projects, dict):
    projects = [projects]

def strip_html(s): return re.sub(r"<[^>]+>", "", s or "").strip()
def people(lst, key="role"):
    return ", ".join(f"{p.get('name','').strip()} ({p.get(key,'')})" for p in (lst or []))
def to_num(x):
    try: return float(x)
    except (TypeError, ValueError): return 0
def slim_team(lst):
    return [{"name": (t.get("name") or "").strip(), "role": t.get("role"), "pocRole": t.get("pocRole")} for t in (lst or [])]

docs = []
for p in projects:
    cust = p.get("customer") or {}
    mis = p.get("misSummary") or {}
    lines = [
        f"Project {p.get('projectCode')} ({p.get('companyName','').strip()}).",
        f"Type: {p.get('type')} / {p.get('leadType')}. Status: {p.get('leadStatus')}.",
        f"Customer: {cust.get('name','').strip()} ({cust.get('emailId')}, {cust.get('mobileNumber')}).",
        f"City: {p.get('city')}, zone {p.get('zone')}. Area: {p.get('areaSft')} sqft.",
        f"Estimated value: {p.get('estimatedValue')}. BOQ value: {p.get('boqValue')}.",
        f"Stage: {p.get('stage')} / sub-stage {p.get('subStage')}. Priority: {p.get('priority')}. Age: {p.get('projectAge')} days.",
        f"Owner: {p.get('owner')}. Scope: {p.get('scope')}. Channel: {p.get('channel')}.",
        f"Notes: {strip_html(p.get('desc'))}.",
        f"Team: {people(p.get('team'))}.",
        f"POC team: {people(p.get('pocTeam'), 'pocRole')}.",
    ]
    if mis:
        lines.append(
            f"MIS: project value w/o tax {mis.get('currentProjectValueWoTax')}, "
            f"design order value {mis.get('currentDesignOrderValue')}, "
            f"build order value {mis.get('tbcBuildOrderValue')}, "
            f"COGS {mis.get('cogs')} ({mis.get('cogsPercentage')}%), lead source {mis.get('leadSource')}."
        )
    docs.append({
        "id": p.get("projectId"),
        "text": "\n".join(lines),
        "metadata": {
            "projectCode": p.get("projectCode"),
            "companyName": p.get("companyName"),
            "city": p.get("city"),
            "stage": p.get("stage"),
            "owner": p.get("owner"),
            "estimatedValue": to_num(p.get("estimatedValue")),
            "boqValue": to_num(p.get("boqValue")),
            "team": slim_team(p.get("team")),
            "pocTeam": slim_team(p.get("pocTeam")),
        },
    })

print(json.dumps({"collection": "collection_gemini", "customerId": customer_id, "documents": docs}))
PY
)

echo "Ingesting $(python3 -c "import json;print(len(json.load(open('$DATA_FILE'))))") projects into collection_gemini ..."
curl -sS -X POST "$HOST/rag/openai/qdrant/ingest" -H "Content-Type: application/json" -d "$BODY"
echo
echo "Creating customerId payload index (safe to re-run) ..."
curl -sS -X PATCH "$HOST/rag/qdrant/collection_gemini/add-index" \
  -H "Content-Type: application/json" -d '{ "field": "customerId", "schema": "keyword" }'
echo
