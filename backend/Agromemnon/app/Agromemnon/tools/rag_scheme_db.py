"""Retrieval over the government agricultural scheme knowledge base.

Calls Bedrock's Retrieve API directly rather than going through
strands_tools.retrieve: that package pulls in sympy, pillow and slack-bolt for tools this
project never uses, which added ~20 MB to the deployment zip and pushed the upload past
its timeout. One boto3 call replaces it.
"""

import logging
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from strands import tool

logger = logging.getLogger(__name__)

# The knowledge base lives in the account the agent is deployed into; both are
# overridable so a redeploy elsewhere does not need a code change.
KB_ID = os.environ.get("SCHEME_KB_ID", "IY3TP8DVNA")
REGION = os.environ.get("SCHEME_KB_REGION", "us-east-1")

# Enough passages to cover two or three schemes without burying the model in near-duplicates.
NUM_RESULTS = 5

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-agent-runtime", region_name=REGION)
    return _client


def _build_filter(level: str, category: str):
    """Build the metadata filter, or None when no filter was asked for."""
    clauses = []
    if level.strip():
        clauses.append({"equals": {"key": "level", "value": level.strip()}})
    if category.strip():
        clauses.append({"equals": {"key": "schemeCategory", "value": category.strip()}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"andAll": clauses}


@tool
def rag_scheme_db(query: str, level: str = "", category: str = "") -> dict:
    """Search the government agricultural scheme database for schemes relevant to a farmer.

    Args:
        query: The farmer's question or situation, in natural language
               (e.g. "relief for a fisherman lost at sea", "subsidy for drip
               irrigation in Maharashtra").
        level: Optional filter — "State" or "Central". Leave blank to search both.
        category: Optional filter — e.g. "Agriculture,Rural & Environment".
                  Leave blank to search all categories.
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "Describe what the farmer needs, e.g. query='drip irrigation subsidy'."}
    if not KB_ID:
        return {"error": "The scheme knowledge base is not configured (SCHEME_KB_ID is unset)."}

    search: dict = {"numberOfResults": NUM_RESULTS}
    metadata_filter = _build_filter(level, category)
    if metadata_filter:
        search["filter"] = metadata_filter

    try:
        response = _get_client().retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query.strip()},
            retrievalConfiguration={"vectorSearchConfiguration": search},
        )
    except (BotoCoreError, ClientError) as e:
        logger.warning("scheme knowledge base retrieve failed (%s in %s): %s", KB_ID, REGION, e)
        return {"error": f"Could not reach the scheme database: {e}"}

    passages = []
    for result in response.get("retrievalResults", []):
        text = (result.get("content") or {}).get("text") or ""
        if not text.strip():
            continue
        entry = {"text": text.strip()}
        # Preserve provenance for the advice agent. Retrieval metadata is evidence, not
        # decoration: it lets the final answer cite the exact passage and distinguish an
        # exact match from a related fallback result without inventing a source.
        metadata = result.get("metadata") or {}
        for key in ("level", "schemeCategory", "schemeName", "source", "sourceUrl", "url"):
            if metadata.get(key):
                entry[key] = metadata[key]
        location = result.get("location") or {}
        if location:
            entry["retrieval_location"] = location
        if result.get("score") is not None:
            entry["retrieval_score"] = result["score"]
        passages.append(entry)

    if not passages:
        return {"query": query, "passages": [],
                "message": "No scheme in the database matched this situation."}

    return {
        "query": query,
        "passages": passages,
        "source": "Government scheme database",
        "source_type": "Bedrock Knowledge Base retrieval",
        "retrieval_region": REGION,
    }
