import os
import sys

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import json

DESCRIPTION = """
Create (or verify) the Elasticsearch indices used by the app with the required
semantic_text mapping — without inserting any data.

This script will:
  1) Connect to your Elastic Cloud using ELASTIC_ENDPOINT and ELASTIC_API_KEY from .env
  2) Use the pre-provisioned ELSER v2 inference deployment with id '.elser-2-elasticsearch'
  3) Create the indices 'us_navy_dive_manual' and 'diving_safety_manual' with a
     'semantic_content' field mapped as semantic_text using that inference id.

Assumptions:
  - The cluster supports semantic_text and ELSER v2 (Elastic Cloud/newest APIs)
  - Your API key has permissions for Inference/ML and index creation

Usage:
  python setup_indices.py

Notes:
- On Elastic Cloud the ELSER v2 endpoint '.elser-2-elasticsearch' is
  pre-provisioned.
"""

INFERENCE_ID = ".elser-2-elasticsearch"


def log(msg: str):
    print(f"[setup] {msg}")


def connect_es() -> Elasticsearch:
    load_dotenv()
    endpoint = os.getenv("ELASTIC_ENDPOINT")
    api_key = os.getenv("ELASTIC_API_KEY")
    if not endpoint or not api_key:
        print("ERROR: ELASTIC_ENDPOINT and/or ELASTIC_API_KEY are not set in .env", file=sys.stderr)
        sys.exit(2)
    es = Elasticsearch(endpoint, api_key=api_key)
    try:
        if not es.ping():
            print("ERROR: Failed to ping Elasticsearch endpoint.", file=sys.stderr)
            sys.exit(2)
        log("Connected to Elasticsearch.")
    except Exception as e:
        print(f"ERROR: Failed to connect to Elasticsearch: {e}", file=sys.stderr)
        sys.exit(2)
    return es


def _request_raw(es: Elasticsearch, method: str, path: str, body: str):
    return es.transport.perform_request(method, path, headers={"content-type": "application/json"}, body=body)


def ensure_ingest_pipeline(es: Elasticsearch, pipeline_id: str = "diving_pipeline"):
    try:
        existing = es.ingest.get_pipeline(id=pipeline_id)
        if isinstance(existing, dict) and pipeline_id in existing:
            log(f"Ingest pipeline '{pipeline_id}' already exists. Skipping creation.")
            return
    except Exception:
        pass

    raw_body = """
{
  "processors": [
    {
      "set": {
        "field": "semantic_content",
        "ignore_empty_value": true,
        "copy_from": "body"
      }
    }
  ]
}
"""
    try:
        payload = json.loads(raw_body)
        es.ingest.put_pipeline(id=pipeline_id, body=payload)
        log(f"Created ingest pipeline '{pipeline_id}'.")
    except Exception as e:
        print(
            f"ERROR: Elasticsearch rejected the ingest pipeline definition for '{pipeline_id}': {e}",
            file=sys.stderr,
        )
        sys.exit(3)

    return


def create_index_if_missing(es: Elasticsearch, index_name: str, inference_id: str):
    if es.indices.exists(index=index_name):
        log(f"Index '{index_name}' already exists. Skipping creation.")
        return

    body = {
        "mappings": {
            "properties": {
                "semantic_content": {
                    "type": "semantic_text",
                    "inference_id": inference_id
                }
            }
        }
    }

    es.indices.create(index=index_name, body=body)
    log(f"Created index '{index_name}' with semantic_text mapping bound to '{inference_id}'.")


def reconcile_index_mapping_inference(es: Elasticsearch, index_name: str, desired_inference_id: str):
    try:
        mapping = es.indices.get_mapping(index=index_name)
        props = mapping.get(index_name, {}).get("mappings", {}).get("properties", {})
        field = props.get("semantic_content", {})
        current = field.get("inference_id") or field.get("inference", {}).get("inference_id")
        if current == desired_inference_id:
            return
        body = {"properties": {"semantic_content": {"type": "semantic_text", "inference_id": desired_inference_id}}}
        es.indices.put_mapping(index=index_name, body=body)
        log(f"Updated mapping for '{index_name}.semantic_content' to inference_id '{desired_inference_id}'.")
    except Exception as e:
        log(f"Warning: Could not reconcile mapping for '{index_name}': {e}")


def main():
    es = connect_es()
    inference_id = INFERENCE_ID
    # Create/verify indices first
    for idx in ("us_navy_dive_manual", "diving_safety_manual"):
        create_index_if_missing(es, idx, inference_id)
        reconcile_index_mapping_inference(es, idx, inference_id)

    ensure_ingest_pipeline(es, pipeline_id="diving_pipeline")

    log("Done. Indices are ready for the app.")


if __name__ == "__main__":
    # Simple entrypoint; no CLI args supported.
    print(DESCRIPTION)
    main()
