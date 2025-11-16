import os
import sys
import pathlib
import shutil
import time
import logging
import warnings
from typing import Iterable, List, Tuple

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers


DESCRIPTION = """
Seed Elasticsearch with the US Navy Diving Manual as described in the README.

Assumptions 
  - setup_indices.py has been run successfully beforehand.

What this script does:
  1) Connects to Elasticsearch using ELASTIC_ENDPOINT and ELASTIC_API_KEY from .env
  2) Downloads the US Navy Diving Manual PDF into ./data if not already present
     URL: https://www.navsea.navy.mil/Portals/103/Documents/SUPSALV/Diving/US%20DIVING%20MANUAL_REV7.pdf
  3) Loads the PDF using LlamaIndex SimpleDirectoryReader
  4) Bulk uploads pages into index 'us_navy_dive_manual_raw' with field 'body'
  5) Triggers a reindex from 'us_navy_dive_manual_raw' → 'us_navy_dive_manual'
     using pipeline 'diving_pipeline' with slices=auto and wait_for_completion=false

Notes:
  - Optional dependencies required: llama-index, tqdm (for nicer progress bars)
    If not installed, you will be prompted with setup instructions.

Usage:
  python seed_data.py
"""


PDF_URL = "https://www.navsea.navy.mil/Portals/103/Documents/SUPSALV/Diving/US%20DIVING%20MANUAL_REV7.pdf"
DATA_DIR = pathlib.Path("./data")
PDF_PATH = DATA_DIR / "US_DIVING_MANUAL_REV7.pdf"
SOURCE_INDEX = "us_navy_dive_manual_raw"
DEST_INDEX = "us_navy_dive_manual"
PIPELINE_ID = "diving_pipeline"


def log(msg: str):
    print(f"[seed] {msg}")


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


def ensure_prereqs(es: Elasticsearch) -> None:
    if not es.indices.exists(index=DEST_INDEX):
        print(
            f"ERROR: Destination index '{DEST_INDEX}' not found. Please run 'python setup_indices.py' first.",
            file=sys.stderr,
        )
        sys.exit(3)
    try:
        _ = es.ingest.get_pipeline(id=PIPELINE_ID)
    except Exception:
        print(
            f"ERROR: Ingest pipeline '{PIPELINE_ID}' not found. Please run 'python setup_indices.py' first.",
            file=sys.stderr,
        )
        sys.exit(3)

def download_pdf(url: str, out_path: pathlib.Path) -> None:
    import urllib.request
    import shutil as _shutil

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        log(f"PDF already present at {out_path} (skipping download).")
        return

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) "
                "Gecko/20100101 Firefox/128.0"
            ),
            "Accept": "application/pdf,application/octet-stream,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "identity",
            "Referer": "https://www.navsea.navy.mil/",
        },
        method="GET",
    )

    try:
        log(f"Downloading PDF to {out_path} ...")
        with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
            _shutil.copyfileobj(resp, f)
        log("Download complete.")
    except Exception:
        print(
            f"Cannot download the file {url}, download it manually to \"data\" folder. Exiting",
            file=sys.stderr,
        )
        sys.exit(4)

def load_documents_from_dir(input_dir: pathlib.Path):
    try:
        from llama_index.core import SimpleDirectoryReader
    except Exception as e:
        print(
            "ERROR: Failed to import LlamaIndex (llama-index).\n"
            "Install the optional dependency first, e.g.:\n"
            "  pip install llama-index\n"
            "Or uncomment it in requirements.txt and reinstall.",
            file=sys.stderr,
        )
        sys.exit(5)

    reader = SimpleDirectoryReader(input_dir=str(input_dir))
    documents = reader.load_data()
    return [doc.to_dict() for doc in list(documents)]


def bulk_upload_to_elasticsearch(
    data: List[dict],
    index_name: str,
    es: Elasticsearch,
    batch_size: int = 500,
    max_workers: int = 10,
) -> Tuple[int, int]:
    # Optional imports for progress bars and threads
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore

    from concurrent.futures import ThreadPoolExecutor, as_completed

    total_documents = len(data)
    if total_documents == 0:
        return 0, 0

    success_bar = tqdm(total=total_documents, desc="Successful uploads", colour="green") if tqdm else None
    failed_bar = tqdm(total=total_documents, desc="Failed uploads", colour="red") if tqdm else None

    def create_action(doc: dict) -> dict:
        return {
            "_index": index_name,
            "_id": doc.get("id_"),
            "body": doc.get("text"),
        }

    def read_and_create_batches(items: List[dict]):
        batch: List[dict] = []
        for d in items:
            batch.append(create_action(d))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def upload_batch(batch: List[dict]):
        try:
            success, failed = helpers.bulk(
                es,
                batch,
                raise_on_error=False,
                request_timeout=45,
            )
            if isinstance(failed, list):
                failed = len(failed)
            return int(success), int(failed)
        except Exception as e:
            print(f"Error during bulk upload: {e}")
            return 0, len(batch)

    total_uploaded = 0
    total_failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_batch, batch): batch for batch in read_and_create_batches(data)}
        for fut in as_completed(futures):
            s, f = fut.result()
            total_uploaded += s
            total_failed += f
            if success_bar:
                success_bar.update(s)
            if failed_bar:
                failed_bar.update(f)

    if success_bar:
        success_bar.close()
    if failed_bar:
        failed_bar.close()

    return total_uploaded, total_failed


def iter_actions(data: Iterable[dict], index_name: str) -> Iterable[dict]:
    for doc in data:
        _id = doc.get("id_") or doc.get("doc_id") or doc.get("id")
        text = doc.get("text") or doc.get("body") or ""
        yield {
            "_index": index_name,
            "_id": _id,
            "body": text,
        }


def bulk_upload(es: Elasticsearch, actions: Iterable[dict]) -> Tuple[int, int]:
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    actions_list: List[dict] = list(actions)
    total = len(actions_list)
    if total == 0:
        return 0, 0

    success_total = 0
    failed_total = 0

    if tqdm:
        success_bar = tqdm(total=total, desc="Successful uploads", colour="green")
        failed_bar = tqdm(total=total, desc="Failed uploads", colour="red")
    else:
        success_bar = failed_bar = None

    try:
        success, failed = helpers.bulk(es, actions_list, raise_on_error=False, request_timeout=90)
        if isinstance(failed, list):
            failed = len(failed)
        success_total += int(success)
        failed_total += int(failed)
        if success_bar:
            success_bar.update(success)
        if failed_bar:
            failed_bar.update(failed)
    except Exception as e:
        print(f"ERROR: Bulk upload failed: {e}", file=sys.stderr)
        failed_total += total

    if success_bar:
        success_bar.close()
    if failed_bar:
        failed_bar.close()

    return success_total, failed_total


def trigger_reindex(es: Elasticsearch, src: str, dest: str, pipeline: str) -> dict:
    body = {
        "source": {
            "index": src,
            # README shows size: 4; keep small batch size hint
            "size": 4,
        },
        "dest": {
            "index": dest,
            "pipeline": pipeline,
        },
        "conflicts": "proceed",
    }
    # Use the high-level API which supports keyword args for slices/wait_for_completion
    return es.reindex(body=body, slices="auto", wait_for_completion=False)


def monitor_reindex_task(
    es: Elasticsearch,
    task_id: str,
    poll_interval: float = 2.0,
    timeout_seconds: int | None = None,
) -> dict:
    """Poll the Tasks API until the reindex task completes, printing progress.

    Returns the final task result dict (from es.tasks.get), which includes a
    "completed": true flag and a "response" with reindex stats.
    """
    # Optional progress bar
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore

    start = time.time()
    bar = None
    last_total = None

    log(f"Monitoring reindex task '{task_id}' ...")
    try:
        while True:
            info = es.tasks.get(task_id=task_id)
            if info.get("completed"):
                # Close progress bar if open
                if bar:
                    bar.close()
                log("Reindex task completed.")
                return info

            task = info.get("task", {})
            status = task.get("status", {}) or {}
            total = status.get("total")
            created = status.get("created", 0)
            updated = status.get("updated", 0)
            deleted = status.get("deleted", 0)
            batches = status.get("batches")

            done = (created or 0) + (updated or 0) + (deleted or 0)

            if tqdm and total and (bar is None or last_total != total):
                # Initialize or reinitialize the bar when we first learn total
                if bar:
                    bar.close()
                bar = tqdm(total=total, desc="Reindexing", colour="blue")
                last_total = total
            if bar:
                # Keep bar in sync (tqdm only increments; compute delta)
                delta = max(0, done - bar.n)
                if delta:
                    bar.update(delta)
                bar.set_postfix({"batches": batches})
            else:
                # Console fallback
                if total:
                    log(f"Reindex progress: {done}/{total} docs (batches={batches})")
                else:
                    log(f"Reindex progress: done={done} (batches={batches})")

            if timeout_seconds is not None and (time.time() - start) > timeout_seconds:
                if bar:
                    bar.close()
                raise TimeoutError(
                    f"Reindex task '{task_id}' did not complete within {timeout_seconds}s"
                )

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        if bar:
            bar.close()
        print("Interrupted by user while monitoring reindex. Task continues on server.")
        return {"interrupted": True, "task_id": task_id}
    except Exception as e:
        if bar:
            bar.close()
        print(f"ERROR while monitoring task {task_id}: {e}", file=sys.stderr)
        return {"error": str(e), "task_id": task_id}


def main():
    print(DESCRIPTION)
    # Keep stdout clean so progress bars are readable
    warnings.simplefilter("ignore", DeprecationWarning)
    for _logger in ("elasticsearch", "elastic_transport", "urllib3"):
        try:
            logging.getLogger(_logger).setLevel(logging.WARNING)
        except Exception:
            pass
    es = connect_es()
    ensure_prereqs(es)

    # Ensure source index exists (we can let ES auto-create, but be explicit)
    if not es.indices.exists(index=SOURCE_INDEX):
        es.indices.create(index=SOURCE_INDEX)
        log(f"Created source index '{SOURCE_INDEX}'.")
    else:
        log(f"Source index '{SOURCE_INDEX}' exists.")

    # Download and load documents
    download_pdf(PDF_URL, PDF_PATH)
    docs = load_documents_from_dir(DATA_DIR)
    log(f"Loaded {len(docs)} document(s) from {DATA_DIR}.")

    # Bulk upload (README-aligned batched threaded bulk)
    log(f"Uploading documents to '{SOURCE_INDEX}' using threaded bulk ...")
    success, failed = bulk_upload_to_elasticsearch(
        docs,
        SOURCE_INDEX,
        es,
        batch_size=16,
        max_workers=10,
    )
    log(f"Bulk upload completed: success={success}, failed={failed}.")

    # Ensure we only start reindex AFTER all source docs are visible to searches
    if failed > 0:
        print(
            f"ERROR: {failed} documents failed to index into '{SOURCE_INDEX}'. Aborting reindex.",
            file=sys.stderr,
        )
        sys.exit(7)

    # Force a refresh and verify counts, with a short wait loop
    try:
        es.indices.refresh(index=SOURCE_INDEX)
    except Exception:
        # Non-fatal; we'll still poll counts
        pass

    expected = len(docs)
    deadline = time.time() + 30.0  # up to 30s to see all docs
    last_count = -1
    while True:
        try:
            cnt = es.count(index=SOURCE_INDEX).get("count", 0)
            last_count = cnt
        except Exception:
            cnt = 0
            last_count = 0

        if cnt >= expected:
            break
        if time.time() > deadline:
            print(
                f"WARNING: Source index '{SOURCE_INDEX}' count={last_count} < expected={expected} after wait. Proceeding with reindex.",
                file=sys.stderr,
            )
            break
        time.sleep(0.5)

    # Trigger reindex to semantic index via pipeline
    log(
        f"Triggering reindex from '{SOURCE_INDEX}' → '{DEST_INDEX}' via pipeline '{PIPELINE_ID}' (slices=auto, async)..."
    )
    try:
        resp = trigger_reindex(es, SOURCE_INDEX, DEST_INDEX, PIPELINE_ID)
        log(f"Reindex started: {resp}")
        task_id = resp.get("task")
        if not task_id:
            log("No task id returned by reindex call (it may have completed immediately).")
        else:
            # Monitor until completion
            final = monitor_reindex_task(es, task_id, poll_interval=2.0)
            if final.get("completed"):
                response = final.get("response", {})
                total = response.get("total")
                created = response.get("created")
                updated = response.get("updated")
                deleted = response.get("deleted")
                conflicts = response.get("version_conflicts")
                log(
                    f"Reindex finished: total={total}, created={created}, updated={updated}, deleted={deleted}, conflicts={conflicts}"
                )
        # Regardless of synchronous or asynchronous completion path above, refresh the destination index
        try:
            es.indices.refresh(index=DEST_INDEX)
            try:
                dest_count = es.count(index=DEST_INDEX).get("count", 0)
                log(f"Refreshed destination index '{DEST_INDEX}'. Document count now: {dest_count}.")
            except Exception:
                # Count is best-effort; not fatal
                pass
        except Exception as e:
            log(f"Warning: Failed to refresh destination index '{DEST_INDEX}': {e}")
    except Exception as e:
        print(f"ERROR: Failed to start reindex: {e}", file=sys.stderr)
        sys.exit(6)
    
    log("Done.")


if __name__ == "__main__":
    main()
