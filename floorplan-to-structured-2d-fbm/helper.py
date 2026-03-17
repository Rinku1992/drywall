import logging
import json
import sys
import os
import time
from pathlib import Path
from contextlib import asynccontextmanager
from ruamel.yaml import YAML
from time import sleep
from random import uniform
import asyncio
import asyncpg
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud.storage import Client as CloudStorageClient
from fastapi.encoders import jsonable_encoder
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, DeadlineExceeded
import re 
import math

from transcriber import Transcriber
import 

# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------

def log_json(severity: str, message: str, **kwargs):
    payload = {"severity": severity, "message": message}
    payload.update(kwargs)
    print(json.dumps(payload, default=str), flush=True)


@asynccontextmanager
async def timed_step(step_name: str, request_id: str = "", **extra):
    start = time.perf_counter()
    log_json("INFO", "STEP_START", step=step_name, request_id=request_id, **extra)
    error_msg = None
    try:
        yield
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if error_msg:
            log_json("ERROR", "STEP_FAILED", step=step_name, request_id=request_id,
                     duration_ms=duration_ms, error=error_msg, **extra)
        else:
            log_json("INFO", "STEP_COMPLETE", step=step_name, request_id=request_id,
                     duration_ms=duration_ms, **extra)


# ---------------------------------------------------------------------------
# Configuration Loaders (called once at startup, cached)
# ---------------------------------------------------------------------------

def load_gcp_credentials() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("gcp.yaml", 'r') as f:
        credentials = yaml.load(f)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials["service_drywall_account_key"]
    return credentials


def load_hyperparameters() -> dict:
    yaml = YAML(typ="safe", pure=True)
    with open("hyperparameters.yaml", 'r') as f:
        hyperparameters = yaml.load(f)
    return hyperparameters


def enable_logging_on_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format='{"severity": "%(levelname)s", "message": "%(message)s"}',
        stream=sys.stdout
    )


# ---------------------------------------------------------------------------
# GCS Client (shared singleton)
# ---------------------------------------------------------------------------

_gcs_client = None

def get_gcs_client() -> CloudStorageClient:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = CloudStorageClient()
        log_json("INFO", "GCS_CLIENT_CREATED")
    return _gcs_client


# ---------------------------------------------------------------------------
# PostgreSQL Connection Pool
# ---------------------------------------------------------------------------

async def create_pg_pool(credentials) -> asyncpg.Pool:
    pg_config = credentials["PostgreSQL"]
    try:
        pool = await asyncpg.create_pool(
            host=pg_config["host"],
            port=pg_config["port"],
            database=pg_config["database"],
            user=pg_config["user"],
            password=pg_config["password"],
            min_size=1,
            max_size=5,
            command_timeout=300,
            max_inactive_connection_lifetime=0,
        )
        log_json("INFO", "PG_POOL_CREATED", host=pg_config["host"],
                 database=pg_config["database"])
        return pool
    except Exception as exc:
        log_json("ERROR", "PG_POOL_FAILED", host=pg_config["host"],
                 error=f"{type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Core DB Helpers
# ---------------------------------------------------------------------------

async def pg_fetch_all(pool, query, params=None, query_name="unnamed"):
    start = time.perf_counter()
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *(params or []))
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("INFO", "DB_FETCH_ALL", query=query_name, duration_ms=duration_ms, row_count=len(rows))
        return rows
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("ERROR", "DB_FETCH_ALL_FAILED", query=query_name, duration_ms=duration_ms,
                 error=f"{type(exc).__name__}: {exc}")
        raise


async def pg_fetch_one(pool, query, params=None, query_name="unnamed"):
    start = time.perf_counter()
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *(params or []))
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("INFO", "DB_FETCH_ONE", query=query_name, duration_ms=duration_ms, found=row is not None)
        return row
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("ERROR", "DB_FETCH_ONE_FAILED", query=query_name, duration_ms=duration_ms,
                 error=f"{type(exc).__name__}: {exc}")
        raise


async def pg_execute(pool, query, params=None, query_name="unnamed"):
    start = time.perf_counter()
    try:
        async with pool.acquire() as conn:
            status = await conn.execute(query, *(params or []))
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("INFO", "DB_EXECUTE", query=query_name, duration_ms=duration_ms, status=status)
        return status
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_json("ERROR", "DB_EXECUTE_FAILED", query=query_name, duration_ms=duration_ms,
                 error=f"{type(exc).__name__}: {exc}")
        raise


# ---------------------------------------------------------------------------
# JSONB Helper
# ---------------------------------------------------------------------------

def parse_jsonb(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    return value


# ---------------------------------------------------------------------------
# Database Operations
# ---------------------------------------------------------------------------

async def insert_model_2d(
    model_2d, scale, page_number, plan_id, user_id, project_id,
    target_drywalls, pool, credentials
):
    page_number = int(page_number)
    scale = scale or ''
    target_drywalls = target_drywalls or ''

    # Use a fresh direct connection for ALL DB operations here.
    # After 5-15 min of Vertex AI processing, pool connections are
    # dead (killed by VPC connector). A fresh connection guarantees success.
    pg_config = credentials["PostgreSQL"]
    conn = None
    for attempt in range(3):
        try:
            conn = await asyncpg.connect(
                host=pg_config["host"],
                port=pg_config["port"],
                database=pg_config["database"],
                user=pg_config["user"],
                password=pg_config["password"],
                timeout=60,
                command_timeout=60,
            )
            log_json("INFO", "DB_DIRECT_CONNECT_SUCCESS", attempt=attempt + 1)
            break
        except Exception as e:
            log_json("WARNING", "DB_DIRECT_CONNECT_RETRY", attempt=attempt + 1,
                     error=f"{type(e).__name__}: {e}")
            if attempt == 2:
                raise
            await asyncio.sleep(2)
    try:
        if not model_2d.get("metadata", None):
            row = await conn.fetchrow(
                "SELECT model_2d->'metadata' AS metadata FROM models "
                "WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2) AND page_number = $3",
                project_id, plan_id, page_number,
            )
            if row:
                metadata = parse_jsonb(row["metadata"])
                if metadata:
                    model_2d["metadata"] = metadata

        model_2d_json = json.dumps(model_2d)

        status = await conn.execute(
            """
            INSERT INTO models (
                plan_id, project_id, user_id, page_number, scale,
                model_2d, model_3d, takeoff, target_drywalls,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6::jsonb, '{}'::jsonb, '{}'::jsonb, $7,
                NOW(), NOW()
            )
            ON CONFLICT (LOWER(project_id), LOWER(plan_id), page_number)
            DO UPDATE SET
                model_2d = EXCLUDED.model_2d,
                scale = CASE WHEN EXCLUDED.scale = '' THEN models.scale ELSE EXCLUDED.scale END,
                user_id = EXCLUDED.user_id,
                updated_at = NOW()
            """,
            plan_id, project_id, user_id, page_number, scale,
            model_2d_json, target_drywalls,
        )
        log_json("INFO", "DB_EXECUTE", query="insert_model_2d", status=status)
    finally:
        await conn.close()
        
async def load_templates(pool, credentials):
    """Load SKU/drywall templates from the sku table."""
    rows = await pg_fetch_all(pool, "SELECT * FROM sku", query_name="load_templates")

    product_templates_target = list()
    cached_templates_sku = list()
    for row in rows:
        product_template = dict(row)
        if product_template["sku_id"] in cached_templates_sku:
            continue
        cached_templates_sku.append(product_template["sku_id"])
        product_template["sku_variant"] = f"{product_template['sku_id']} - {product_template['sku_description']}"
        color_code = product_template["color_code"]
        if isinstance(color_code, str):
            color_code = json.loads(color_code)
        product_template["color_code"] = [color_code['b'], color_code['g'], color_code['r']]
        product_templates_target.append(product_template)

    log_json("INFO", "TEMPLATES_LOADED", template_count=len(product_templates_target))
    return jsonable_encoder(product_templates_target)


# ---------------------------------------------------------------------------
# Non-DB Helpers
# ---------------------------------------------------------------------------

def download_floorplan(user_id, plan_id, project_id, credentials, index, destination_path="/tmp/floor_plan_wall_processed.png"):
    client = get_gcs_client()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/floor_plan.png"
    blob = bucket.blob(blob_path)
    destination_path = Path(destination_path)
    destination_path = destination_path.parent.joinpath(project_id).joinpath(plan_id).joinpath(user_id).joinpath(destination_path.name)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_path)
    return destination_path


def upload_floorplan(plan_path, plan_id, project_id, credentials, index=None, directory=None):
    client = get_gcs_client()
    page_number = Path(plan_path.stem).suffix
    if page_number:
        blob_object_name = Path(str(plan_path).replace(page_number, '')).name
    else:
        blob_object_name = plan_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if directory:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{directory}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{directory}/{blob_object_name}"
    else:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{index}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{blob_object_name}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(plan_path)
    return f"gs://{credentials['CloudStorage']['bucket_name']}/{blob_path}"


def load_vertex_ai_client(credentials, region="us-central1"):
    with open(credentials["VertexAI"]["service_account_key"], 'r') as f:
        project_id = json.load(f)["project_id"]
    vertexai.init(project=project_id, location=region)
    vertex_ai_client = GenerativeModel(credentials["VertexAI"]["llm"]["model_name"])
    generation_config = credentials["VertexAI"]["llm"]["parameters"]
    return vertex_ai_client, generation_config


def transcribe(credentials, hyperparameters, floor_plan_path):
    transcriber = Transcriber(credentials, hyperparameters)
    return transcriber.transcribe(floor_plan_path, [0, 1, -1, -2])


def phoenix_call(generate_content_lambda, max_retry=5, base_delay=1.0, pydantic_model=None):
    n_iterations = 0
    temperature = 0
    while n_iterations < max_retry:
        try:
            response = generate_content_lambda(temperature)
            # --- NEW: Safely extract text even if SDK chunks it ---
            if hasattr(response.candidates[0].content, 'parts'):
                raw_text = "".join([part.text for part in response.candidates[0].content.parts])
            else:
                raw_text = response.text
            # ------------------------------------------------------
                
            if pydantic_model:
                # Use raw_text instead of response.text
                json_response = json.loads(raw_text.strip("`json\n").replace("{{", '{').replace("}}", '}'))
                response_json_pydantic = pydantic_model(**json_response)
                return response_json_pydantic, json_response
            return raw_text
            
        except (ResourceExhausted, ServiceUnavailable, DeadlineExceeded) as e:
            n_iterations += 1
            log_json("WARNING", "PHOENIX_CALL_RETRY", attempt=n_iterations,
                     max_retry=max_retry, error=str(e), reason="rate_limit_or_unavailable")
            if n_iterations >= max_retry:
                raise e
            sleep_time = base_delay * (2 ** (n_iterations - 1)) + uniform(0, 0.5)
            sleep(sleep_time)
        except Exception as e:
            n_iterations += 1
            log_json("WARNING", "PHOENIX_CALL_RETRY", attempt=n_iterations,
                     max_retry=max_retry, error=str(e), reason="parse_or_generation_error")
            if n_iterations >= max_retry:
                raise e
            temperature = min(0.5 * (n_iterations + 1) / max_retry, 0.5)




# ---------------------------------------------------------------------------
# Dimension Pre-Processing Utilities
# ---------------------------------------------------------------------------

def parse_dimension_text(text):
    """Parse architectural dimension text to feet.
    Handles: 12'-6" → 12.5, 12' → 12.0, 6" → 0.5
    Returns float in feet, or None if unparseable.
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    # Try feet-inches format: 12'-6", 12'6"
    match = re.match(r"(\d+)'-?(\d+)?\"?", text)
    if match:
        feet = int(match.group(1))
        inches = int(match.group(2) or 0)
        return round(feet + inches / 12, 2)
    # Try inches-only format: 6"
    match = re.match(r"(\d+)\"", text)
    if match:
        return round(int(match.group(1)) / 12, 2)
    return None


def point_to_line_distance(point, wall_endpoints):
    """Perpendicular distance from point (x, y) to line segment [x1, y1, x2, y2].
    Returns distance in pixels.
    """
    px, py = point
    x1, y1, x2, y2 = wall_endpoints
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def extract_wall_dimension_candidates(wall_endpoints, ocr_entries, pixel_aspect_ratio, max_distance=150):
    """Pre-compute likely dimensions for a wall from nearby OCR text.
    
    Args:
        wall_endpoints: [x1, y1, x2, y2]
        ocr_entries: dict {text: (centroid_x, centroid_y)}
        pixel_aspect_ratio: dict with 'horizontal' and 'vertical' keys
        max_distance: max pixel distance for OCR to be considered
    
    Returns:
        List of top 3 candidates sorted by confidence:
        [{"dimension_ft": 12.5, "confidence": "high", "source_text": "12'-6\""}, ...]
    """
    x1, y1, x2, y2 = wall_endpoints
    wall_length_px = math.hypot(x2 - x1, y2 - y1)
    if wall_length_px == 0:
        return []

    candidates = []
    for text, centroid in ocr_entries.items():
        cx, cy = centroid[0], centroid[1]
        dim_ft = parse_dimension_text(text)
        if dim_ft is None:
            continue
        dist = point_to_line_distance((cx, cy), wall_endpoints)
        if dist > max_distance:
            continue
        # Estimate expected pixel length from parsed dimension
        dim_px_expected = dim_ft / pixel_aspect_ratio.get("horizontal", 0.07)
        dimension_match = abs(dim_px_expected - wall_length_px) / max(wall_length_px, 1)

        if dist < 50 and dimension_match < 0.10:
            confidence = "high"
        elif dist < 100 and dimension_match < 0.25:
            confidence = "medium"
        else:
            confidence = "low"

        candidates.append({
            "dimension_ft": dim_ft,
            "confidence": confidence,
            "distance_px": round(dist, 1),
            "source_text": text
        })

    confidence_order = {"high": 0, "medium": 1, "low": 2}
    candidates.sort(key=lambda c: (confidence_order[c["confidence"]], c["distance_px"]))
    return candidates[:3]
