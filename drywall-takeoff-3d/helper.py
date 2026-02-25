import json
import hashlib
from pathlib import Path
import cv2
from json.decoder import JSONDecodeError

# from google.cloud import bigquery
from google.cloud.storage import Client as CloudStorageClient
import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content

from prompt import ARCHITECTURAL_DRAWING_CLASSIFIER
import asyncpg
import asyncio


_pg_pool = None

async def get_pg_pool(credentials):
    global _pg_pool
    if _pg_pool is None:
        _pg_pool = await asyncpg.create_pool(
            host=credentials["PostgreSQL"]["host"],
            port=credentials["PostgreSQL"]["port"],
            database=credentials["PostgreSQL"]["database"],
            user=credentials["PostgreSQL"]["user"],
            password=credentials["PostgreSQL"]["password"],
            min_size=2,
            max_size=10
        )
    return _pg_pool

def load_pg_pool_sync(credentials):
    """Synchronous wrapper for startup — creates the pool once."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(get_pg_pool(credentials))

# def load_bigquery_client(credentials):
#     bigquery_client = bigquery.Client.from_service_account_json(credentials["GBQServer"]["service_account_key"])
#     return bigquery_client



# def bigquery_run(credentials, bigquery_client, GBQ_query, job_config=dict()):
#     job_config = bigquery.QueryJobConfig(
#         destination_encryption_configuration=bigquery.EncryptionConfiguration(
#             kms_key_name=credentials["GBQServer"]["KMS_key"]
#         ),
#         **job_config
#     )
#     query_output = bigquery_client.query(GBQ_query, job_config=job_config)
#     return query_output

def sha256(path, chunk_size=8192):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def upload_floorplan(plan_path, user_id, plan_id, project_id, credentials, index=None, directory=None):
    client = CloudStorageClient()
    page_number = Path(plan_path.stem).suffix
    if page_number:
        blob_object_name = Path(str(plan_path).replace(page_number, '')).name
    else:
        blob_object_name = plan_path.name
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    if directory:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/{index}/{directory}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/{directory}/{blob_object_name}"
    else:
        if index:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/{index}/{blob_object_name}"
        else:
            blob_path = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/{blob_object_name}"
    blob = bucket.blob(blob_path)

    blob.upload_from_filename(plan_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"


async def insert_model_2d(
    model_2d, scale, page_number, plan_id, user_id, project_id,
    GCS_URL_floorplan_page, GCS_URL_target_drywalls_page,
    pg_pool, credentials
):
    if not model_2d.get("metadata", None):
        row = await pg_pool.fetchrow(
            """SELECT model_2d->'metadata' AS metadata FROM models
               WHERE LOWER(project_id) = LOWER($1)
               AND LOWER(plan_id) = LOWER($2)
               AND page_number = $3""",
            project_id, plan_id, page_number
        )
        if row and row["metadata"]:
            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            model_2d["metadata"] = metadata

    await pg_pool.execute(
        """INSERT INTO models (plan_id, project_id, user_id, page_number, scale,
               model_2d, model_3d, takeoff, source, target_drywalls, created_at, updated_at)
           VALUES ($1, $2, $3, $4, $5, $6::jsonb, '{}'::jsonb, '{}'::jsonb, $7, $8, now(), now())
           ON CONFLICT (project_id, plan_id, page_number)
           DO UPDATE SET
               model_2d = $6::jsonb,
               scale = COALESCE(NULLIF($5, ''), models.scale),
               user_id = $3,
               updated_at = now()""",
        plan_id, project_id, user_id, page_number, scale,
        json.dumps(model_2d), GCS_URL_floorplan_page, GCS_URL_target_drywalls_page
    )


async def is_duplicate(pg_pool, credentials, pdf_path, project_id):
    sha_256 = sha256(pdf_path)
    rows = await pg_pool.fetch(
        "SELECT plan_id, sha256, status FROM plans WHERE LOWER(project_id) = LOWER($1)",
        project_id
    )
    for plan_target in rows:
        if not plan_target["sha256"]:
            await delete_plan(credentials, pg_pool, plan_target["plan_id"], project_id)
            continue
        if plan_target["sha256"] == sha_256:
            if plan_target["status"] == "FAILED":
                await delete_plan(credentials, pg_pool, plan_target["plan_id"], project_id)
                return False
            return plan_target["plan_id"]
    return False
# def is_duplicate(bigquery_client, credentials, pdf_path, project_id):
#     sha_256 = sha256(pdf_path)
#     GBQ_query = f"SELECT plan_id, sha256, status FROM `drywall_takeoff.plans` WHERE LOWER(project_id) = LOWER('{project_id}');"
#     query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
#     for plan_target in list(query_output):
#         if not plan_target.sha256:
#             delete_plan(credentials, bigquery_client, plan_target.plan_id, project_id)
#             continue
#         if plan_target.sha256 == sha_256:
#             if plan_target.status == "FAILED":
#                 delete_plan(credentials, bigquery_client, plan_target.plan_id, project_id)
#                 return False
#             return plan_target.plan_id
#     return False

# def delete_plan(credentials, bigquery_client, plan_id, project_id):
#     GBQ_query = f"DELETE FROM `drywall_takeoff.plans` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}');"
#     query_output = bigquery_run(credentials, bigquery_client, GBQ_query).result()
#     return query_output

async def delete_plan(credentials, pg_pool, plan_id, project_id):
    await pg_pool.execute(
        "DELETE FROM plans WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        project_id, plan_id
    )

def load_floorplan_to_structured_2d_ID_token(credentials):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_drywall_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["floorplan_to_structured_2d"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token
    return id_token

def load_vertex_ai_client(credentials, region="us-central1"):
    with open(credentials["VertexAI"]["service_account_key"], 'r') as f:
        project_id = json.load(f)["project_id"]
    vertexai.init(project=project_id, location=region)
    vertex_ai_client = GenerativeModel(credentials["VertexAI"]["llm"]["model_name"])
    generation_config = credentials["VertexAI"]["llm"]["parameters"]
    return vertex_ai_client, generation_config

def classify_plan(plan_path, vertex_ai_client):
    plan_BGR = cv2.imread(plan_path)
    _, canvas_buffer_array = cv2.imencode(".png", plan_BGR)
    bytes_canvas = canvas_buffer_array.tobytes()
    system = Content(role="model", parts=[Part.from_text(ARCHITECTURAL_DRAWING_CLASSIFIER)])
    query = Content(role="user", parts=[
        Part.from_data(data=bytes_canvas, mime_type="image/png")
    ])
    response = vertex_ai_client(contents=[system, query])
    try:
        plan_type = json.loads(response.text.strip("`json").replace("{{", '{').replace("}}", '}'))
    except (JSONDecodeError, ValueError):
        plan_type = dict(plan_type="FLOOR_PLAN")

    return plan_type
