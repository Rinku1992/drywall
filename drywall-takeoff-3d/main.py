import os
import sys
import logging
from functools import partial
from datetime import timedelta, datetime, date, time
from decimal import Decimal
from base64 import b64encode
from ruamel.yaml import YAML
from pathlib import Path
import json
from time import time as from_unix_epoch
from time import sleep
from collections import defaultdict
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pydantic_core import ValidationError
from concurrent.futures import ThreadPoolExecutor

from google.cloud.storage import Client as CloudStorageClient
# from google.cloud import secretmanager
import asyncio
import pandas as pd
import numpy as np
import math
from preprocessing import preprocess
from extrapolate_3d import Extrapolate3D
from helper import (
    get_pg_pool,
    sha256,
    upload_floorplan,
    insert_model_2d,
    is_duplicate,
    delete_plan,
    load_floorplan_to_structured_2d_ID_token,
    load_vertex_ai_client,
    classify_plan,
)


def respond_with_UI_payload(payload, status_code=200):
    return JSONResponse(
        content=json.loads(json.dumps(payload)),
        status_code=status_code,
        media_type="application/json",
    )


def download_floorplan(user_id, plan_id, project_id, credentials, destination_path="/tmp/floor_plan.PDF"):
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)

    blob.download_to_filename(destination_path)
    return f"gs://{credentials["CloudStorage"]["bucket_name"]}/{blob_path}"

async def insert_model_2d_revision(model_2d, scale, page_number, plan_id, user_id, project_id, pg_pool, credentials):
    if not model_2d.get("metadata", None):
        row = await pg_pool.fetchrow(
            """SELECT model_2d->'metadata' AS metadata FROM models
               WHERE LOWER(project_id) = LOWER($1)
               AND LOWER(plan_id) = LOWER($2) AND page_number = $3""",
            project_id, plan_id, page_number
        )
        if row and row["metadata"]:
            metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
            model_2d["metadata"] = metadata

    row = await pg_pool.fetchrow(
        """SELECT COALESCE(MAX(revision_number), 0) AS rev
           FROM model_revisions_2d
           WHERE LOWER(project_id) = LOWER($1)
           AND LOWER(plan_id) = LOWER($2) AND page_number = $3""",
        project_id, plan_id, page_number
    )
    revision_number = row["rev"] + 1

    await pg_pool.execute(
        """INSERT INTO model_revisions_2d
           (plan_id, project_id, user_id, page_number, scale, model, created_at, revision_number)
           VALUES ($1, $2, $3, $4, $5, $6::jsonb, now(), $7)""",
        plan_id, project_id, user_id, page_number, scale,
        json.dumps(model_2d), revision_number
    )

async def insert_model_3d_revision(model_3d, scale, page_number, plan_id, user_id, project_id, pg_pool, credentials):
    row = await pg_pool.fetchrow(
        """SELECT COALESCE(MAX(revision_number), 0) AS rev
           FROM model_revisions_3d
           WHERE LOWER(project_id) = LOWER($1)
           AND LOWER(plan_id) = LOWER($2) AND page_number = $3""",
        project_id, plan_id, page_number
    )
    revision_number = row["rev"] + 1

    await pg_pool.execute(
        """INSERT INTO model_revisions_3d
           (plan_id, project_id, user_id, page_number, scale, model, takeoff, created_at, revision_number)
           VALUES ($1, $2, $3, $4, $5, $6::jsonb, '{}'::jsonb, now(), $7)""",
        plan_id, project_id, user_id, page_number, scale,
        json.dumps(model_3d), revision_number
    )


async def insert_model_3d(model_3d, scale, page_number, plan_id, user_id, project_id, pg_pool, credentials):
    await pg_pool.execute(
        """UPDATE models SET
               model_3d = $1::jsonb,
               scale = COALESCE(NULLIF($2, ''), scale),
               user_id = $3,
               updated_at = now()
           WHERE LOWER(project_id) = LOWER($4)
           AND LOWER(plan_id) = LOWER($5)
           AND page_number = $6""",
        json.dumps(model_3d), scale, user_id, project_id, plan_id, page_number
    )

async def delete_floorplan(project_id, plan_id, user_id, pg_pool, credentials):
    # Use a transaction — all deletes succeed or none do
    async with pg_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM plans WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND LOWER(user_id)=LOWER($3)",
                project_id, plan_id, user_id
            )
            await conn.execute(
                "DELETE FROM models WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND LOWER(user_id)=LOWER($3)",
                project_id, plan_id, user_id
            )
            await conn.execute(
                "DELETE FROM model_revisions_2d WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND LOWER(user_id)=LOWER($3)",
                project_id, plan_id, user_id
            )
            await conn.execute(
                "DELETE FROM model_revisions_3d WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND LOWER(user_id)=LOWER($3)",
                project_id, plan_id, user_id
            )
    # GCS deletion — NO CHANGE, stays exactly as-is
    client = CloudStorageClient()
    bucket = client.bucket(credentials["CloudStorage"]["bucket_name"])
    prefix = f"{project_id.lower()}/{plan_id.lower()}/{user_id.lower()}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        bucket.delete_blobs(blobs)

async def insert_takeoff(takeoff, page_number, plan_id, user_id, project_id, revision_number, pg_pool, credentials):
    await pg_pool.execute(
        """UPDATE models SET takeoff = $1::jsonb, updated_at = now(), user_id = $2
           WHERE LOWER(project_id)=LOWER($3) AND LOWER(plan_id)=LOWER($4) AND page_number=$5""",
        json.dumps(takeoff), user_id, project_id, plan_id, page_number
    )
    if revision_number:
        await pg_pool.execute(
            """UPDATE model_revisions_3d SET takeoff = $1::jsonb, user_id = $2
               WHERE LOWER(project_id)=LOWER($3) AND LOWER(plan_id)=LOWER($4)
               AND page_number=$5 AND revision_number=$6""",
            json.dumps(takeoff), user_id, project_id, plan_id, page_number, revision_number
        )

async def insert_plan(
    project_id, user_id, status, pg_pool, credentials,
    payload_plan=None, plan_id=None, size_in_bytes=None,
    GCS_URL_floorplan=None, n_pages=None
):
    sha_256 = ''
    if plan_id:
        pdf_path = Path("/tmp/floor_plan.PDF")
        download_floorplan(user_id, plan_id, project_id, credentials, destination_path=pdf_path)
        sha_256 = sha256(pdf_path)
    if not plan_id:
        plan_id = payload_plan.plan_id
    plan_name = payload_plan.plan_name if payload_plan else ''
    plan_type = payload_plan.plan_type if payload_plan else ''
    file_type = payload_plan.file_type if payload_plan else ''
    n_pages = n_pages or 0
    GCS_URL_floorplan = GCS_URL_floorplan or ''
    size_in_bytes = size_in_bytes or 0

    await pg_pool.execute(
        """INSERT INTO plans (plan_id, project_id, user_id, status, plan_name, plan_type,
               file_type, pages, size_in_bytes, source, sha256, created_at, updated_at)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11, now(), now())
           ON CONFLICT (project_id, plan_id)
           DO UPDATE SET
               pages=$8, source=$10, sha256=$11, status=$4,
               size_in_bytes=$9, user_id=$3, updated_at=now()""",
        plan_id, project_id, user_id, status, plan_name, plan_type,
        file_type, n_pages, size_in_bytes, GCS_URL_floorplan, sha_256
    )


async def insert_project(payload_project, pg_pool, credentials):
    row = await pg_pool.fetchrow(
        """INSERT INTO projects (project_id, project_name, project_location, fbm_branch,
               project_type, project_area, contractor_name, created_at, created_by)
           VALUES ($1,$2,$3,$4,$5,$6,$7, now(), $8)
           ON CONFLICT (project_id) DO NOTHING
           RETURNING created_at""",
        payload_project.project_id, payload_project.project_name,
        payload_project.project_location, payload_project.FBM_branch,
        payload_project.project_type, payload_project.project_area,
        payload_project.contractor_name, payload_project.created_by
    )
    if row:
        return row["created_at"].isoformat()
    # If already existed (DO NOTHING), fetch created_at
    row = await pg_pool.fetchrow(
        "SELECT created_at FROM projects WHERE project_id = $1",
        payload_project.project_id
    )
    return row["created_at"].isoformat()


def floorplan_to_structured_2d(credentials, id_token, project_id, plan_id, user_id, page_number):
    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }
    requests.post(
        f"{credentials["CloudRun"]["APIs"]["floorplan_to_structured_2d"]}/floorplan_to_structured_2d",
        headers=headers,
        json=dict(
            project_id=project_id,
            plan_id=plan_id,
            user_id=user_id,
            page_number=page_number
        ),
        timeout=(10, 7200)
    )


def load_UI_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.map(lambda x: float(x) if isinstance(x, Decimal) else x)
    df = df.map(lambda x: "null" if isinstance(x, int) and (pd.isna(x) or math.isnan(x) or math.isinf(x) or np.isnan(x) or np.isinf(x)) else x)
    df = df.map(lambda x: "null" if isinstance(x, float) and (pd.isna(x) or math.isnan(x) or math.isinf(x) or np.isnan(x) or np.isinf(x)) else x)
    df = df.map(lambda x: x.date().isoformat() if isinstance(x, datetime) else x)
    df = df.map(lambda x: x.isoformat() if isinstance(x, date) else x)
    df = df.map(lambda x: x.isoformat() if isinstance(x, time) else x)
    df = df.map(lambda x: b64encode(x).decode("utf-8") if isinstance(x, bytes) else x)

    return df


def enable_logging_on_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format='{"severity": "%(levelname)s", "message": "%(message)s"}',
        stream=sys.stdout
    )


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


app = FastAPI(title="Drywall Takeoff (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# bigquery_client = load_bigquery_client(CREDENTIALS)
pg_pool = None 
@app.on_event("startup")
async def startup():
    global pg_pool
    pg_pool = await get_pg_pool(CREDENTIALS)

@app.on_event("shutdown")
async def shutdown():
    global pg_pool
    if pg_pool:
        await pg_pool.close()

class PayloadProject(BaseModel):
    project_id: str
    project_name: str
    project_location: str
    project_area: str
    project_type: str
    contractor_name: str
    FBM_branch: str
    created_by: str

class PayloadPlan(BaseModel):
    plan_id: str
    plan_name: str
    plan_type: str
    file_type: str


@app.post("/generate_project")
async def generate_project(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    try:
        payload_project = PayloadProject(**parameters)
    except ValidationError:
        payload_project = PayloadProject(**body)
    # created_at = insert_project(payload_project, bigquery_client, CREDENTIALS)
    created_at = await insert_project(payload_project, pg_pool, CREDENTIALS)
    logging.info(f"SYSTEM: New Project {payload_project.project_name} generated successfully")
    return respond_with_UI_payload(
        dict(
            project_id=payload_project.project_id,
            project_name=payload_project.project_name,
            created_at=created_at,
        )
    )


@app.post("/load_projects")
async def load_projects(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()

    # GBQ_query = f"SELECT * FROM `{CREDENTIALS["GBQServer"]["table_name_projects"]}`"
    # projects = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())
    rows = await pg_pool.fetch("SELECT * FROM projects")
    projects = [dict(r) for r in rows]
    for project in projects:
        if project.get("created_at"):
            project["created_at"] = project["created_at"].isoformat()
    logging.info("SYSTEM: Project Metaaata retrieved successfully")
    return respond_with_UI_payload(
        jsonable_encoder({
            "projects": projects
        })
    )

@app.post("/load_project_plans")
async def load_project_plans(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")

    project_row = await pg_pool.fetchrow(
        "SELECT * FROM projects WHERE LOWER(project_id) = LOWER($1)",
        project_id
    )
    if not project_row:
        return respond_with_UI_payload(dict(project_metadata=dict(), project_plans=list()))

    project_metadata = dict(project_row)
    if project_metadata.get("created_at"):
        project_metadata["created_at"] = project_metadata["created_at"].isoformat()

    plan_rows = await pg_pool.fetch(
        "SELECT * FROM plans WHERE LOWER(project_id) = LOWER($1)",
        project_id
    )
    project_plans = [dict(r) for r in plan_rows]
    for plan in project_plans:
        if plan.get("created_at"):
            plan["created_at"] = plan["created_at"].isoformat()
        if plan.get("updated_at"):
            plan["updated_at"] = plan["updated_at"].isoformat()

    logging.info("SYSTEM: Project Plans Data retrieved successfully")
    return respond_with_UI_payload(
        jsonable_encoder({
            "project_metadata": project_metadata,
            "project_plans": project_plans
        })
    )

@app.post("/generate_floorplan_upload_signed_URL")
async def generate_floorplan_upload_signed_URL(request: Request) -> str:
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    payload_plan = parameters.get("plan") or body.get("plan")
    user_id = parameters.get("user_id") or body.get("user_id")
    payload_plan = PayloadPlan(**payload_plan)
    logging.info("SYSTEM: Received Signed Floorplan upload URL generation Request")

    await insert_plan(
        project_id,
        user_id,
        "NOT STARTED",
        pg_pool,
        CREDENTIALS,
        payload_plan=payload_plan
    )

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"{project_id.lower()}/{payload_plan.plan_id.lower()}/{user_id.lower()}/floor_plan.PDF"
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="PUT",
        content_type="application/octet-stream",
    )

    return url

@app.post("/load_plan_pages")
async def load_plan_pages(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")

    rows = await pg_pool.fetch(
        "SELECT * FROM models WHERE LOWER(project_id) = LOWER($1) AND LOWER(plan_id) = LOWER($2)",
        project_id, plan_id
    )
    plan_pages_data = [dict(r) for r in rows]
    plan_metadata = plan_pages_data[0] if plan_pages_data else dict()

    # Convert datetime objects for JSON serialization
    for page in plan_pages_data:
        if page.get("created_at"):
            page["created_at"] = page["created_at"].isoformat()
        if page.get("updated_at"):
            page["updated_at"] = page["updated_at"].isoformat()

    logging.info(f"SYSTEM: Plan Pages Data retrieved successfully")
    return respond_with_UI_payload(dict(plan_metadata=plan_metadata, plan_pages=plan_pages_data))


@app.post("/floorplan_to_2d")
async def floorplan_to_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    verbose = parameters.get("verbose") or body.get("verbose")
    logging.info("SYSTEM: Received a Floorplan 2D Model Generation Request")

    pdf_path = Path("/tmp/floor_plan.PDF")
    GCS_URL_floorplan = download_floorplan(user_id, plan_id, project_id, CREDENTIALS, destination_path=pdf_path)
    logging.info("SYSTEM: Floorplan Downloaded")
    # plan_duplicate = is_duplicate(bigquery_client, CREDENTIALS, pdf_path, project_id)
    # if plan_duplicate:
    #     delete_plan(CREDENTIALS, bigquery_client, plan_id, project_id)
    #     return respond_with_UI_payload(dict(error="Floor Plan already exists"))
    
    plan_duplicate = await is_duplicate(pg_pool, CREDENTIALS, pdf_path, project_id)
    if plan_duplicate:
        await delete_plan(CREDENTIALS, pg_pool, plan_id, project_id)
        return respond_with_UI_payload(dict(error="Floor Plan already exists"))

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob_path = f"tmp/{user_id.lower()}/{project_id.lower()}/{plan_id.lower()}/floorplan_structured_2d.json"
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.delete()

    size_in_bytes = Path(pdf_path).stat().st_size
    floor_plan_paths_vector, floor_plan_paths_preprocessed = preprocess(pdf_path)

    await insert_plan(
        project_id,
        user_id,
        "IN PROGRESS",
        pg_pool,
        CREDENTIALS,
        plan_id=plan_id,
        size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=len(floor_plan_paths_preprocessed),
    )
   
    logging.info("SYSTEM: Floorplan Preprocessing Completed")

    walls_2d_all = dict(pages=list())
    status = "COMPLETED"
    vertex_ai_client, generation_config = load_vertex_ai_client(CREDENTIALS)
    vertex_ai_client_partial = partial(vertex_ai_client.generate_content, generation_config=generation_config)
    try:
        id_token = load_floorplan_to_structured_2d_ID_token(CREDENTIALS)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = list()
            floorplan_baseline_page_sources = list()
            floorplan_page_sources = list()
            plan_types = list()
            for index, (floor_plan_vector, floor_plan_path) in enumerate(zip(floor_plan_paths_vector, floor_plan_paths_preprocessed)):
                plan_type = classify_plan(floor_plan_path, vertex_ai_client_partial)
                plan_types.append(plan_type)
                floorplan_baseline_page_source = upload_floorplan(floor_plan_vector, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
                floorplan_baseline_page_sources.append(floorplan_baseline_page_source)
                floorplan_page_source = upload_floorplan(floor_plan_path, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
                floorplan_page_sources.append(floorplan_page_source)
                if plan_type["plan_type"].upper().find("FLOOR") == -1:
                    continue
                futures.append(
                    executor.submit(
                        floorplan_to_structured_2d,
                        CREDENTIALS,
                        id_token,
                        project_id,
                        plan_id,
                        user_id,
                        index
                    )
                )
            for page_number, (plan_type, _, floorplan_page_source) in enumerate(zip(plan_types, floorplan_baseline_page_sources, floorplan_page_sources)):
                if plan_type["plan_type"].upper().find("FLOOR") == -1:
                    continue
                timeout = from_unix_epoch() + 7200
                while from_unix_epoch() < timeout:
                 
                    row = await pg_pool.fetchrow(
                        "SELECT scale, model_2d FROM models WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND page_number=$3",
                        project_id, plan_id, page_number
                    )
                    # if query_output:
                    #     break
                    # sleep(2)
                    if row:
                        break
                    await asyncio.sleep(2)
                # walls_2d = json.loads(query_output[0].model_2d) if isinstance(query_output[0].model_2d, str) else query_output[0].model_2d
                walls_2d = json.loads(row["model_2d"]) if isinstance(row["model_2d"], str) else row["model_2d"]
                
                if not walls_2d["polygons"] or not walls_2d["walls_2d"]:
                    # GBQ_query = f"DELETE FROM `{CREDENTIALS["GBQServer"]["table_name_models"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
                    # bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()
                    await pg_pool.execute(
                        "DELETE FROM models WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND page_number=$3",
                        project_id, plan_id, page_number
                    )
                    continue
                # GBQ_query = f"UPDATE `{CREDENTIALS["GBQServer"]["table_name_models"]}` SET source = '{floorplan_page_source}' WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
                # bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result()
                await pg_pool.execute(
                        "UPDATE models SET source = $1 WHERE LOWER(project_id)=LOWER($2) AND LOWER(plan_id)=LOWER($3) AND page_number=$4",
                        floorplan_page_source, project_id, plan_id, page_number
                    )
                page = dict(
                    plan_id=plan_id,
                    page_number=page_number,
                    page_type=plan_type["plan_type"].upper(),
                    # scale=query_output[0].scale,
                    scale=row["scale"],
                    walls_2d=walls_2d["walls_2d"],
                    polygons=walls_2d["polygons"],
                    **walls_2d["metadata"]
                )

                walls_2d_all["pages"].append(page)
    except Exception as e:
        logging.info(f"SYSTEM: Floorplan extraction failed with error: {e}")
        status = "FAILED"

    await insert_plan(
        project_id,
        user_id,
        status,
        pg_pool,
        CREDENTIALS,
        plan_id=plan_id,
        size_in_bytes=size_in_bytes,
        GCS_URL_floorplan=GCS_URL_floorplan,
        n_pages=len(floor_plan_paths_preprocessed),
    )

    with open("/tmp/floorplan_structured_2d.json", 'w') as f:
        json.dump(walls_2d_all, f, indent=4)
    blob.upload_from_filename("/tmp/floorplan_structured_2d.json")
    return respond_with_UI_payload(walls_2d_all)


@app.post("/load_2d_revision")
async def load_2d_revision(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    revision_number = parameters.get("revision_number") or body.get("revision_number")
    logging.info(f"SYSTEM: Received Floorplan 2D Model (Revision: {revision_number}) Load Request")

    # GBQ_query = f"SELECT model FROM `{CREDENTIALS["GBQServer"]["table_name_model_revisions_2d"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number} AND revision_number = {revision_number};"
    # query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())
    # walls_2d_JSON = dict()
    # if query_output and query_output[0].model is not None:
    #     walls_2d_JSON = json.loads(query_output[0].model)
    row = await pg_pool.fetchrow(
        """SELECT model FROM model_revisions_2d
        WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)
        AND page_number=$3 AND revision_number=$4""",
        project_id, plan_id, int(page_number), int(revision_number)
    )
    walls_2d_JSON = dict()
    if row and row["model"] is not None:
        walls_2d_JSON = json.loads(row["model"]) if isinstance(row["model"], str) else row["model"]

    return respond_with_UI_payload(walls_2d_JSON)


@app.post("/load_available_revision_numbers_2d")
async def load_available_revision_numbers_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info(f"SYSTEM: Received Available Revisions Load Request for 2D Model")

    # GBQ_query = f"SELECT revision_number FROM `{CREDENTIALS["GBQServer"]["table_name_model_revisions_2d"]}` WHERE LOWER(project_id) = LOWER('{project_id}') AND LOWER(plan_id) = LOWER('{plan_id}') AND page_number = {page_number};"
    # query_output = list(bigquery_run(CREDENTIALS, bigquery_client, GBQ_query).result())

    # revision_numbers = list()
    # if query_output:
    #     for revision in query_output:
    #         if revision.revision_number is not None:
    #             revision_numbers.append(revision.revision_number)
    
    rows = await pg_pool.fetch(
        "SELECT revision_number FROM model_revisions_2d WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND page_number=$3",
        project_id, plan_id, int(page_number)
    )
    revision_numbers = [r["revision_number"] for r in rows if r["revision_number"] is not None]

    return respond_with_UI_payload(revision_numbers)

@app.post("/load_2d_all")
async def load_2d_all(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received All Floorplan 2D Models Load Request")

    status = "IN PROGRESS"
    row = await pg_pool.fetchrow(
        "SELECT pages FROM plans WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)",
        project_id, plan_id
    )
    if not row:
        return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
    n_pages = row["pages"] if row else 0

    timeout = from_unix_epoch() + (n_pages * 120)
    while from_unix_epoch() < timeout:
        status_row = await pg_pool.fetchrow(
            "SELECT status FROM plans WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)",
            project_id, plan_id
        )
        if not status_row:
            return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
        status = status_row["status"]
        if status == "COMPLETED":
            break
        await asyncio.sleep(2)

    if status != "COMPLETED":
        return respond_with_UI_payload(dict(error="Floor Plan extraction not completed within 15 minutes"), status_code=500)

    walls_2d_all = dict(pages=list())
    rows = await pg_pool.fetch(
        """SELECT page_number, scale, model_2d FROM models
           WHERE project_id=$1 AND plan_id=$2 ORDER BY page_number""",
        project_id, plan_id
    )
    for row in rows:
        if not row["model_2d"]:
            continue
        walls_2d = json.loads(row["model_2d"]) if isinstance(row["model_2d"], str) else row["model_2d"]
        page = {
            "plan_id": plan_id,
            "page_number": row["page_number"],
            "scale": row["scale"],
            "walls_2d": walls_2d.get("walls_2d", list()),
            "polygons": walls_2d.get("polygons", list()),
            **walls_2d.get("metadata", dict()),
        }
        walls_2d_all["pages"].append(page)

    return respond_with_UI_payload(walls_2d_all)


@app.post("/update_floorplan_to_2d")
async def update_floorplan_to_2d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    polygons_JSON = parameters.get("polygons") or body.get("polygons")
    scale = parameters.get("scale") or body.get("scale")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    index = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Floorplan 2D Model Update Request")

    # insert_model_2d(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, None, None, bigquery_client, CREDENTIALS)
    # insert_model_2d_revision(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, bigquery_client, CREDENTIALS)
    await insert_model_2d(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, None, None, pg_pool, CREDENTIALS)
    await insert_model_2d_revision(dict(walls_2d=walls_2d_JSON, polygons=polygons_JSON), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
    
    logging.info("SYSTEM: Floorplan 2D Model Updated Successfully")

    logging.info("SYSTEM: Generating Floorplan 3D Model")
    model_2d_path = "/tmp/walls_2d.json"
    with open(model_2d_path, 'w') as f:
        json.dump(walls_2d_JSON, f)
    polygons_path = "/tmp/polygons.json"
    with open(polygons_path, 'w') as f:
        json.dump(polygons_JSON, f)
    hyperparameters = load_hyperparameters()
    floor_plan_modeller_3d = Extrapolate3D(hyperparameters)
    walls_3d, polygons_3d, walls_3d_path, polygons_3d_path = floor_plan_modeller_3d.extrapolate(model_2d_path=model_2d_path, polygons_path=polygons_path)
    walls_3d, polygons_3d = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d, polygons_3d)
    gltf_paths = floor_plan_modeller_3d.gltf(model_2d_path=model_2d_path, polygons_path=polygons_path)
    model_3d_path = floor_plan_modeller_3d.save_plot_3d(walls_3d_path, polygons_3d_path)
    upload_floorplan(model_3d_path, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
    for gltf_path in gltf_paths:
        upload_floorplan(gltf_path, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2), directory="gltf")
    
    # insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, bigquery_client, CREDENTIALS)
    await insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)

    logging.info("SYSTEM: A 3D Model of the Floorplan Generated Successfully")

    return respond_with_UI_payload(dict(walls_3d=walls_3d, polygons=polygons_3d))


@app.post("/floorplan_to_3d")
async def floorplan_to_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_2d_JSON = parameters.get("walls_2d") or body.get("walls_2d")
    polygons_JSON = parameters.get("polygons") or body.get("polygons")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    scale = parameters.get("scale") or body.get("scale")
    index = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Floorplan 3D Model Generation Request")

    model_2d_path = "/tmp/walls_2d.json"
    with open(model_2d_path, 'w') as f:
        json.dump(walls_2d_JSON, f)
    polygons_path = "/tmp/polygons.json"
    with open(polygons_path, 'w') as f:
        json.dump(polygons_JSON, f)
    hyperparameters = load_hyperparameters()
    floor_plan_modeller_3d = Extrapolate3D(hyperparameters)
    walls_3d, polygons_3d, walls_3d_path, polygons_3d_path = floor_plan_modeller_3d.extrapolate(model_2d_path=model_2d_path, polygons_path=polygons_path)
    walls_3d, polygons_3d = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d, polygons_3d)
    gltf_paths = floor_plan_modeller_3d.gltf(model_2d_path=model_2d_path, polygons_path=polygons_path)
    model_3d_path = floor_plan_modeller_3d.save_plot_3d(walls_3d_path, polygons_3d_path)
    upload_floorplan(model_3d_path, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2))
    for gltf_path in gltf_paths:
        upload_floorplan(gltf_path, user_id, plan_id, project_id, CREDENTIALS, index=str(index).zfill(2), directory="gltf")
    # insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, bigquery_client, CREDENTIALS)
    await insert_model_3d(dict(walls_3d=walls_3d, polygons=polygons_3d), scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: A 3D Model of the Floorplan Generated Successfully")

    return respond_with_UI_payload(dict(walls_3d=walls_3d, polygons=polygons_3d))

@app.post("/load_3d_revision")
async def load_3d_revision(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    revision_number = parameters.get("revision_number") or body.get("revision_number")
    logging.info(f"SYSTEM: Received Floorplan 3D Model (Revision: {revision_number}) Load Request")

    row = await pg_pool.fetchrow(
        """SELECT model FROM model_revisions_3d
           WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)
           AND LOWER(user_id)=LOWER($3) AND page_number=$4 AND revision_number=$5""",
        project_id, plan_id, user_id, int(page_number), int(revision_number)
    )
    walls_3d_JSON = dict()
    if row and row["model"] is not None:
        walls_3d_JSON = json.loads(row["model"]) if isinstance(row["model"], str) else row["model"]

    return respond_with_UI_payload(walls_3d_JSON)

@app.post("/load_available_revision_numbers_3d")
async def load_available_revision_numbers_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    page_number = parameters.get("page_number") or body.get("page_number")
    logging.info(f"SYSTEM: Received Available Revisions Load Request for 3D Model")

    rows = await pg_pool.fetch(
        """SELECT revision_number FROM model_revisions_3d
           WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)
           AND LOWER(user_id)=LOWER($3) AND page_number=$4""",
        project_id, plan_id, user_id, int(page_number)
    )
    revision_numbers = [r["revision_number"] for r in rows if r["revision_number"] is not None]

    return respond_with_UI_payload(revision_numbers)
@app.post("/load_3d_all")
async def load_3d_all(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received All Floorplan 3D Models Load Request")

    walls_3d_all = dict(pages=list())
    rows = await pg_pool.fetch(
        """SELECT page_number, scale, model_3d FROM models
           WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)
           AND LOWER(user_id)=LOWER($3)""",
        project_id, plan_id, user_id
    )
    for row in rows:
        if not row["model_3d"]:
            continue
        walls_3d = json.loads(row["model_3d"]) if isinstance(row["model_3d"], str) else row["model_3d"]
        page = dict(page_number=row["page_number"], walls_3d=walls_3d, page_name='', scale=row["scale"])
        walls_3d_all["pages"].append(page)

    return respond_with_UI_payload(walls_3d_all)


@app.post("/update_floorplan_to_3d")
async def update_floorplan_to_3d(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_3d_JSON = parameters.get("walls_3d") or body.get("walls_3d")
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    scale = parameters.get("scale") or body.get("scale")
    index = parameters.get("page_number") or body.get("page_number")
    logging.info("SYSTEM: Received a Floorplan 3D Model Update Request")

    await insert_model_3d(walls_3d_JSON, scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
    await insert_model_3d_revision(walls_3d_JSON, scale, index, plan_id, user_id, project_id, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: Floorplan 3D Model Updated Successfully")

@app.post("/generate_drywall_overlaid_floorplan_download_signed_URL")
async def generate_drywall_overlaid_floorplan_download_signed_URL(request: Request) -> str:
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    index = parameters.get("page_number") or body.get("page_number")
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received Signed Floorplan download URL generation Request")

    status = "IN PROGRESS"
    row = await pg_pool.fetchrow(
        "SELECT pages FROM plans WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)",
        project_id, plan_id
    )
    if not row:
        return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
    n_pages = row["pages"]

    timeout = from_unix_epoch() + (n_pages * 120)
    while from_unix_epoch() < timeout:
        status_row = await pg_pool.fetchrow(
            "SELECT status FROM plans WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)",
            project_id, plan_id
        )
        if not status_row:
            return respond_with_UI_payload(dict(error="Floor Plan does not exist"), status_code=500)
        status = status_row["status"]
        if status == "COMPLETED":
            break
        await asyncio.sleep(2)

    if status != "COMPLETED":
        return respond_with_UI_payload(dict(error="Floor Plan extraction not completed within 15 minutes"), status_code=500)

    row = await pg_pool.fetchrow(
        "SELECT target_drywalls FROM models WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2) AND page_number=$3",
        project_id, plan_id, int(index)
    )
    drywall_overlaid_floorplan_source_path = row["target_drywalls"]
    _, _, _, blob_path = drywall_overlaid_floorplan_source_path.split('/', 3)

    client = CloudStorageClient()
    bucket = client.bucket(CREDENTIALS["CloudStorage"]["bucket_name"])
    blob = bucket.blob(blob_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=CREDENTIALS["CloudStorage"]["expiration_in_minutes"]),
        method="GET",
    )

    return url

@app.post("/remove_floorplan")
async def remove_floorplan(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    project_id = parameters.get("project_id") or body.get("project_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    logging.info("SYSTEM: Received a Floorplan Deletion Request")

    await delete_floorplan(project_id, plan_id, user_id, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: Floorplan Deleted Successfully")

@app.post("/compute_takeoff")
async def compute_takeoff(request: Request):
    enable_logging_on_stdout()
    parameters = dict(request.query_params)
    try:
        body = await request.json()
    except Exception:
        body = dict()
    walls_3d_JSON = parameters.get("walls_3d", list()) or body.get("walls_3d", list())
    polygons_JSON = parameters.get("polygons", list()) or body.get("polygons", list())
    index = parameters.get("page_number") or body.get("page_number")
    project_id = parameters.get("project_id") or body.get("project_id")
    plan_id = parameters.get("plan_id") or body.get("plan_id")
    user_id = parameters.get("user_id") or body.get("user_id")
    revision_number = parameters.get("revision_number", '') or body.get("revision_number", '')
    logging.info("SYSTEM: Received a Drywall Takeoff computation Request")

    if not walls_3d_JSON:
        if revision_number:
            row = await pg_pool.fetchrow(
                """SELECT model FROM model_revisions_3d
                   WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)
                   AND LOWER(user_id)=LOWER($3) AND page_number=$4 AND revision_number=$5""",
                project_id, plan_id, user_id, int(index), int(revision_number)
            )
            walls_3d_JSON = json.loads(row["model"]) if isinstance(row["model"], str) else row["model"]
        else:
            row = await pg_pool.fetchrow(
                """SELECT model_3d FROM models
                   WHERE LOWER(project_id)=LOWER($1) AND LOWER(plan_id)=LOWER($2)
                   AND LOWER(user_id)=LOWER($3) AND page_number=$4""",
                project_id, plan_id, user_id, int(index)
            )
            walls_3d_JSON = json.loads(row["model_3d"]) if isinstance(row["model_3d"], str) else row["model_3d"]

        if walls_3d_JSON is None:
            walls_3d_JSON = list()

    hyperparameters = load_hyperparameters()
    floor_plan_modeller_3d = Extrapolate3D(hyperparameters)
    walls_3d_JSON, polygons_JSON = floor_plan_modeller_3d.extrapolate_wall_heights_given_polygons(walls_3d_JSON, polygons_JSON)
    drywall_takeoff = dict(total=dict(roof=0, wall=0), per_drywall=dict(roof=defaultdict(lambda: 0), wall=defaultdict(lambda: 0)))
    for wall in walls_3d_JSON:
        surface_area = wall["height"] * wall["length"]
        drywall_count = 0
        for drywall in wall["surfaces_drywall"]:
            if drywall["enabled"]:
                drywall_takeoff["per_drywall"]["wall"][drywall["type"]] += surface_area
                drywall_count += 1
        drywall_takeoff["total"]["wall"] += drywall_count * surface_area
    for polygon in polygons_JSON:
        surface_area = floor_plan_modeller_3d.compute_updated_area_polygon(
            polygon["vertices"],
            polygon["area"],
            polygon["slope"],
            polygon["tilt_axis"]
        )
        drywall_takeoff["per_drywall"]["roof"][polygon["surface_drywall"]["type"]] += surface_area
        drywall_takeoff["total"]["roof"] += surface_area

    drywall_takeoff["total"]["wall"] = round(drywall_takeoff["total"]["wall"], 2)
    drywall_takeoff["total"]["roof"] = round(drywall_takeoff["total"]["roof"], 2)
    for key in drywall_takeoff["per_drywall"]["wall"]:
        drywall_takeoff["per_drywall"]["wall"][key] = round(drywall_takeoff["per_drywall"]["wall"][key], 2)
    for key in drywall_takeoff["per_drywall"]["roof"]:
        drywall_takeoff["per_drywall"]["roof"][key] = round(drywall_takeoff["per_drywall"]["roof"][key], 2)

    await insert_takeoff(drywall_takeoff, index, plan_id, user_id, project_id, revision_number, pg_pool, CREDENTIALS)
    logging.info("SYSTEM: Drywall Takeoff Computed Successfully for the provided Floorplan")
    return respond_with_UI_payload(drywall_takeoff)