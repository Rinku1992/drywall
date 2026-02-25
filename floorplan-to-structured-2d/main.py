import logging
from pathlib import Path
import json
import requests
from functools import partial
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor

import google.auth.transport.requests
from google.oauth2.service_account import IDTokenCredentials
# from google.cloud import secretmanager

from modeller_2d import FloorPlan2D
from helper import (
    enable_logging_on_stdout,
    load_vertex_ai_client,
    load_gcp_credentials,
    load_hyperparameters,
    transcribe,
    upload_floorplan,
    download_floorplan,
    insert_model_2d,
    get_pg_pool,
)

def respond_with_UI_payload(payload, status_code=200):
    return JSONResponse(
        content=json.loads(json.dumps(payload)),
        status_code=status_code,
        media_type="application/json",
    )


def floorplan_to_walls(credentials, project_id, plan_id, user_id, page_number, output_path=None):
    auth_req = google.auth.transport.requests.Request()
    service_account_credentials = IDTokenCredentials.from_service_account_file(
        credentials["service_compute_account_key"],
        target_audience=credentials["CloudRun"]["APIs"]["wall_detector"]
    )
    service_account_credentials.refresh(auth_req)
    id_token = service_account_credentials.token

    headers = {
        "Authorization": f"Bearer {id_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{credentials["CloudRun"]["APIs"]["wall_detector"]}/detect_wall",
        headers=headers,
        json=dict(
            project_id=project_id,
            plan_id=plan_id,
            user_id=user_id,
            page_number=page_number
        )
    )

    if not output_path:
        output_path  = Path("/tmp/floor_plan_wall_segmented.png")
    with open(output_path, "wb") as f:
        f.write(response.content)
    return Path(output_path)


app = FastAPI(title="Floorplan-to-Structured-2D (Cloud Run)")

CREDENTIALS = load_gcp_credentials()
app.add_middleware(
    CORSMiddleware,
    allow_origins=CREDENTIALS["CloudRun"]["origins_cors"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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



@app.post("/floorplan_to_structured_2d")
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
    page_number = parameters.get("page_number") or body.get("page_number")
    verbose = parameters.get("verbose") or body.get("verbose")
    logging.info("SYSTEM: Received a Floorplan 2D Model Generation Request")

    floor_plan_processed_path = download_floorplan(user_id, plan_id, project_id, CREDENTIALS, str(page_number).zfill(2))
    logging.info(f"SYSTEM: Processed Floorplan Downloaded: Page Number: {page_number}")

    hyperparameters = load_hyperparameters()
    vertex_ai_client, generation_config = load_vertex_ai_client(CREDENTIALS)
    vertex_ai_client_partial = partial(vertex_ai_client.generate_content, generation_config=generation_config)
    floor_plan_modeller_2d = FloorPlan2D(hyperparameters, vertex_ai_client_partial)

    futures = dict()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures["floorplan_to_walls"] = executor.submit(
            floorplan_to_walls,
            CREDENTIALS,
            project_id,
            plan_id,
            user_id,
            page_number,
            output_path=f"/tmp/{project_id}/{plan_id}/{user_id}/floor_plan_wall_segmented_{str(page_number).zfill(2)}.png"
        )
        futures["transcriber"] = executor.submit(
            transcribe,
            CREDENTIALS,
            hyperparameters,
            floor_plan_processed_path,
        )
    wall_segmented_path = futures["floorplan_to_walls"].result()
    upload_floorplan(wall_segmented_path, user_id, plan_id, project_id, CREDENTIALS, index=str(page_number).zfill(2))
    logging.info(f"SYSTEM: Wall Detection Completed from PAGE: {page_number}")

    transcription_block_with_centroids, transcription_headers_and_footers = futures["transcriber"].result()
    logging.info(f"SYSTEM: Transcription Completed from PAGE: {page_number}")

    walls_2d, polygons, metadata, floorplan_baseline_page_source = None, None, None, None
    if not floor_plan_modeller_2d.is_none(wall_segmented_path):
        walls_2d, polygons, walls_2d_path, external_contour = floor_plan_modeller_2d.model(
            image_path=wall_segmented_path,
            model_2d_path=f"/tmp/{project_id}/{plan_id}/{user_id}/walls_2d_{str(page_number).zfill(2)}.json",
            floor_plan_path=floor_plan_processed_path,
            transcription_block_with_centroids=transcription_block_with_centroids,
            transcription_headers_and_footers=transcription_headers_and_footers
        )
        if walls_2d and polygons:
            floor_plan_modeller_2d.load_drywall_choices(walls_2d, polygons)
            floor_plan_modeller_2d.load_ceiling_choices(polygons)
            #if verbose.upper() == "TRUE":
            model_2d_path = floor_plan_modeller_2d.save_plot_2d(walls_2d_path, floor_plan_path=floor_plan_processed_path)
            upload_floorplan(model_2d_path, user_id, plan_id, project_id, CREDENTIALS, index=str(page_number).zfill(2))
            #model_2d_path_overlay_enabled = floor_plan_modeller_2d.save_plot_2d(walls_2d_path, floor_plan_path=floor_plan_processed_path, overlay_enabled=True)
            #upload_floorplan(model_2d_path_overlay_enabled, user_id, plan_id, project_id, CREDENTIALS, index=str(page_number).zfill(2))
            floorplan_baseline, floorplan_page_statistics = floor_plan_modeller_2d.scale_to(floor_plan_path=floor_plan_processed_path)
            floorplan_baseline_page_source = upload_floorplan(floorplan_baseline, user_id, plan_id, project_id, CREDENTIALS, index=str(page_number).zfill(2))

            metadata = dict(
                size_in_bytes=floorplan_page_statistics["size"],
                height_in_pixels=floorplan_page_statistics["height"],
                width_in_pixels=floorplan_page_statistics["width"],
                origin=["LEFT", "TOP"],
                offset=(0, 0),
                contour_root_vertices=external_contour,
            )
    # bigquery_client = load_bigquery_client(CREDENTIALS)
    # insert_model_2d(
    #     dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata),
    #     floor_plan_modeller_2d.scale,
    #     page_number,
    #     plan_id,
    #     user_id,
    #     project_id,
    #     floorplan_baseline_page_source,
    #     bigquery_client,
    #     CREDENTIALS
    # )

    await insert_model_2d(
        dict(walls_2d=walls_2d, polygons=polygons, metadata=metadata),
        floor_plan_modeller_2d.scale,
        page_number,
        plan_id,
        user_id,
        project_id,
        floorplan_baseline_page_source,
        pg_pool,
        CREDENTIALS
    )
    logging.info(f"SYSTEM: A 2D Model of the Floorplan from PAGE: {page_number} Generated Successfully")
