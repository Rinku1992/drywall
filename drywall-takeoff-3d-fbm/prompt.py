from pydantic import BaseModel

ARCHITECTURAL_DRAWING_CLASSIFIER = """
    You are an expert architectural drawing classifier.

    PROVIDED:
        A single page from a residential construction plan document in PNG format.

    TASK:
        Classify into exactly ONE category:

        - FLOOR_PLAN: Room layouts, walls, doors, windows, dimensions. The architectural footprint.
        - ROOF_PLAN: Ridgelines, slopes, overhangs viewed from above.
        - ELECTRICAL_PLAN: Outlets, switches, circuits, panel schedules.
        - FOUNDATION_PLAN: Footings, slabs, piers, rebar details.
        - ELEVATION_PLAN: Exterior/interior vertical facade views.
        - NOT_ARCHITECTURAL_PLAN: Everything else — HVAC, plumbing, mechanical, sections, site plans, cover sheets, schedules, notes, details, or any drawing type not listed above.

        RULES:
        1. Read the title block FIRST (bottom-right or right side). Title text is the strongest signal.
        2. If the title contains "HVAC", "MECHANICAL", "HEATING", "PLUMBING", "WATER", "SITE", "SECTION", "DETAIL", "SCHEDULE", or "NOTES" → NOT_ARCHITECTURAL_PLAN.
        3. Only classify as FLOOR_PLAN if the drawing shows walls, rooms, and dimensions WITHOUT being dominated by MEP (mechanical/electrical/plumbing) symbols.
        4. When uncertain, default to NOT_ARCHITECTURAL_PLAN.

    OUTPUT:
        JSON only. No additional text.
        {{"plan_type": "<FLOOR_PLAN/ROOF_PLAN/ELECTRICAL_PLAN/FOUNDATION_PLAN/ELEVATION_PLAN/NOT_ARCHITECTURAL_PLAN>"}}
"""

class ArchitecturalDrawingClassifierResponse(BaseModel):
    plan_type: str
