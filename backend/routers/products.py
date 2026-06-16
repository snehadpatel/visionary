"""
Visionary API — Products Router.
Handles product search and matching endpoints.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api", tags=["products"])


@router.get("/products/{job_id}")
async def get_products(job_id: str):
    """
    Get matched products for a completed redesign job.
    Products are included in the main job result, but this endpoint
    allows refreshing product search independently.
    """
    from routers.redesign import jobs
    
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    
    if job["status"] != "done":
        return JSONResponse({"error": "Job not complete yet"}, status_code=400)
    
    result = job.get("result", {})
    return {
        "matched_products": result.get("matched_products", []),
        "budget_plan": result.get("budget_plan", {}),
    }
