"""
Budget Engine — Visionary's money brain.
Takes total budget in INR and scene graph,
outputs a smart allocation plan per item with priority scoring.
"""

# Priority weights: how much visual impact does replacing this item have?
ITEM_PRIORITY = {
    "couch":        10,
    "sofa":         10,
    "bed":          10,
    "dining table": 8,
    "chair":        6,
    "curtain":      7,
    "rug":          7,
    "lamp":         5,
    "desk":         5,
    "wardrobe":     6,
    "shelf":        4,
    "pillow":       3,
    "vase":         2,
    "potted plant": 2,
    "clock":        1,
    "tv":           4,
    "mirror":       5,
}

# Typical INR price ranges per item category
PRICE_RANGES = {
    "couch":        {"min": 8000,  "mid": 18000, "max": 45000},
    "sofa":         {"min": 8000,  "mid": 18000, "max": 45000},
    "bed":          {"min": 6000,  "mid": 15000, "max": 40000},
    "dining table": {"min": 5000,  "mid": 12000, "max": 30000},
    "chair":        {"min": 1500,  "mid": 4000,  "max": 12000},
    "curtain":      {"min": 800,   "mid": 2500,  "max": 8000},
    "rug":          {"min": 1200,  "mid": 4000,  "max": 15000},
    "lamp":         {"min": 500,   "mid": 2000,  "max": 8000},
    "desk":         {"min": 3000,  "mid": 8000,  "max": 20000},
    "wardrobe":     {"min": 8000,  "mid": 20000, "max": 50000},
    "shelf":        {"min": 1000,  "mid": 3500,  "max": 10000},
    "pillow":       {"min": 200,   "mid": 600,   "max": 2000},
    "vase":         {"min": 200,   "mid": 700,   "max": 3000},
    "potted plant": {"min": 150,   "mid": 500,   "max": 2000},
    "clock":        {"min": 300,   "mid": 1000,  "max": 4000},
    "tv":           {"min": 12000, "mid": 25000, "max": 60000},
    "mirror":       {"min": 800,   "mid": 2500,  "max": 8000},
    "wall paint":   {"min": 1500,  "mid": 3000,  "max": 8000},
    "flooring":     {"min": 3000,  "mid": 8000,  "max": 25000},
}

# Always suggest painting at minimum
ALWAYS_INCLUDE = ["wall paint"]


def allocate_budget(
    total_budget_inr: float,
    scene_graph: dict,
    target_style: str,
    user_prompt: list[str] | str | None = None,
) -> dict:
    """
    Smart budget allocation across all redesign items.
    
    Strategy:
    1. Score each detected item by visual impact priority
    2. Boost items explicitly mentioned by the user
    3. Allocate budget proportionally by priority weight
    4. Clamp allocations to realistic price ranges
    5. Keep a 5% buffer for contingency
    
    Args:
        total_budget_inr: Total available budget in Indian Rupees
        scene_graph: Structured scene graph from pipeline
        target_style: Target design style
        user_priorities: Items user explicitly wants to change
    
    Returns:
        Complete budget allocation plan with per-item breakdowns
    """
    # Support both full scene graph (`objects`) and lightweight preview graph (`cv_objects`).
    object_nodes = scene_graph.get("objects", [])
    if not object_nodes:
        object_nodes = scene_graph.get("cv_objects", [])
    detected_labels = [obj["label"] for obj in object_nodes if isinstance(obj, dict) and "label" in obj]

    # Accept user priorities as either a list or a free-form string.
    normalized_priorities = set()
    if isinstance(user_prompt, str):
        text = user_prompt.lower()
        for item_name in ITEM_PRIORITY.keys():
            if item_name in text:
                normalized_priorities.add(item_name)
    elif isinstance(user_prompt, (list, tuple, set)):
        normalized_priorities = {str(x).lower() for x in user_prompt}
    items_to_budget = list(set(detected_labels + ALWAYS_INCLUDE))

    # Score each item
    scored = []
    for item in items_to_budget:
        priority = ITEM_PRIORITY.get(item, 3)
        # Boost priority if user explicitly mentioned it
        if normalized_priorities and item.lower() in normalized_priorities:
            priority += 5
        # Only include items we have price data for
        if item in PRICE_RANGES:
            scored.append({"item": item, "priority": priority})

    # Sort by priority descending
    scored.sort(key=lambda x: x["priority"], reverse=True)

    # Determine budget tier
    tier = _get_tier(total_budget_inr)

    # Allocate proportionally by priority
    total_priority = sum(s["priority"] for s in scored) or 1
    allocation_plan = []
    remaining = total_budget_inr
    reserve = total_budget_inr * 0.05  # keep 5% as buffer
    spendable = total_budget_inr - reserve

    for item_data in scored:
        item = item_data["item"]
        weight = item_data["priority"] / total_priority
        raw_alloc = spendable * weight
        price_range = PRICE_RANGES[item]

        # Clamp to realistic price range
        alloc = max(price_range["min"], min(raw_alloc, price_range["max"]))
        alloc = min(alloc, remaining - reserve)

        if alloc < price_range["min"]:
            # Can't afford minimum — skip this item
            continue

        remaining -= alloc

        allocation_plan.append({
            "item": item,
            "allocated_inr": round(alloc),
            "tier": _item_tier(alloc, price_range),
            "price_range": price_range,
            "search_keywords": _build_search_keywords(item, target_style, tier),
            "priority_score": item_data["priority"],
        })

        if remaining <= reserve:
            break

    return {
        "total_budget_inr": total_budget_inr,
        "total_allocated_inr": round(total_budget_inr - remaining),
        "buffer_inr": round(remaining),
        "budget_tier": tier,
        "items": allocation_plan,
    }


def _get_tier(budget: float) -> str:
    """Classify overall budget tier."""
    if budget < 15000:
        return "budget"
    elif budget < 50000:
        return "mid"
    else:
        return "premium"


def _item_tier(alloc: float, price_range: dict) -> str:
    """Classify individual item allocation tier."""
    if alloc <= price_range["min"] * 1.3:
        return "budget"
    elif alloc <= price_range["mid"] * 1.2:
        return "mid"
    else:
        return "premium"


def _build_search_keywords(item: str, style: str, tier: str) -> list[str]:
    """Generate search keywords for product matching."""
    style_kw = {
        "scandinavian": "scandinavian minimalist",
        "industrial": "industrial metal",
        "bohemian": "bohemian rattan",
        "mid-century modern": "mid century retro",
        "japandi": "japandi minimalist wood",
        "minimalist": "minimalist clean",
        "coastal": "coastal white wicker",
        "luxury": "luxury premium velvet",
        "rustic": "rustic reclaimed wood",
        "traditional": "traditional classic",
        "modern": "modern contemporary",
        "contemporary": "contemporary sleek",
    }.get(style, "modern")

    tier_kw = {"budget": "affordable", "mid": "quality", "premium": "luxury premium"}

    return [
        f"{tier_kw.get(tier, '')} {style_kw} {item}".strip(),
        f"{item} buy online India",
        f"{style} {item} furniture",
    ]
