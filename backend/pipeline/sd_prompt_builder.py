"""
Visionary Pipeline — Stable Diffusion Prompt Builder.
Builds the SD prompt from VLM analysis + redesign plan.
No LLM needed — pure structured string assembly.
"""

STYLE_FRAGMENTS = {
    "scandinavian": "Scandinavian minimalist hygge interior, light oak wood, linen textures, white walls",
    "industrial":   "industrial loft interior, exposed brick, raw concrete, matte black steel",
    "bohemian":     "bohemian eclectic room, rattan furniture, macrame, layered textiles, warm terracotta",
    "mid-century modern": "mid-century modern room, teak wood, tapered legs, sunburst clock, avocado green",
    "japandi":      "Japandi wabi-sabi interior, bamboo, natural linen, warm white clay walls",
    "minimalist":   "ultra minimalist interior, clean lines, pure white, negative space, barely-there decor",
    "coastal":      "coastal beach house interior, whitewashed wood, sea glass accents, ocean blue",
    "luxury":       "ultra-luxury high-end interior, white marble floors, gold leaf accents, premium velvet upholstery, architectural digest style, boutique hotel aesthetic",
    "rustic":       "rustic farmhouse interior, reclaimed wood, stone walls, plaid textiles",
    "traditional":  "traditional classic interior, crown moulding, wingback chairs, antique mahogany",
    "modern":       "high-end modern interior, vibrant contrast, warm architectural lighting, designer furniture, polished surfaces",
    "contemporary": "contemporary interior, curved furniture, neutral tones, artistic accents",
}

QUALITY_SUFFIX = (
    "architectural digest, photorealistic, 8k, sharp focus"
)

NEGATIVE_BASE = (
    "low quality, blurry, distorted, watermark, text overlay, "
    "cartoon, sketch, painting, deformed, extra objects, ugly, "
    "oversaturated, flat lighting, amateur photo"
)


def build_sd_prompt(
    target_style: str,
    budget_plan: dict,
    scene_graph: dict | None = None,
    user_prompt: str = "",
    wall_treatment: str = "matte white paint with warm undertones",
    flooring: str = "light oak hardwood",
    lighting: str = "soft warm ambient lighting",
    accessories: list[str] = None,
) -> tuple[str, str]:
    """
    Build Stable Diffusion prompt and negative prompt from redesign plan.
    
    Args:
        target_style: Target interior design style
        budget_plan: Budget allocation plan from budget engine
        wall_treatment: Description of wall treatment
        flooring: Description of flooring
        lighting: Description of lighting
        accessories: List of decorative accessories
    
    Returns:
        Tuple of (positive_prompt, negative_prompt)
    """
    if accessories is None:
        accessories = ["potted plants", "throw cushions", "decorative vase"]
    
    style_frag = STYLE_FRAGMENTS.get(target_style, "modern interior")
    
    # Build furniture description from budget plan
    furniture_desc = ", ".join(
        f"{item['item'].replace('_', ' ')}"
        for item in budget_plan.get("items", [])
        if item.get("tier") in ("mid", "premium")
    )
    
    accessory_desc = ", ".join(accessories[:5])
    
    # Style-specific negative prompts
    style_negative = {
        "scandinavian": "dark colours, heavy drapes, ornate, cluttered",
        "industrial":   "pastel, floral, ornate decor",
        "bohemian":     "stark minimalism, cold grey, bare walls",
        "minimalist":   "clutter, patterns, multiple colours",
        "japandi":      "clutter, bright neon colours, heavy furniture",
        "coastal":      "dark colours, heavy furniture, urban elements",
        "luxury":       "cheap materials, basic design, cluttered",
        "rustic":       "modern sleek, glass and chrome, minimalist",
    }.get(target_style, "clutter, dated design")

    scene_hint = ""
    if scene_graph:
        room_type = scene_graph.get("room_type")
        if room_type:
            scene_hint = f"{room_type} interior"

    prompt = ", ".join(filter(None, [
        user_prompt.strip(),
        scene_hint,
        style_frag,
        f"walls: {wall_treatment}",
        f"floor: {flooring}",
        f"new furniture: {furniture_desc}" if furniture_desc else "",
        f"decor: {accessory_desc}" if accessory_desc else "",
        f"lighting: {lighting}",
        QUALITY_SUFFIX,
    ]))

    negative = f"{style_negative}, {NEGATIVE_BASE}"

    return prompt, negative
