"""
redesign_spec_engine.py
=======================
Converts a scene graph + user text prompt into a Stable Diffusion prompt
entirely using rule-based NLP and lookup tables. No LLM, no API, no internet.
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Style → SD prompt fragments
# ---------------------------------------------------------------------------

STYLE_PROMPTS: dict[str, dict] = {
    "scandinavian": {
        "adjectives": "Scandinavian minimalist, hygge, light oak wood, white walls, linen textures",
        "lighting": "soft diffused natural light, large windows, warm afternoon sun",
        "colors": "white, pale grey, warm beige, dusty blue accents",
        "negative": "dark colours, heavy drapes, cluttered, ornate",
    },
    "industrial": {
        "adjectives": "industrial loft, exposed brick, raw concrete, steel beams, matte black metal",
        "lighting": "Edison bulb pendant lights, dramatic shadows, amber warm glow",
        "colors": "charcoal grey, rust, burnt sienna, aged copper",
        "negative": "pastel colours, floral patterns, ornate decor",
    },
    "bohemian": {
        "adjectives": "bohemian eclectic, layered textiles, macrame wall art, rattan furniture",
        "lighting": "warm golden light, fairy lights, candles",
        "colors": "terracotta, mustard yellow, deep teal, burgundy",
        "negative": "stark minimalism, cold grey tones, bare walls",
    },
    "mid-century modern": {
        "adjectives": "mid-century modern, teak wood furniture, tapered legs, abstract art prints",
        "lighting": "arc floor lamp, indirect warm lighting",
        "colors": "avocado green, burnt orange, walnut brown, cream",
        "negative": "contemporary minimalism, chrome, ultra-modern",
    },
    "japandi": {
        "adjectives": "Japandi style, wabi-sabi, natural linen, bamboo, minimal decor",
        "lighting": "soft indirect light, paper lanterns, zen calm",
        "colors": "warm white, clay, sage green, charcoal, natural wood",
        "negative": "bright saturated colours, heavy ornaments, clutter",
    },
    "minimalist": {
        "adjectives": "ultra minimalist, clean lines, barely-there decor, negative space",
        "lighting": "bright even daylight, recessed ceiling lights",
        "colors": "pure white, light grey, warm cream, black accents",
        "negative": "clutter, patterns, multiple colours, ornate details",
    },
    "coastal": {
        "adjectives": "coastal beach house, whitewashed wood, sea glass, nautical accents",
        "lighting": "bright airy natural light, breezy atmosphere",
        "colors": "ocean blue, sandy beige, coral, sea foam green, white",
        "negative": "dark colours, heavy fabrics, urban industrial elements",
    },
    "luxury": {
        "adjectives": "luxury contemporary, marble surfaces, velvet upholstery, gold accents",
        "lighting": "dramatic chandeliers, layered ambient lighting, golden glow",
        "colors": "deep navy, emerald green, champagne gold, ivory white",
        "negative": "budget materials, plain flat colours, minimal bare look",
    },
    "rustic": {
        "adjectives": "rustic farmhouse, reclaimed wood, stone walls, plaid textiles",
        "lighting": "warm candlelight, exposed filament bulbs, cosy glow",
        "colors": "warm brown, cream, forest green, brick red",
        "negative": "modern chrome, glass furniture, cold grey tones",
    },
    "traditional": {
        "adjectives": "traditional classic interior, crown moulding, wingback chairs, antique wood",
        "lighting": "table lamps, warm incandescent glow, classic chandelier",
        "colors": "navy, burgundy, forest green, cream, mahogany",
        "negative": "contemporary minimalism, raw concrete, industrial elements",
    },
}

STYLE_KEYWORD_MAP: dict[str, str] = {
    "scandinavian": "scandinavian", "scandi": "scandinavian", "nordic": "scandinavian",
    "industrial": "industrial", "loft": "industrial",
    "boho": "bohemian", "bohemian": "bohemian", "eclectic": "bohemian",
    "mid century": "mid-century modern", "mid-century": "mid-century modern", "retro": "mid-century modern",
    "japandi": "japandi", "japanese": "japandi", "zen": "japandi", "wabi": "japandi",
    "minimalist": "minimalist", "minimal": "minimalist", "clean": "minimalist",
    "coastal": "coastal", "beach": "coastal", "nautical": "coastal",
    "luxury": "luxury", "glam": "luxury", "glamorous": "luxury",
    "rustic": "rustic", "farmhouse": "rustic", "country": "rustic",
    "traditional": "traditional", "classic": "traditional",
}

FURNITURE_REPLACEMENTS: dict[str, dict[str, str]] = {
    "couch": {
        "scandinavian": "low-profile linen sofa in pale grey",
        "industrial": "leather Chesterfield sofa in dark brown",
        "bohemian": "overstuffed velvet sofa in terracotta with throw pillows",
        "mid-century modern": "teak-legged sofa in burnt orange fabric",
        "japandi": "low platform sofa in natural linen",
        "minimalist": "clean-lined white sofa with no armrests",
        "coastal": "slipcovered sofa in sandy white linen",
        "luxury": "deep velvet sofa in emerald green with gold legs",
        "rustic": "distressed leather sofa in warm cognac",
        "traditional": "rolled-arm sofa in navy blue fabric",
    },
    "chair": {
        "scandinavian": "Wegner-style oak dining chair",
        "industrial": "metal mesh chair with black powder coat",
        "bohemian": "rattan peacock chair with cushion",
        "mid-century modern": "fibreglass shell chair on eiffel base",
        "japandi": "simple wooden stool with rush seat",
        "minimalist": "moulded plastic chair in white",
        "coastal": "wicker armchair with white cushion",
        "luxury": "velvet accent chair in dusty rose with brass legs",
        "rustic": "Windsor chair in natural pine",
        "traditional": "wingback armchair in deep green fabric",
    },
    "bed": {
        "scandinavian": "platform bed in white oak with white linen bedding",
        "industrial": "metal bed frame in matte black with grey bedding",
        "bohemian": "canopy bed with hanging macrame and layered quilts",
        "mid-century modern": "walnut platform bed with hairpin legs",
        "japandi": "low futon-style bed in natural wood, white bedding",
        "minimalist": "floating bed frame in white with single white duvet",
        "coastal": "whitewashed wood bed with blue and white stripe bedding",
        "luxury": "upholstered bed in ivory velvet with tufted headboard",
        "rustic": "reclaimed wood log bed with plaid flannel bedding",
        "traditional": "carved mahogany four-poster bed with floral bedding",
    },
    "dining table": {
        "scandinavian": "round white dining table with tulip base",
        "industrial": "rectangular table with raw steel legs and reclaimed wood top",
        "bohemian": "wooden dining table with mismatched chairs and runner",
        "mid-century modern": "oval teak dining table on splayed legs",
        "japandi": "low kotatsu-style wooden table",
        "minimalist": "slim white rectangular table",
        "coastal": "whitewashed rectangular dining table",
        "luxury": "oval marble dining table with gold base",
        "rustic": "farmhouse trestle table in reclaimed pine",
        "traditional": "oval mahogany dining table with turned legs",
    },
}

STYLE_ACCESSORIES: dict[str, list[str]] = {
    "scandinavian": ["potted eucalyptus plant", "geometric candle holders", "sheepskin throw rug"],
    "industrial": ["exposed pipe shelf", "vintage Edison bulb lamp", "metal wire basket"],
    "bohemian": ["macrame wall hanging", "multiple potted plants", "layered area rugs", "hanging plants"],
    "mid-century modern": ["abstract art print", "arc floor lamp", "sunburst clock"],
    "japandi": ["bonsai tree", "ceramic vase with dried pampas grass", "paper floor lamp"],
    "minimalist": ["single stem vase", "one abstract painting", "sculptural bowl"],
    "coastal": ["woven seagrass basket", "driftwood art", "coastal blue throw blanket"],
    "luxury": ["large floral arrangement", "gold-framed mirrors", "crystal table lamp"],
    "rustic": ["galvanised metal planters", "vintage wooden crates", "hand-woven throw blanket"],
    "traditional": ["porcelain table lamp", "framed botanical prints", "decorative bookends"],
}

_PLANT_KEYWORDS = {"plant", "plants", "green", "nature", "greenery", "botanical", "floral"}
_LIGHT_KEYWORDS = {"light", "lighting", "bright", "cosy", "cozy", "warm", "dark", "moody", "airy"}
_COLOR_KEYWORDS = {"white", "black", "grey", "gray", "blue", "green", "red", "yellow",
                   "pink", "purple", "orange", "beige", "brown", "cream", "teal", "gold"}

def build_redesign_spec(scene_graph: dict, user_prompt: str, style_override: str = "auto", room_type: str = "room") -> dict:
    target_style = _resolve_style(user_prompt, style_override, scene_graph)
    style_data = STYLE_PROMPTS[target_style]
    words = _tokenize(user_prompt)

    objects = scene_graph.get("objects", [])
    # -----------------------------------------------------------------------
    # FALLBACK LOGIC: If no objects detected, assume the room needs its core furniture.
    # -----------------------------------------------------------------------
    if not objects:
        if room_type == "bedroom":
            objects = [{"label": "bed", "id": 0}]
        elif room_type == "living room":
            objects = [{"label": "couch", "id": 0}]

    furniture_changes = _compute_furniture_changes(objects, target_style)
    accessories = _compute_accessories(target_style, words)
    wall_treatment = _compute_wall(target_style, words)
    flooring = _compute_flooring(target_style)
    lighting = _compute_lighting(target_style, words)

    sd_prompt = _build_sd_prompt(
        style_data, furniture_changes, accessories,
        wall_treatment, flooring, lighting, room_type
    )

    return {
        "target_style": target_style,
        "wall_treatment": wall_treatment,
        "flooring": flooring,
        "lighting": lighting,
        "furniture_changes": furniture_changes,
        "accessories": accessories,
        "sd_prompt": sd_prompt,
        "negative_sd_prompt": (
            f"{style_data['negative']}, "
            "low quality, blurry, distorted, watermark, text, "
            "lowres, grainy, noisy, over-saturated, jpeg artifacts, "
            "cartoon, sketch, dark lighting, messy, ugly"
        ),
    }

def _tokenize(text: str) -> set[str]:
    return set(re.sub(r"[^a-z\s]", "", text.lower()).split())

def _resolve_style(prompt: str, override: str, scene_graph: dict) -> str:
    if override and override != "auto":
        key = override.lower().strip()
        if key in STYLE_PROMPTS:
            return key
        for kw, mapped in STYLE_KEYWORD_MAP.items():
            if kw in key:
                return mapped

    prompt_lower = prompt.lower()
    for kw, mapped in STYLE_KEYWORD_MAP.items():
        if kw in prompt_lower:
            return mapped

    return scene_graph.get("current_style", "scandinavian")

def _compute_furniture_changes(objects: list[dict], style: str) -> list[dict]:
    changes = []
    seen_labels = set()
    for obj in objects:
        label = obj.get("label", "unknown")
        if label in seen_labels:
            continue
        seen_labels.add(label)
        replacement = FURNITURE_REPLACEMENTS.get(label, {}).get(style)
        if replacement:
            changes.append({
                "action": "replace",
                "object": label,
                "replacement": replacement,
            })
        else:
            changes.append({"action": "keep", "object": label, "replacement": None})
    return changes

def _compute_accessories(style: str, words: set[str]) -> list[str]:
    base = list(STYLE_ACCESSORIES.get(style, []))
    if words & _PLANT_KEYWORDS and "potted plant" not in " ".join(base):
        base.append("multiple lush indoor plants in terracotta pots")
    return base

def _compute_wall(style: str, words: set[str]) -> str:
    color_hit = next((w for w in words if w in _COLOR_KEYWORDS), None)
    defaults = {
        "scandinavian": "matte white paint",
        "industrial": "exposed brick with grey concrete plaster",
        "bohemian": "warm terracotta paint with woven wall hangings",
        "mid-century modern": "avocado green accent wall",
        "japandi": "warm white with subtle clay texture",
        "minimalist": "pure white matte paint",
        "coastal": "whitewashed wood panelling",
        "luxury": "textured ivory wallpaper with subtle gold thread",
        "rustic": "exposed stone with white plaster",
        "traditional": "deep navy or forest green wallpaper with subtle pattern",
    }
    base = defaults.get(style, "white matte paint")
    if color_hit:
        base = f"{color_hit}-toned {base}"
    return base

def _compute_flooring(style: str) -> str:
    return {
        "scandinavian": "light blonde hardwood floors",
        "industrial": "polished concrete floors",
        "bohemian": "colourful patterned tile with layered rugs",
        "mid-century modern": "warm walnut parquet flooring",
        "japandi": "pale bamboo flooring with tatami mat",
        "minimalist": "light grey polished concrete",
        "coastal": "whitewashed wide-plank oak floors",
        "luxury": "large-format white marble tiles",
        "rustic": "wide reclaimed pine plank flooring",
        "traditional": "rich mahogany herringbone parquet",
    }.get(style, "natural wood floors")

def _compute_lighting(style: str, words: set[str]) -> str:
    base = STYLE_PROMPTS[style]["lighting"]
    if words & _LIGHT_KEYWORDS:
        if any(w in words for w in ["warm", "cosy", "cozy"]):
            base += ", extra warm amber tones"
        if any(w in words for w in ["bright", "airy"]):
            base += ", maximised natural daylight"
        if any(w in words for w in ["moody", "dark"]):
            base += ", low-key moody atmosphere with accent lighting"
    return base

def _build_sd_prompt(
    style_data: dict,
    furniture_changes: list[dict],
    accessories: list[str],
    wall_treatment: str,
    flooring: str,
    lighting: str,
    room_type: str = "room",
) -> str:
    furniture_desc = ", ".join(
        c["replacement"] for c in furniture_changes
        if c["action"] == "replace" and c["replacement"]
    )
    accessory_desc = ", ".join(accessories)

    parts = [
        style_data["adjectives"],
        wall_treatment,
        flooring,
        furniture_desc if furniture_desc else "",
        accessory_desc if accessory_desc else "",
        lighting,
        style_data["colors"],
        "8k, ultra-detailed, photorealistic",
    ]
    return ", ".join(p for p in parts if p)
