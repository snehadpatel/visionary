"""
Visionary Products — Product Matcher.
Match each budget allocation item to the best available product from scrapers.
Picks best value: closest to allocated budget without exceeding it.
"""
from products.scraper import search_amazon_in, search_pepperfry


def find_products_for_plan(budget_plan: dict, target_style: str) -> list[dict]:
    """
    For each item in the budget plan, search for real products and pick the best match.
    
    Strategy: 
    - Search using style-appropriate keywords
    - Pick the product closest to the allocated budget (best quality within budget)
    
    Args:
        budget_plan: Budget allocation plan from budget engine
        target_style: Target interior design style
    
    Returns:
        List of items with matched products
    """
    matched = []
    
    for item_data in budget_plan.get("items", []):
        item = item_data["item"]
        max_price = item_data["allocated_inr"]
        keywords = item_data.get("search_keywords", [f"{item} buy online India"])

        all_results = []
        for kw in keywords[:2]:  # Use top 2 keywords
            all_results.extend(search_amazon_in(kw, max_price))
            all_results.extend(search_pepperfry(kw, max_price))

        # Deduplicate by title (rough)
        seen_titles = set()
        unique_results = []
        for r in all_results:
            title_key = r["title"][:50].lower()
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_results.append(r)

        # Pick best value product
        best = _pick_best(unique_results, max_price)

        matched.append({
            "item": item,
            "allocated_budget_inr": max_price,
            "product": best,
            "tier": item_data.get("tier", "mid"),
            "priority_score": item_data.get("priority_score", 0),
        })

    return matched


def _pick_best(results: list[dict], max_price: int) -> dict | None:
    """
    Pick the best product within budget.
    Strategy: closest to max_price (maximize value without exceeding budget).
    """
    eligible = [r for r in results if r["price_inr"] <= max_price]
    if not eligible:
        return None
    # Pick the one closest to max_price (best quality within budget)
    eligible.sort(key=lambda r: abs(r["price_inr"] - max_price))
    return eligible[0]
