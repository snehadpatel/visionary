"""
Visionary Products — Product scraper.
Fetches real products from Indian e-commerce sites.
Uses requests + BeautifulSoup. Results are cached to avoid repeated calls.
"""
import requests
from bs4 import BeautifulSoup
from products.cache import cache_key, read_cache, write_cache

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def search_amazon_in(query: str, max_price_inr: int) -> list[dict]:
    """
    Search Amazon.in for products matching query within price range.
    Results are cached for 24 hours.
    """
    key = cache_key(query, "amazon")
    cached = read_cache(key)
    if cached:
        return cached

    url = f"https://www.amazon.in/s?k={query.replace(' ', '+')}&rh=p_36%3A-{max_price_inr}00"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for item in soup.select('[data-component-type="s-search-result"]')[:5]:
            title_el = item.select_one("h2 a span")
            price_el = item.select_one(".a-price-whole")
            link_el = item.select_one("h2 a")
            img_el = item.select_one(".s-image")

            if title_el and price_el and link_el:
                price_text = price_el.text.replace(",", "").strip()
                try:
                    price = int(price_text)
                except ValueError:
                    continue
                if price <= max_price_inr:
                    results.append({
                        "title": title_el.text.strip(),
                        "price_inr": price,
                        "url": "https://www.amazon.in" + link_el.get("href", ""),
                        "image_url": img_el.get("src", "") if img_el else "",
                        "source": "Amazon.in",
                    })

        write_cache(key, results)
        return results

    except Exception as e:
        print(f"[Scraper] Amazon.in error: {e}")
        return []


def search_pepperfry(query: str, max_price_inr: int) -> list[dict]:
    """
    Search Pepperfry for furniture products matching query within price range.
    Results are cached for 24 hours.
    """
    key = cache_key(query, "pepperfry")
    cached = read_cache(key)
    if cached:
        return cached

    url = f"https://www.pepperfry.com/site_product/search?q={query.replace(' ', '+')}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []

        for item in soup.select(".search-result-item, .clipdatabox, .product-card")[:5]:
            title_el = item.select_one(".sku-title, .clipProdName, .product-card__title")
            price_el = item.select_one(".final-price, .clipProdPrice, .product-card__price")
            link_el = item.select_one("a")
            img_el = item.select_one("img")

            if title_el and price_el and link_el:
                price_text = price_el.text.replace("\u20b9", "").replace(",", "").strip()
                try:
                    price = int(float(price_text))
                except ValueError:
                    continue
                if price <= max_price_inr:
                    results.append({
                        "title": title_el.text.strip(),
                        "price_inr": price,
                        "url": "https://www.pepperfry.com" + link_el.get("href", ""),
                        "image_url": img_el.get("src", "") if img_el else "",
                        "source": "Pepperfry",
                    })

        write_cache(key, results)
        return results

    except Exception as e:
        print(f"[Scraper] Pepperfry error: {e}")
        return []


def search_all(query: str, max_price_inr: int) -> list[dict]:
    """Search across all supported Indian e-commerce sites."""
    results = []
    results.extend(search_amazon_in(query, max_price_inr))
    results.extend(search_pepperfry(query, max_price_inr))
    return results
