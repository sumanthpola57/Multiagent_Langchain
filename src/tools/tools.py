from langchain.tools import tool
import requests
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import re
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ── Domains that cannot/shouldn't be scraped ──────────────────────────────────
UNSCRAPPABLE_DOMAINS = {
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "instagram.com", "facebook.com",
    "linkedin.com", "tiktok.com",
    "reddit.com",
    "researchgate.net",
    "academia.edu",
    "jstor.org",
    "springer.com",     # paywall
    "nature.com",       # paywall on deep articles
    "sciencedirect.com",
}

def _is_scrappable(url: str) -> bool:
    url_lower = url.lower()
    if not url_lower.startswith("http"):
        return False
    if url_lower.endswith(".pdf"):
        return False
    for domain in UNSCRAPPABLE_DOMAINS:
        if domain in url_lower:
            return False
    return True


# ── Tool 1: Tavily Web Search (Primary) ───────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information on a topic.
    Use this for current events, news, recent developments, and general research.
    Returns titles, URLs and snippets.
    Input should be a clear search query string.
    """
    try:
        response = tavily.search(query=query, max_results=6)

        if isinstance(response, dict):
            items = response.get('results', [])
        else:
            items = getattr(response, 'results', [])

        out = []
        for r in items:
            if isinstance(r, dict):
                title   = r.get('title', '')
                url     = r.get('url', '')
                snippet = r.get('content', '')[:400]
            else:
                title   = getattr(r, 'title', '')
                url     = getattr(r, 'url', '')
                snippet = getattr(r, 'content', '')[:400]
            out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")

        return "\n----\n".join(out)
    except Exception as e:
        return f"Tavily search failed: {str(e)}. Try using wikipedia or arxiv instead."


# ── Tool 2: DuckDuckGo Search (Fallback) ─────────────────────────────────────
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    ddg_search = DuckDuckGoSearchRun(
        name="ddg_search",
        description=(
            "Fallback web search using DuckDuckGo. "
            "Use this when web_search fails or returns poor results. "
            "Free and reliable for general information."
        )
    )
    print("✅ DuckDuckGo loaded")
except Exception as e:
    ddg_search = None
    print(f"⚠️ DuckDuckGo unavailable: {e}")


# ── Tool 3: Wikipedia ─────────────────────────────────────────────────────────
wikipedia = WikipediaQueryRun(
    name="wikipedia",
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=3000
    ),
    description=(
        "Search Wikipedia for background information, definitions, history, and concepts. "
        "Use this for foundational knowledge on any topic. "
        "Best for: scientific concepts, historical events, technical definitions, biographies."
    )
)


# ── Tool 4: ArXiv Research Papers ─────────────────────────────────────────────
arxiv = ArxivQueryRun(
    name="arxiv",
    api_wrapper=ArxivAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=2000
    ),
    description=(
        "Search ArXiv for academic research papers and scientific studies. "
        "Use this for AI, machine learning, physics, mathematics, and other technical topics. "
        "Best for: finding latest research, academic citations, scientific evidence."
    )
)


# ── Tool 5: URL Scraper ───────────────────────────────────────────────────────
@tool
def scrape_url(url: str) -> str:
    """
    Scrape and extract clean readable content from a URL.
    Use this to get full content from specific web pages found in search results.

    IMPORTANT — do NOT use this on:
    - YouTube URLs (youtube.com, youtu.be)
    - Social media (twitter.com, x.com, instagram.com, facebook.com, linkedin.com, tiktok.com)
    - Reddit (reddit.com)
    - Paywalled academic sites (researchgate.net, academia.edu, jstor.org)
    - PDF files (anything ending in .pdf)

    If one URL fails or is on the blocklist above, try a different one.
    Input should be a valid URL string starting with http:// or https://
    """
    # Hard block inside the tool — even if the agent ignores its instructions
    if not _is_scrappable(url):
        return (
            f"Skipped '{url}' — this domain cannot be scraped "
            f"(YouTube/social/paywall/PDF). Please try a different URL."
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html = response.text

        # Strategy 1 — trafilatura (best for articles/blogs)
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True
        )
        if extracted and len(extracted.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', extracted)
            return cleaned[:6000]

        # Strategy 2 — readability
        doc = Document(html)
        clean_html = doc.summary()
        soup = BeautifulSoup(clean_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        if text and len(text.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', text)
            return cleaned[:6000]

        # Strategy 3 — full page fallback
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        cleaned = re.sub(r'\s+', ' ', text)
        if cleaned:
            return cleaned[:6000]

        return "Could not extract meaningful content from this page. Try a different URL."

    except requests.exceptions.Timeout:
        return "Request timed out. Try a different URL."
    except requests.exceptions.HTTPError as e:
        return f"HTTP error {str(e)}. This site may block scrapers. Try a different URL."
    except Exception as e:
        return f"Could not scrape URL: {str(e)}. Try a different URL."


# ── Tool 6: Save Report ───────────────────────────────────────────────────────
@tool
def save_report(content: str, filename: str = "report.md") -> str:
    """
    Save the research report to a local file.
    Use this after the report is fully written.
    Input: content (the report text), filename (optional, default: report.md)
    """
    try:
        os.makedirs("reports", exist_ok=True)
        filepath = f"reports/{filename}"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Report saved successfully to {filepath}"
    except Exception as e:
        return f"Failed to save report: {str(e)}"