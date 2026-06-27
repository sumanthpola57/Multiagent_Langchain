from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from src.tools.tools import (
    web_search,
    ddg_search,
    scrape_url,
    wikipedia,
    arxiv,
    save_report,
)
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# ── Cache setup ───────────────────────────────────────────────────────────────
set_llm_cache(InMemoryCache())

# ── Models ────────────────────────────────────────────────────────────────────
llm_fast  = ChatGroq(model="llama-3.1-8b-instant",    temperature=0)
llm_smart = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)


# ── Agent 1: Search Agent ─────────────────────────────────────────────────────
def build_search_agent():
    return create_react_agent(
        model=llm_fast,
        tools=[web_search, ddg_search, wikipedia, arxiv],
        prompt=(
            "You are an expert AI and technology research assistant. "
            "When given a topic, always assume it is related to AI, machine learning, "
            "or technology unless clearly stated otherwise."
            "\n\nYour search strategy:"
            "\n1. First use web_search for recent news and articles"
            "\n2. Use wikipedia for background and definitions"
            "\n3. Use arxiv for academic papers (especially for AI/ML topics)"
            "\n4. If web_search fails, use ddg_search as backup"
            "\n5. Always do minimum 3 searches with different query angles"
            "\n\nIMPORTANT: RAG = Retrieval-Augmented Generation in AI. "
            "Never confuse with music ragas."
            "\n\nReturn a comprehensive summary with ALL URLs found. "
            "Organize results by source type (news, academic, reference)."
        ),
    )


# ── Agent 2: Reader Agent ─────────────────────────────────────────────────────
def build_reader_agent():
    return create_react_agent(
        model=llm_fast,
        tools=[scrape_url, wikipedia],
        prompt=(
            "You are an expert web content extractor and researcher. "
            "Given a list of URLs and search context, extract deep, detailed content."
            "\n\nYour strategy:"
            "\n1. Only scrape URLs that are explicitly provided in the user message"
            "\n2. Scrape the top 3 URLs using scrape_url"
            "\n3. If a URL returns an error or skip message, move to the next one immediately"
            "\n4. If all URLs fail, use wikipedia to get information directly"
            "\n5. Never give up — always return something useful"
            "\n\nSTRICT URL RULES — NEVER scrape these, ever:"
            "\n- youtube.com or youtu.be (videos cannot be scraped)"
            "\n- twitter.com or x.com"
            "\n- instagram.com, facebook.com, tiktok.com, linkedin.com"
            "\n- reddit.com"
            "\n- researchgate.net, academia.edu, jstor.org"
            "\n- Any URL ending in .pdf"
            "\nIf you see any of the above in the list, skip them and use the next URL."
            "\n\nExtract and return: key facts, statistics, quotes, expert opinions, "
            "and any data tables or numbers you find. Be thorough and detailed."
        ),
    )


# ── Writer Chain ──────────────────────────────────────────────────────────────
writer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a world-class research writer and analyst. "
     "You write clear, detailed, well-structured reports that are factual, insightful, and engaging. "
     "Always use proper markdown formatting with headers, bullet points, bold key terms, and tables where relevant. "
     "Focus on AI and technology context unless clearly stated otherwise. "
     "CRITICAL: Only write content based on the research provided. "
     "Never make up or assume information not present in the research. "
     "If the research lists specific types, names, or categories — use exactly those. "
     "If the research mentions specific examples — include exactly those. "
     "The research is your only source of truth. "
     "Do NOT include YouTube links in the Sources section — only include article, blog, and academic URLs."),
    ("human", """Write a comprehensive, professional research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report using this exact format:

# Research Report: {topic}

## Introduction
(3 paragraphs: what it is, why it matters, what this report covers)

## Key Findings
(Minimum 5 detailed findings. Each finding should have:
 - A bold heading
 - 2-3 sentences of explanation
 - Supporting evidence or examples from the research)

## Types / Categories
(If applicable, list and explain different types, variants, or categories with a comparison table)

## Real-World Applications
(3-5 concrete use cases with examples)

## Limitations & Challenges
(Honest discussion of current problems and open questions)

## Future Outlook
(Where is this heading? What developments are expected?)

## Conclusion
(Strong 2-paragraph summary and key takeaways)

## Sources
(List all article and academic URLs from the research, numbered. Skip YouTube and social media links.)

---
Write minimum 800 words. Be detailed, factual, and professional. Use markdown tables where comparisons are needed."""),
])

writer_chain = writer_prompt | llm_smart | StrOutputParser()


# ── Critic Chain ──────────────────────────────────────────────────────────────
critic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a sharp, constructive research critic, editor, and fact-checker. "
     "You give honest, specific, and actionable feedback that helps improve research quality."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact markdown format:

## Score: X/10

## Strengths
- (specific strength with example from report)
- (specific strength with example from report)
- (specific strength with example from report)

## Areas to Improve
- (specific issue with suggestion on how to fix it)
- (specific issue with suggestion on how to fix it)
- (specific issue with suggestion on how to fix it)

## Missing Information
(What important aspects of the topic were not covered? Be specific.)

## Fact Check Flags
(List any claims that seem questionable, outdated, or need verification. If all facts seem solid, say so.)

## Depth Assessment
Rate each section: Introduction / Key Findings / Analysis / Conclusion
Format: Section Name: X/10 — one line comment

## One Line Verdict
(Honest single sentence summary of the report quality)

## Suggested Title
(A more compelling, specific title for this report)

## Quick Fix Priority
(The single most important thing to fix to improve this report)"""),
])

critic_chain = critic_prompt | llm_smart | StrOutputParser()