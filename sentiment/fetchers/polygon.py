import asyncio
import os
from collections import Counter
from typing import Literal

import httpx

BASE_URL = "https://api.polygon.io/v2/reference/news"


def get_dominant_sentiment(
    sentiments: list[Literal["positive", "negative", "neutral"]],
) -> str:
    """
    Summarizes a list of sentiment strings into a single sentiment.

    Args:
        sentiments: List of sentiment strings.

    Returns:
        A single summarized sentiment string.
    """
    if not sentiments:
        return "neutral"

    return Counter(sentiments).most_common(1)[0][0]


async def fetch_news_for_ticker(
    client: httpx.AsyncClient, ticker: str, limit: int
) -> list[dict[str, str]]:
    """
    Asynchronously fetches news for a single ticker.

    Args:
        client: Shared HTTP client.
        ticker: Stock ticker symbol.
        limit: Number of news articles to fetch.

    Returns:
        A list of news articles as dictionaries.
    """
    params = {
        "apiKey": os.environ.get("POLYGON_API_SECRET_ACCESS_KEY"),
        "limit": limit,
        "ticker": ticker.upper(),
        "sort": "published_utc",
        "order": "desc",
    }

    try:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        return [
            {
                "title": item.get("title", ""),
                "summary": item.get("description", ""),
                "sentiment": get_dominant_sentiment(
                    [insight.get("sentiment") for insight in item.get("insights")]
                ),
                "published_utc": item.get("published_utc", ""),
                "url": item.get("article_url", ""),
            }
            for item in data.get("results", [])
        ]

    except httpx.HTTPStatusError as http_err:
        print(f"[ERROR] HTTP error for {ticker}: {http_err.response.status_code}")
    except httpx.RequestError as req_err:
        print(f"[ERROR] Network error for {ticker}: {req_err}")
    except Exception as e:
        print(f"[ERROR] Unexpected error for {ticker}: {e}")

    return []


async def fetch_polygon_news(
    tickers: list[str], limit: int = 10
) -> list[dict[str, str]]:
    """
    Asynchronously fetches news for multiple tickers using batched async calls.

    Args:
        limit: Number of news articles per ticker.

    Returns:
        Combined list of news articles from all tickers.
    """
    all_news: list[dict[str, str]] = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = [fetch_news_for_ticker(client, ticker, limit) for ticker in tickers]
        results = await asyncio.gather(*tasks)

        for result in results:
            all_news.extend(result)

    return all_news
