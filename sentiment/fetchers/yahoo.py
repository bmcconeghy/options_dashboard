from datetime import datetime

import yfinance as yf

from sentiment.datatypes import NewsArticle


def get_news(ticker: str, num_articles: int = 10) -> list[NewsArticle]:
    """
    Fetches news articles for a given ticker using yfinance.

    Args:
        ticker: Stock ticker symbol.
        num_articles: Number of news articles to fetch.
    Returns:
        A list of news articles as dictionaries.
    """
    news_items = yf.Ticker(ticker).get_news(count=num_articles, tab="all")

    return [
        NewsArticle(
            title=item.get("content", {}).get("title", ""),
            summary=item.get("content", {}).get("summary", ""),
            url=item.get("content", {}).get("canonicalUrl", {}).get("url", ""),
            published_date=datetime.fromisoformat(
                item.get("content", {}).get(
                    "pubDate", "1999-01-01T00:00:00Z"
                )  # Default to a very old date so we can still get a date object
            ).date(),
            source="Yahoo Finance",
        )
        for item in news_items
    ]
