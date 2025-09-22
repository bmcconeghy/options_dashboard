import asyncio

import attr

from sentiment.datatypes import NewsArticle
from sentiment.fetchers import yahoo
from sentiment.finbert_analyzer import FinBERTAnalyzer
from sentiment.utils import clean_for_finbert

ANALYZER = FinBERTAnalyzer()


def assign_sentiment(posts: list[NewsArticle]) -> list[NewsArticle]:
    """Assigns sentiment to a list of NewsArticle objects using FinBERT."""
    texts = [clean_for_finbert(post.title + " " + post.summary) for post in posts]
    results = ANALYZER.analyze(texts)

    return [
        attr.evolve(
            posts[i],
            sentiment=results[i]["label"].lower(),
            confidence=results[i]["score"],
        )
        for i in range(len(posts))
    ]


async def main() -> None:
    posts = yahoo.get_news("CRCL", num_articles=100)
    return assign_sentiment(posts)


if __name__ == "__main__":
    asyncio.run(main())
