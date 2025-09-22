from __future__ import annotations

from datetime import datetime

import attr


@attr.define(frozen=True)
class NewsArticle:
    title: str
    summary: str
    url: str
    published_date: datetime
    sentiment: str = attr.field(default="")
    confidence: float = attr.field(default=0.0)
    source: str = attr.field(default="")
