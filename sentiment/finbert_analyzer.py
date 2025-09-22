import attr
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Pipeline,
    pipeline,
)


@attr.s(auto_attribs=True, slots=True)
class FinBERTAnalyzer:
    """
    An analyzer class for financial sentiment using the FinBERT model.
    Implements lazy loading of the pipeline using HuggingFace Transformers.
    """

    model_name: str = "ProsusAI/finbert"
    pipeline: Pipeline = attr.ib(init=False)

    def __attrs_post_init__(self) -> None:
        """
        Initializes the tokenizer, model, and pipeline after attribute setup.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        device = 0 if torch.cuda.is_available() else -1

        self.pipeline = pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer, device=device
        )

    def analyze(self, texts: list[str]) -> list[dict[str, str | float]]:
        """
        Analyzes sentiment for a batch of texts.

        Args:
            texts: List of cleaned strings.

        Returns:
            List of dictionaries containing sentiment label and score.
        """
        outputs = self.pipeline(texts)
        return [
            {"label": output["label"], "score": round(output["score"], 4)}
            for output in outputs
        ]
