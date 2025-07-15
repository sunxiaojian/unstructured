"""
Text chunking implementation using LangChain text splitters.
This module provides various strategies for splitting text into chunks,
including character-based, token-based, and semantic-based approaches.
"""

from __future__ import annotations

import re
from typing import Iterable, Any, List, Dict, Optional, Callable, Tuple, Union
from langchain.text_splitter import (
    TokenTextSplitter,
    TextSplitter,
    Language,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    LatexTextSplitter,
    PythonCodeTextSplitter,
    KonlpyTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    split_text_on_tokens,
    SentenceTransformersTokenTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    CharacterTextSplitter
)
from unstructured.documents.elements import Element, ElementMetadata, Text
from pydantic import BaseModel, Field


def chunk_by_langchain(
    elements: Iterable[Element],
    *,
    max_characters: Optional[int] = 2000,
    overlap: Optional[int] = 200,
    chunk_strategy: str = "recursive",
    **splitter_kwargs: Any
) -> List[Element]:
    """
    Split document elements into chunks using LangChain text splitters.

    Args:
        elements: Iterator of document elements to chunk
        chunk_strategy: Name of chunking strategy to use, defaults to "recursive"
        **splitter_kwargs: Additional parameters to pass to the splitter

    Returns:
        List of chunked document elements

    Raises:
        ValueError: If an invalid chunking strategy is specified
    """
    # Convert elements to text and metadata lists
    texts = [str(e) for e in elements]
    metadata = [e.metadata if hasattr(e, 'metadata') else ElementMetadata() for e in elements]

    # Get appropriate splitter instance
    splitter = _get_langchain_splitter(chunk_strategy, chunk_size= max_characters, chunk_overlap=overlap, **splitter_kwargs)

    chunks = []
    for text, meta in zip(texts, metadata):
        # Split each text and create new Text elements with original metadata
        text_chunks = splitter.split_text(text)
        chunks.extend(Text(text=chunk, metadata=meta) for chunk in text_chunks)
    return chunks


def _get_langchain_splitter(chunk_strategy: str, chunk_size: int, chunk_overlap: int, **kwargs: Any) -> Any:
    """
    Get LangChain splitter instance based on strategy.

    Args:
        chunk_strategy: Name of chunking strategy
        **kwargs: Parameters to pass to the splitter

    Returns:
        Splitter instance

    Raises:
        ValueError: If an invalid strategy is specified
    """
    # Define mapping of strategy names to splitter classes
    strategy_to_splitter_map = {
        "character": CharacterTextSplitter,
        "recursive": RecursiveCharacterTextSplitter,
        "token": TokenTextSplitter,
        "markdown": MarkdownTextSplitter,
        "python": PythonCodeTextSplitter,
        "latex": LatexTextSplitter,
        "nltk": NLTKTextSplitter,
        "spacy": SpacyTextSplitter,
        "html_header": HTMLHeaderTextSplitter,
        "sentence_transformers": SentenceTransformersTokenTextSplitter,
        "language": Language
    }

    if chunk_strategy not in strategy_to_splitter_map:
        raise ValueError(f"Invalid chunking strategy: {chunk_strategy}. Valid options: {list(strategy_to_splitter_map.keys())}")

    # Set default parameters
    defaults = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }

    # Special strategy defaults
    if chunk_strategy == "sentence_transformers":
        defaults["model_name"] = "sentence-transformers/all-mpnet-base-v2"
    elif chunk_strategy == "language":
        defaults["language"] = Language.PYTHON

    # Merge user parameters with defaults
    params = {**defaults, **kwargs}

    # Get allowed parameters for strategy
    allows = get_splitter_params(chunk_strategy)
    allows_kwargs = {k: v for k, v in params.items() if k in allows.keys()}

    # Special handling for specific strategies
    if chunk_strategy == "spacy":
        return SpacyTextSplitter(**allows_kwargs)
    if chunk_strategy == "nltk":
        return NLTKTextSplitter(**allows_kwargs)
    if chunk_strategy == "html_header":
        return HTMLHeaderTextSplitter(**allows_kwargs)
    if chunk_strategy == "sentence_transformers":
        return SentenceTransformersTokenTextSplitter(**allows_kwargs)
    if chunk_strategy == "language":
        return RecursiveCharacterTextSplitter.from_language(**allows_kwargs)

    # Create and return splitter instance
    return strategy_to_splitter_map[chunk_strategy](**allows_kwargs)

def get_available_strategies() -> List[str]:
    """
    Get all available chunking strategy names.

    Returns:
        List of strategy names
    """
    return [
        "character",
        "recursive",
        "token",
        "markdown",
        "python",
        "latex",
        "nltk",
        "spacy",
        "html_header",
        "sentence_transformers",
        "language"
    ]


def get_splitter_params(strategy: str) -> Dict[str, Any]:
    """
    Get parameter documentation for specified chunking strategy.

    Args:
        strategy: Strategy name

    Returns:
        Dictionary of parameter documentation

    Raises:
        ValueError: If invalid strategy specified
    """

    base_params = {
        "chunk_size": ("Target chunk size in characters", 4000),
        "chunk_overlap": ("Overlap size between chunks in characters", 200),
        "length_function": ("Function to calculate text length", len),
        "keep_separator": ("Keep separator in output chunks", False),
        "add_start_index": ("Add start index to metadata", False),
        "strip_whitespace": ("Strip whitespace from chunks", True),
    }

    param_docs = {
        "character": {
            **base_params,
            "is_separator_regex": ("Whether separator is a regex pattern", False),
            "separator": ("Text separator, defaults to '\\n\\n'", "\n\n"),
        },
        "recursive": {
            **base_params,
            "separators": ("List of separators for recursive splitting", ["\n\n", "\n", " ", ""]),
            "is_separator_regex": ("Whether separators are regex patterns", False),
        },
        "token": {
            **base_params,
            "encoding_name": ("Tokenizer name (e.g., 'cl100k_base')", "cl100k_base"),
            "model_name": ("Hugging Face model name", None),
            "allowed_special": ("Allowed special tokens", "all"),
            "disallowed_special": ("Disallowed special tokens", None)
        },
        "markdown": {
            **base_params,
            "separators": ("List of separators for recursive splitting", ["\n\n", "\n", " ", ""]),
            "is_separator_regex": ("Whether separators are regex patterns", False),
        },
        "python": {
            **base_params,
            "separators": ("List of separators for recursive splitting", ["\n\n", "\n", " ", ""]),
            "is_separator_regex": ("Whether separators are regex patterns", False),
        },
        "latex": {
            **base_params,
            "separators": ("List of separators for recursive splitting", ["\n\n", "\n", " ", ""]),
            "is_separator_regex": ("Whether separators are regex patterns", False),
        },
        "nltk": {
            **base_params,
            "separator": ("Custom separator (default: NLTK sentence tokenizer)", "\n\n"),
            "language": ("", "english"),
            "use_span_tokenize" : ("Use span tokenization instead of sentence tokenization", False)
        },
        "spacy": {
            **base_params,
            "pipeline": ("spaCy pipeline name", "en_core_web_sm"),
            "separator": ("Custom separator (default: spaCy sentence tokenizer)", "\n\n"),
            "max_length": ("Maximum length of sentence", 1_000_000),
        },
        "html_header": {
            "headers_to_split_on": ("HTML tags to split on", []),
            "return_each_element": ("Return each element separately", False),
        },
        "sentence_transformers": {
            **base_params,
            "chunk_overlap": ("Overlap size between chunks in characters", 50),  # 覆盖基础值
            "model_name": ("Sentence transformer model name", "sentence-transformers/all-mpnet-base-v2"),
            "tokens_per_chunk": ("Maximum number of tokens per chunk", None),
        },
        "language": {
            **base_params,
            "is_separator_regex": ("Whether separators are regex patterns", False),
            "separators": ("Custom separator list", None),
            "language": ("Programming language enum", "python"),
        }
    }

    if strategy not in param_docs:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return param_docs[strategy]


class ChunkingConfig(BaseModel):
    """Configuration model for text chunking."""

    strategy: str = Field(
        default="recursive",
        description="Name of chunking strategy"
    )
    chunk_size: int = Field(
        default=1000,
        description="Target chunk size",
        gt=0
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap size between chunks",
        ge=0
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model name for sentence_transformers strategy"
    )
    separators: Optional[List[str]] = Field(
        default=None,
        description="Separator list for recursive strategy"
    )
    headers: Optional[List[Tuple[str, str]]] = Field(
        default=None,
        description="Header configuration for html_header strategy"
    )
    language: Optional[str] = Field(
        default=None,
        description="Programming language for language strategy"
    )

    class Config:
        schema_extra = {
            "example": {
                "strategy": "recursive",
                "chunk_size": 1500,
                "chunk_overlap": 300,
                "separators": ["\n\n", "\n", " ", ""]
            }
        }


def create_chunking_pipeline(config: ChunkingConfig) -> Callable:
    """
    Create a text chunking pipeline from configuration.

    Args:
        config: Chunking configuration object

    Returns:
        Chunking pipeline function
    """
    def chunking_pipeline(elements: Iterable[Element]) -> List[Element]:
        # Extract configuration parameters
        kwargs = config.dict(exclude={"strategy"})

        # Remove irrelevant parameters
        if config.strategy != "sentence_transformers":
            kwargs.pop("model_name", None)
        if config.strategy != "html_header":
            kwargs.pop("headers", None)
        if config.strategy != "language":
            kwargs.pop("language", None)

        # Set default separators for recursive strategy
        if config.strategy == "recursive" and not kwargs.get("separators"):
            kwargs["separators"] = ["\n\n", "\n", " ", ""]

        # Execute chunking
        return chunk_by_langchain(elements, chunk_strategy=config.strategy, **kwargs)

    return chunking_pipeline
