import pytest
from unstructured.documents.elements import Text
from unstructured.chunking.langchain import (
    chunk_by_langchain,
    get_available_strategies,
    get_splitter_params,
    ChunkingConfig,
    create_chunking_pipeline
)

@pytest.fixture
def sample_elements():
    elements = [
        Text("This is a test document.\nIt has multiple lines.\nAnd some content."),
        Text("Another document with different content.\nMore lines here.")
    ]
    return elements


@pytest.fixture
def html_elements():
    html_text = """
    <h1>Main Title</h1>
    <p>Some content under main title.</p>
    <h2>Subtitle</h2>
    <p>Content under subtitle.</p>
    """
    return [Text(html_text)]


@pytest.fixture
def markdown_elements():
    markdown_text = """
    # Main Title
    Some content under main title.
    ## Subtitle
    Content under subtitle.
    """
    return [Text(markdown_text)]


@pytest.fixture
def latex_elements():
    latex_text = r"""
    \section{Main Title}
    Some content under main title.
    \subsection{Subtitle}
    Content under subtitle.
    """
    return [Text(latex_text)]


@pytest.fixture
def code_elements():
    python_code = """
def test_function():
    print("Hello")
    return True
    """
    return [Text(python_code)]


def test_chunk_by_langchain_character(sample_elements):
    chunks = chunk_by_langchain(sample_elements, chunk_strategy="character", max_characters=20, overlap=5)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_chunk_by_langchain_recursive(sample_elements):
    chunks = chunk_by_langchain(sample_elements, chunk_strategy="recursive", max_characters=20, overlap=5)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)
    assert all(len(str(chunk)) <= 25 for chunk in chunks)


def test_chunk_by_langchain_token(sample_elements):
    chunks = chunk_by_langchain(sample_elements, chunk_strategy="token", max_characters=10, overlap=2)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_chunk_by_langchain_markdown(markdown_elements):
    chunks = chunk_by_langchain(markdown_elements, chunk_strategy="markdown", max_characters=100, overlap=20)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_chunk_by_langchain_python(code_elements):
    chunks = chunk_by_langchain(code_elements, chunk_strategy="python", max_characters=50, overlap=10)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_chunk_by_langchain_latex(latex_elements):
    chunks = chunk_by_langchain(latex_elements, chunk_strategy="latex", max_characters=50, overlap=10)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_chunk_by_langchain_nltk(sample_elements):
    chunks = chunk_by_langchain(sample_elements, chunk_strategy="nltk", max_characters=50, overlap=10)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_chunk_by_langchain_spacy(sample_elements):
    chunks = chunk_by_langchain(sample_elements, chunk_strategy="spacy", max_characters=50, overlap=10)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_chunk_by_langchain_html_header(html_elements):
    chunks = chunk_by_langchain(html_elements, chunk_strategy="html_header")
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_chunk_by_langchain_sentence_transformers(sample_elements):
    chunks = chunk_by_langchain(
        sample_elements,
        chunk_strategy="sentence_transformers",
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_chunk_by_langchain_language(code_elements):
    chunks = chunk_by_langchain(code_elements, chunk_strategy="language", language="python")
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_chunk_by_langchain_invalid_strategy(sample_elements):
    with pytest.raises(ValueError):
        chunk_by_langchain(sample_elements, chunk_strategy="invalid")

def test_get_available_strategies():
    strategies = get_available_strategies()
    assert isinstance(strategies, list)
    assert len(strategies) > 0
    expected_strategies = [
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
    assert all(strategy in strategies for strategy in expected_strategies)


def test_get_splitter_params():
    strategies = get_available_strategies()
    for strategy in strategies:
        params = get_splitter_params(strategy)
        assert isinstance(params, dict)
        assert len(params) > 0
        if strategy != "html_header":
            assert "chunk_size" in params
            assert "chunk_overlap" in params


def test_get_splitter_params_invalid():
    with pytest.raises(ValueError):
        get_splitter_params("invalid")


def test_chunking_config():
    config = ChunkingConfig(
        strategy="recursive",
        max_characters=1500,
        overlap=300,
        separators=["\n\n", "\n", " ", ""]
    )
    assert config.strategy == "recursive"
    assert config.chunk_size == 1500
    assert config.chunk_overlap == 300
    assert config.separators == ["\n\n", "\n", " ", ""]


def test_chunking_config_defaults():
    config = ChunkingConfig()
    assert config.strategy == "recursive"
    assert config.chunk_size == 1000
    assert config.chunk_overlap == 200
    assert config.model_name is None
    assert config.separators is None
    assert config.headers is None
    assert config.language is None

def test_create_chunking_pipeline(sample_elements):
    config = ChunkingConfig(
        strategy="recursive",
        max_characters=20,
        chunk_overlap=5
    )
    pipeline = create_chunking_pipeline(config)
    chunks = pipeline(sample_elements)
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_empty_elements():
    empty_elements = [Text("")]
    chunks = chunk_by_langchain(empty_elements, chunk_strategy="recursive")
    assert len(chunks) == 0


def test_large_text_chunking():
    large_text = " ".join(["word"] * 1000)
    elements = [Text(large_text)]
    chunks = chunk_by_langchain(
        elements,
        chunk_strategy="recursive",
        max_characters=100,
        overlap=20
    )
    assert len(chunks) > 1
    assert all(len(str(chunk)) <= 150 for chunk in chunks)


def test_special_characters_handling():
    special_chars = "Text with special chars: !@#$%^&*()\n\t"
    elements = [Text(special_chars)]
    chunks = chunk_by_langchain(elements, chunk_strategy="recursive")
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_html_header_with_custom_headers(html_elements):
    custom_headers = [
        ("h1", "First Level"),
        ("h2", "Second Level"),
        ("h3", "Third Level")
    ]
    chunks = chunk_by_langchain(
        html_elements,
        chunk_strategy="html_header",
        headers_to_split_on=custom_headers
    )
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_language_specific_chunking_with_different_languages(code_elements):
    languages = ["python", "js", "java", "go"]
    for lang in languages:
        chunks = chunk_by_langchain(
            code_elements,
            chunk_strategy="language",
            language=lang
        )
        assert len(chunks) > 0
        assert all(isinstance(chunk, Text) for chunk in chunks)


def test_recursive_chunking_with_custom_separators(sample_elements):
    custom_separators = ["\n\n", "\n", ".", " "]
    chunks = chunk_by_langchain(
        sample_elements,
        chunk_strategy="recursive",
        separators=custom_separators,
        max_characters=20,
        overlap=4
    )
    assert len(chunks) > 0
    assert all(isinstance(chunk, Text) for chunk in chunks)


def test_token_chunking_with_different_models(sample_elements):
    encoding_names = ["gpt2", "cl100k_base"]
    for encoding in encoding_names:
        chunks = chunk_by_langchain(
            sample_elements,
            chunk_strategy="token",
            encoding_name=encoding,
            max_characters=10,
            overlap=2
        )
        assert len(chunks) > 0
        assert all(isinstance(chunk, Text) for chunk in chunks)
