from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import stopwords
import re
from transformers import AutoTokenizer


def clean_text(input_string):
    "clean text for a vector db search"

    # lower case
    input_string = input_string.lower()

    # Remove punctuation
    clean_string = re.sub(r"[^\w\s]", "", input_string)

    # Convert to lowercase
    clean_string = clean_string.lower()

    # Split into words
    words = clean_string.split()

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]

    # Join the words back into a string
    result = " ".join(filtered_words)

    return result


def chunk_text(
    text, tokenizer_name="meta-llama/Llama-2-7b-hf", chunk_size=700, overlap=150
):
    """
    Tokenizes text using a LLaMA tokenizer, splits it into overlapping chunks,
    and then reconstructs the text from chunks.

    Args:
        text: str: text to process.
        tokenizer_name: str: Hugging Face model name for the  tokenizer.
        chunk_size: in): Number of tokens per chunk.
        overlap: int: Overlap size between consecutive chunks.

    Returns:
        list: List of chunked texts.
    """

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the text into token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Split into chunks with overlap
    chunks = []
    for i in range(0, len(token_ids), chunk_size - overlap):
        chunk = token_ids[i : i + chunk_size]
        chunks.append(chunk)

    # Decode each chunk back into text
    chunked_texts = [
        tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks
    ]

    return chunked_texts


def yield_docs(polars_df):
    "yield the documents of a corpus for doc2vec"

    counter = 0
    for row in polars_df.iter_rows():
        row_dict = dict(zip(polars_df.columns, row))
        yield TaggedDocument(words=row_dict["chunk"].lower().split(), tags=[counter])
        counter += 1
