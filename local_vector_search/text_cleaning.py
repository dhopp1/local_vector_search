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


def replace_newpage_with_occurrence(text):
    # Define a function that replaces [newpage] with [newpage n]
    def replace_match(match, count=[0]):
        count[0] += 1  # Increment the count
        return f"[newpage {count[0]}]"

    # Use re.sub to replace all occurrences of [newpage]
    result = re.sub(r"\[newpage\]", replace_match, text)
    return result


def is_markdown_table(s):
    "determine if a string is a markdown table"

    lines = [line.strip() for line in s.strip().splitlines()]
    if len(lines) < 2:
        return False
    if not all("|" in line for line in lines[:2]):
        return False
    if re.match(r"^\s*\|?[\s:-]+\|[\s|:-]+\|?\s*$", lines[1]):
        return True
    return False


def chunk_text(
    text,
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    chunk_size=700,
    overlap=150,
    include_metadata=False,
    metadata_string=None,
):
    """
    Tokenizes text using a LLaMA tokenizer, splits it into overlapping chunks,
    and then reconstructs the text from chunks.

    Args:
        text: str: text to process.
        tokenizer_name: str: Hugging Face model name for the  tokenizer.
        chunk_size: in): Number of tokens per chunk.
        overlap: int: Overlap size between consecutive chunks.
        include_metadata: bool: whether nor not to include the metadata in the chunk so it will be searched in the vector search

    Returns:
        list: List of chunked texts.
    """

    # replace [newpage] with [newpage page_num]
    text = replace_newpage_with_occurrence(text)

    # determine if is markdown table
    is_markdown = is_markdown_table(text)

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the text into token IDs
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # markdown info
    if is_markdown:
        col_line = text.split("\n")[0] + "\n" + text.split("\n")[1]  # column headers
        max_tokens_one_line = max(
            [
                len(tokenizer.encode(_, add_special_tokens=False))
                for _ in text.split("\n")
            ]
        )  # see how many tokens one line is
        lines_per_chunk = max(1, int(chunk_size / max_tokens_one_line))

    # Split into chunks with overlap
    chunks = []
    page_nums = []
    last_page_num = "NA"
    if is_markdown:
        loop_range = list(
            range(0, len(text.split("\n")) - 1, lines_per_chunk)
        )  # - 1 to account for header row
    else:
        loop_range = list(range(0, len(token_ids), chunk_size - overlap))

    for i in loop_range:
        if not (is_markdown):
            # add the metadata to the chunk if required
            if include_metadata:
                metadata_tokenized = tokenizer.encode(
                    "metadata: " + metadata_string, add_special_tokens=False
                )
            else:
                metadata_tokenized = []

            # getting last page num
            chunk = token_ids[i : i + chunk_size]
            pages = re.findall(
                r"\[newpage \d+\]", tokenizer.decode(chunk, skip_special_tokens=True)
            )
            if len(pages) > 0:
                pages = [int(_.replace("[newpage", "").replace("]", "")) for _ in pages]
                last_page_num = pages[-1]
                if len(pages) > 1:
                    page_num_text = f"{pages[0]}-{pages[1]}"
                else:
                    page_num_text = str(pages[0])
            else:
                page_num_text = str(last_page_num)
            page_nums.append(page_num_text)

            # adding page number to chunk if desired
            if include_metadata:
                chunk = (
                    metadata_tokenized
                    + tokenizer.encode(
                        f" | page number(s): {page_num_text}'\n\n",
                        add_special_tokens=False,
                    )
                    + token_ids[i : i + chunk_size]
                )

            # remove [new page *] from the chunk
            chunk = tokenizer.decode(
                chunk, skip_special_tokens=True
            )  # convert back into text
            chunk = re.sub(r"\[newpage [^\]]+\]", "", chunk)  # remove newpage

            # reencode chunk
            chunk = tokenizer.encode(chunk, add_special_tokens=False)  # re-encode

        else:  # markdown table
            chunk = "\n".join(
                "\n".join(text.split("\n")[2:]).split("\n")[i : (i + (lines_per_chunk))]
            )

            if include_metadata:
                chunk = (
                    "metadata: " + metadata_string + "\n\n" + col_line + "\n" + chunk
                )
            else:
                chunk = col_line + "\n" + chunk

            chunk = tokenizer.encode(chunk, add_special_tokens=False)
            page_nums.append(f"rows {i}-{i + lines_per_chunk}")

        chunks.append(chunk)

    # Decode each chunk back into text
    chunked_texts = [
        tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks
    ]

    return chunked_texts, page_nums


def yield_docs(polars_df):
    "yield the documents of a corpus for doc2vec"

    counter = 0
    for row in polars_df.iter_rows():
        row_dict = dict(zip(polars_df.columns, row))
        yield TaggedDocument(words=row_dict["chunk"].lower().split(), tags=[counter])
        counter += 1
