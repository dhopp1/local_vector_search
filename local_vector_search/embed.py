import os
import polars as pl
from scipy.spatial.distance import cosine, euclidean

from local_vector_search.misc import robust_detect, pickle_save
from local_vector_search.text_cleaning import chunk_text, clean_text, yield_docs


def embed_docs(
    metadata,
    files_path,
    filepath_col_name,
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    chunk_size=700,
    chunk_overlap=150,
    quiet=True,
    write_path=None,
    model_path=None,
    model=None,
    include_metadata=False,
    include_chunk_id_metadata_string=False,
    text_ids=None,
):
    # chunking
    counter = 0

    if text_ids is None:
        files = [_ for _ in os.listdir(files_path) if ".txt" in _]
    else:
        files = metadata.filter(pl.col("text_id").is_in(text_ids))["filepath"].to_list()

    languages = []
    for file_name in files:
        if not (quiet):
            print(f"Chunking and embedding doc {counter+1}/{len(files)}")

        with open(f"{files_path}{file_name}", "r") as file:
            s = file.read()
            languages.append(robust_detect(s))

            doc_metadata = metadata.filter(pl.col(filepath_col_name) == file_name).drop(
                filepath_col_name,
                "vector_weight",
            )

            doc_weight = (
                metadata.filter(pl.col(filepath_col_name) == file_name)
                .select("vector_weight")
                .item()
            )

            if include_chunk_id_metadata_string:
                exclude_cols = ["text_id"]
            else:
                exclude_cols = ["text_id", "chunk_id"]

            if len(doc_metadata.columns) > 0:
                metadata_string = " | ".join(
                    f"{col}: {val}"
                    for col, val in zip(doc_metadata.columns, doc_metadata.row(0))
                    if col not in exclude_cols
                )
            else:
                metadata_string = ""

            # chunk the document
            chunks, page_nums = chunk_text(
                s,
                tokenizer_name,
                chunk_size,
                chunk_overlap,
                include_metadata,
                metadata_string,
            )

            df = pl.DataFrame({"chunk": chunks, "page_num": page_nums})

            # adding page numbers to metadata
            df = df.with_columns(
                pl.col("page_num")
                .map_elements(
                    lambda x: f"{metadata_string} | page number(s): {x}",
                    return_dtype=pl.String,
                )
                .alias("metadata_string")
            )

            for col in doc_metadata.columns:
                df = df.with_columns(pl.lit(doc_metadata.select(col).item()).alias(col))

            # adding weight
            df = df.with_columns(pl.lit(doc_weight).alias("vector_weight"))

            df = df.select(
                doc_metadata.columns + ["metadata_string", "chunk", "vector_weight"]
            )

            if str(type(model)) != "<class 'gensim.models.doc2vec.Doc2Vec'>":
                df = df.with_columns(
                    pl.Series("embedding", model.encode(df.to_pandas()["chunk"]))
                )

            if counter == 0:
                final_df = df
            else:
                final_df = final_df.vstack(df)

            counter += 1

    # doc2vec
    if str(type(model)) == "<class 'gensim.models.doc2vec.Doc2Vec'>":
        corpus = [x for x in yield_docs(final_df)]
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count, epochs=10)

        # adding embeddings to final_df
        final_df = final_df.with_columns(pl.Series("embedding", model.dv.vectors))

        if model_path is not None:
            pickle_save(model, model_path)

    # adding chunk ids
    final_df = final_df.with_columns(
        pl.int_range(0, final_df.height, dtype=pl.Int64).alias("chunk_id")
    )
    final_df = final_df.select(
        ["chunk_id"] + [col for col in final_df.columns if col != "chunk_id"]
    )

    # writing out parquet file
    if write_path is not None:
        final_df.write_parquet(write_path)

    # getting the most common corpus language
    corpus_language = max(set(languages), key=languages.count)

    return final_df, corpus_language


def get_top_n(
    query,
    final_df,
    text_ids=[],
    clean_text_function=None,
    model=None,
    top_n=3,
    distance_metric="cosine",
    chunk_text_format="Excerpt metadata: {}\n\nExcerpt: {}\n\n\n\n",
    include_metadata=False,
):
    "return the top chunks based on distance"

    if distance_metric == "cosine":
        dist_func = cosine
    elif distance_metric == "euclidean":
        dist_func = euclidean

    if clean_text_function is None:
        clean_text_function = clean_text

    transformed_query = clean_text_function(query)

    # filter for desired text ids
    if len(text_ids) > 0:
        final_df = final_df.filter(pl.col("text_id").is_in(text_ids))

    if top_n is None:
        top_n = len(final_df)

    # docv2vec
    try:
        new_vector = model.infer_vector(transformed_query.split())
    # transformer
    except:
        new_vector = model.encode(transformed_query)

    # calculating distance
    return_df = (
        final_df.with_columns(
            (
                pl.col("embedding").map_elements(
                    lambda x: dist_func(x, new_vector), return_dtype=pl.Float64
                )
                * (1 / pl.col("vector_weight"))
            ).alias("vector_distance")
        )
        .sort(by="vector_distance", descending=False)
        .limit(top_n)
    )

    chunk_ids = return_df["chunk_id"].to_list()

    # does the text format have a |, in which case this comes first
    if "|" in chunk_text_format:
        return_string = chunk_text_format.split("|")[0]
        chunk_text_format = chunk_text_format.split("|")[1]
    else:
        return_string = ""

    for row in return_df.iter_rows():
        row_dict = dict(zip(return_df.columns, row))

        if (
            include_metadata
        ):  # don't include in this case because metadata is in the chunk itself
            metadata_string = ""
        else:
            metadata_string = row_dict["metadata_string"]

        return_string += chunk_text_format.format(metadata_string, row_dict["chunk"])

    return {
        "response": return_string,
        "chunk_ids": chunk_ids,
    }


def retrieve_chunks(embeddings_df, chunk_ids):
    "retrieve text of chunks given a list of chunk_ids"
    filtered_df = embeddings_df.filter(pl.col("chunk_id").is_in(chunk_ids))
    filtered_df = (
        filtered_df.with_columns(
            pl.col("chunk_id")
            .map_elements(
                lambda x: {v: i for i, v in enumerate(chunk_ids)}.get(x, len(chunk_ids))
            )
            .alias("sort_key")
        )
        .sort("sort_key")
        .drop("sort_key")
    )

    return {
        "metadata": filtered_df["metadata_string"].to_list(),
        "chunks": filtered_df["chunk"].to_list(),
    }
