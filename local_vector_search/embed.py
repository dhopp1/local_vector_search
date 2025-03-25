import os
import polars as pl
from scipy.spatial.distance import cosine, euclidean

from local_vector_search.misc import pickle_save
from local_vector_search.text_cleaning import chunk_text, clean_text, yield_docs


def embed_docs(
    metadata_path,
    files_path,
    filepath_col_name,
    tokenizer_name="meta-llama/Llama-2-7b-hf",
    chunk_size=700,
    chunk_overlap=150,
    quiet=True,
    write_path=None,
    model_path=None,
    model=None,
):
    metadata = pl.read_csv(metadata_path)

    # chunking
    counter = 0
    files = [_ for _ in os.listdir(files_path) if ".txt" in _]
    for file_name in files:
        if not (quiet):
            print(f"Chunking and embedding doc {counter+1}/{len(files)}")

        with open(f"{files_path}{file_name}", "r") as file:
            s = file.read()

            doc_metadata = metadata.filter(pl.col(filepath_col_name) == file_name).drop(
                filepath_col_name
            )
            metadata_string = " | ".join(
                f"{col}: {val}"
                for col, val in zip(doc_metadata.columns, doc_metadata.row(0))
            )
            chunks = chunk_text(s, tokenizer_name, chunk_size, chunk_overlap)

            df = pl.DataFrame({"chunk": chunks})
            df = df.with_columns(pl.lit(metadata_string).alias("metadata_string"))
            for col in doc_metadata.columns:
                df = df.with_columns(pl.lit(doc_metadata.select(col).item()).alias(col))

            df = df.select(doc_metadata.columns + ["metadata_string", "chunk"])

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

    # writing out parquet file
    if write_path is not None:
        final_df.write_parquet(write_path)

    return final_df


def get_top_n(
    query,
    final_df,
    clean_text_function=None,
    model=None,
    top_n=3,
    distance_metric="cosine",
    chunk_text_format="Excerpt metadata: {}\n\nExcerpt: {}\n\n\n\n",
):
    "return the top chunks based on distance"

    if clean_text_function is None:
        clean_text_function = clean_text

    transformed_query = clean_text_function(query)

    if top_n is None:
        top_n = len(final_df)

    # docv2vec
    try:
        new_vector = model.infer_vector(transformed_query.split())
    # transformer
    except:
        new_vector = model.encode(transformed_query)

    if distance_metric == "cosine":
        return_df = (
            final_df.with_columns(
                pl.col("embedding")
                .map_elements(lambda x: cosine(x, new_vector), return_dtype=pl.Float64)
                .alias("vector_distance")
            )
            .sort(by="vector_distance", descending=False)
            .limit(top_n)
        )
    elif distance_metric == "euclidean":
        return_df = (
            final_df.with_columns(
                pl.col("embedding")
                .map_elements(
                    lambda x: euclidean(x, new_vector), return_dtype=pl.Float64
                )
                .alias("vector_distance")
            )
            .sort(by="vector_distance", descending=False)
            .limit(top_n)
        )

    return_string = ""
    for row in return_df.iter_rows():
        row_dict = dict(zip(return_df.columns, row))
        return_string += chunk_text_format.format(
            row_dict["metadata_string"], row_dict["chunk"]
        )

    return return_string
