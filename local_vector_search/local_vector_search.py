from importlib import import_module
from mtranslate import translate
import polars as pl
from sentence_transformers import SentenceTransformer

from local_vector_search.misc import robust_detect


class local_vs:
    """Primary class of the library, manage and embed the corpus
    parameters:
        :metadata_path: str: path to the metadata.csv file. Must have at least the column "filepath" corresponding to the name of the files in the 'files_path' directory. Include a column named 'vector_weight' in the metadata csv to weight distances. 1 = no alteration to the distance, 0.5 = distance will be multiplied by 1/0.5, so be made 2x farther. Not including this column will result in equal weight for all documents.
        :files_path: str: folder path containing the .txt files of the documents
        :filepath_col_name: str: name of the column in the metadata that contains the file names
        :model: gensim.models.doc2vec.Doc2Vec or str: if using a doc2vec model, the model with its hyperparameters, if using a pre-trained embedding model, its name
        :tokenizer_name: str: name of the tokenizer, options: https://huggingface.co/docs/transformers/model_doc/auto
        :clean_text_function: function: function that takes a single input string and returns an output string. Text will go through this process before being passed as a query for the vector similarity search
        :embeddings_path: str: if already generated the embeddings, the path to the parquet file where they are saved.
        :doc2vec_path: str: if already generated the embeddings, the path to the doc2vec pickle model
        :include_metadata: bool: whether nor not to include the metadata in the chunk so it will be searched in the vector search
        :include_chunk_id_metadata_string: bool: whether or not to include the chunk id in the metadata string at embedding time
    """

    def __init__(
        self,
        metadata_path=None,
        files_path=None,
        filepath_col_name="filepath",
        model="all-MiniLM-L6-v2",
        tokenizer_name="meta-llama/Llama-2-7b-hf",
        clean_text_function=None,
        embeddings_path=None,
        doc2vec_path=None,
        include_metadata=False,
        include_chunk_id_metadata_string=False,
    ):
        self.embed = import_module("local_vector_search.embed")
        self.misc = import_module("local_vector_search.misc")
        self.text_cleaning = import_module("local_vector_search.text_cleaning")

        self.metadata_path = metadata_path
        metadata = pl.read_csv(metadata_path)

        # add a text_id column if it's not already there
        if "text_id" not in metadata.columns:
            metadata = metadata.with_row_index(name="text_id", offset=1)

        # add a weight column if it's not already there
        if "vector_weight" not in metadata.columns:
            metadata = metadata.with_columns(pl.lit(1.0).alias("vector_weight"))

        metadata.write_csv(metadata_path)
        self.metadata = metadata

        self.files_path = files_path
        self.filepath_col_name = filepath_col_name

        if str(type(model)) != "<class 'gensim.models.doc2vec.Doc2Vec'>":
            self.model = SentenceTransformer(model)
        else:
            self.model = model

        self.tokenizer_name = tokenizer_name
        self.clean_text_function = clean_text_function
        self.embeddings_path = embeddings_path
        self.doc2vec_path = doc2vec_path
        self.include_metadata = include_metadata
        self.include_chunk_id_metadata_string = include_chunk_id_metadata_string

        if embeddings_path is not None:
            self.embeddings_df = pl.read_parquet(embeddings_path)
            # finding corpus language
            languages = []
            for row in self.embeddings_df.sample(
                n=min(100, len(self.embeddings_df)), shuffle=True
            ).iter_rows():  # only do max 100 rows of the embeddings df to save time
                row_dict = dict(zip(self.embeddings_df.columns, row))
                languages.append(robust_detect(row_dict["chunk"]))
            try:
                self.corpus_language = max(set(languages), key=languages.count)
            except:
                self.corpus_language = "en"

    def embed_docs(
        self,
        chunk_size=700,
        chunk_overlap=150,
        embeddings_path=None,
        model_path=None,
        text_ids=None,
        quiet=True,
    ):
        """Chunk and embed the documents
        parameters:
            :chunk_size: int: how many tokens each chunk should be
            :chunk_overlap: int: how much overlap each chunk should have
            :write_path: str: where to write out the parquet file that will contain the embeddings
            :model_path: str: if using doc2vec, where to write the doc2vec model out
            :text_ids: list[int]: list of text ids in case want to embed in a loop for status printing
            :quiet: bool: whether or not to print out the embedding progress
        """

        final_df, corpus_language = self.embed.embed_docs(
            metadata=self.metadata,
            files_path=self.files_path,
            filepath_col_name=self.filepath_col_name,
            tokenizer_name=self.tokenizer_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            quiet=quiet,
            write_path=embeddings_path,
            model_path=model_path,
            model=self.model,
            include_metadata=self.include_metadata,
            include_chunk_id_metadata_string=self.include_chunk_id_metadata_string,
            text_ids=text_ids,
        )

        self.embeddings_path = embeddings_path
        try:
            self.embeddings_df = pl.read_parquet(embeddings_path)
        except:
            pass
        self.corpus_language = corpus_language

        if model_path is not None:
            self.model = self.misc.pickle_load(model_path)

        if embeddings_path is None:
            return final_df

    def get_top_n(
        self,
        query,
        text_ids=[],
        top_n=3,
        distance_metric="cosine",
        chunk_text_format="Here is the context information:\n\n|Excerpt metadata: '{}'\n\nExcerpt: '{}'\n\n\n\n",
    ):
        """Retrieve top n chunks according to a query
        parameters:
            :query: str: the new query
            :text_ids: list[int]: list of text ids to include in the search. Leave an empty list to search all
            :top_n: int: top n chunks to retrieve
            :distance_metric: str: "cosine" or "euclidean"
            :chunk_text_format: str: how to format the retrieved chunks, two {}'s, first will insert the metadata, second will insert the chunk. Anything you put in frot of a '|' will only appear in the beginning of the retrieval, after tha will appear for every chunk
        returns:
            :dict: dictionary, 'response' = text of chunks, 'chunk_ids' = list of top n closesrtt chunk ids
        """

        query_lang = robust_detect(query)

        if query_lang != self.corpus_language:
            query = translate(query, self.corpus_language, query_lang)

        response = self.embed.get_top_n(
            query=query,
            final_df=self.embeddings_df,
            text_ids=text_ids,
            clean_text_function=self.clean_text_function,
            model=self.model,
            top_n=top_n,
            distance_metric=distance_metric,
            chunk_text_format=chunk_text_format,
            include_metadata=self.include_metadata,
        )

        return response

    def retrieve_chunks(self, chunk_ids):
        "Retrieve metadata strings and chunks given chunk ids"
        return self.embed.retrieve_chunks(self.embeddings_df, chunk_ids)
