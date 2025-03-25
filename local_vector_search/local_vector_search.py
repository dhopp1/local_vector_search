from importlib import import_module
from langdetect import detect
from mtranslate import translate
import polars as pl
from sentence_transformers import SentenceTransformer


class local_vs:
    """Primary class of the library, manage and embed the corpus
    parameters:
        :metadata_path: str: path to the metadata.csv file. Must have at least the column "filepath" corresponding to the name of the files in the 'files_path' directory
        :files_path: str: folder path containing the .txt files of the documents
        :filepath_col_name: str: name of the column in the metadata that contains the file names
        :model: gensim.models.doc2vec.Doc2Vec or str: if using a doc2vec model, the model with its hyperparameters, if using a pre-trained embedding model, its name
        :tokenizer_name: str: name of the tokenizer, options: https://huggingface.co/docs/transformers/model_doc/auto
        :clean_text_function: function: function that takes a single input string and returns an output string. Text will go through this process before being passed as a query for the vector similarity search
        :embeddings_path: str: if already generated the embeddings, the path to the parquet file where they are saved.
        :doc2vec_path: str: if already generated the embeddings, the path to the doc2vec pickle model
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
    ):
        self.embed = import_module("local_vector_search.embed")
        self.misc = import_module("local_vector_search.misc")
        self.text_cleaning = import_module("local_vector_search.text_cleaning")

        self.metadata_path = metadata_path
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

        if embeddings_path is not None:
            self.embeddings_df = pl.read_parquet(embeddings_path)
            # finding corpus language
            languages = []
            for row in self.embeddings_df.sample(
                n=min(100, len(self.embeddings_df)), shuffle=True
            ).iter_rows():  # only do max 100 rows of the embeddings df to save time
                row_dict = dict(zip(self.embeddings_df.columns, row))
                languages.append(detect(row_dict["chunk"]))
            self.corpus_language = max(set(languages), key=languages.count)

    def embed_docs(
        self,
        chunk_size=700,
        chunk_overlap=150,
        embeddings_path=None,
        model_path=None,
        include_metadata=False,
        quiet=True,
    ):
        """Chunk and embed the documents
        parameters:
            :chunk_size: int: how many tokens each chunk should be
            :chunk_overlap: int: how much overlap each chunk should have
            :write_path: str: where to write out the parquet file that will contain the embeddings
            :model_path: str: if using doc2vec, where to write the doc2vec model out
            :include_metadata: bool: whether nor not to include the metadata in the chunk so it will be searched in the vector search
            :quiet: bool: whether or not to print out the embedding progress
        """

        self.include_metadata = include_metadata

        final_df, corpus_language = self.embed.embed_docs(
            metadata_path=self.metadata_path,
            files_path=self.files_path,
            filepath_col_name=self.filepath_col_name,
            tokenizer_name=self.tokenizer_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            quiet=quiet,
            write_path=embeddings_path,
            model_path=model_path,
            model=self.model,
            include_metadata=include_metadata,
        )

        self.embeddings_path = embeddings_path
        self.embeddings_df = pl.read_parquet(embeddings_path)
        self.corpus_language = corpus_language

        if model_path is not None:
            self.model = self.misc.pickle_load(model_path)

    def get_top_n(
        self,
        query,
        top_n=3,
        distance_metric="cosine",
        chunk_text_format="Here is the context information:\n\n|Excerpt metadata: '{}'\n\nExcerpt: '{}'\n\n\n\n",
    ):
        """Retrieve top n chunks according to a query
        parameters:
            :query: str: the new query
            :top_n: int: top n chunks to retrieve
            :distance_metric: str: "cosine" or "euclidean"
            :chunk_text_format: str: how to format the retrieved chunks, two {}'s, first will insert the metadata, second will insert the chunk. Anything you put in frot of a '|' will only appear in the beginning of the retrieval, after tha will appear for every chunk
        """

        query_lang = detect(query)

        if query_lang != self.corpus_language:
            query = translate(query, self.corpus_language, query_lang)

        response = self.embed.get_top_n(
            query=query,
            final_df=self.embeddings_df,
            clean_text_function=self.clean_text_function,
            model=self.model,
            top_n=top_n,
            distance_metric=distance_metric,
            chunk_text_format=chunk_text_format,
            include_metadata=self.include_metadata,
        )

        return response
