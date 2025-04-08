# local_vector_search
Vector search for input to RAG without a vector database.

## Quick usage
```py
from local_vector_search.local_vector_search import local_vs# using transfomer pre-trained model
vs = local_vs(        metadata_path = path_to_metadata_csv,        files_path = path_to_files,        model = "all-MiniLM-L6-v2", # can be any transformers embedding model, https://huggingface.co/sentence-transformers        tokenizer_name = "meta-llama/Llama-2-7b-hf", # can be any tokenizer model, for tokenizing and getting appropriate chunk sizes according to tokens, https://huggingface.co/docs/transformers/model_doc/auto        clean_text_function=None, # any function that takes in a string and outputs a string, in case you want to edit the queries for searching the vector database
        include_metadata = False, # set to true to include metadata in the chunks, so they will be searched as well    )
    
# embed the documents
vs.embed_docs(
	chunk_size = 700, 
	chunk_overlap = 150, 
	embeddings_path = "path_to_save_embeddings.parquet",
	quiet=False,
)

# perform the vector similarity search
vs.get_top_n(query, top_n=3, distance_metric="cosine") # 'cosine' or 'euclidean' # returns the chunks in a single string, as well as the ids of the chunks

# perform the search, but only include text ids 2 and 4 in the search
vs.get_top_n(query, text_ids=[2,4], top_n=3, distance_metric="cosine") # 'cosine' or 'euclidean'

# return selected chunks from a corpus
vs.retrieve_chunks(chunk_ids=[1,5,7]) # retrieve the metadata and text of these chunks

# instantiate a vs with an already calculated embeddings dataset
vs = local_vs(embeddings_path = "path_to_save_embeddings.parquet")

# using a doc2vec custom embedding model
from gensim.models.doc2vec import Doc2Vecmodel = Doc2Vec(vector_size=100, window=5, min_count=2, epochs=10, workers=4)
vs = local_vs(        metadata_path = path_to_metadata_csv,        files_path = path_to_files,        model = model,        tokenizer_name = "meta-llama/Llama-2-7b-hf",        clean_text_function=None,    )# embed the documents
vs.embed_docs(
	chunk_size = 700, 
	chunk_overlap = 150, 
	embeddings_path = "path_to_save_embeddings.parquet",
	model_path = "path_to_save_doc2vec_model.pickle", 
	quiet = False,
)# load an already trained doc2vec embedding model
from local_vector_search.misc import pickle_loadvs = local_vs(
	embeddings_path = "path_to_save_embeddings.parquet",
	model = pickle_load("path_to_save_doc2vec_model.pickle"),
)
```

### Input data
- two main things are required for constructing the corpora:
	- `metadata.csv` file: this contains at least one column, `filepath`, which has the names of the files. It can include other columns with information. Include a column named `vector_weight` with a value between 1 and 0. With this column, each document can be given a weight in the chunk retrieval. A value of 1 leaves the distance calculation as is, a value of 0.5 will do the following: the documents' distances to the query will be multiplied by `(1/0.5)=2`, making their vectors twice as distant, and thereby giving them less weight/likelihood to be retrieved.
	- for a unique identifier, include a `text_id` column. If you don't provide one, it will default to `1`, `2`, etc.
	- `files_path`: the directory where the documents are. `.txt` files only. Ideally, these should have been converted with [nlp_pipeline](https://github.com/dhopp1/nlp_pipeline). If present, it will use the `[newpage]` tag to determine which page of the PDF the chunk comes from.
	- If the text file is determined to be a markdown table (e.g., if you converted a CSV or Excel file with [nlp_pipeline](https://github.com/dhopp1/nlp_pipeline)), the chunks will automatically include the column headers in each chunk so the LLM has full, self-contained context within each chunk, and so that column names will also be considered in the vector similarity search.

### Functionality
- The corpus's language will be determined when embedded. If the query language does not match the corpus language, the query will be translated to the corpus's language to ensure the retrieval of relevant results.
- Any transformers embedding model or custom doc2vec model can be used for generating the embeddings
- You can customize the formatting of the chunk retrieval by changing the `chunk_text_format` parameter of the `get_top_n` function. Run `help()` on it for more information.