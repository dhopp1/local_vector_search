from fastapi import FastAPI
from local_vector_search.local_vector_search import local_vs
import os
from pydantic import BaseModel


# app
app = FastAPI()
corpora_path = "corpora/"  # change to location of corpora on local machine
corpora = [
    _.split("_")[1].split(".")[0] for _ in os.listdir(corpora_path) if ".parquet" in _
]

# read in all corpora on load to avoid doing it for every call
corpora_dict = {}
for corpus in corpora:
    corpora_dict[corpus] = local_vs(
        embeddings_path=f"{corpora_path}/embeddings_{corpus}.parquet"
    )
    corpora_dict[corpus].include_metadata = False


@app.get("/api/v1/which_corpora/")
async def which_corpora():
    return corpora


class SearchRequest(BaseModel):
    which_corpus: str
    query: str
    distance_metric: str = "cosine"
    top_n: int = 3


@app.post("/api/v1/vector_search/")
async def vector_search(request: SearchRequest):
    return corpora_dict[request.which_corpus].get_top_n(
        request.query, top_n=int(request.top_n), distance_metric=request.distance_metric
    )
