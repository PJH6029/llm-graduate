from typing import Generator, Any
from wasabi import msg

from langchain_community.callbacks import get_openai_callback
from langchain.globals import set_debug

from rag.rag_manager import RAGManager
from rag import util
from rag.type import *

recent_chunks = None
recent_translated_query = None
rag_manager = None

def init(config: dict=None):
    # set_debug(True)
    
    util.load_secrets()
    global rag_manager
    rag_manager = RAGManager()
    _config = {
        "global": {
            "context-hierarchy": False, # used in selecting retriever and generation prompts
            "target-filtering": True,
        },
        "ingestion": { # optional
            "ingestor": "pinecone-multivector",
            "embeddings": "solar-embedding-1-large",
            "namespace": "parent",
            "sub-namespace": "child",
        },
        "transformation": { # optional
            "model": "gpt-4o-mini",
            "enable": {
                "translation": False,
                "rewriting": True,
                "expansion": False,
                "hyde": True,
            },
        },
        "retrieval": { # mandatory
            # "retriever": ["pinecone-multivector", "kendra"],
            "retriever": ["pinecone-multivector"],
            # "retriever": ["kendra"],
            # "weights": [0.5, 0.5],
            
            "namespace": "parent",
            "sub-namespace": "child",
            
            "embeddings": "solar-embedding-1-large", # may be optional
            "top_k": 5, # for multi-vector retriever, context size is usually big. Use small top_k
            "post_retrieval": {
                "rerank": True,
                # TODO
            }
        },
        "generation": { # mandatory
            "model": "gpt-4o",
        },
        "fact_verification": { # optional
            "model": "gpt-4o-mini",
            "enable": False,
        },
    } if config is None else config
    
    rag_manager.set_config(_config)

init()
    

def query(query: str, history: list[ChatLog]=None) -> GenerationResult:
    history = history or []
    with get_openai_callback() as cb:
        queries = rag_manager.transform_query(query, history)
        translated_query = queries["translation"] # first query is the translated query
        chunks = rag_manager.retrieve(queries)
        
        global recent_chunks, recent_translated_query
        recent_chunks = chunks
        recent_translated_query = translated_query

        generation_response = rag_manager.generate(translated_query, history, chunks)
        verification_response = rag_manager.verify_fact(generation_response, chunks)
        
        print(cb)
    return {"transformation": queries, "retrieval": chunks, "generation": generation_response, "fact_verification": verification_response}
    

def query_stream(query: str, history: list[ChatLog]=None) -> Generator[GenerationResult, None, None]:
    history = history or []
    with get_openai_callback() as cb:
        queries = rag_manager.transform_query(query, history)
        yield {"transformation": queries}
        
        translated_query = queries["translation"] # first query is the translated query
        chunks = rag_manager.retrieve(queries)
        yield {"retrieval": chunks}
        
        global recent_chunks, recent_translated_query
        recent_chunks = chunks
        recent_translated_query = translated_query

        generation_response = ""
        for response in rag_manager.generate_stream(translated_query, history, chunks):
            yield {"generation": response}
            generation_response += response
        
        # TODO visualize the result in frontend
        for response in rag_manager.verify_fact_stream(generation_response, chunks):
            yield {"fact_verification": response}
        
        print(cb)
    
def upload_data(file_path: str, object_location: str) -> bool:
    return rag_manager.upload_data(file_path, object_location)

def ingest_data(s3_url: str) -> int:
    return rag_manager.ingest(s3_url)

def ingest_data_from_local(file_path: str) -> int:
    return rag_manager.ingest_from_local(file_path)

async def aingest_data(s3_url: str) -> int:
    return await rag_manager.aingest(s3_url)

def ingest_from_backup(backup_dir: str, object_location: str) -> int:
    return rag_manager.ingest_from_backup(backup_dir, object_location)
