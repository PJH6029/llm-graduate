# LLM Project

# Configuration Guide

```python
# rag.api.py
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
}
```