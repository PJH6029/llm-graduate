from typing import Optional


from rag.type import *
from rag.component.ingestor.base import BaseRAGIngestor
from rag.component.vectorstore.PineconeVectorstore import PineconeVectorstore
from rag.component import embeddings

class PineconeVectorstoreIngestor(BaseRAGIngestor):
    def __init__(
        self,
        embeddings: Embeddings, 
        namespace: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.vectorstore = PineconeVectorstore(
            embeddings=embeddings,
            namespace=namespace,
        )
        self.namespace = namespace
    
    def ingest(self, chunks: list[Chunk]) -> int:
        return self.vectorstore.ingest(chunks)
    
    @classmethod
    def from_config(cls, config: dict) -> "PineconeVectorstoreIngestor":
        embeddings_name = config.get("embeddings")
        namespace = config.get("namespace")
        
        embeddings_model = embeddings.get_model(embeddings_name)
        
        return cls(
            embeddings=embeddings_model,
            namespace=namespace,
        )