from wasabi import msg
from typing import Type, Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel, RunnableLambda

from rag.component.retriever.base import BaseRAGRetriever, FilterUtil
from rag.type import *
from rag.component import llm, prompt
from rag import util

class TargetFilterRetriever(BaseRAGRetriever):
    available_options = ["복수전공", "부전공", "주전공(다전공)", "주전공(단일전공)"]
    
    @classmethod
    def from_retriever(cls, retriever: BaseRAGRetriever) -> "TargetFilterRetriever":
        return cls(retriever)
    
    def __init__(
        self, 
        retriever: BaseRAGRetriever, 
        **kwargs
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.top_k = self.retriever.top_k
    
    def classify_targets(self, queries: list[str]) -> list[str]:
        def parse_output(_dict):
            targets = _dict["targets"]
            targets = targets.split("\n")
            return list(filter(lambda x: x in self.available_options, targets))
        
        llm_model = llm.get_model("gpt-4o-mini", temperature=0.0)
        if llm_model is None:
            msg.warn("LLM model not found. Skipping classification.")
            return ""
        
        _prompt = prompt.classfiy_target_prompt
        
        chain = _prompt | llm_model | StrOutputParser() | {'targets': RunnablePassthrough()} | RunnableLambda(parse_output)
        return chain.invoke({"queries": queries, "options": self.available_options})
    
    def retrieve(self, queries: dict[str, str | list[str]], filter: Filter | None = None) -> list[Chunk]:
        _queries = queries.copy()
        _queries.pop("hyde", None) # remove verbose answer
        
        # targets = self.classify_targets(util.flatten_dict(_queries))
        # print(targets)
        
        # chunks = self.retriever.retrieve(
        #     queries, filter=FilterUtil.from_dict({"in": {"key": "target", "value": targets}})
        # )
        print(filter)
        chunks = self.retriever.retrieve(
            queries=queries, filter=filter
        )
        
        return chunks
