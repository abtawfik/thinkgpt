import re
from typing import List, Optional, Type, Dict, Any, Union, Iterable

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import LLMResult, BaseOutputParser, Generation
from langchain.embeddings import OpenAIEmbeddings
from docarray import DocumentArray, Document
from pydantic.config import Extra
from thinkgpt.helper import PythonREPL

from thinkgpt.abstract import AbstractMixin, AbstractChain
from thinkgpt.condition import ConditionMixin, ConditionChain
from thinkgpt.infer import InferMixin, InferChain
from thinkgpt.memory import MemoryMixin, ExecuteWithContextChain
from thinkgpt.refine import RefineMixin, RefineChain
from thinkgpt.gpt_select import SelectChain, SelectMixin

from thinkgpt.summarize import SummarizeMixin, SummarizeChain



class ThinkGPT(ChatOpenAI,
               MemoryMixin,
               AbstractMixin,
               RefineMixin,
               ConditionMixin,
               SelectMixin,
               InferMixin,
               SummarizeMixin,
               extra=Extra.allow):
    """Wrapper around OpenAI large language models to augment it with memory

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key.
    """

    def __init__(self,
                 chatllm,
                 embeddings_model,
                 memory_name,
                 persist_directory=None,
                 verbose=True):
        super().__init__(**kwargs)
        # TODO: offer more docarray backends
        self.openai = chatllm
        self.embeddings_model = embeddings_model
        self.memory = Chroma(embedding_function=self.embeddings_model,
                             collection_name=memory_name,
                             persist_directory=persist_directory) 
        self.execute_with_context_chain = ExecuteWithContextChain(llm=self.openai, verbose=verbose)
        self.abstract_chain = AbstractChain(llm=self.openai, verbose=verbose)
        self.refine_chain = RefineChain(llm=self.openai, verbose=verbose)
        self.condition_chain = ConditionChain(llm=self.openai, verbose=verbose)
        self.select_chain = SelectChain(llm=self.openai, verbose=verbose)
        self.infer_chain = InferChain(llm=self.openai, verbose=verbose)
        self.summarize_chain = SummarizeChain(llm=self.openai, verbose=verbose)  # Add this line
        self.mem_cnt = 0


    def generate(self,
                 prompts: List[List[str]],
                 stop: Optional[List[str]] = None,
                 remember: Union[int, List[str]] = 0) -> LLMResult:
        # only works for single prompt
        if len(prompts) > 1:
            raise Exception('only works for a single prompt')
        prompt = prompts[0][0]
        if isinstance(remember, int) and remember > 0:
            remembered_elements = self.remember(prompt, limit=5)
            result = self.execute_with_context_chain.predict(prompt=prompt,
                                                             context='\n'.join(remembered_elements)
                                                             if remembered_elements else 'Nothing')
        elif isinstance(remember, list):
            result = self.execute_with_context_chain.predict(prompt=prompt,
                                                             context='\n'.join(remember))
        else:
            result = self.execute_with_context_chain.predict(prompt=prompt,
                                                             context='Nothing')

        return LLMResult(generations=[[Generation(text=result)]])

    def predict(self,
                prompt: str,
                stop: Optional[List[str]] = None,
                remember: Union[int, List[str]] = 0) -> str:
        return self.generate([[prompt]], remember=remember).generations[0][0].text


if __name__ == '__main__':
    llm = ThinkGPT(model_name="gpt-3.5-turbo")

    rules = llm.abstract(observations=[
        "in tunisian, I did not eat is \"ma khditech\"",
        "I did not work is \"ma khdemtech\"",
        "I did not go is \"ma mchitech\"",
    ], instruction_hint="output the rule in french")
    llm.memorize(rules)

    llm.memorize("in tunisian, I went is \"mchit\"")

    task = "translate to Tunisian: I didn't go"
    print(llm.predict(task, remember=llm.remember(task)))
