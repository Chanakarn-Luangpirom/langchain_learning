import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import os



CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print('abc:',results)
    print('def:',results[0][1])
    if len(results) == 0 or results[0][1] < 0.3:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(type(prompt))



    # Model

    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta",task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    }
)
    template = """{query}"""
    prompt_template = PromptTemplate(
        input_variables=["query"], template=template
    )
    conv_chain = LLMChain(llm=llm, prompt=prompt_template)    
    print(prompt_template)
    print(conv_chain.run(prompt))
    
    model_url = "https://huggingface.co/openthaigpt/openthaigpt-1.0.0-beta-13b-chat-gguf/resolve/main/ggml-model-q4_0.gguf"

    from llama_index import (
        SimpleDirectoryReader,
        VectorStoreIndex,
        ServiceContext,
    )
    from llama_index.llms import LlamaCPP
    from llama_index.llms.llama_utils import (
        messages_to_prompt,
        completion_to_prompt,
    )

    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=model_url,
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 43},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    print(prompt)
    response = llm.complete(prompt)
    print(response.text) 
    
if __name__ == "__main__":
    main()
