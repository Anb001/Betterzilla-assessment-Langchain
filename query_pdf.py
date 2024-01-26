from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate

from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from langchain.llms import OpenAI

import os
from constants import openai_key

os.environ['OPENAI_API_KEY'] = openai_key

pdfreader = PdfReader('data.pdf')

raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

document_search = FAISS.from_texts(texts, embeddings)

st.title('Question Answering based on a pdf')

query = st.text_input("Write your Question")
if query:
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = document_search.similarity_search(query=query, k=2)
    response = chain.run(input_documents=docs, question=query)

    st.write(response)
