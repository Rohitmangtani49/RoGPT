import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from typing import Union, List, Dict
import openai



class Agent:
    def __init__(self, openai_api_key: Union[str, None] = None) -> None:
        # if openai_api_key is None, then it will look for the environment variable OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.chat_history = []
        self.chain = None
        self.db = None

        # Initialize the conversation history with the theme
        self.conversation_history = [{"role": "system", "content": "Answer the users question"}]

    def ask(self, question: str, temperature: float = 0.2) -> str:
        if self.chain is None:
            response = "Please, add a document."
        else:
            # First, retrieve a response using the chain
            chain_response = self.chain({"question": question, "chat_history": self.chat_history})
            chain_response = chain_response["answer"].strip()
            
            # Add the chain response and question to the conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": chain_response})
            
            # Now, refine the response using the OpenAI model
            response = self.generate_response(self.conversation_history, temperature)
            
            # Update the chat history
            self.chat_history.append((question, response))
        return response

    def ingest(self, file_path: os.PathLike) -> None:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitted_documents = self.text_splitter.split_documents(documents)

        if self.db is None:
            self.db = FAISS.from_documents(splitted_documents, self.embeddings)
            self.chain = ConversationalRetrievalChain.from_llm(self.llm, self.db.as_retriever())
            self.chat_history = []
        else:
            self.db.add_documents(splitted_documents)

    def forget(self) -> None:
        self.db = None
        self.chain = None
        self.chat_history = []
        self.conversation_history = [{"role": "system", "content": "Answer the users question"}]

    def generate_response(self, messages: List[Dict[str, str]], temp: float) -> str:
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=messages,
            n=1,
            stop=None,
            temperature=temp
        )
        reply = chat.choices[0].message['content']
        return reply
