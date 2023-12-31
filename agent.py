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
    def __init__(self, openai_api_key: str | None = None) -> None:
        # if openai_api_key is None, then it will look the enviroment variable OPENAI_API_KEY
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        if openai_api_key:
            openai.api_key = openai_api_key
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

        self.chat_history = None
        self.chain = None
        self.db = None

        # Initialize the conversation history with the theme
        self.conversation_history = [{"role": "system", "content": "Answer the users question using the referenced information only, DRAW CONCLUSIONS. Essentially act as a clone of the user and draw insights to access the correct data and answers. You are only trained on this dataset. Give long descriptive responses with examples."}]

    def ask(self, question: str, temperature: float = 0.7) -> str:
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
        self.conversation_history = [{"role": "system", "content": "Answer the users question using the referenced information only, DRAW CONCLUSIONS. Essentially act as a clone of the user and draw insights to access the correct data and answers. You are only trained on this dataset. Give long descriptive responses with examples."}]

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
    def extract_from_pdf(self, user_text: str) -> str:
        # Search the ingested PDFs for relevant information based on the user's question
        # For simplicity, we'll use the chain to retrieve a response
        chain_response = self.chain({"question": user_text, "chat_history": self.chat_history})
        return chain_response["answer"].strip()

    def ask_with_context(self, user_text: str, extracted_info: str, temperature: float = 0.3) -> str:
        # Craft a new prompt for GPT-3.5 using the user's question and the extracted information
        refined_question = f"{user_text} (Based on the document: {extracted_info})"
        
        # Add the refined question to the conversation history
        self.conversation_history.append({"role": "user", "content": refined_question})
        
        # Now, get the response using the OpenAI model
        response = self.generate_response(self.conversation_history, temperature)
        
        # Update the chat history
        self.chat_history.append((refined_question, response))
        return response
