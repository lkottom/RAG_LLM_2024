from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
import time
from uuid import uuid4
from langchain_core.documents import Document
import json
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

class Chatbot():

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

    # Define Index Name
    index_name = "langchain-main" 

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    def load_documents_from_json(json_file_path):
        with open(json_file_path, 'r') as json_file:
            documents_data = json.load(json_file)

        documents = []
        for doc_data in documents_data:
            document = Document(
                page_content=doc_data['page_content'],
                metadata=doc_data['metadata']
            )
            documents.append(document)

        return documents

    # Example usage:
    json_file_path = 'documents_formatted.json'
    documents = load_documents_from_json(json_file_path)

    # Generate unique UUIDs for each document
    uuids = [str(uuid4()) for _ in range(len(documents))]

    # Assuming you have a vector store ready for adding documents
    vector_store.add_documents(documents=documents, ids=uuids)

    # Define the repo ID and connect to Mixtral model on Huggingface
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
        repo_id=repo_id, 
        model_kwargs={"temperature": 0.8, "top_k": 50}, 
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_ACCESS_TOKEN')
    )

    # You are a chatbot that will help answer questions about sermons from the webiste
    #     The Humans will ask you a questions about the Bible, sermons, faith, and topics realted to the sermons.
    #     Use following piece of context, and only this context from the website, to answer the question. 
    #     If you don't know the answer witin the context, just say you don't know. 
    #     Please base all your answers of the context. 
    #     Keep the answer to only 5 clear sentences that always end in a period.
    #     The answer cannot have any numbers in it. 

    template = """
        You are a Biblical chatbot that will answer questions exclusively from the collection of sermons in your trained database from the website Insightful Sermons. If the answer is not found within this specific context, respond concisely with "Sorry, I do not have enough information to answer that question." Your responses should always be clear, concise, positive, and correct, in sentence format, ending with a period. Each response should be limited to a maximum of eight sentences. If a question involves a pronoun or ambiguous reference, ask for clarification without providing any additional commentary. Make sure that your answers never pull from any external data or previous training outside of the specified sermon collection. You cannot use any outside information or previous knowledge. 
        Context: {context}
        Question: {question}
        Answer: 
        """

    prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )

    rag_chain = (
        {"context": vector_store.as_retriever(),  "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser() 
    )


def format_answer(input_text):
    # Split the input text by "Answer:"
    result = input_text.split("Answer:")[-1].strip()

    # Replace any numbered lists with spaces
    formatted_result = result.replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "")

    # Remove any extra spaces or punctuation
    formatted_result = " ".join(formatted_result.split()).strip(", ")

    # Ensure that the text ends with a complete sentence
    if not formatted_result.endswith('.'):
        # Find the last period in the text
        last_period_index = formatted_result.rfind('.')
        if last_period_index != -1:
            # Cut off everything after the last period
            formatted_result = formatted_result[:last_period_index + 1]
        else:
            # If no period exists, just return the entire string as is
            pass

    return formatted_result


if __name__ == "__main__":

    bot = Chatbot()
    input = input("Ask me anything: ")
    result = bot.rag_chain.invoke(input)
    # print(result)
    answer = result.split("Answer:")[-1].strip()
    print("----ANSWER-----")
    print(format_answer(answer))