import os
import time
import json
from uuid import uuid4
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.documents import Document
from flask_cors import CORS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "langchain-main"

# Check if index exists; if not, create it
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    # Wait for index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Get index object
index = pc.Index(index_name)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
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

# Load your documents
json_file_path = 'documents_formatted.json'
documents = load_documents_from_json(json_file_path)

# Uncomment the following line if you need to add documents again
vector_store.add_documents(documents=documents, ids=[str(uuid4()) for _ in documents])

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.8, "top_k": 50}, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_ACCESS_TOKEN')
)

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
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def format_answer(input_text):
    result = input_text.split("Answer:")[-1].strip()
    # Basic formatting if needed
    formatted_result = result.replace("1.", "").replace("2.", "").replace("3.", "").replace("4.", "")
    formatted_result = " ".join(formatted_result.split()).strip(", ")
    if not formatted_result.endswith('.'):
        last_period_index = formatted_result.rfind('.')
        if last_period_index != -1:
            formatted_result = formatted_result[:last_period_index + 1]
    return formatted_result

def find_most_similar_source(response, json_file_path):
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        sermon_data = json.load(file)
    
    # Extract contents and URLs
    contents = [sermon['content'] for sermon in sermon_data.values()]
    urls = [sermon['url'] for sermon in sermon_data.values()]
    url_category_pairs = [(sermon['url'], sermon['category']) for sermon in sermon_data.values()]

    # Get a random url to provide to the reader
    random_pair = random.choice(url_category_pairs)
    random_url, random_category = random_pair


    # Add the response to the list of contents
    all_texts = contents + [response]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer().fit_transform(all_texts)
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
    
    # Find the index of the most similar content
    most_similar_index = np.argmax(cosine_similarities)
    
    # Return the URL of the most similar content along with a set of catagories 
    return urls[most_similar_index], random_url, random_category


def answer_with_source(formatted_result, source_url, random_url, random_category):
    source_phrases = [
        f"For more information on this topic, please refer to: {source_url}",
        f"This answer draws from the following source: {source_url}",
        f"The insights provided are based on content available at: {source_url}",
        f"For a deeper dive into this matter, consider visiting: {source_url}",
        f"Additional context and details can be found here: {source_url}",
        f"To verify and expand on this information, please check: {source_url}",
        f"The full sermon that inspired this answer is available at: {source_url}",
        f"For the complete biblical context, you're encouraged to visit: {source_url}",
        f"This interpretation is grounded in the sermon found at: {source_url}"
    ]
    second_source = [
        f"If you're interested in exploring the topic of {random_category}, you might find this resource enlightening: {random_url}",
        f"For those curious about {random_category}, here's an additional perspective: {random_url}",
        f"To broaden your understanding of {random_category}, consider reading: {random_url}",
        f"Delve deeper into {random_category} with this insightful sermon: {random_url}",
        f"Expand your knowledge on {random_category} by visiting: {random_url}",
        f"For a related discussion on {random_category}, we recommend: {random_url}",
        f"If {random_category} piques your interest, you may appreciate this sermon: {random_url}",
        f"To complement your study of {random_category}, explore this resource: {random_url}",
        f"Gain additional insights on {random_category} from this sermon: {random_url}",
        f"For a different angle on {random_category}, check out: {random_url}",
        f"Enrich your understanding of {random_category} with this complementary reading: {random_url}",
        f"If you'd like to learn more about {random_category}, this sermon offers valuable perspectives: {random_url}",
        f"For those seeking to deepen their knowledge of {random_category}, we suggest: {random_url}",
        f"Discover more about {random_category} in this thought-provoking sermon: {random_url}"
    ]

    chosen_phrase = random.choice(source_phrases)
    second_phrase = random.choice(second_source)
    return f"{formatted_result}\n\n{chosen_phrase}\n\n{second_phrase}"

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '').strip()
    if not user_input:
        return jsonify({"answer": "Please ask a question."})
    
    result = rag_chain.invoke(user_input)
    formatted_result = format_answer(result)
    
    if "enough information" in formatted_result or "Sorry, I do not have" in formatted_result \
     or "I'm really sorry" in formatted_result or "unable to assist" in formatted_result.lower():
        return jsonify({"answer": formatted_result})  # Return the response without adding a source
    
    source_url, random_url, random_category = find_most_similar_source(formatted_result, '/Users/lukekottom/Desktop/RAG LLM Final /sermon_data.json')
    final_answer = answer_with_source(formatted_result, source_url, random_url, random_category)
    
    return jsonify({"answer": final_answer})

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Chatbot API. POST to /chat with a JSON message to interact."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
