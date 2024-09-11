from main_rag_bot import Chatbot
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import random

json_file_path = '/Users/lukekottom/Desktop/RAG LLM Final /sermon_data.json'  # Update this path
 
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


bot = Chatbot()

st.set_page_config(page_title="Biblical Sermon Bot")
with st.sidebar:
    st.title('Biblical Sermon Bot')

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    formatted_result = format_answer(result)  # Apply the format_answer function here

    if "enough information" in formatted_result or "Sorry, I do not have" in formatted_result \
     or "I'm really sorry" in formatted_result or "unable to assit" in formatted_result:
        return formatted_result  # Return the response without adding a source
    
    source_url, random_url, random_category = find_most_similar_source(formatted_result, json_file_path)
    return answer_with_source(formatted_result, source_url, random_url, random_category)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, \
                                  I am here to help you answer any questions \
                                  related to the context of https://www.insightfulsermons.com/"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Getting your answer......"):
            response = generate_response(input)  # Get the formatted response
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)