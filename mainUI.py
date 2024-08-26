from main_rag_bot import Chatbot
import streamlit as st

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

bot = Chatbot()

st.set_page_config(page_title="Biblical Sermon Bot")
with st.sidebar:
    st.title('Biblical Sermon Bot')

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    formatted_result = format_answer(result)  # Apply the format_answer function here
    return formatted_result

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