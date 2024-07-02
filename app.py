import streamlit as st
from streamlit_option_menu import option_menu  # Make sure to install this package
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
import ollama

# Initialize the embeddings models
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Initialize the Qdrant clients
url = "http://localhost:6333"
client = QdrantClient(
    url=url, prefer_grpc=False
)

# Initialize the Qdrant database
db = Qdrant(client=client, embeddings=embeddings, collection_name="chatbot")

# Streamlit app
st.title("Llama Contextual Search Engine")
# st.image("img.jpeg", caption="RAG with LLM")  # Replace with a valid image URL

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    page = option_menu(
        "Navigation", 
        ["Home", "Generate Response", "Contact Us"],
        icons=["house", "search", "envelope"],
        menu_icon="cast",
        default_index=0,
    )

if page == "Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("img.jpg", width=600, caption="RAG baseed LLM", use_column_width=True)  # Replace with a valid image URL
    st.subheader("Home")
    st.success("Retrieval-Augmented Generation (RAG) using the Hugging Face Open Source Embeeding Model integrated with LLAMA3.")
    st.write("Welcome to the Chat application. Select 'Generate Response' from the menu to get started.")
    st.write("Using cutting-edge AI technology, It is an independent LLM-based Query agent. By utilizing cutting-edge models like the Open Source Model from Hugging Face and the BGE-Large-EN embeddings from BAAI, BookFinder provides an advanced retrieval-augmented generation (RAG) capacity. Through a conversational chat interface, users may interact and receive tailored suggestions based on their inquiries. Semantic search and deep learning models are easily integrated by the application to deliver precise and contextually relevant results. It was created with the user in mind, combining the effectiveness of AI-driven recommendation algorithms with natural user interfaces to improve the experience.")

elif page == "Generate Response":
    st.subheader("Chat with Llama")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

    # Display all messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Say something"):
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Generating response..."):
                # Perform a semantic search
                docs = db.similarity_search_with_score(query=prompt, k=5)

                # Prepare the context for the Llama3 model
                context = "\n".join([doc.page_content for doc, score in docs])

                # Generate a response using Llama3
                response = ollama.chat(model='llama3', messages=[
                    {
                        'role': 'user',
                        'content': f"Using the following context, write about {prompt}:\n{context}",
                    },
                ])

                # Build full response
                full_response += response['message']['content']
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

elif page == "Contact Us":
    st.markdown("***")

    st.header("Contact Me")
    st.write("Please fill out the form below to get in touch with me.")

    # Input fields for user's name, email, and message
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message", height=150)

    # Submit button
    if st.button("Submit"):
        if name.strip() == "" or email.strip() == "" or message.strip() == "":
            st.warning("Please fill out all the fields.")
        else:
            send_email_to = 'kumawatharsh2004@gmail.com'
            st.success("Your message has been sent successfully!")

st.sidebar.success("This app demonstrates Retrieval-Augmented Generation (RAG) using the Llama3 model.")
st.sidebar.warning("Developed by [Harsh Kumawat](https://www.linkedin.com/in/harsh-k04/)")
