import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os
import re
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import base64
from pydub import AudioSegment
import pypdf

# Add RAG-specific imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import tempfile
import uuid

# Load environment variables
load_dotenv()

# --------- 🔁 Conversation Chain ---------
def get_conversation_chain(model_name: str, username: str = None):
    """Create a new conversation chain with memory"""

    name_clause = f"The user's name is {username}." if username else ""

    template = f"""You are a helpful AI assistant. Be friendly and professional.
Always format code in markdown with syntax highlighting.

{name_clause}

Current conversation:
{{chat_history}}
Human: {{input}}
AI:"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    llm = ChatOllama(model=model_name)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,  # Keep last 5 messages in memory
        return_messages=True
    )
    
    # Create a chain with just the prompt and the LLM
    chain = prompt | llm
    
    return {
        "chain": chain,
        "memory": memory
    }

# --------- 📚 RAG Pipeline ---------
def setup_embeddings():
    """Set up the embedding model"""
    # Use Ollama's embedding model instead of HuggingFace
    return OllamaEmbeddings(model="tinydolphin")

def process_document(file, file_type):
    """Process an uploaded document and create a vector store"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
        temp_file.write(file.getvalue())
        temp_path = temp_file.name
    
    # Load document based on file type
    try:
        if file_type == 'txt':
            loader = TextLoader(temp_path)
        elif file_type == 'pdf':
            loader = PyPDFLoader(temp_path)
        elif file_type == 'csv':
            loader = CSVLoader(temp_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and store in Chroma
        embeddings = setup_embeddings()
        
        # Create a unique ID for the vectorstore
        vectorstore_id = str(uuid.uuid4())
        persist_directory = f"chroma_db_{vectorstore_id}"
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return vectorstore, persist_directory
    
    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

def get_rag_response(query, vectorstore):
    """Get a response using RAG"""
    llm = ChatOllama(model=st.session_state.current_model)
    
    # Get relevant documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create a simple RAG chain
    relevant_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Get chat history from memory
    chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
    
    # Build a template that includes the retrieved context
    template = f"""You are a helpful research assistant who answers questions based on the provided context.
Use the following context to answer the human's question:

CONTEXT:
{context}

If the context doesn't contain the information needed to answer the question, just say so.
Do not make up an answer if the context is insufficient.
"""
    
    # Get response using the context-enhanced prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    return response.content

# --------- 🎤 Audio Functions ---------
def record_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak now.")
        audio = recognizer.listen(source, phrase_time_limit=5)
        st.success("✅ Audio captured, transcribing...")

    try:
        text = recognizer.recognize_google(audio)
        st.success(f"📝 Transcribed: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand your voice.")
        return None
    except sr.RequestError:
        st.error("Speech Recognition API error.")
        return None

def text_to_speech(text):
    try:
        tts = gTTS(text)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        st.audio(audio_fp, format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio could not be generated: {str(e)}")
        st.info("Text-to-speech functionality is currently unavailable. Please read the response instead.")

# --------- 🔍 Name Extraction ---------
def extract_name(text: str) -> str:
    match = re.search(r"(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None

# --------- 🧠 Response Generation ---------
def get_response(user_input: str, model_name: str) -> str:
    try:
        # Handle name inquiry
        if "what is my name" in user_input.lower():
            if st.session_state.username:
                return f"Your name is {st.session_state.username}."
            elif st.session_state.summary_memory:
                extracted = extract_name(st.session_state.summary_memory)
                return f"I think your name might be {extracted}." if extracted else "I don't know your name yet."
            else:
                return "I don't know your name yet."

        # Detect and store user's name
        name = extract_name(user_input)
        if name:
            st.session_state.username = name
            st.session_state.summary_memory += f"\n{name}"
            conv = get_conversation_chain(model_name, username=name)
            st.session_state.conversation = conv["chain"]
            st.session_state.memory = conv["memory"]
            return f"Nice to meet you, {name}! How can I assist you today?"

        # Check if we have a vectorstore and should use RAG
        if hasattr(st.session_state, 'vectorstore') and st.session_state.vectorstore:
            # Use RAG for document-related queries
            try:
                rag_response = get_rag_response(user_input, st.session_state.vectorstore)
                
                # Update memory
                st.session_state.memory.save_context(
                    {"input": user_input}, 
                    {"output": rag_response}
                )
                
                return rag_response
            except Exception as e:
                st.warning(f"RAG query failed, falling back to regular response: {str(e)}")
                # Fall through to regular response if RAG fails
        
        # Add file context if available (legacy method)
        if hasattr(st.session_state, 'file_context') and st.session_state.file_context:
            user_input = f"{user_input}\n\n[Context from file: {st.session_state.file_context}]"

        # Get chat history from memory
        chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
        
        # Get response using the new syntax
        ai_message = st.session_state.conversation.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        # Simply extract the content from the AIMessage
        response_text = ai_message.content
        
        # Update memory
        st.session_state.memory.save_context(
            {"input": user_input}, 
            {"output": response_text}
        )
        
        return response_text

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"I'm sorry, I encountered an error: {str(e)}"

# --------- 🚀 Main App ---------
# Set Streamlit page config
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# --------- 🔧 SESSION STATE SETUP ---------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "username" not in st.session_state:
    st.session_state.username = None

if "summary_memory" not in st.session_state:
    st.session_state.summary_memory = ""

if "current_model" not in st.session_state:
    st.session_state.current_model = "tinydolphin"  # Default to TinyDolphin

if "file_context" not in st.session_state:
    st.session_state.file_context = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "conversation" not in st.session_state:
    conv = get_conversation_chain(st.session_state.current_model, st.session_state.username)
    st.session_state.conversation = conv["chain"]
    st.session_state.memory = conv["memory"]

# --------- ⚙️ Sidebar ---------
AVAILABLE_MODELS = {
    "TinyDolphin (Fast)": "tinydolphin",
    "Llama 2 (Balanced)": "llama2",
    "Mistral (Advanced)": "mistral"
}

with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model selection
    selected_model_name = st.selectbox(
        "🤖 Select AI Model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.values()).index(st.session_state.current_model) if st.session_state.current_model in AVAILABLE_MODELS.values() else 0
    )

    selected_model_id = AVAILABLE_MODELS[selected_model_name]

    if st.session_state.current_model != selected_model_id:
        st.session_state.current_model = selected_model_id
        conv = get_conversation_chain(selected_model_id, username=st.session_state.username)
        st.session_state.conversation = conv["chain"]
        st.session_state.memory = conv["memory"]
        st.success(f"Switched to {selected_model_name} model")

    with st.expander("💭 Conversation Summary"):
        if hasattr(st.session_state.memory, "buffer"):
            st.text_area("Summary Memory",
                         value=st.session_state.memory.buffer,
                         height=200)

    if st.button("Clear Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.username = None
        st.session_state.summary_memory = ""
        
        # Clear the vectorstore too
        if st.session_state.vectorstore is not None:
            st.session_state.vectorstore = None
        
        # Clear file context
        if hasattr(st.session_state, 'file_context'):
            st.session_state.file_context = None
        
        conv = get_conversation_chain(st.session_state.current_model)
        st.session_state.conversation = conv["chain"]
        st.session_state.memory = conv["memory"]
        st.rerun()

    uploaded_file = st.sidebar.file_uploader("📄 Upload a file to interact with", type=["txt", "pdf", "csv"])

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        with st.spinner(f"Processing {uploaded_file.name} with RAG pipeline..."):
            # Show processing steps
            st.sidebar.info(f"Loading and processing document...")
            
            # Process the document
            vectorstore, persist_directory = process_document(uploaded_file, file_type)
            
            # Store in session state
            st.session_state.vectorstore = vectorstore
            
            # Clear the old file_context since we're using RAG now
            st.session_state.file_context = None
            
            st.sidebar.success(f"✅ {uploaded_file.name} processed with RAG pipeline")
            st.sidebar.info("You can now ask questions about the document")
            
    except Exception as e:
        st.sidebar.error(f"Error processing document: {type(e).__name__}: {str(e)}")
        import traceback
        st.sidebar.error(f"Error details: {str(e)}")
        
        # Fallback to the old method if RAG processing fails
        try:
            file_content = ""
            if file_type == "txt":
                file_content = uploaded_file.getvalue().decode("utf-8")
            elif file_type == "pdf":
                # Handle PDF files
                from PyPDF2 import PdfReader
                pdf_reader = PdfReader(uploaded_file)
                file_content = "\n".join([page.extract_text() for page in pdf_reader.pages])
            elif file_type == "csv":
                # Handle CSV files
                import pandas as pd
                df = pd.read_csv(uploaded_file)
                file_content = df.to_string()
            
            st.session_state.file_context = file_content[:2000]  # Limit context size
            st.sidebar.warning(f"⚠️ Fallback: Loaded {uploaded_file.name} using simple context method")
        except Exception as fallback_error:
            st.sidebar.error(f"Could not process document at all: {str(fallback_error)}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = get_response(prompt, st.session_state.current_model)
        st.markdown(response)
        
        # Convert response to speech
        text_to_speech(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Voice input button
if st.sidebar.button("🎙️ Use Voice Input"):
    user_input = record_voice_input()
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = get_response(user_input, st.session_state.current_model)
            st.markdown(response)
            
            # Convert response to speech
            text_to_speech(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
