import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import re
import os

def clean_text(text):
    # Remove surrogate pairs and other problematic characters
    return re.sub(r'[\ud800-\udfff]', '', text)

def get_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            try:
                text += page.extract_text()
            except UnicodeEncodeError:
                continue
    return clean_text(text)

# splitting text into small chunks to create embeddings
def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function = len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# using google's embeddings model to create embeddings and FAISS to store the embeddings
def get_vectorstore(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        st.info("Available embedding models for Google Generative AI:")
        st.info("1. models/embedding-001 (recommended)")
        st.info("2. models/text-embedding-001")
        st.info("3. models/text-embedding-002")
        return None
    
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first!")
        return
        
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write("👤 You:", message.content)
            else:
                st.write("🤖 Assistant:", message.content)
                
        # Show source documents if available
        if 'source_documents' in response and response['source_documents']:
            with st.expander("📄 Source Documents Used"):
                for i, doc in enumerate(response['source_documents']):
                    st.write(f"**Source {i+1}:**")
                    st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    st.write("---")
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        st.info("Troubleshooting steps:")
        st.info("1. Check your Google API key in .env file (GOOGLE_API_KEY=your_key_here)")
        st.info("2. Ensure you have API access permissions")
        st.info("3. Verify stable internet connection")
        st.info("4. Try reprocessing your PDFs")

# Storing conversations as chain of outputs with Gemini Flash 2.0
def get_conversation_chain(vectorstore):
    try:
        # Updated to use Gemini Flash 2.0 model
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp',  # Updated model name for Gemini Flash 2.0
            temperature=0.3,  # Lower temperature for more consistent responses
            top_p=0.9,        # Slightly higher for better diversity
            top_k=40,
            max_tokens=8192  # Increased token limit for longer responses
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'  # Specify output key for better memory handling
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True,
            chain_type="stuff"  # Use "stuff" chain type for better handling
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        st.info("Available Gemini models:")
        st.info("1. gemini-2.0-flash-exp (Flash 2.0 - Latest)")
        st.info("2. gemini-1.5-flash (Flash 1.5)")
        st.info("3. gemini-1.5-pro (Pro 1.5)")
        st.info("4. gemini-pro (Legacy)")
        
        # Fallback to Gemini 1.5 Flash if 2.0 Flash is not available
        try:
            st.warning("Trying fallback to Gemini 1.5 Flash...")
            llm_fallback = ChatGoogleGenerativeAI(
                model='gemini-1.5-flash',
                temperature=0.3,
                top_p=0.9,
                top_k=40
            )
            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                output_key='answer'
            )
            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm_fallback,
                retriever=vectorstore.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 4}
                ),
                memory=memory,
                return_source_documents=True,
                verbose=True,
                chain_type="stuff"
            )
            st.success("Using Gemini 1.5 Flash as fallback")
            return conversation_chain
        except Exception as fallback_error:
            st.error(f"Fallback also failed: {str(fallback_error)}")
            return None

def main():
    load_dotenv()
    st.set_page_config(
        page_title="Chat with Multiple PDFs", 
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = False

    # Header
    st.title("📚 Chat with Multiple PDFs")
    st.markdown("*Powered by Gemini Flash 2.0*")
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ Google API Key not found!")
        st.info("Please add your Google API key to a .env file:")
        st.code("GOOGLE_API_KEY=your_api_key_here")
        st.info("You can get your API key from: https://makersuite.google.com/app/apikey")
        return

    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Questions About Your Documents")
        user_question = st.text_input(
            "Enter your question:", 
            placeholder="What is the main topic discussed in the documents?",
            help="Ask anything about the content of your uploaded PDFs"
        )
        
        if user_question:
            handle_userinput(user_question)

    # Sidebar for document upload
    with st.sidebar:
        st.subheader('📄 Your Documents')
        
        # Model info
        st.info("🚀 Using Gemini Flash 2.0 for fast, accurate responses")
        
        pdf_docs = st.file_uploader(
            "Upload your PDFs here:", 
            accept_multiple_files=True,
            type=['pdf'],
            help="You can upload multiple PDF files at once"
        )
        
        if pdf_docs:
            st.write(f"📁 {len(pdf_docs)} file(s) selected")
            for pdf in pdf_docs:
                st.write(f"• {pdf.name}")
        
        if st.button("🔄 Process Documents", type="primary"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file first!")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        # Extract text
                        raw_text = get_text(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("No text could be extracted from the PDFs. Please check if they contain readable text.")
                            return
                        
                        # Create text chunks
                        text_chunks = get_text_chunks(raw_text)
                        st.success(f"✅ Created {len(text_chunks)} text chunks")
                        
                        # Create vector store
                        vectorstore = get_vectorstore(text_chunks)
                        
                        if vectorstore:
                            st.success("✅ Vector store created successfully!")
                            
                            # Create conversation chain
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            
                            if st.session_state.conversation:
                                st.success("✅ Ready to answer your questions!")
                                st.session_state.processed_docs = True
                                st.balloons()
                            else:
                                st.error("❌ Failed to create conversation chain")
                        else:
                            st.error("❌ Failed to create vector store")
                            
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
        
        # Status indicator
        if st.session_state.processed_docs:
            st.success("🟢 Documents processed and ready!")
        else:
            st.info("🔵 Upload and process documents to start chatting")
        
        # Additional info
        with st.expander("ℹ️ How it works"):
            st.write("""
            1. **Upload**: Select one or more PDF files
            2. **Process**: Extract and chunk the text content
            3. **Embed**: Create vector embeddings using Google's model
            4. **Chat**: Ask questions and get AI-powered answers
            5. **Source**: See which parts of your documents were used
            """)
        
        with st.expander("🔧 Technical Details"):
            st.write("""
            - **Model**: Gemini Flash 2.0 (with 1.5 Flash fallback)
            - **Embeddings**: Google Embedding-001
            - **Vector Store**: FAISS
            - **Chunk Size**: 1000 characters
            - **Overlap**: 200 characters
            """)

if __name__ == '__main__':
    main()

# Process Flow:
# 1. Convert PDF into text chunks
# 2. Convert text chunks into embeddings  
# 3. Store embeddings in FAISS vector database
# 4. Generate embeddings for user questions
# 5. Search database for relevant content
# 6. Generate responses using Gemini Flash 2.0