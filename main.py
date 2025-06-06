import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import re
import os

def clean_text(text):
    # Remove surrogate pairs and other problematic characters
    return re.sub(r'[\ud800-\udfff]', '', text)

def get_text_with_metadata(pdf_docs):
    """Extract text while preserving document and page metadata"""
    documents = []
    for pdf_idx, pdf in enumerate(pdf_docs):
        pdf_reader = PdfReader(pdf)
        for page_idx, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    documents.append(Document(
                        page_content=clean_text(page_text),
                        metadata={
                            "source": pdf.name,
                            "page": page_idx + 1,
                            "doc_id": pdf_idx
                        }
                    ))
            except UnicodeEncodeError:
                continue
    return documents

def get_text(pdf_docs):
    """Original function - get all text as single string"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            try:
                text += page.extract_text()
            except UnicodeEncodeError:
                continue
    return clean_text(text)

# APPROACH 1: Semantic/Intelligent Chunking
def get_semantic_chunks(documents, chunk_size=2000, chunk_overlap=400):
    """Create larger, more intelligent chunks that preserve context"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Split on paragraphs first, then sentences
    )
    
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            all_chunks.append(Document(
                page_content=chunk,
                metadata=doc.metadata
            ))
    
    return all_chunks

# APPROACH 2: Hierarchical Chunking
def get_hierarchical_chunks(documents):
    """Create both page-level and chunk-level embeddings"""
    chunks = []
    
    # Page-level documents (full pages)
    for doc in documents:
        chunks.append(Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "chunk_type": "page", "chunk_size": len(doc.page_content)}
        ))
    
    # Smaller chunks for detailed search
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    for doc in documents:
        small_chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(small_chunks):
            chunks.append(Document(
                page_content=chunk,
                metadata={**doc.metadata, "chunk_type": "section", "section_id": i}
            ))
    
    return chunks

# APPROACH 3: Full Document Context (for smaller PDFs)
def get_full_document_context(documents, max_tokens=6000):
    """Combine entire documents or create very large chunks"""
    full_docs = []
    current_doc = ""
    current_metadata = {}
    
    for doc in documents:
        # If adding this page would exceed token limit, save current and start new
        if len(current_doc + doc.page_content) > max_tokens and current_doc:
            full_docs.append(Document(
                page_content=current_doc,
                metadata=current_metadata
            ))
            current_doc = doc.page_content
            current_metadata = doc.metadata
        else:
            if not current_doc:
                current_metadata = doc.metadata
            current_doc += "\n\n" + doc.page_content
    
    # Add the last document
    if current_doc:
        full_docs.append(Document(
            page_content=current_doc,
            metadata=current_metadata
        ))
    
    return full_docs

# APPROACH 4: Multi-level Retrieval
def create_multi_level_vectorstore(documents, approach="semantic"):
    """Create vectorstore based on selected approach"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        
        if approach == "semantic":
            chunks = get_semantic_chunks(documents, chunk_size=2000, chunk_overlap=400)
        elif approach == "hierarchical":
            chunks = get_hierarchical_chunks(documents)
        elif approach == "full_context":
            chunks = get_full_document_context(documents)
        elif approach == "page_level":
            chunks = documents  # Use full pages
        else:
            # Default to semantic
            chunks = get_semantic_chunks(documents)
        
        # Create texts and metadatas for FAISS
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        vectorstore = FAISS.from_texts(
            texts=texts, 
            embedding=embeddings,
            metadatas=metadatas
        )
        
        return vectorstore, len(chunks)
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None, 0

# Enhanced conversation chain with better retrieval
def get_enhanced_conversation_chain(vectorstore, retrieval_strategy="similarity"):
    try:
        llm = ChatGoogleGenerativeAI(
            model='gemini-2.0-flash-exp',
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            max_tokens=8192
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        # Configure retriever based on strategy
        if retrieval_strategy == "mmr":
            retriever = vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": 6,
                    "fetch_k": 20,
                    "lambda_mult": 0.5
                }
            )
        elif retrieval_strategy == "similarity_threshold":
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": 0.5,
                    "k": 8
                }
            )
        else:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True,
            chain_type="stuff"
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def handle_enhanced_userinput(user_question):
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
                
        # Enhanced source document display
        if 'source_documents' in response and response['source_documents']:
            with st.expander(f"📄 Source Documents Used ({len(response['source_documents'])} sources)"):
                for i, doc in enumerate(response['source_documents']):
                    metadata = doc.metadata
                    st.write(f"**Source {i+1}:** {metadata.get('source', 'Unknown')} - Page {metadata.get('page', 'N/A')}")
                    
                    # Show chunk type if available
                    if 'chunk_type' in metadata:
                        st.write(f"*Chunk Type: {metadata['chunk_type']}*")
                    
                    # Truncate very long content
                    content = doc.page_content
                    if len(content) > 500:
                        st.write(content[:500] + "...")
                        if st.button(f"Show full content for Source {i+1}", key=f"show_full_{i}"):
                            st.text_area(f"Full content - Source {i+1}", content, height=200, key=f"full_content_{i}")
                    else:
                        st.write(content)
                    st.write("---")
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")

def main():
    load_dotenv()
    st.set_page_config(
        page_title="Enhanced PDF Chat", 
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
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    # Header
    st.title("📚 Enhanced PDF Chat - No Data Loss")
    st.markdown("*Multiple strategies to preserve complete document context*")
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ Google API Key not found!")
        st.info("Please add your Google API key to a .env file:")
        st.code("GOOGLE_API_KEY=your_api_key_here")
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
            handle_enhanced_userinput(user_question)

    # Sidebar for document upload and configuration
    with st.sidebar:
        st.subheader('📄 Your Documents')
        
        # Processing strategy selection
        st.subheader('🔧 Processing Strategy')
        processing_approach = st.selectbox(
            "Choose processing approach:",
            [
                "semantic",
                "hierarchical", 
                "full_context",
                "page_level"
            ],
            help="""
            - Semantic: Larger chunks with intelligent splitting
            - Hierarchical: Both page-level and section-level chunks
            - Full Context: Very large chunks or whole documents
            - Page Level: Each page as separate chunk
            """
        )
        
        # Retrieval strategy
        retrieval_strategy = st.selectbox(
            "Choose retrieval strategy:",
            ["similarity", "mmr", "similarity_threshold"],
            help="""
            - Similarity: Standard similarity search
            - MMR: Maximum Marginal Relevance (diverse results)
            - Similarity Threshold: Only above confidence threshold
            """
        )
        
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
                with st.spinner(f"Processing documents using {processing_approach} approach..."):
                    try:
                        # Extract text with metadata
                        documents = get_text_with_metadata(pdf_docs)
                        
                        if not documents:
                            st.error("No text could be extracted from the PDFs.")
                            return
                        
                        st.success(f"✅ Extracted {len(documents)} pages")
                        
                        # Create vector store with selected approach
                        vectorstore, chunk_count = create_multi_level_vectorstore(
                            documents, 
                            approach=processing_approach
                        )
                        
                        if vectorstore:
                            st.success(f"✅ Created {chunk_count} chunks using {processing_approach} approach")
                            st.session_state.chunk_count = chunk_count
                            
                            # Create conversation chain
                            st.session_state.conversation = get_enhanced_conversation_chain(
                                vectorstore, 
                                retrieval_strategy=retrieval_strategy
                            )
                            
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
            st.info(f"📊 Using {st.session_state.chunk_count} chunks")
        else:
            st.info("🔵 Upload and process documents to start chatting")
        
        # Strategy explanations
        with st.expander("📖 Processing Strategies Explained"):
            st.write("""
            **Semantic Chunking:**
            - Uses 2000-character chunks with 400 overlap
            - Splits on paragraphs and sentences intelligently
            - Preserves more context than standard chunking
            
            **Hierarchical Chunking:**
            - Creates both full-page and section-level chunks
            - Allows retrieval at different granularities
            - Best for documents with clear structure
            
            **Full Context:**
            - Creates very large chunks (up to 6000 tokens)
            - Minimal splitting, maximum context preservation
            - Best for smaller documents or when context is crucial
            
            **Page Level:**
            - Each page is a separate chunk
            - No splitting within pages
            - Preserves complete page context
            """)
        
        with st.expander("🔍 Retrieval Strategies"):
            st.write("""
            **Similarity:** Standard cosine similarity search
            
            **MMR (Maximum Marginal Relevance):** 
            - Balances relevance and diversity
            - Reduces redundant results
            
            **Similarity Threshold:**
            - Only returns results above confidence threshold
            - Ensures high-quality matches
            """)

if __name__ == '__main__':
    main()