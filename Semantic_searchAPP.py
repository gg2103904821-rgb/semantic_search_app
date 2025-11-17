# semantic_search_app.py
import streamlit as st
import pandas as pd
from pinecone import Pinecone
from openai import AzureOpenAI
import time

# ============================
# Configuration Section - Using your provided API keys
# ============================
#PINECONE_API_KEY = "pcsk_dzwMp_JVByLNDFEeHhQfcv4qR7xeBqAZCW2aRUqW9DZN5cTjGuCu3VqsRZo8TMSEe2aup"
PINECONE_API_KEY ="pcsk_JPQMS_zQZ9MfrD4aSEe8b69PoxsjcsvoSPEHpzgYGt4GPm8bv7ED95Wjy4u7vPmxSnjj"
AZURE_ENDPOINT = "https://hkust.azure-api.net"
AZURE_API_KEY = "92a7b72844fd4d59bf05e69941f2d6d4"
AZURE_API_VERSION = "2023-05-15"
PINECONE_INDEX_NAME = "developer-quickstart-py"  
PINECONE_HOST = "https://developer-quickstart-py-9d1pu2j.svc.aped-4627-b74a.pinecone.io" 

# ============================
# Search Engine Core Class
# ============================
class SemanticSearchEngine:
    def __init__(self):
        """Initialize search engine"""
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients"""
        try:
            # Azure OpenAI client
            self.openai_client = AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_version=AZURE_API_VERSION,
                api_key=AZURE_API_KEY
            )
            
            # Pinecone client
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self.index = self.pc.Index(PINECONE_INDEX_NAME, host=PINECONE_HOST)
            
            st.success("✅ API clients initialized successfully")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize clients: {str(e)}")
            raise
    
    def get_embedding(self, text):
        """Get text embedding vector"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"Failed to get embedding vector: {str(e)}")
    
    def search_documents(self, query, top_k=3):
        """Search for relevant documents in Pinecone"""
        try:
            query_embedding = self.get_embedding(query)
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                fields=["category", "chunk_text"],
                include_metadata=True
            )
            
            return results
        except Exception as e:
            raise Exception(f"Document search failed: {str(e)}")
    
    def generate_answer(self, query, context_documents):
        """Generate answer based on query and retrieved documents"""
        try:
            # Build context
            context = "\n\n".join([
                f"Document {i+1}: {match.metadata.get('content', '')}" 
                for i, match in enumerate(context_documents.matches)
            ])
            
            # Build enhanced prompt
            prompt = f"""
            Based on the following document content, accurately answer the user's question.
            
            Relevant document content:
            {context}
            
            User question: {query}
            
            Requirements:
            1. Answer based on the provided document content
            2. If there is no relevant information in the documents, please state so
            3. The answer should be accurate, useful, and concise
            4. Do not fabricate information that does not exist in the documents
            
            Answer:
            """
            
            # Call Azure OpenAI API to generate answer
            response = self.openai_client.chat.completions.create(
                model="gpt-35-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional assistant that accurately answers user questions based on provided document content."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            raise Exception(f"Failed to generate answer: {str(e)}")
    
    def format_search_results(self, results):
        """Format search results for display"""
        formatted_results = []
        
        for i, match in enumerate(results.matches):
            content = match.metadata.get('content', '')
            # Smart truncation
            if len(content) > 300:
                truncated_content = content[:300]
                last_period = truncated_content.rfind('。')
                if last_period > 250:
                    content = content[:last_period + 1] + "..."
                else:
                    content = truncated_content + "..."
            
            result = {
                "rank": i + 1,
                "score": match.score,
                "content": content,
                "source": match.metadata.get('source', 'Unknown'),
                "id": match.id
            }
            formatted_results.append(result)
        
        return formatted_results

# ============================
# Input Interface Function
# ============================
def render_input_section(search_engine):
    """Render user input interface"""
    # Application title
    st.title("🔍 Intelligent Semantic Search Application")
    st.markdown("Using Pinecone + Azure OpenAI for semantic search")
    
    # Display API status
    with st.sidebar:
        st.header("🔧 API Status")
        st.success("✅ Pinecone: Connected")
        st.success("✅ Azure OpenAI: Connected")
        
        st.header("⚙️ Search Configuration")
        top_k = st.slider(
            "Number of documents to return", 
            min_value=1, 
            max_value=10, 
            value=3
        )
        
        st.markdown("---")
        st.header("💡 Usage Tips")
        st.info("""
        - Enter complete question statements
        - More specific questions yield more accurate results
        - Supports both Chinese and English queries
        - System generates answers based on relevant documents
        """)
    
    # Main search area
    st.subheader("📝 Enter Your Question")
    
    user_query = st.text_area(
        "Please enter your question:",
        placeholder="For example:\n• What is HKUST?\n• What is machine learning?",
        height=120,
        key="user_query"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        search_button = st.button(
            "🚀 Start Search", 
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        if st.button("🔄 Test Connection", use_container_width=True):
            with st.spinner("Testing API connection..."):
                try:
                    # Test connection
                    test_query = "Test"
                    test_embedding = search_engine.get_embedding(test_query)
                    st.success("✅ API connection test successful!")
                except Exception as e:
                    st.error(f"❌ Connection test failed: {str(e)}")
    
    # Search button logic
    if search_button:
        if not user_query.strip():
            st.error("❌ Please enter a valid question")
        else:
            with st.spinner("🔍 Searching for relevant documents and generating answer..."):
                try:
                    # Execute search
                    search_results = search_engine.search_documents(
                        user_query.strip(), 
                        top_k=top_k
                    )
                    
                    # Generate answer
                    answer = search_engine.generate_answer(
                        user_query.strip(), 
                        search_results
                    )
                    
                    # Format results
                    formatted_results = search_engine.format_search_results(
                        search_results
                    )
                    
                    # Save to session state
                    st.session_state.last_search = {
                        "query": user_query.strip(),
                        "answer": answer,
                        "results": formatted_results,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    # Display success message and auto redirect
                    st.success("✅ Search completed! Displaying results...")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error occurred during search: {str(e)}")

# ============================
# Output Interface Function
# ============================
def render_output_section():
    """Render search results output interface"""
    # Check if there are search results
    if 'last_search' not in st.session_state:
        st.warning("🔍 No search results yet, please search first")
        if st.button("Return to Search"):
            st.session_state.pop('last_search', None)
            st.rerun()
        return
    
    current_search = st.session_state.last_search
    
    # Page title and navigation
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("📊 Search Results")
        st.caption(f"Query: **{current_search['query']}**")
    
    with col2:
        st.write("")  # Vertical spacing
        if st.button("🔄 New Search", use_container_width=True):
            st.session_state.pop('last_search', None)
            st.rerun()
    
    st.markdown("---")
    
    # AI answer area
    st.subheader("🤖 AI Intelligent Answer")
    
    # Create answer card
    with st.container():
        st.markdown(
            f"""
            <div style='
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 25px;
                border-radius: 10px;
                color: white;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            '>
            <h4 style='color: white; margin-top: 0;'>💡 Intelligent Answer Based on Semantic Search</h4>
            <div style='font-size: 16px; line-height: 1.6;'>
            {current_search['answer']}
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Relevant documents area
    st.subheader(f"📄 Relevant Documents ({len(current_search['results'])} found)")
    
    # Document similarity statistics
    if current_search['results']:
        scores = [result['score'] for result in current_search['results']]
        avg_score = sum(scores) / len(scores)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Highest Similarity", f"{max(scores):.3f}")
        with col2:
            st.metric("Average Similarity", f"{avg_score:.3f}")
        with col3:
            st.metric("Relevant Documents", len(current_search['results']))
    
    # Display detailed information for each document
    for i, result in enumerate(current_search['results']):
        with st.expander(
            f"📋 Document {result['rank']} | Similarity: {result['score']:.3f} | Source: {result['source']}",
            expanded=(i == 0)
        ):
            # Document content
            st.write("**Content:**")
            st.info(result['content'])
            
            # Metadata
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Document ID:** {result['id']}")
            with col2:
                st.write(f"**Similarity Score:** {result['score']:.3f}")
                st.progress(result['score'])
    
    st.markdown("---")
    
    # Search statistics
    st.subheader("📈 Search Statistics")
    
    # Create similarity distribution chart
    if current_search['results']:
        df = pd.DataFrame(current_search['results'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(df.set_index('rank')['score'])
            st.caption("Similarity Distribution by Document")
        
        with col2:
            # Display search information
            st.write("**Search Information:**")
            st.write(f"- 📝 Query Content: {current_search['query']}")
            st.write(f"- 📊 Documents Returned: {len(current_search['results'])}")
            st.write(f"- 🎯 Highest Similarity: {df['score'].max():.3f}")
            st.write(f"- 📍 Average Similarity: {df['score'].mean():.3f}")
            st.write(f"- ⏰ Search Time: {current_search['timestamp']}")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Results", use_container_width=True):
            # Save to search history
            if 'search_history' not in st.session_state:
                st.session_state.search_history = []
            
            st.session_state.search_history.append(current_search)
            st.success("✅ Results saved to search history")
    
    with col2:
        if st.button("🔄 Search Again", use_container_width=True):
            st.session_state.pop('last_search', None)
            st.rerun()
    
    with col3:
        if st.button("🏠 Return Home", use_container_width=True):
            st.session_state.pop('last_search', None)
            st.rerun()

# ============================
# Main Application Function
# ============================
def main():
    """Main application function"""
    # Set page configuration
    st.set_page_config(
        page_title="Intelligent Semantic Search Application",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize search engine
    if 'search_engine' not in st.session_state:
        try:
            st.session_state.search_engine = SemanticSearchEngine()
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            return
    
    # Decide which page to display based on whether there are search results
    if 'last_search' in st.session_state:
        render_output_section()
    else:
        render_input_section(st.session_state.search_engine)

# ============================
# Application Launch
# ============================
if __name__ == "__main__":
    main()