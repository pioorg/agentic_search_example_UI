import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
import io
import sys

load_dotenv()

# Page config
st.set_page_config(
    page_title="Diving Assistant",
    page_icon="🤿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode CSS
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #262730;
        border-radius: 0.5rem;
    }
    
    /* User messages */
    .stChatMessage[data-testid="user-message"] {
        background-color: #1e2329;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1d23;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #363a42;
    }
    
    .stButton > button:hover {
        background-color: #363a42;
        border: 1px solid #4a4e58;
    }
    
    /* Input field */
    .stChatInput > div > div > input {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #363a42;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #262730;
        color: #fafafa;
    }
    
    .streamlit-expanderContent {
        background-color: #1a1d23;
        color: #fafafa;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #fafafa;
    }
    
    /* Divider */
    hr {
        border-color: #363a42;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1a1d23;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_agent():
    class Googler:
        def __init__(self):
            self.service = build('customsearch', 'v1', developerKey=os.getenv("GCP_API_KEY"))
        
        def scrape(self, url):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    return '\n'.join(chunk for chunk in chunks if chunk)[:5000]
                return None
            except:
                return None
        
        def search(self, query, n=5):
            results = self.service.cse().list(q=query, cx=os.getenv("GCP_PSE_ID"), num=n).execute()
            scraped_data = []
            for item in results.get('items', []):
                url = item['link']
                title = item['title']
                content = self.scrape(url) or item['snippet']
                scraped_data.append(f"Page: {title}\nURL: {url}\n\n{content}\n")
            return "\n".join(scraped_data)

    class ElasticSearcher:
        def __init__(self):
            self.client = Elasticsearch(
                os.environ.get("ELASTIC_ENDPOINT"),
                api_key=os.environ.get("ELASTIC_API_KEY")
            )
        
        def search(self, query, index="us_navy_dive_manual", size=10):
            response = self.client.search(
                index=index,
                body={
                    "size": size,
                    "query": {
                        "semantic": {
                            "field": "semantic_content",
                            "query": query
                        }
                    }
                }
            )
            return "\n".join([hit["_source"].get("body", "No Body") 
                             for hit in response["hits"]["hits"]])

    # Initialize searchers
    googler = Googler()
    elastic = ElasticSearcher()

    # Create tools
    tools = [
        Tool(
            name="WebSearch",
            func=lambda q: googler.search(q, n=3),
            description="Search the web for information. Use for current events or general knowledge or to complement with additional information."
        ),
        Tool(
            name="NavyDiveManual",
            func=lambda q: elastic.search(q, index="us_navy_dive_manual"),
            description="Search the Operations Dive Manual. Use for diving procedures, advanced or technical operational planning, resourcing, and technical information."
        ),
        Tool(
            name="DivingSafetyManual",
            func=lambda q: elastic.search(q, index="diving_safety_manual"),
            description="Search the Diving Safety Manual. Use for generic diving safety protocols and best practices."
        )
    ]

    # Create agent
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        deployment_name=os.getenv("AZURE_OPENAI_MODEL"),
        streaming=False
    )

    prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

You should use multiple tools in conjunction to promote completeness of information.

Be comprehensive in your answer.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}""")

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    return agent_executor

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("🤿 Initializing diving assistant..."):
        st.session_state.agent = initialize_agent()

if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = False

# Header
st.title("🤿 Diving Assistant")
st.markdown("Ask me anything about diving! I can search the web, Navy Dive Manual, and Diving Safety Manual.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.show_reasoning = st.checkbox(
        "Show agent reasoning", 
        value=st.session_state.show_reasoning,
        help="Display the agent's thought process and tool usage"
    )
    
    st.markdown("---")
    
    st.header("💡 Example Questions")
    example_questions = [
        "When should NITROX be used in diving operations?",
        "List of recommended equipment for ice/cold water diving operations",
        "List of best regulators available for diving",
        "Tell me about the Apeks XTX50 regulator",
        "What are the risk factors when snorkeling?",
        "How to perform emergency ascent procedures?",
        "What are decompression stops and why are they important?"
    ]
    
    
    for question in example_questions:
        if st.button(f"🔍 {question}", key=question, use_container_width=True):
            st.session_state.prompt = question

# Chat interface
chat_container = st.container()

# Display chat history
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show reasoning if it exists for this message
            if message["role"] == "assistant" and "reasoning" in message and st.session_state.show_reasoning:
                with st.expander("🧠 Agent Reasoning", expanded=False):
                    st.code(message["reasoning"], language="text")

# Input box
if prompt := st.chat_input("Ask me about diving...") or st.session_state.get("prompt"):
    # Clear the prompt from session state
    if "prompt" in st.session_state:
        del st.session_state.prompt
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Show thinking message
            with message_placeholder.container():
                st.markdown("🤔 Thinking...")
            
            # Capture verbose output if reasoning is enabled
            reasoning_text = ""
            
            if st.session_state.show_reasoning:
                # Capture stdout
                old_stdout = sys.stdout
                sys.stdout = buffer = io.StringIO()
                
                try:
                    # Set verbose to True temporarily
                    st.session_state.agent.verbose = True
                    response = st.session_state.agent.invoke({"input": prompt})
                    final_answer = response["output"]
                    
                    # Get the captured output
                    reasoning_text = buffer.getvalue()
                    
                except Exception as e:
                    final_answer = f"I encountered an error: {str(e)}. Please try again."
                finally:
                    sys.stdout = old_stdout
                    st.session_state.agent.verbose = False
            else:
                # Run without verbose output
                try:
                    response = st.session_state.agent.invoke({"input": prompt})
                    final_answer = response["output"]
                except Exception as e:
                    final_answer = f"I encountered an error: {str(e)}. Please try again."
            
            # Display final answer
            message_placeholder.markdown(final_answer)
            
            # Add to history with reasoning
            message_data = {"role": "assistant", "content": final_answer}
            if reasoning_text:
                message_data["reasoning"] = reasoning_text
            st.session_state.messages.append(message_data)
            
            # Show reasoning if enabled
            if st.session_state.show_reasoning and reasoning_text:
                with st.expander("🧠 Agent Reasoning", expanded=True):
                    st.code(reasoning_text, language="text")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

