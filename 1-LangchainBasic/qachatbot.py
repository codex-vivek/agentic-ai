import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

# ---------- Page config (must be the first Streamlit command) ----------
st.set_page_config(
    page_title="LangChain Q&A Chatbot with Groq",
    page_icon="🤖",
)

st.title("🤖 Simple LangChain Chatbot with Groq")
st.markdown("Learn LangChain basics with Groq's ultra-fast inference!")

@st.cache_data(ttl=3600, show_spinner=False)
def list_models(api_key: str):
    """Fetch currently available chat models from Groq."""
    try:
        from groq import Groq

        models = Groq(api_key=api_key).models.list().data
        skip = ("whisper", "guard", "tts", "orpheus", "playai")
        return sorted(m.id for m in models if not any(s in m.id for s in skip))
    except Exception:
        return []


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get a free API key at console.groq.com",
    )

    available_models = list_models(api_key) if api_key else []
    if available_models:
        model_name = st.selectbox("Model", available_models)
    else:
        model_name = st.text_input(
            "Model",
            value="llama-3.3-70b-versatile",
            help="API key daalte hi available models ki list yahan aa jayegi.",
        )

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

    memory_limit = st.slider(
        "Memory (last N messages)",
        2, 30, 10, 2,
        help="Bot sirf itne pichle messages yaad rakhega. Kam rakhoge to context error nahi aayega.",
    )

    max_tokens = st.slider(
        "Max reply length (tokens)",
        128, 4096, 1024, 128,
    )

    system_prompt = st.text_area(
        "System prompt",
        value=(
            "You are a helpful assistant powered by Groq. "
            "Always reply in Hinglish: Hindi written in English (Roman) letters, "
            "for example 'Haan bhai, ye aise kaam karta hai'. "
            "NEVER use Urdu script or Arabic script. "
            "Do not use Devanagari either, only Roman letters. "
            "Keep the tone friendly and simple."
        ),
        height=100,
    )

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of HumanMessage / AIMessage


# ---------- Build the chain ----------
@st.cache_resource
def get_chain(
    api_key: str,
    model_name: str,
    temperature: float,
    system_prompt: str,
    max_tokens: int,
):
    if not api_key:
        return None

    llm = ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    return prompt | llm | StrOutputParser()


chain = get_chain(api_key, model_name, temperature, system_prompt, max_tokens)

# ---------- Show previous messages ----------
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ---------- Handle new input ----------
if question := st.chat_input("Ask me anything..."):
    if chain is None:
        st.warning("Please enter your Groq API key in the sidebar first.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                chain.stream(
                    {
                        "question": question,
                        "history": st.session_state.messages[-memory_limit:],
                    }
                )
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # Save to memory only after a successful reply
    st.session_state.messages.append(HumanMessage(content=question))
    st.session_state.messages.append(AIMessage(content=response))