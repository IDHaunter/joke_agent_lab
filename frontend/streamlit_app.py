import os
from pathlib import Path
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("FRONTEND_REQUEST_TIMEOUT_SECONDS", "900"))

st.set_page_config(
    page_title="Joke Agent Lab",
    page_icon="😄",
    layout="centered",
)

st.title("😄 Joke Agent Lab")
st.caption("A small LangChain and LangGraph learning project")


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# Clear the server-side conversation and start a new frontend session.
if st.button("Clear chat", use_container_width=True):
    reset_warning = None
    try:
        response = httpx.post(
            f"{BACKEND_URL}/api/v1/reset",
            json={"session_id": st.session_state.session_id},
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        reset_warning = (
            "The local chat was cleared, but the backend could not confirm removal "
            "of the old checkpoint. A new session has been started. "
            f"Details: {exc}"
        )

    st.session_state.messages = []
    st.session_state.session_id = str(uuid4())

    if reset_warning:
        st.session_state.reset_warning = reset_warning
    else:
        st.session_state.pop("reset_warning", None)

    st.rerun()


if warning := st.session_state.pop("reset_warning", None):
    st.warning(warning)


# Render the existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Streamlit's chat input includes a send control and submits on Enter.
if prompt := st.chat_input("Ask for a joke or just say hello..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(
                    f"{BACKEND_URL}/api/v1/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                    },
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                answer = response.json()["answer"]
            except httpx.HTTPStatusError as exc:
                try:
                    detail = exc.response.json().get("detail", "The backend returned an error.")
                except ValueError:
                    detail = "The backend returned an error."
                answer = f"⚠️ {detail}"
                st.error(answer)
            except httpx.RequestError:
                answer = (
                    "⚠️ Could not connect to the backend. Make sure FastAPI is running "
                    "and BACKEND_URL is configured correctly."
                )
                st.error(answer)
            else:
                st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    st.rerun()