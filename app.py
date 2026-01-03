import streamlit as st
import json
import os
import requests
from datetime import datetime

# OCR imports
import pytesseract
import cv2
import numpy as np
from PIL import Image

#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------------- CONFIG -----------------------
st.set_page_config(
    page_title="CODE GEN AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

HISTORY_FILE = "chat_history.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

# ----------------------- TESSERACT PATH (Windows only) -----------------------
# Uncomment if required
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# ----------------------- SAVE / LOAD HISTORY -----------------------
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# ----------------------- OLLAMA FUNCTION -----------------------
def ollama_chat(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"⚠️ Ollama error: {e}"

# ----------------------- OCR FUNCTION -----------------------
def extract_text_from_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, lang="eng")
    return text

# ----------------------- TEMPLATE FEATURE -----------------------
TEMPLATES = {
    "None": "",
    "Explain Code": "Explain the following code in simple terms:\n\n{}",
    "Fix Bugs": "Find and fix bugs in the following code:\n\n{}",
    "Optimize Code": "Optimize the following code for better performance:\n\n{}",
    "Write Code": "Write code for the following requirement:\n\n{}",
    "Generate Documentation": "Generate documentation for this code:\n\n{}"
}

# ----------------------- SESSION STATE INIT -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = load_history()

if "selected_template" not in st.session_state:
    st.session_state.selected_template = "None"

# ----------------------- SIDEBAR -----------------------
with st.sidebar:
    st.title(" CODE GEN AI")

    st.subheader(" Prompt Template")
    selected = st.selectbox("Choose a template:", list(TEMPLATES.keys()))
    st.session_state.selected_template = selected

    if st.button("➕ New Chat"):
        if st.session_state.messages:
            st.session_state.history.append({
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages
            })
            save_history(st.session_state.history)
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑 Clear Previous Sessions"):
        st.session_state.history = []
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.success("✔ Previous sessions cleared!")
        st.rerun()

    st.subheader("Previous Sessions")
    for idx, chat in enumerate(st.session_state.history):
        title = chat["messages"][0]["content"][:25] + "..." if chat["messages"] else f"Chat {idx+1}"
        if st.button(title, key=f"load_{idx}"):
            st.session_state.messages = chat["messages"]
            st.rerun()

# ----------------------- MAIN CHAT UI -----------------------
st.title(" Code Gen AI")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'><b>You:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'><b>Code Gen AI:</b><br>{msg['content']}</div>", unsafe_allow_html=True)

# ----------------------- MESSAGE INPUT + OCR ICON -----------------------
default_text = ""
if st.session_state.selected_template != "None":
    default_text = TEMPLATES[st.session_state.selected_template].format("<< your code/text here >>")

col1, col2 = st.columns([9, 1])

with col1:
    user_input = st.text_input(
        "Enter your request:",
        value=default_text,
        placeholder="Ask me anything about coding..."
    )

#with col2:
    uploaded_image = st.file_uploader("📎", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

# ----------------------- SEND MESSAGE -----------------------
if st.button("Send"):
    final_prompt = ""

    # Case 1: Image uploaded (OCR)
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        extracted_text = extract_text_from_image(image)

        final_prompt = f"""
The following text was extracted from an image using OCR.
Analyze it and respond accordingly:

{extracted_text}
"""

        st.session_state.messages.append({
            "role": "user",
            "content": f"[Image OCR Input]\n{extracted_text}"
        })

    # Case 2: Normal text input
    elif user_input.strip():
        final_prompt = user_input
        st.session_state.messages.append({"role": "user", "content": user_input})

    # Call Ollama if prompt exists
    if final_prompt:
        bot_reply = ollama_chat(final_prompt)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        st.rerun()
