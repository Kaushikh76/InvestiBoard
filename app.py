import streamlit as st
import pyrebase
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import json
import uuid
import PyPDF2
import docx
import io
import logging
import networkx as nx
import base64
from web3 import Web3
import os
from dotenv import load_dotenv
import requests
import time

load_dotenv()

# Pinata setup
PINATA_API_KEY = os.getenv("PINATA_API_KEY")
PINATA_SECRET_API_KEY = os.getenv("PINATA_SECRET_API_KEY")
PINATA_JWT = os.getenv("PINATA_JWT")
PINATA_BASE_URL = "https://api.pinata.cloud"

# Web3 setup
WEB3_PROVIDER_URL = os.getenv("WEB3_PROVIDER_URL")
web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URL))

# Set up OpenAI client
AI71_BASE_URL = "https://api.ai71.ai/v1/"
AI71_API_KEY = "api71-api-ee730785-33fe-41ae-a5ac-cc66e8c0d02d"
client = openai.OpenAI(
    api_key=AI71_API_KEY,
    base_url=AI71_BASE_URL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Firebase Configuration
firebaseConfig = {
    "apiKey": "AIzaSyC-G1JRKBZtWjtu02WKCmizdBmXMMP3PAA",
    "authDomain": "investiboard.firebaseapp.com",
    "projectId": "investiboard",
    "storageBucket": "investiboard.appspot.com",
    "messagingSenderId": "556809275550",
    "appId": "1:556809275550:web:a37e3770466ea4011b138f",
    "measurementId": "G-DGNP1CF7QK",
    "databaseURL" : "https://investiboard-default-rtdb.asia-southeast1.firebasedatabase.app/"
}

# Initialize Firebase
try:
    firebase = pyrebase.initialize_app(firebaseConfig)
    auth = firebase.auth()
    db = firebase.database()
except Exception as e:
    st.error(f"Error initializing Firebase: {str(e)}")
    st.stop()

vectorizer = TfidfVectorizer()

def initialize_session_state():
    if 'user' not in st.session_state:
        st.session_state.user = None
    if "response_added" not in st.session_state:
        st.session_state.response_added = False
    if 'authentication_status' not in st.session_state:
        st.session_state.authentication_status = None
    if 'cards' not in st.session_state:
        st.session_state.cards = []
    if 'connections' not in st.session_state:
        st.session_state.connections = []
    if 'graph' not in st.session_state:
        st.session_state.graph = nx.Graph()
    if 'card_updates' not in st.session_state:
        st.session_state.card_updates = None
    if 'show_rag' not in st.session_state:
        st.session_state.show_rag = False
    if 'card_just_created' not in st.session_state:
        st.session_state.card_just_created = False
    if 'wallet_connected' not in st.session_state:
        st.session_state.wallet_connected = False
    if 'wallet_address' not in st.session_state:
        st.session_state.wallet_address = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def connect_wallet():
    st.session_state.wallet_address = st.text_input("Enter your Ethereum wallet address:")
    if st.session_state.wallet_address and web3.isAddress(st.session_state.wallet_address):
        st.session_state.wallet_connected = True
        st.success(f"Wallet connected: {st.session_state.wallet_address}")
    else:
        st.error("Invalid Ethereum address. Please enter a valid address.")
        st.session_state.wallet_connected = False
        st.session_state.wallet_address = None

def disconnect_wallet():
    st.session_state.wallet_connected = False
    st.session_state.wallet_address = None
    st.success("Wallet disconnected successfully.")

def upload_to_ipfs(file):
    try:
        files = {
            'file': file.getvalue()
        }
        headers = {
            'Authorization': f'Bearer {PINATA_JWT}'
        }
        response = requests.post(
            PINATA_BASE_URL + "/pinning/pinFileToIPFS",
            files=files,
            headers=headers
        )
        if response.status_code == 200:
            return response.json()['IpfsHash']
        else:
            st.error(f"Error uploading to IPFS: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error uploading to IPFS: {str(e)}")
        return None

def get_ipfs_files():
    try:
        headers = {
            'Authorization': f'Bearer {PINATA_JWT}'
        }
        response = requests.get(
            PINATA_BASE_URL + "/data/pinList?status=pinned",
            headers=headers
        )
        if response.status_code == 200:
            return response.json()['rows']
        else:
            st.error(f"Error fetching IPFS files: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching IPFS files: {str(e)}")
        return []
    
def sign_up():
    st.subheader("Create New Account")
    with st.form("signup_form"):
        new_user_email = st.text_input("Email Address", key="signup_email")
        new_user_password = st.text_input("Password", type='password', key="signup_password")
        submit_button = st.form_submit_button("Sign Up")
    
    if submit_button:
        try:
            user = auth.create_user_with_email_and_password(new_user_email, new_user_password)
            st.success("Account created successfully!")
            st.session_state.user = user
            st.session_state.authentication_status = True
        except Exception as e:
            st.error(f"Error: {e}")

def sign_in():
    st.subheader("Sign In to Existing Account")
    with st.form("signin_form"):
        user_email = st.text_input("Email Address", key="signin_email")
        user_password = st.text_input("Password", type='password', key="signin_password")
        submit_button = st.form_submit_button("Sign In")
    
    if submit_button:
        try:
            user = auth.sign_in_with_email_and_password(user_email, user_password)
            st.success("Signed in successfully!")
            st.session_state.user = user
            st.session_state.authentication_status = True
            load_canvas_data()  # Load canvas data immediately after successful sign-in
        except Exception as e:
            st.error(f"Error: {e}")

def sign_out():
    st.session_state.user = None
    st.session_state.authentication_status = None
    st.session_state.cards = []
    st.session_state.connections = []
    st.session_state.graph = nx.Graph()
    st.success("Signed out successfully!")


def google_sign_in():
    st.subheader("Sign In with Google")
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?client_id={firebaseConfig['clientId']}&redirect_uri={firebaseConfig['redirectUri']}&response_type=code&scope=email%20profile"
    
    if st.button("Sign In with Google"):
        st.markdown(f'<a href="{auth_url}" target="_self">Click here to sign in with Google</a>', unsafe_allow_html=True)

def save_canvas_data(canvas_data):
    if st.session_state.user:
        user_id = st.session_state.user['localId']
        db.child("users").child(user_id).child("canvas").set(canvas_data)
        st.session_state.cards = canvas_data["cards"]
        st.session_state.connections = canvas_data["connections"]

def load_canvas_data():
    if st.session_state.user:
        user_id = st.session_state.user['localId']
        canvas_data = db.child("users").child(user_id).child("canvas").get().val()
        if canvas_data is None:
            canvas_data = {"cards": [], "connections": []}
        
        st.session_state.cards = canvas_data.get("cards", [])
        st.session_state.connections = canvas_data.get("connections", [])
        
        return canvas_data
    return {"cards": [], "connections": []}



def create_event_card():
    with st.form("event_card_form"):
        title = st.text_input("Title/Name")
        date = st.date_input("Date")
        time = st.time_input("Time")
        location = st.text_input("Location")
        description = st.text_area("Description")
        related_suspects = st.multiselect("Related Suspects", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Suspect'])
        related_evidence = st.multiselect("Related Evidence", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Evidence'])
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Add Event")
    
    if submitted:
        if not all([title, date, time, location, description]):
            st.error("Please fill in all required fields.")
            return None
        return {
            "id": str(uuid.uuid4()),
            "type": "Event",
            "title": title,
            "date_time": datetime.combine(date, time).isoformat(),
            "location": location,
            "description": description,
            "related_suspects": related_suspects,
            "related_evidence": related_evidence,
            "notes": notes,
            "x": 50,
            "y": len(st.session_state.cards) * 220
        }
    return None

def create_suspect_card():
    with st.form("suspect_card_form"):
        name = st.text_input("Name")
        photo = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
        dob = st.date_input("Date of Birth")
        occupation = st.text_input("Occupation")
        address = st.text_input("Address")
        alibi = st.text_area("Alibi")
        description = st.text_area("Description")
        related_events = st.multiselect("Related Events", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Event'])
        related_evidence = st.multiselect("Related Evidence", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Evidence'])
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Add Suspect")
    
    if submitted:
        if not all([name, dob, occupation, address, description]):
            st.error("Please fill in all required fields.")
            return None
        suspect_card = {
            "id": str(uuid.uuid4()),
            "type": "Suspect",
            "name": name,
            "dob": dob.isoformat(),
            "occupation": occupation,
            "address": address,
            "alibi": alibi,
            "description": description,
            "related_events": related_events,
            "related_evidence": related_evidence,
            "notes": notes,
            "x": 50,
            "y": len(st.session_state.cards) * 220
        }
        if photo:
            suspect_card["photo"] = base64.b64encode(photo.read()).decode()
        return suspect_card
    return None

def create_evidence_card():
    with st.form("evidence_card_form"):
        title = st.text_input("Title/Name")
        evidence_type = st.text_input("Type")
        date = st.date_input("Date")
        time = st.time_input("Time")
        location = st.text_input("Location")
        description = st.text_area("Description")
        related_events = st.multiselect("Related Events", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Event'])
        related_suspects = st.multiselect("Related Suspects", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Suspect'])
        condition = st.text_input("Condition")
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Add Evidence")
    
    if submitted:
        if not all([title, evidence_type, date, time, location, description, condition]):
            st.error("Please fill in all required fields.")
            return None
        return {
            "id": str(uuid.uuid4()),
            "type": "Evidence",
            "title": title,
            "evidence_type": evidence_type,
            "date_time": datetime.combine(date, time).isoformat(),
            "location": location,
            "description": description,
            "related_events": related_events,
            "related_suspects": related_suspects,
            "condition": condition,
            "notes": notes,
            "x": 50,
            "y": len(st.session_state.cards) * 220
        }
    return None

def create_card(card_type):
    with st.form(key=f"create_{card_type.lower()}_card"):
        st.subheader(f"Create {card_type} Card")
        new_card = {"id": str(uuid.uuid4()), "type": card_type, "x": 50, "y": 50}
        
        if card_type == "Event":
            new_card["title"] = st.text_input("Title/Name")
            new_card["date"] = st.date_input("Date").isoformat()
            new_card["time"] = st.time_input("Time").isoformat()
            new_card["location"] = st.text_input("Location")
            new_card["description"] = st.text_area("Description")
            new_card["related_suspects"] = st.multiselect("Related Suspects", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Suspect'])
            new_card["related_evidence"] = st.multiselect("Related Evidence", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Evidence'])
            new_card["notes"] = st.text_area("Notes")
            required_fields = ["title", "date", "time", "location", "description"]
        
        elif card_type == "Suspect":
            new_card["name"] = st.text_input("Name")
            new_card["dob"] = st.date_input("Date of Birth").isoformat()
            new_card["occupation"] = st.text_input("Occupation")
            new_card["address"] = st.text_input("Address")
            new_card["alibi"] = st.text_area("Alibi")
            new_card["description"] = st.text_area("Description")
            new_card["related_events"] = st.multiselect("Related Events", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Event'])
            new_card["related_evidence"] = st.multiselect("Related Evidence", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Evidence'])
            new_card["notes"] = st.text_area("Notes")
            required_fields = ["name", "dob", "occupation", "address", "description"]
        
        elif card_type == "Evidence":
            new_card["title"] = st.text_input("Title/Name")
            new_card["evidence_type"] = st.text_input("Evidence Type")
            new_card["date"] = st.date_input("Date Found").isoformat()
            new_card["time"] = st.time_input("Time Found").isoformat()
            new_card["location"] = st.text_input("Location")
            new_card["description"] = st.text_area("Description")
            new_card["related_events"] = st.multiselect("Related Events", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Event'])
            new_card["related_suspects"] = st.multiselect("Related Suspects", [get_card_identifier(card) for card in st.session_state.cards if card['type'] == 'Suspect'])
            new_card["condition"] = st.text_input("Condition")
            new_card["notes"] = st.text_area("Notes")
            required_fields = ["title", "evidence_type", "date", "location", "description"]
        
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_data = uploaded_file.read()
            new_card["image"] = base64.b64encode(image_data).decode()
        
        submitted = st.form_submit_button("Create Card")
        
        if submitted:
            if all(new_card.get(field) for field in required_fields):
                if "cards" not in st.session_state:
                    st.session_state.cards = []
                st.session_state.cards.append(new_card)
                save_canvas_data({"cards": st.session_state.cards, "connections": st.session_state.get("connections", [])})
                st.success(f"{card_type} card created successfully.")
                return True
            else:
                st.error("Please fill in all required fields.")
                return False
        return False


def update_card_position(card_id, x, y):
    canvas_data = load_canvas_data()
    for card in canvas_data["cards"]:
        if card['id'] == card_id:
            card['x'] = x
            card['y'] = y
            break
    save_canvas_data(canvas_data)


def delete_card(card_id):
    canvas_data = load_canvas_data()
    canvas_data["cards"] = [card for card in canvas_data["cards"] if card['id'] != card_id]
    canvas_data["connections"] = [conn for conn in canvas_data["connections"] if conn[0] != card_id and conn[1] != card_id]
    save_canvas_data(canvas_data)
    st.success("Card deleted successfully.")


def edit_card():
    if 'edit_card_id' in st.session_state:
        card_id = st.session_state.edit_card_id
        card = next((card for card in st.session_state.cards if card['id'] == card_id), None)
        if card:
            with st.form(key=f"edit_card_{card_id}"):
                st.subheader(f"Edit {card['type']} Card")
                updated_card = {}
                for key, value in card.items():
                    if key not in ['id', 'type', 'x', 'y']:
                        updated_card[key] = st.text_input(key.capitalize(), value)
                
                if st.form_submit_button("Save Changes"):
                    card.update(updated_card)
                    update_card(card_id, card)
                    del st.session_state.edit_card_id
                    st.success("Card updated successfully.")
                    st.rerun()

def clear_all_cards():
    # Clear cards from session state
    st.session_state.cards = []
    st.session_state.connections = []
    
    # Clear cards from Firebase
    if st.session_state.user:
        user_id = st.session_state.user['localId']
        db.child("users").child(user_id).child("canvas").set({})
    
    st.success("All cards have been cleared from the board and database.")
    st.experimental_rerun()

def connect_cards():
    if len(st.session_state.cards) < 2:
        st.warning("You need at least two cards to make a connection.")
        return

    card_identifiers = [get_card_identifier(card) for card in st.session_state.cards]
    
    card1 = st.selectbox("Select first card", card_identifiers, key="card1")
    card2 = st.selectbox("Select second card", card_identifiers, key="card2")
    
    if st.button("Connect Cards"):
        if card1 != card2 and (card1, card2) not in st.session_state.connections and (card2, card1) not in st.session_state.connections:
            st.session_state.connections.append((card1, card2))
            save_canvas_data({"cards": st.session_state.cards, "connections": st.session_state.connections})
            st.success("Cards connected successfully!")
            st.experimental_rerun()
        else:
            st.error("Invalid connection or already exists.")   

def get_card_identifier(card):
    return card.get('title') or card.get('name') or 'Unnamed Card'

def get_card_identifier(card):
    return card.get('title') or card.get('name') or 'Unnamed Card'

def process_document(file, file_type):
    if file_type == 'pdf':
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    elif file_type == 'docx':
        doc = docx.Document(io.BytesIO(file.read()))
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = file.getvalue().decode('utf-8')
    
    return text

def store_document(text, metadata):
    doc_id = str(uuid.uuid4())
    doc_data = {
        'text': text,
        'metadata': metadata
    }
    db.child('rag_documents').child(doc_id).set(doc_data)
    return doc_id

def retrieve_similar_documents(query_vector, top_k=3):
    all_docs = db.child('rag_documents').get().val()
    if not all_docs:
        return []
    
    all_texts = [doc_data['text'] for doc_data in all_docs.values()]
    all_vectors = vectorizer.fit_transform(all_texts)
    
    similarities = cosine_similarity(query_vector, all_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    similar_docs = []
    for idx in top_indices:
        doc_id = list(all_docs.keys())[idx]
        similar_docs.append((doc_id, similarities[idx], all_docs[doc_id]))
    
    return similar_docs

def generate_response(query, similar_docs):
    context = "Here are the most relevant documents for the query:\n\n"
    for i, (doc_id, similarity, doc_data) in enumerate(similar_docs, 1):
        context += f"Document {i} (Similarity: {similarity:.2f}):\n"
        context += f"Title: {doc_data['metadata'].get('filename', 'Untitled')}\n"
        context += f"Content: {doc_data['text'][:500]}...\n\n"

    prompt = f"""Based on the following context, please answer the user's question. If the answer is not directly stated in the context, use the information provided to infer a response. If you cannot answer based on the given context, please say so.

Context:
{context}

User Question: {query}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="tiiuae/falcon-180b-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context. Always refer to the provided documents when answering."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in generate_response: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request. Please try again later."

def rag_query(query):
    all_docs = db.child('rag_documents').get().val()
    if not all_docs:
        return "No documents have been uploaded yet. Please upload some documents before asking questions."
    
    all_texts = [doc_data['text'] for doc_data in all_docs.values()]
    vectorizer.fit(all_texts)
    
    query_vector = vectorizer.transform([query])
    
    similar_docs = retrieve_similar_documents(query_vector)
    
    response = generate_response(query, similar_docs)
    
    return response

def rag_interface():
    st.subheader("Document-based Question Answering System")
    
    uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, or TXT)", type=['pdf', 'docx', 'txt'])
    if uploaded_file:
        try:
            file_type = uploaded_file.type.split('/')[-1]
            text = process_document(uploaded_file, file_type)
            metadata = {
                'filename': uploaded_file.name,
                'filetype': file_type
            }
            doc_id = store_document(text, metadata)
            st.success(f"Document uploaded and processed. ID: {doc_id}")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
    
    query = st.text_input("Ask a question based on the uploaded documents:")
    if st.button("Get Answer"):
        if query:
            with st.spinner("Generating answer..."):
                response = rag_query(query)
            st.write("Answer:", response)
        else:
            st.warning("Please enter a question.")

def display_draggable_cards():
    canvas_data = load_canvas_data()
    card_data = json.dumps(canvas_data.get("cards", []))
    connection_data = json.dumps(canvas_data.get("connections", []))
    
    # HTML and JavaScript for draggable cards
    html_content = f"""
    <style>
    #outer-container {{
        position: relative;
        width: 100%;
        height: 800px;
        overflow: hidden;
        border: 1px solid #ccc;
    }}
    #canvas-container {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: #f0f0f0;
        cursor: move;
    }}
    #connectionCanvas {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 1;
    }}
    #cardContainer {{
        position: absolute;
        top: 0;
        left: 0;
    }}

    .card-image {{
    max-width: 100%;
    height: auto;
    margin-top: 10px;
    }}

    .card {{
        position: absolute;
        width: 300px;
        min-height: 200px;
        padding: 15px;
        border: 1px solid #999;
        border-radius: 8px;
        font-family: Arial, sans-serif;
        font-size: 14px;
        cursor: move;
        opacity: 0.9;
        transition: box-shadow 0.3s ease;
        background-color: #ffffff;
        z-index: 2;
    }}
    .card:hover {{
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
    }}
    .card-content {{
        display: table;
        width: 100%;
    }}
    .card-row {{
        display: table-row;
    }}
    .card-cell {{
        display: table-cell;
        padding: 5px;
        border-bottom: 1px solid #eee;
    }}
    .card-cell:first-child {{
        font-weight: bold;
        width: 30%;
    }}
    .delete-btn {{
        position: absolute;
        top: 10px;
        right: 10px;
        background-color: #ff4d4d;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }}
    .delete-btn:hover {{
        background-color: #ff1a1a;
    }}
    .card-image {{
        max-width: 100%;
        height: auto;
        margin-top: 10px;
    }}
    #mini-map {{
        position: absolute;
        bottom: 10px;
        right: 10px;
        width: 200px;
        height: 150px;
        background-color: rgba(255, 255, 255, 0.8);
        border: 1px solid #999;
        z-index: 3;
    }}
    #mini-map-canvas {{
        width: 100%;
        height: 100%;
    }}
    #zoom-controls {{
        position: absolute;
        top: 10px;
        left: 10px;
        z-index: 3;
    }}
    .zoom-btn {{
        width: 30px;
        height: 30px;
        font-size: 20px;
        margin: 5px;
    }}
    </style>
    <div id="outer-container">
        <div id="canvas-container">
            <canvas id="connectionCanvas"></canvas>
            <div id="cardContainer"></div>
        </div>
        <div id="mini-map">
            <canvas id="mini-map-canvas"></canvas>
        </div>
        <div id="zoom-controls">
            <button class="zoom-btn" onclick="zoomIn()">+</button>
            <button class="zoom-btn" onclick="zoomOut()">-</button>
        </div>
    </div>

    <script>
    const cardData = {card_data};
    const connectionData = {connection_data};
    const outerContainer = document.getElementById('outer-container');
    const canvasContainer = document.getElementById('canvas-container');
    const cardContainer = document.getElementById('cardContainer');
    const canvas = document.getElementById('connectionCanvas');
    const ctx = canvas.getContext('2d');
    const miniMap = document.getElementById('mini-map');
    const miniMapCanvas = document.getElementById('mini-map-canvas');
    const miniMapCtx = miniMapCanvas.getContext('2d');

    let scale = 1;
    let containerWidth = 5000;  // Increased initial width
    let containerHeight = 3000;  // Increased initial height
    let offsetX = 0;
    let offsetY = 0;

    function resizeCanvas() {{
        canvas.width = containerWidth;
        canvas.height = containerHeight;
        cardContainer.style.width = containerWidth + 'px';
        cardContainer.style.height = containerHeight + 'px';
        drawConnections();
        updateMiniMap();
    }}

    function zoomIn() {{
        scale = Math.min(scale * 1.2, 3);
        applyZoom();
    }}

    function zoomOut() {{
        scale = Math.max(scale / 1.2, 0.5);
        applyZoom();
    }}

    function applyZoom() {{
        cardContainer.style.transform = `scale(${{scale}})`;
        cardContainer.style.transformOrigin = '0 0';
        drawConnections();
        updateMiniMap();
    }}

    function updateMiniMap() {{
        miniMapCanvas.width = miniMap.offsetWidth;
        miniMapCanvas.height = miniMap.offsetHeight;
        miniMapCtx.clearRect(0, 0, miniMapCanvas.width, miniMapCanvas.height);

        const scaleX = miniMapCanvas.width / containerWidth;
        const scaleY = miniMapCanvas.height / containerHeight;

        // Draw cards
        cardData.forEach(card => {{
            const x = card.x * scaleX;
            const y = card.y * scaleY;
            miniMapCtx.fillStyle = 'blue';
            miniMapCtx.fillRect(x, y, 5, 5);
        }});

        // Draw visible area
        const visibleWidth = outerContainer.offsetWidth / scale;
        const visibleHeight = outerContainer.offsetHeight / scale;
        const visibleX = -offsetX / scale * scaleX;
        const visibleY = -offsetY / scale * scaleY;
        miniMapCtx.strokeStyle = 'red';
        miniMapCtx.strokeRect(visibleX, visibleY, visibleWidth * scaleX, visibleHeight * scaleY);
    }}

    function getCardIdentifier(card) {{
        return card.title || card.name || 'Unnamed Card';
    }}

    function createCard(card) {{
        const cardElement = document.createElement('div');
        cardElement.className = 'card';
        cardElement.id = card.id;
        cardElement.style.backgroundColor = card.type === 'Event' ? 'rgba(255,179,186,0.9)' : 
                                            card.type === 'Suspect' ? 'rgba(186,255,201,0.9)' : 'rgba(186,225,255,0.9)';
        cardElement.style.left = (card.x || 50) + 'px';
        cardElement.style.top = (card.y || 50) + 'px';
        
        let cardContent = `<div class="card-content">`;
        cardContent += `<div class="card-row"><div class="card-cell">Type:</div><div class="card-cell">${{card.type}}</div></div>`;
        for (const [key, value] of Object.entries(card)) {{
            if (!['id', 'type', 'x', 'y', 'image'].includes(key)) {{
                cardContent += `<div class="card-row"><div class="card-cell">${{key}}:</div><div class="card-cell">${{value}}</div></div>`;
            }}
        }}
        cardContent += `</div>`;
        
        if (card.image) {{
            cardContent += `<img src="data:image/jpeg;base64,${{card.image}}" class="card-image" alt="Card Image">`;
        }}
        
        cardElement.innerHTML = cardContent;
        cardElement.addEventListener('mousedown', startDragging);
        cardContainer.appendChild(cardElement);
    }}
    let isDragging = false;
    let isPanning = false;
    let currentCard = null;
    let startX, startY;

    function startDragging(e) {{
        isDragging = true;
        currentCard = e.currentTarget;
        const rect = currentCard.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
        
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', stopDragging);
    }}

    function onMouseMove(e) {{
        if (isDragging) {{
            const containerRect = cardContainer.getBoundingClientRect();
            let newX = (e.clientX - containerRect.left - startX) / scale;
            let newY = (e.clientY - containerRect.top - startY) / scale;
            
            newX = Math.max(0, Math.min(newX, containerWidth - currentCard.offsetWidth));
            newY = Math.max(0, Math.min(newY, containerHeight - currentCard.offsetHeight));
            
            currentCard.style.left = newX + 'px';
            currentCard.style.top = newY + 'px';
            
            drawConnections();
        }}
    }}

    function stopDragging() {{
        if (isDragging) {{
            isDragging = false;
            updateCardPosition(currentCard.id, parseFloat(currentCard.style.left), parseFloat(currentCard.style.top));
            drawConnections();
        }}
        currentCard = null;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', stopDragging);
    }}

    function updateCardPosition(cardId, x, y) {{
        const cardIndex = cardData.findIndex(c => c.id === cardId);
        if (cardIndex !== -1) {{
            cardData[cardIndex].x = x;
            cardData[cardIndex].y = y;
            updateStreamlit({{ type: 'update', cardId: cardId, x: x, y: y }});
            drawConnections();
        }}
    }}


    function drawConnections() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'rgba(0, 123, 255, 0.7)';
        ctx.lineWidth = 10;

        connectionData.forEach(([c1, c2]) => {{
            const card1 = cardData.find(c => getCardIdentifier(c) === c1);
            const card2 = cardData.find(c => getCardIdentifier(c) === c2);
            if (card1 && card2) {{
                const elem1 = document.getElementById(card1.id);
                const elem2 = document.getElementById(card2.id);
                if (elem1 && elem2) {{
                    const pos1 = getCardPosition(elem1);
                    const pos2 = getCardPosition(elem2);

                    drawArrow(pos1.x, pos1.y, pos2.x, pos2.y);
                }}
            }}
        }});
    }}

    function getCardPosition(element) {{
        const rect = element.getBoundingClientRect();
        const canvasRect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / canvasRect.width;
        const scaleY = canvas.height / canvasRect.height;
        
        return {{
            x: (rect.left - canvasRect.left + rect.width / 2) * scaleX,
            y: (rect.top - canvasRect.top + rect.height / 2) * scaleY
        }};
    }}

    function drawArrow(fromX, fromY, toX, toY) {{
        const headLength = 10;
        const dx = toX - fromX;
        const dy = toY - fromY;
        const angle = Math.atan2(dy, dx);

        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI / 6), toY - headLength * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI / 6), toY - headLength * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
    }}

    function deleteCard(cardId) {{
        updateStreamlit({{ type: 'delete', cardId: cardId }});
    }}

    function updateStreamlit(action) {{
        window.parent.postMessage({{
            type: "streamlit:setComponentValue",
            value: JSON.stringify({{ action: action }}),
        }}, "*");
    }}

    // Initialize
    resizeCanvas();
    cardData.forEach(createCard);
    drawConnections();
    updateMiniMap();

    // Panning functionality
    canvasContainer.addEventListener('mousedown', (e) => {{
        if (e.target === canvasContainer || e.target === canvas) {{
            isPanning = true;
            startX = e.clientX;
            startY = e.clientY;
        }}
    }});

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', stopDragging);

    // Mini-map interaction
    miniMapCanvas.addEventListener('mousedown', (e) => {{
        const rect = miniMapCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const scaleX = containerWidth / miniMapCanvas.width;
        const scaleY = containerHeight / miniMapCanvas.height;
        const centerX = x * scaleX;
        const centerY = y * scaleY;
        
        offsetX = -centerX + outerContainer.offsetWidth / (2 * scale);
        offsetY = -centerY + outerContainer.offsetHeight / (2 * scale);
        
        cardContainer.style.left = offsetX + 'px';
        cardContainer.style.top = offsetY + 'px';
        
        updateMiniMap();
    }});

    // Mouse wheel zoom
    outerContainer.addEventListener('wheel', (e) => {{
        e.preventDefault();
        const delta = e.deltaY;
        if (delta > 0) {{
            zoomOut();
        }} else {{
            zoomIn();
        }}
    }});

    // Touch events for mobile devices
    let touchStartX, touchStartY;
    outerContainer.addEventListener('touchstart', (e) => {{
        if (e.touches.length === 1) {{
            isPanning = true;
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
        }}
    }});

    outerContainer.addEventListener('touchmove', (e) => {{
        if (e.touches.length === 1 && isPanning) {{
            e.preventDefault();
            const dx = e.touches[0].clientX - touchStartX;
            const dy = e.touches[0].clientY - touchStartY;
            offsetX += dx;
            offsetY += dy;
            cardContainer.style.left = offsetX + 'px';
            cardContainer.style.top = offsetY + 'px';
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
            updateMiniMap();
        }}
    }});

    outerContainer.addEventListener('touchend', () => {{
        isPanning = false;
    }});

    // Pinch-to-zoom for mobile devices
    let initialDistance = 0;
    outerContainer.addEventListener('touchstart', (e) => {{
        if (e.touches.length === 2) {{
            initialDistance = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY
            );
        }}
    }});

    outerContainer.addEventListener('touchmove', (e) => {{
        if (e.touches.length === 2) {{
            e.preventDefault();
            const currentDistance = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY
            );
            if (initialDistance > 0) {{
                if (currentDistance > initialDistance) {{
                    zoomIn();
                }} else {{
                    zoomOut();
                }}
            }}
            initialDistance = currentDistance;
        }}
    }});

    window.addEventListener('resize', drawConnections);
    cardContainer.addEventListener('scroll', drawConnections);

    // Initialize the board
    resizeCanvas();
    cardData.forEach(createCard);
    drawConnections();
    updateMiniMap();
    </script>
    """

    st.components.v1.html(html_content, height=820)

    # Handle updates from JavaScript
    if st.session_state.card_updates:
        try:
            data = json.loads(st.session_state.card_updates)
            action = data.get('action', {})
            
            if action.get('type') == 'delete':
                delete_card(action['cardId'])
            elif action.get('type') == 'update':
                update_card_position(action['cardId'], action['x'], action['y'])
            
            st.session_state.card_updates = None  # Clear the updates after processing
        except json.JSONDecodeError:
            st.error("Error processing card updates. Please try again.")
        except KeyError as e:
            st.error(f"Error accessing data: {str(e)}. Please check the data structure.")

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

def chatbot():
    st.subheader("Detective's Assistant Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()
    input_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            st.text(f"{message['role'].capitalize()}: {message['content']}")

    with input_container:
        with st.form(key="chat_form"):
            prompt = st.text_input("What would you like to know about the case?", key="chat_input")
            col1, col2 = st.columns([1,1])
            with col1:
                submit_button = st.form_submit_button("Send")
            with col2:
                clear_button = st.form_submit_button("Clear Chat")

        if submit_button and prompt:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Prepare context with all card information
            context = "Case Information:\n"
            if "cards" in st.session_state and st.session_state.cards:
                for card in st.session_state.cards:
                    context += f"{card['type']} Card:\n"
                    for key, value in card.items():
                        if key not in ['id', 'x', 'y', 'image']:
                            context += f"  {key}: {value}\n"
                    context += "\n"
            else:
                context += "No cards have been added to the investigation board yet.\n"

            # Add connection information if available
            if "connections" in st.session_state and st.session_state.connections:
                context += "Connections:\n"
                for connection in st.session_state.connections:
                    context += f"  {connection[0]} is connected to {connection[1]}\n"

            ai_messages = [
                {"role": "system", "content": "You are a helpful assistant for a detective. Use the detailed case information to answer questions. If there's no relevant information in the case details, say so."},
                {"role": "user", "content": f"{context}\n\nQuestion: {prompt}"}
            ]

            try:
                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model="tiiuae/falcon-180b-chat",
                        messages=ai_messages
                    )
                    full_response = response.choices[0].message.content
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                full_response = "I'm sorry, but I encountered an error. Please try again later."

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Rerun to update the chat display
            st.experimental_rerun()

        if clear_button:
            st.session_state.messages = []
            st.experimental_rerun()

    # Debugging: Print the contents of st.session_state.cards
    st.write("Debug - Cards in session state:", st.session_state.get("cards", []))


def main():
    st.set_page_config(layout="wide")
    st.title("Detective's Investigation Tool")

    initialize_session_state()
    
    # Reset the card_just_created flag at the beginning of each run
    st.session_state.card_just_created = False

    if 'show_ipfs' not in st.session_state:
        st.session_state.show_ipfs = False

    if st.session_state.authentication_status != True:
        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])
        
        with tab1:
            sign_in()
        
        with tab2:
            sign_up()
    
    else:
        st.write(f"Welcome, {st.session_state.user['email']}!")
        if st.button("Sign Out", key="signout_button"):
            sign_out()
            st.experimental_rerun()
        
        # Create a two-column layout
        col1, col2 = st.columns([3, 1], gap="large")
        
        # Investigation Board Column
        with col1:
            st.header("Investigation Board")
            board_container = st.container()
            with board_container:
                display_draggable_cards()
        
        # Tools and Features Column
        with col2:
            st.header("Tools and Features")
            
            with st.expander("Add New Card"):
                card_type = st.selectbox("Select card type", ["Event", "Suspect", "Evidence"], key="card_type_select")
                create_card(card_type)

            with st.expander("Connect Cards"):
                connect_cards()
            
            with st.expander("Chat with Detective Bot"):
                chatbot()
            
            
            
            # Toggle button for IPFS File Management
            if st.button("Toggle IPFS File Management"):
                st.session_state.show_ipfs = not st.session_state.show_ipfs
            
            if st.session_state.show_ipfs:
                st.subheader("IPFS File Management")
                
                if not st.session_state.get('wallet_connected', False):
                    connect_wallet()
                else:
                    st.success(f"Wallet connected: {st.session_state.wallet_address}")
                    if st.button("Disconnect Wallet"):
                        disconnect_wallet()
                        st.experimental_rerun()
                
                if st.session_state.get('wallet_connected', False):
                    ipfs_tabs = st.tabs(["Upload File", "View Files"])
                    
                    with ipfs_tabs[0]:
                        st.subheader("Upload File to IPFS")
                        uploaded_file = st.file_uploader("Choose a file to upload", type=['pdf', 'docx', 'txt', 'jpg', 'png'])
                        if uploaded_file is not None:
                            if st.button("Upload to IPFS"):
                                with st.spinner("Uploading file to IPFS..."):
                                    ipfs_hash = upload_to_ipfs(uploaded_file)
                                    if ipfs_hash:
                                        st.success(f"File uploaded to IPFS. Hash: {ipfs_hash}")
                                        # Store the IPFS hash in the user's data
                                        user_id = st.session_state.user['localId']
                                        db.child("users").child(user_id).child("ipfs_files").push({"name": uploaded_file.name, "hash": ipfs_hash})
                    
                    with ipfs_tabs[1]:
                        st.subheader("View IPFS Files")
                        user_id = st.session_state.user['localId']
                        user_files = db.child("users").child(user_id).child("ipfs_files").get().val()
                        if user_files:
                            for file_key, file_data in user_files.items():
                                st.write(f"File: {file_data['name']}")
                                st.write(f"IPFS Hash: {file_data['hash']}")
                                st.markdown(f"[View on IPFS](https://gateway.pinata.cloud/ipfs/{file_data['hash']})")
                        else:
                            st.info("No files uploaded yet.")
                else:
                    st.warning("Please connect your wallet to access IPFS features.")
        
            
            if st.button("Clear All Cards"):
                clear_all_cards()
            if st.button("Toggle RAG Interface"):
                st.session_state.show_rag = not st.session_state.get('show_rag', False)
                st.experimental_rerun()
        
        if st.session_state.show_rag:
            st.markdown("---")
            rag_interface()

if __name__ == "__main__":
    main()