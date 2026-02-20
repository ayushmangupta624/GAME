import google.generativeai as genai
from sentence_transformers import SentenceTransformer

API_KEY = ""
genai.configure(api_key=API_KEY)

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "block_none"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "block_none"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "block_none"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "block_none"}
]

GENERATION_CONFIG = genai.types.GenerationConfig(
    candidate_count=1,
    temperature=0
)

model = genai.GenerativeModel('gemini-2.0-flash')
sim_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

TRANSLIT_LOOKUP = {
    'करता': 'karta',
    'होता': 'hota',
    'किया': 'kiya',
    'किये': 'kiye'
}

EXCEPTIONS = ["se", "sakti", "tune"]