import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware 

# --- 1. Load Model and Tokenizer Globally ---
# This loads the model once when the server starts, not for every request.
# Make sure this path is correct relative to your main.py file!
MODEL_PATH = "gpt2-genius-tutor-finetuned-final" # No './'

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True) # Added local_files_only=True
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True) # Added local_files_only=True
    # Check for CUDA (GPU) and use it if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure your model files are in the correct path: {MODEL_PATH}")
    # If the model fails to load, exit the application.
    import sys
    sys.exit(1) # Exit with an error code

# --- 2. Initialize FastAPI App ---
app = FastAPI()

# --- CORS Middleware (Crucial for Frontend-Backend Communication) ---
# This allows your web browser (frontend) to make requests to your API backend.
# In a production environment, you would replace "*" with the specific domain(s)
# where your frontend is hosted for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, including your local HTML file
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- 3. Define Request Body Model ---
# This defines the structure of the JSON data your API expects for a prompt.
class PromptRequest(BaseModel):
    prompt_text: str

# --- 4. API Endpoint for Text Generation ---
# This endpoint receives a POST request with a prompt and returns generated text.
@app.post("/generate")
async def generate_text_api(request: PromptRequest):
    prompt = request.prompt_text
    print(f"Received prompt: '{prompt}'") # Log the incoming prompt in the terminal

    # Encode the prompt and get attention mask, move to appropriate device
    encoded_input = tokenizer.encode_plus(prompt, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    # Generate text using the loaded model
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,            # Max length of generated sequence (prompt + new text)
        num_return_sequences=1,    # Generate 1 sequence for simplicity
        temperature=0.7,           # Controls randomness: lower for more predictable, higher for more creative
        top_p=0.9,                 # Nucleus sampling: picks from words with cumulative probability p
        top_k=0,                   # Disable Top-K when using Top-P
        pad_token_id=tokenizer.eos_token_id, # Important for GPT-2 generation
        do_sample=True,            # Enable sampling
    )

    decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Generated text: '{decoded_text}'") # Log the generated text in the terminal

    # Return the generated text in a JSON response
    return {"generated_text": decoded_text}

# --- 5. Serve Static Files (Frontend HTML/CSS/JS) ---
# This tells FastAPI to serve files from the 'static' directory.
# For example, if you have 'static/index.html', it can be accessed via /static/index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 6. Root Endpoint to Serve the HTML Page ---
# When a user visits the base URL (e.g., http://127.0.0.1:8000/),
# this endpoint will serve your index.html file.
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Ensure 'static' folder exists and 'index.html' is inside it
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)