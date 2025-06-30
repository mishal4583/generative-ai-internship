✅ README.md

# 🧠 Mentra AI – Personalized Learning Assistant

Welcome to _Mentra AI, a lightweight personalized assistant that uses a fine-tuned GPT-2 model to generate intelligent, thoughtful responses to user queries. This project was developed as \*\*Task 1_ during the _Generative AI Internship at Prodigy InfoTech_.

---

## 🌟 Features

- 🚀 Fast and lightweight inference with FastAPI
- 💬 Interactive prompt-based response generation
- 🎨 Futuristic UI with glowing neon effects
- 🔐 Secure backend with CORS enabled
- 🧠 Local model loading for privacy and offline use

---

## 🛠 Tech Stack

| Layer      | Technology                                  |
| ---------- | ------------------------------------------- |
| Backend    | Python, FastAPI, Hugging Face Transformers  |
| Frontend   | HTML5, CSS3, Vanilla JavaScript             |
| Model      | Fine-tuned GPT-2 (local .safetensors model) |
| Deployment | Runs locally on http://127.0.0.1:8000/      |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mishal4583/generative-ai-internship-t1.git
cd generative-ai-internship-t1
2. Set Up Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate   # On Windows
# Or
source venv/bin/activate  # On Unix/macOS

pip install -r requirements.txt
✅ You must have the fine-tuned model saved in:
gpt2-genius-tutor-finetuned-final/
(This folder should include config.json, pytorch_model.bin or model.safetensors, tokenizer files, etc.)

🧪 Running the App
bash
Copy code
uvicorn main:app --reload
Now visit:
🌐 http://127.0.0.1:8000/
to interact with Mentra AI in your browser.

📁 Project Structure
php
Copy code
.
├── main.py                    # FastAPI backend
├── static/
│   ├── index.html             # Frontend UI
│   └── img3.png               # AI avatar image
├── gpt2-genius-tutor-finetuned-final/
│   ├── config.json
│   ├── tokenizer.json
│   └── model.safetensors      # (Not committed due to size)
└── README.md
⚠ Note on Large Files
Due to GitHub's file size limits, the following are not committed:

Trained model weights (*.safetensors, *.pt)

Virtual environments (venv/)

PyTorch library binaries

To load the model, make sure to place it locally under:
gpt2-genius-tutor-finetuned-final/

📸 Preview


👨‍💻 Author
Mishal K S
🎓 MCA Student, Jain University
💼 Generative AI Intern @ Prodigy InfoTech

📜 License
This project is for academic and demonstration purposes only.


```
