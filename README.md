This gives users an overview, setup guide, and usage instrctions.
# 🤖 Advanced AI Chatbot (PyTorch + NLTK)

An intelligent, domain-aware chatbot built with **PyTorch** and **NLTK**, trained on `intents.json` to handle queries about greetings, weather, education, healthcare, and e-commerce.

## 🚀 Features

- Neural network trained on intent-based data  
- Supports multiple domains (general, health, education, etc.)  
- Easily extendable via `intents.json`  
- Offline, privacy-friendly — no API required  
- Uses bag-of-words + PyTorch neural model 

---

## ⚙️ Setup Instructions

1. Install dependencies:

pip install -r requirements.txt


2. Download NLTK data:

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

🧠 Training the Chatbot
Train the model (it saves the model + dimensions automatically):
python ai_chatbot.py

This will:
1. Parse your intents.json
2. Prepare training data
3. Train the PyTorch model
4. Save chatbot_model.pth and dimensions.json



💬 Chat with Your Bot
After training, you can chat interactively:

You: Hello
Bot: Hey there! How can I assist you today?

You: What's the date today?
Bot: Today is November 4, 2025. 📅

Type /quit to exit.


🧩 Folder Structure
advanced_ai_chatbot/
│
├── ai_chatbot.py          # Main chatbot code
├── intents.json           # Training data (intents + responses)
├── requirements.txt       # Dependencies
├── README.md              # Documentation
├── chatbot_model.pth      # Saved PyTorch model (after training)
└── dimensions.json        # Model metadata


🧠 Add New Intents
You can easily expand the chatbot by adding new entries in intents.json, for example:

{
  "tag": "motivation",
  "patterns": ["motivate me", "say something inspiring"],
  "responses": ["Keep going — your hard work will pay off 💪"]
}

Then re-run:
python ai_chatbot.py

AUTHOR
NANCY AHAKE