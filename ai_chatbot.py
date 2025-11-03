import os 
import json
import random

import nltk # type: ignore

# 🧠 Make sure the right data is installed
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

import numpy as np

import torch # type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore
import torch.optim as optim #type: ignore
from torch.utils.data import DataLoader, TensorDataset # type: ignore


class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings

        self.X = None
        self.y = None

        # New attributes used later
        self.all_words = []
        self.tags = []
        self.xy = []

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum()]
        return words

    @staticmethod
    def bag_of_words(words, vocabulary):
        return [1 if word in words else 0 for word in vocabulary]

    def parse_intents(self):
        # ✅ Step 1: Load the JSON file
        with open(self.intents_path, "r", encoding="utf-8") as f:
            intents_data = json.load(f)

        all_words = []
        tags = []
        xy = []

        # ✅ Step 2: Tokenize and collect all words, tags, and pattern/tag pairs
        for intent in intents_data['intents']:
            tag = intent['tag']
            for pattern in intent['patterns']:
                words = self.tokenize_and_lemmatize(pattern)
                all_words.extend(words)
                xy.append((words, tag))
            tags.append(tag)
            self.intents_responses[tag] = intent.get("responses", [])

        # ✅ Step 3: Sort and deduplicate
        all_words = sorted(set(all_words))
        tags = sorted(set(tags))

        # ✅ Step 4: Store as attributes
        self.all_words = all_words
        self.tags = tags
        self.xy = xy

        print(f"Parsed {len(tags)} intents and {len(all_words)} unique words.")

    def prepare_data(self):
        bags = []
        indices = []

        for (words, tag) in self.xy:
            bag = self.bag_of_words(words, self.all_words)
            tag_index = self.tags.index(tag)
            bags.append(bag)
            indices.append(tag_index)

        self.X = np.array(bags)
        self.y = np.array(indices)
        print(f"Prepared data: {len(self.X)} samples, {len(self.all_words)} features.")

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatbotModel(self.X.shape[1], len(self.tags))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.tags)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=False))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words, self.all_words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_index = torch.argmax(predictions, dim=1).item()
        predicted_tag = self.tags[predicted_index]

        if self.function_mappings and predicted_tag in self.function_mappings:
            self.function_mappings[predicted_tag]()

        if self.intents_responses.get(predicted_tag):
            return random.choice(self.intents_responses[predicted_tag])
        else:
            return "I'm not sure how to respond to that."

            
if __name__ == '__main__':
    assistant = ChatbotAssistant('intents.json')
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)

    assistant.save_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input('You: ')
        if message.lower() == '/quit':
            break
        print("Bot:", assistant.process_message(message))
