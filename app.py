from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
import torch
import datetime
import json
import os
import pymongo
import bcrypt

# === Flask App
app = Flask(__name__)
CORS(app)

# === MongoDB Setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["mindmitra"]
users_col = db["users"]
chat_col = db["chat_history"]

# === Load Main Chatbot (DialoGPT)
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# === Load Emotion Classifier (MentalBERT)
emotion_tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
emotion_model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")

# === Load Predefined Dataset
with open("chat_dataset.json", "r", encoding="utf-8") as f:
    predefined_data = json.load(f)

# === Detect Emotion Using MentalBERT
def detect_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True)
    outputs = emotion_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    label = emotion_model.config.id2label[predicted_class]
    return label.lower()

# === Sanitize Response
def sanitize_reply(reply):
    blocked = [
        "because you're a man", "kill yourself", "worthless", "you deserve it",
        "nobody cares", "die", "stupid"
    ]
    for phrase in blocked:
        if phrase in reply.lower():
            return "I'm really sorry you're feeling this way. You're not alone — I'm here with you."
    return reply

# === Save Chat
def save_log(user, msg, reply, emotion, lang):
    chat_col.insert_one({
        "timestamp": datetime.datetime.now(),
        "user": user,
        "message": msg,
        "reply": reply,
        "emotion": emotion,
        "language": lang
    })

# === Find predefined answer
def find_predefined_reply(msg):
    msg_lower = msg.lower()
    for item in predefined_data:
        if item["prompt"].lower() in msg_lower:
            return item["response"]
    return None

# === Get user chat context
def get_user_history(username, limit=3):
    past = list(chat_col.find({"user": username}).sort("timestamp", -1).limit(limit))
    return [chat["message"] for chat in reversed(past)]

# === CHAT Route
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get("message")
    lang = data.get("lang", "en")
    user = data.get("username", "Guest")

    msg_en = GoogleTranslator(source=lang, target="en").translate(user_msg)

    reply_pre = find_predefined_reply(msg_en)
    if reply_pre:
        reply = GoogleTranslator(source="en", target=lang).translate(reply_pre)
    else:
        context = get_user_history(user)
        chat_input = " ".join(context + [msg_en])
        input_ids = chat_tokenizer.encode(chat_input + chat_tokenizer.eos_token, return_tensors='pt')
        output_ids = chat_model.generate(input_ids, max_length=1000, pad_token_id=chat_tokenizer.eos_token_id)
        reply_en = chat_tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        reply_en = sanitize_reply(reply_en)
        reply = GoogleTranslator(source="en", target=lang).translate(reply_en)

    emotion = detect_emotion(msg_en)

    suggestions = []
    if emotion == "sad":
        suggestions = ["🧘 Let's try breathing together.", "🌧️ Want to talk more about it?"]
    elif emotion == "angry":
        suggestions = ["💡 Try expressing your anger.", "☕ Take a short break."]
    elif emotion == "happy":
        suggestions = ["😊 Keep that smile going!", "🎉 Want to share your joy?"]
    elif emotion == "fear":
        suggestions = ["😌 You're safe here.", "🌙 Let’s calm that fear together."]
    elif emotion == "love":
        suggestions = ["💖 That's lovely!", "🌟 Want to talk more about that?"]
    elif emotion == "surprise":
        suggestions = ["😮 That sounds unexpected!", "✨ Tell me more."]

    save_log(user, user_msg, reply, emotion, lang)

    return jsonify({
        "reply": reply,
        "emotion": emotion,
        "suggestions": suggestions
    })

# === Feedback API
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    with open("feedback.json", "a") as f:
        f.write(json.dumps(data) + "\n")
    return jsonify({"status": "feedback received"})

# === Signup
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    if users_col.find_one({"email": email}):
        return jsonify({"success": False, "message": "Email already exists"})
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users_col.insert_one({"email": email, "password": hashed_pw})
    return jsonify({"success": True, "message": "Signup successful"})

# === Login
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    user = users_col.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"})

@app.route('/stats', methods=['GET'])
def stats():
    username = request.args.get("username")
    if not username:
        return jsonify({"error": "Username is required"}), 400

    user_logs = list(chat_col.find({"user": username}))
    total = len(user_logs)
    emotion_counter = {}
    sessions = set()

    for entry in user_logs:
        emotion = entry.get("emotion", "neutral")
        emotion_counter[emotion] = emotion_counter.get(emotion, 0) + 1
        sessions.add(entry["timestamp"].date())

    return jsonify({
        "totalMessages": total,
        "sessions": len(sessions),
        "emotionCount": emotion_counter
    })


# === Run App
if __name__ == "__main__":
    app.run(debug=True)
