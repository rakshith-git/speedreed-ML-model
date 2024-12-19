from flask import Flask, request, jsonify
import spacy
from simple_model import SimpleRSVPTrainer
import torch

# Load spaCy model
nlp = spacy.load("en_core_web_lg")

app = Flask(__name__)

# Initialize the RSVP model
class RSVPPredictor:
    def __init__(self, model_path="best_simple_rsvp_model.pth"):
        self.trainer = SimpleRSVPTrainer()
        self.trainer.model.load_state_dict(torch.load(model_path))
        self.trainer.model.eval()

    def get_delays(self, sentence):
        predictions = self.trainer.predict(sentence,0.1)
        return {
            "sentence": sentence,
            "delays": [
                {"word": word, "delay": float(delay)}
                for word, delay in predictions
            ]
        }

predictor = RSVPPredictor()

@app.route('/process-text', methods=['POST'])
def process_text():
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    text = data['text']
    
    # Use spaCy to split text into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Process each sentence
    results = []
    for sentence in sentences:
        if sentence:  # Skip empty sentences
            sentence_delays = predictor.get_delays(sentence)
            results.append(sentence_delays)
    
    return jsonify({
        "results": results
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)