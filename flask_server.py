from flask import Flask, request, jsonify
import spacy
from simple_model import SimpleRSVPTrainer
import torch
from typing import List, Dict, Any

# Load spaCy model
nlp = spacy.load("en_core_web_lg")
app = Flask(__name__)

class RSVPPredictor:
    def __init__(self, model_path: str = "best_simple_rsvp_model.pth"):
        self.trainer = SimpleRSVPTrainer()
        # Use the new load_model method with proper device handling
        self.trainer.load_model(model_path)
        self.trainer.model.eval()
    
    def get_delays(self, sentence: str, base_delay: float = 0.2) -> Dict[str, Any]:
        """
        Get word delays for a sentence
        
        Args:
            sentence: Input sentence to process
            base_delay: Base delay in seconds
            
        Returns:
            Dictionary containing sentence and word delays
        """
        predictions = self.trainer.predict(sentence, base_delay)
        return {
            "sentence": sentence,
            "delays": [
                {"word": word, "delay": float(delay)}
                for word, delay in predictions
            ]
        }

# Initialize predictor
predictor = RSVPPredictor()

@app.route('/process-text', methods=['POST'])
def process_text() -> tuple[Any, int]:
    """
    Process text endpoint
    
    Expects JSON with 'text' field
    Returns JSON with processed sentences and word delays
    """
    # Validate request
    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json"
        }), 400
    
    data = request.get_json()
    if 'text' not in data:
        return jsonify({
            "error": "Missing 'text' field in request"
        }), 400
    
    text = data['text']
    
    try:
        # Use spaCy to split text into sentences
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Process each sentence
        results = []
        for sentence in sentences:
            if sentence:  # Skip empty sentences
                try:
                    sentence_delays = predictor.get_delays(sentence)
                    results.append(sentence_delays)
                except Exception as e:
                    # Log the error but continue processing other sentences
                    print(f"Error processing sentence '{sentence}': {str(e)}")
                    continue
        
        return jsonify({
            "results": results
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Error processing text: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
