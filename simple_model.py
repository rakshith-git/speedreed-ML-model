import spacy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
import textstat
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler

class SimpleFeatureExtractor:
    """
    Enhanced feature extractor with dependency parsing, word vectors, and sentiment
    """
    def __init__(self, nlp_model='en_core_web_lg'):
        self.nlp = spacy.load(nlp_model)
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
        # Scaler for word vectors
        self.vector_scaler = MinMaxScaler()
        
        # Original POS weights
        self.pos_weights = {
            'NOUN': 1.2,
            'VERB': 1.1,
            'ADJ': 1.3,
            'ADV': 1.15,
            'PROPN': 1.4,
            'NUM': 1.25,
            'PUNCT': 0.5,
            'DET': 0.6,
        }
        
        # NER weights
        self.ner_weights = {
            'PERSON': 1.4,
            'ORG': 1.3,
            'GPE': 1.3,
            'DATE': 1.2,
            'TIME': 1.2,
            'MONEY': 1.25,
            'PERCENT': 1.25,
            'PRODUCT': 1.3,
            'EVENT': 1.35,
            'WORK_OF_ART': 1.4,
        }
        
        # New dependency parsing weights
        self.dep_weights = {
            'ROOT': 1.4,     # Main verb
            'nsubj': 1.3,    # Subject
            'dobj': 1.2,     # Direct object
            'iobj': 1.2,     # Indirect object
            'amod': 1.1,     # Adjectival modifier
            'compound': 1.2,  # Compound words
            'conj': 1.1,     # Conjunction
            'cc': 0.8,       # Coordinating conjunction
            'det': 0.7,      # Determiner
            'mark': 0.9,     # Marker
            'case': 0.8,     # Case marking
            'punct': 0.5,    # Punctuation
        }
    
    def get_sentiment_score(self, sentence: str) -> float:
        """Get sentiment score normalized to 0-1 range"""
        result = self.sentiment_analyzer(sentence)[0]
        # Convert sentiment to score (positive = higher score)
        score = result['score'] if result['label'] == 'POSITIVE' else 1 - result['score']
        return score
    
    def process_word_vector(self, vector: np.ndarray) -> float:
        """Process word vector to a single meaningful value"""
        # Use vector magnitude as a measure of semantic richness
        magnitude = np.linalg.norm(vector)
        # Normalize to a reasonable range (assuming most magnitudes fall between 0-20)
        return min(magnitude / 20.0, 1.0)
    
    def extract_features(self, sentence: str) -> np.ndarray:
        """
        Extract enhanced linguistic features including dependency parsing,
        word vectors, and sentiment
        """
        doc = self.nlp(sentence)
        
        # Get sentence-level features
        flesch_score = textstat.flesch_reading_ease(sentence) / 100.0
        sentiment_score = self.get_sentiment_score(sentence)
        
        features = []
        # Create entity lookup for faster processing
        entity_dict = {}
        for ent in doc.ents:
            for i in range(ent.start, ent.end):
                entity_dict[i] = (ent.label_, ent.text)
        
        for i, token in enumerate(doc):
            # Original features
            word_length = textstat.lexicon_count(token.text, removepunct=True) / 20.0
            is_stop = float(token.is_stop)
            pos_weight = self.pos_weights.get(token.pos_, 1.0)
            syllable_count = textstat.syllable_count(token.text) / 10.0
            
            # NER feature
            ner_weight = self.ner_weights.get(entity_dict.get(i, (None, None))[0], 1.0)
            
            # New dependency parsing feature
            dep_weight = self.dep_weights.get(token.dep_, 1.0)
            
            # Word vector feature
            vector_importance = self.process_word_vector(token.vector)
            
            # Distance from root (normalized by sentence length)
            root_distance = min(len(list(token.ancestors)) / len(doc), 1.0)
            
            # Combine all features for this word
            word_features = [
                word_length,        # 1. Normalized word length
                is_stop,           # 2. Is it a stop word
                pos_weight,        # 3. Part of speech weight
                syllable_count,    # 4. Syllable count
                ner_weight,        # 5. Named entity weight
                flesch_score,      # 6. Flesch reading ease score
                dep_weight,        # 7. Dependency parsing weight
                vector_importance, # 8. Word vector importance
                root_distance,     # 9. Distance from root
                sentiment_score    # 10. Sentiment score
            ]
            features.append(word_features)
        
        # Convert to numpy array and pad/truncate to fixed length
        features = np.array(features)
        target_length = 20  # Max sentence length we'll consider
        
        if len(features) > target_length:
            features = features[:target_length]
        elif len(features) < target_length:
            padding = np.zeros((target_length - len(features), 10))  # 10 features per word
            features = np.vstack([features, padding])
        
        return features.flatten()
class SimpleRSVPModel(nn.Module):
    """
    Enhanced RSVP model for handling more features
    """
    def __init__(self, input_dim):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),  # Increased size for more features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 20),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x) + 0.5

class SimpleRSVPTrainer:
    """
    Training process updated for enhanced feature set
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = SimpleFeatureExtractor()
        
        # Input dim is 200 (20 words * 10 features per word)
        self.model = SimpleRSVPModel(input_dim=200).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
    
    def train(self, sentences: List[str], multipliers: List[List[float]], 
              epochs: int = 100, batch_size: int = 4):
        """
        Train the model on sentences and their speed multipliers
        """
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            # Process in small batches
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                batch_multipliers = multipliers[i:i + batch_size]
                
                batch_features = []
                batch_targets = []
                
                # Prepare batch data
                for sent, mults in zip(batch_sentences, batch_multipliers):
                    # Extract features
                    features = self.feature_extractor.extract_features(sent)
                    batch_features.append(features)
                    
                    # Prepare targets (pad/truncate to 20 words)
                    target = np.array(mults[:20] + [1.0] * (20 - len(mults)))
                    batch_targets.append(target)
                
                # Convert to tensors
                features_tensor = torch.FloatTensor(batch_features).to(self.device)
                targets_tensor = torch.FloatTensor(batch_targets).to(self.device)
                
                # Training step
                self.optimizer.zero_grad()
                predictions = self.model(features_tensor)
                loss = self.loss_fn(predictions, targets_tensor)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # Print progress
            avg_loss = total_loss / batch_count
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), "best_simple_rsvp_model.pth")
    
    def predict(self, sentence: str, base_delay: float = 0.25) -> List[tuple]:
        """
        Predict reading speed multipliers for a sentence
        """
        # Extract features
        features = self.feature_extractor.extract_features(sentence)
        features_tensor = torch.FloatTensor([features]).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            multipliers = self.model(features_tensor).cpu().numpy()[0]
        
        # Apply predictions to words
        words = sentence.split()[:20]  # Consider only first 20 words
        delays = [(word, base_delay * mult) for word, mult in zip(words, multipliers[:len(words)])]
        
        return delays

# Example usage remains the same
def train_and_test(sentences, multipliers):
    trainer = SimpleRSVPTrainer()
    
    # Train the model
    trainer.train(sentences, multipliers)
    
    # Test prediction
    test_sentence = "Let's test this model."
    predictions = trainer.predict(test_sentence)

    print("\nPredictions for test sentence:")
    for word, delay in predictions:
        print(f"{word}: {delay:.3f}s,")

if __name__ == "__main__":
    pass
    # train_and_test()