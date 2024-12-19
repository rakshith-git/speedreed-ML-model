import spacy
import textstat
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from typing import List, Dict, Any
from copy import deepcopy
import matplotlib.pyplot as plt

class FeatureDimensionNormalizer:
    """
    Ensures consistent feature dimensionality
    """
    def __init__(self, target_dim=256):
        self.target_dim = target_dim
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize input features to a consistent dimension
        
        Args:
            features (np.ndarray): Input feature array
        
        Returns:
            np.ndarray: Normalized feature array
        """
        current_dim = features.shape[1]
        
        if current_dim > self.target_dim:
            # Truncate features if too many
            return features[:, :self.target_dim]
        elif current_dim < self.target_dim:
            # Pad features with zeros if too few
            pad_width = ((0, 0), (0, self.target_dim - current_dim))
            return np.pad(features, pad_width, mode='constant')
        
        return features

class AdvancedFeatureWeights:
    """
    Comprehensive feature weighting system for linguistic annotations
    """
    def __init__(self):
        # Detailed POS tag processing time multipliers
        self.pos_weights = {
            # Cognitive processing complexity weights
            'NOUN': 1.2,      # Nouns require more processing
            'VERB': 1.1,      # Verbs have moderate complexity
            'ADJ': 1.3,       # Descriptive words need more time
            'ADV': 1.15,      # Adverbs moderately complex
            'PROPN': 1.4,     # Proper nouns need extra attention
            'NUM': 1.25,      # Numbers require careful reading
            'PUNCT': 0.5,     # Punctuation needs less time
            'ADP': 0.7,       # Prepositions are quick to process
            'CONJ': 0.6,      # Conjunctions are quick
            'DET': 0.6,       # Determiners are very fast
        }
        
        # Named Entity type processing complexity
        self.ner_weights = {
            # Cognitive load for different entity types
            'PERSON': 1.4,    # Names require more careful reading
            'ORG': 1.3,       # Organization names need attention
            'GPE': 1.2,       # Geo-political entities are complex
            'LOC': 1.1,       # Locations moderately complex
            'DATE': 1.25,     # Dates require careful parsing
            'MONEY': 1.3,     # Financial terms need focus
            'TIME': 1.2,      # Time expressions need attention
            '': 1.0           # Default for non-entities
        }

class SentimentFeatureExtractor:
    """
    Advanced sentiment and emotional complexity analysis
    """
    def __init__(self):
        self.sentiment_model = transformers.pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        Extract comprehensive sentiment and emotional features
        """
        # Sentiment analysis
        sentiment_result = self.sentiment_model(text)[0]
        
        return {
            'sentiment_score': (
                1 if sentiment_result['label'] == 'POSITIVE' 
                else -1
            ) * sentiment_result['score'],
            'emotional_complexity': len(text.split()) / 10  # Rough complexity metric
        }

class MetaLearningRSVPModel(nn.Module):
    def __init__(self, input_dim: int, feature_weights: AdvancedFeatureWeights):
        super().__init__()
        self.feature_weights = feature_weights
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.multiplier_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
            
        )
        
        # Meta-learning adaptation layer
        self.adaptation_layer = nn.Linear(128, 128)
    
    def forward(self, x, adaptation=False):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        features = self.feature_extractor(x)
        
        if adaptation:
            features = torch.relu(self.adaptation_layer(features))
        
        # Scale output to reasonable delay multiplier range
        multiplier = self.multiplier_predictor(features) *2
        return multiplier

class AdvancedRSVPMetaLearner:
    def __init__(self, nlp_model='en_core_web_lg'):
        # Ensure GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device set to use {self.device}")
        
        self.nlp = spacy.load(nlp_model)
        self.feature_weights = AdvancedFeatureWeights()
        self.sentiment_extractor = SentimentFeatureExtractor()
        
        # Feature dimension normalizer
        self.dimension_normalizer = FeatureDimensionNormalizer(target_dim=256)
        
        self.model = None
        self.optimizer = None
        self.loss_fn = None
    
    def extract_enhanced_features(self, sentence: str) -> Dict[str, Any]:
        """
        Extract comprehensive linguistic and contextual features
        """
        doc = self.nlp(sentence)
        
        features = {
            'pos_features': [],
            'ner_features': [],
            'linguistic_features': {
                'word_lengths': [],
                'is_stop_word': [],
                'pos_weights': [],
                'ner_weights': [],
                'syllable_counts': []
            },
            'readability_metrics': {
                'flesch_reading_ease': textstat.flesch_reading_ease(sentence),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(sentence),
                'gunning_fog': textstat.gunning_fog(sentence),
                'smog_index': textstat.smog_index(sentence),
                'automated_readability_index': textstat.automated_readability_index(sentence),
                'coleman_liau_index': textstat.coleman_liau_index(sentence),
                'linsear_write_formula': textstat.linsear_write_formula(sentence),
                'dale_chall_readability_score': textstat.dale_chall_readability_score(sentence),
                'difficult_words_ratio': len(textstat.difficult_words_list(sentence)) / len(sentence.split()),
                'avg_syllables_per_word': textstat.avg_syllables_per_word(sentence)
            }
        }
        
        # Sentiment features
        sentiment_data = self.sentiment_extractor.extract_sentiment_features(sentence)
        
        for token in doc:
            # POS tag features with custom weighting
            pos_tag = token.pos_
            pos_weight = self.feature_weights.pos_weights.get(pos_tag, 1.0)
            
            # NER features with custom weighting
            ner_type = token.ent_type_
            ner_weight = self.feature_weights.ner_weights.get(ner_type, 1.0)
            
            # Calculate syllable count for the token
            syllable_count = textstat.syllable_count(token.text)
            
            features['pos_features'].append(pos_tag)
            features['ner_features'].append(ner_type)
            
            features['linguistic_features']['word_lengths'].append(len(token.text))
            features['linguistic_features']['is_stop_word'].append(int(token.is_stop))
            features['linguistic_features']['pos_weights'].append(pos_weight)
            features['linguistic_features']['ner_weights'].append(ner_weight)
            features['linguistic_features']['syllable_counts'].append(syllable_count)
        
        # Add global sentiment features
        features['sentiment'] = sentiment_data
        features['flesch_score'] = textstat.flesch_reading_ease(sentence)
        
        return features
    
    def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Enhanced feature preprocessing with readability metrics
        """
        if not features['linguistic_features']['word_lengths']:
            return np.zeros((1, 256))
        
        # One-hot encode categorical features
        pos_encoded = pd.get_dummies(features['pos_features'])
        ner_encoded = pd.get_dummies(features['ner_features'])
        
        # Get readability metrics
        readability_metrics = np.array([
            features['readability_metrics']['flesch_reading_ease'],
            features['readability_metrics']['flesch_kincaid_grade'],
            features['readability_metrics']['gunning_fog'],
            features['readability_metrics']['smog_index'],
            features['readability_metrics']['automated_readability_index'],
            features['readability_metrics']['coleman_liau_index'],
            features['readability_metrics']['linsear_write_formula'],
            features['readability_metrics']['dale_chall_readability_score'],
            features['readability_metrics']['difficult_words_ratio'],
            features['readability_metrics']['avg_syllables_per_word']
        ])
        
        # Normalize readability metrics
        readability_metrics = (readability_metrics - np.mean(readability_metrics)) / np.std(readability_metrics)
        
        # Process linguistic features
        word_lengths = np.array(features['linguistic_features']['word_lengths'])
        is_stop_word = np.array(features['linguistic_features']['is_stop_word'])
        pos_weights = np.array(features['linguistic_features']['pos_weights'])
        ner_weights = np.array(features['linguistic_features']['ner_weights'])
        syllable_counts = np.array(features['linguistic_features']['syllable_counts'])
        
        # Get sentiment features
        sentiment_score = features['sentiment']['sentiment_score']
        emotional_complexity = features['sentiment']['emotional_complexity']
        
        # Ensure all features match the primary dimension
        primary_dim = len(word_lengths)
        
        # Combine numerical features
        numerical_features = np.column_stack([
            word_lengths,
            is_stop_word,
            pos_weights,
            ner_weights,
            syllable_counts,
            np.full(primary_dim, sentiment_score),
            np.full(primary_dim, emotional_complexity),
            # Repeat readability metrics for each word
            np.tile(readability_metrics, (primary_dim, 1))
        ])
        
        # Combine all features
        processed_features = np.hstack([
            pos_encoded,
            ner_encoded,
            numerical_features
        ])
        
        # Normalize to consistent dimension
        return self.dimension_normalizer.normalize(processed_features)
    
    def meta_train_step(self, support_sentences, support_multipliers, 
                        query_sentences, query_multipliers, base_delay=0.3):
        """
        Custom loss function for meta-learning
        """
        def custom_loss(predictions, targets, base_delay=0.3):
            # Penalize deviations from base delay more strongly
            deviation_loss = torch.mean(torch.square(predictions - base_delay))
            # Ensure predictions are close to targets
            target_loss = torch.mean(torch.square(predictions - targets))
            
            # Combine losses with weights
            return 0.5 * target_loss + 0.5 * deviation_loss
        
        # Set custom loss function
        self.loss_fn = custom_loss
        
        # Prepare support set (inner loop)
        support_features = []
        support_targets = []
        
        for sentence, multipliers in zip(support_sentences, support_multipliers):
            features = self.extract_enhanced_features(sentence)
            features['multipliers'] = multipliers  # Add multipliers to features
            processed_features = torch.tensor(
                self.preprocess_features(features), 
                dtype=torch.float32
            ).to(self.device)
            
            # Pad or truncate multipliers to match feature length
            padded_multipliers = torch.zeros(processed_features.shape[0], dtype=torch.float32)
            padded_multipliers[:len(multipliers)] = torch.tensor(multipliers)
            
            support_features.append(processed_features)
            support_targets.append(padded_multipliers)
        
        # Inner loop adaptation
        adapted_model = deepcopy(self.model)
        inner_optimizer = optim.Adam(adapted_model.parameters(), lr=0.001)
        
        for features, targets in zip(support_features, support_targets):
            targets = targets.to(self.device)
            inner_optimizer.zero_grad()
            predictions = adapted_model(features, adaptation=True)
            
            # Ensure predictions match targets in size
            inner_loss = self.loss_fn(predictions.squeeze(), targets)
            inner_loss.backward()
            inner_optimizer.step()
        
        # Outer loop (meta-update)
        query_losses = []
        for sentence, multipliers in zip(query_sentences, query_multipliers):
            features = self.extract_enhanced_features(sentence)
            processed_features = torch.tensor(
                self.preprocess_features(features), 
                dtype=torch.float32
            ).to(self.device)
            
            # Pad or truncate multipliers to match feature length
            padded_multipliers = torch.zeros(processed_features.shape[0], dtype=torch.float32)
            padded_multipliers[:len(multipliers)] = torch.tensor(multipliers)
            targets = padded_multipliers.to(self.device)
            
            predictions = self.model(processed_features)
            query_loss = self.loss_fn(predictions.squeeze(), targets)
            query_losses.append(query_loss)
        
        # Aggregate meta-loss
        meta_loss = torch.mean(torch.stack(query_losses))
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        
        return meta_loss.item()

def meta_training_step(sents, multipliers, train_ratio=0.7, epochs=100, base_delay=0.3):
    # Ensure inputs are aligned
    assert len(sents) == len(multipliers), "Sentences and multipliers must have the same length."

    # Split data into support (training) and query (testing) sets
    split_idx = int(len(sents) * train_ratio)
    support_sentences = sents[:split_idx]
    support_multipliers = multipliers[:split_idx]
    query_sentences = sents[split_idx:]
    query_multipliers = multipliers[split_idx:]

    print(f"Training on {len(support_sentences)} support sentences")
    print(f"Validating on {len(query_sentences)} query sentences")
    
    # Initialize meta learner and model
    meta_learner = AdvancedRSVPMetaLearner()
    
    # Ensure all lists in multipliers are floats
    support_multipliers = [[float(m) for m in ms] for ms in support_multipliers]
    query_multipliers = [[float(m) for m in ms] for ms in query_multipliers]
    
    # Prepare initial features for input dimension
    sample_features = meta_learner.extract_enhanced_features(support_sentences[0])
    sample_features['multipliers'] = support_multipliers[0]
    sample_processed = meta_learner.preprocess_features(sample_features)
    input_dim = sample_processed.shape[1]
    
    # Feature weights and model initialization
    feature_weights = AdvancedFeatureWeights()
    meta_learner.model = MetaLearningRSVPModel(input_dim, feature_weights).to(meta_learner.device)
    meta_learner.optimizer = optim.Adam(meta_learner.model.parameters(), lr=0.01)
    
    # Track losses for monitoring
    epoch_losses = []
    
    # Run multiple epochs
    for epoch in range(epochs):
        # Perform meta-learning training step
        loss = meta_learner.meta_train_step(
            support_sentences,
            support_multipliers,
            query_sentences,
            query_multipliers,
            base_delay
        )
        epoch_losses.append(loss)
        
        # Print progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Meta-Training Loss = {loss}")
    
    # Plot loss progression
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses)
    plt.title('Meta-Learning Loss Progression')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('training_loss.png')
    
    # Save final model
    torch.save(meta_learner.model.state_dict(), "trained_rsvp_model.pth")
    print("Model saved to 'trained_rsvp_model.pth'")
    
    return meta_learner

def predict_word_delays(model, meta_learner, sentence, base_delay=0.25):
    # Extract features
    features = meta_learner.extract_enhanced_features(sentence)
    processed_features = torch.tensor(
        meta_learner.preprocess_features(features), 
        dtype=torch.float32
    ).to(meta_learner.device)
    
    words = sentence.split()
    
    # Predict multipliers
    with torch.no_grad():
        multipliers = model(processed_features).cpu().numpy()
    
    # Ensure multipliers are close to base_delay
    multipliers = np.clip(multipliers, 0.5, 1.5)  # Constrain multipliers
    
    # Combine words with their predicted delay multipliers
    padded_multipliers = np.ones(len(words)) * base_delay
    padded_multipliers[:len(multipliers)] = multipliers.squeeze()
    
    word_delays = [
        (word, base_delay * multiplier) 
        for word, multiplier in zip(words, padded_multipliers)
    ]
    
    return word_delays