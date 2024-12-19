import torch
import numpy as np
import spacy
import model
def load_rsvp_model(model_path='trained_rsvp_model.pth', nlp_model='en_core_web_lg'):
    """
    Load the trained RSVP model for making predictions
    
    Args:
    - model_path: Path to the saved model weights
    - nlp_model: Spacy language model to use
    
    Returns:
    - Tuple of (meta_learner, model) ready for predictions
    """
    # Initialize meta learner
    meta_learner = model.AdvancedRSVPMetaLearner(nlp_model)
    
    # Prepare a sample to get input dimension
    sample_sentence = "A sample sentence to initialize the model"
    sample_features = meta_learner.extract_enhanced_features(sample_sentence)
    sample_processed = meta_learner.preprocess_features(sample_features)
    input_dim = sample_processed.shape[1]
    
    # Initialize feature weights and model
    feature_weights = model.AdvancedFeatureWeights()
    meta_learner.model = model.MetaLearningRSVPModel(input_dim, feature_weights)
    
    # Load saved weights
    meta_learner.model.load_state_dict(torch.load(model_path))
    meta_learner.model.eval()  # Set to evaluation mode
    
    return meta_learner

def predict_word_reading_times(sentence, 
                                meta_learner=None, 
                                model_path='trained_rsvp_model.pth', 
                                base_delay=0.3):
    """
    Predict reading times for each word in a sentence
    
    Args:
    - sentence: Input sentence to analyze
    - meta_learner: Optional pre-loaded meta learner
    - model_path: Path to saved model weights
    - base_delay: Base reading time per word
    
    Returns:
    - List of (word, predicted_delay) tuples
    """
    # Load model if not provided
    if meta_learner is None:
        meta_learner = load_rsvp_model(model_path)
    
    # Extract features
    features = meta_learner.extract_enhanced_features(sentence)
    processed_features = torch.tensor(
        meta_learner.preprocess_features(features), 
        dtype=torch.float32
    )
    
    # Tokenize the sentence
    words = sentence.split()
    
    # Predict multipliers
    with torch.no_grad():
        multipliers = meta_learner.model(processed_features)
    print(multipliers)
    # Ensure multipliers match word count
    if len(multipliers) > len(words):
        multipliers = multipliers[:len(words)]
    elif len(multipliers) < len(words):
        # Pad with default multiplier
        padded_multipliers = torch.ones(len(words))
        padded_multipliers[:len(multipliers)] = multipliers
        multipliers = padded_multipliers
    
    # Calculate word-level delays
    word_delays = [
        (word, base_delay * float(multiplier)) 
        for word, multiplier in zip(words, multipliers)
    ]
    
    return word_delays

def visualize_reading_times(word_delays, output_path='reading_times.png'):
    """
    Create a bar plot of word reading times
    
    Args:
    - word_delays: List of (word, delay) tuples
    - output_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    words, delays = zip(*word_delays)
    plt.figure(figsize=(12, 6))
    plt.bar(words, delays)
    plt.title('Predicted Word Reading Times')
    plt.xlabel('Words')
    plt.ylabel('Reading Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Example usage
def main():
    # Load the model
    meta_learner = load_rsvp_model()
    
    # Example sentences
    sentences = [
        "Robotics engineers design advanced machines capable of complex tasks autonomously.",
        "Quantum computing promises exponential increases in processing power for solving intricate problems.",
        "Technology surrounds us constantly, shaping our lives in countless ways.",
        "Philosophical inquiries into consciousness challenge our fundamental understanding of human perception.",
    ]
    
    # Predict and visualize for each sentence
    for sentence in sentences:
        print(f"\nAnalyzing sentence: {sentence}")
        word_delays = predict_word_reading_times(
            sentence, 
            meta_learner=meta_learner
        )
        
        # Print detailed results
        total_time = 0
        print("\nWord-level Reading Times:")
        for word, delay in word_delays:
            print(f"{word}: {delay:.3f} seconds")
            total_time += delay
        
        print(f"\nTotal estimated reading time: {total_time:.3f} seconds")
        
        # Optional: Visualize reading times
        visualize_reading_times(
            word_delays, 
            output_path=f'reading_times_{hash(sentence)}.png'
        )

if __name__ == "__main__":
    main()