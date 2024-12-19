import torch
import matplotlib.pyplot as plt
from simple_model import SimpleRSVPTrainer  # Import your model class
import seaborn as sns
import numpy as np
class RSVPVisualizer:
    def __init__(self, model_path="best_simple_rsvp_model.pth"):
        self.trainer = SimpleRSVPTrainer()
        
        # Load the saved model
        self.trainer.model.load_state_dict(torch.load(model_path))
        self.trainer.model.eval()  # Set to evaluation mode
    
    def visualize_sentence(self, sentence: str, base_delay: float = 0.1):
        """
        Visualize the word delays for a sentence
        """
        # Get predictions
        predictions = self.trainer.predict(sentence,base_delay)
        
        # Separate words and delays
        words, delays = zip(*predictions)
        
        # Create figure with larger size
        plt.figure(figsize=(15, 6))
        
        # Create bar plot
        bars = plt.bar(range(len(words)), delays)
        
        # Customize the plot
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.ylabel('Delay (seconds)')
        plt.title(f'Word Delays for: "{sentence}"')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s',
                    ha='center', va='bottom')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Show grid for better readability
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        return plt.gcf()  # Return the figure
    

def test_sentences():
    # Initialize visualizer
    visualizer = RSVPVisualizer()
    
    # Test sentences
    test_sentences = [
        "It will simplify your job and help the audience better understand the core of the problem",
        "Campaigns involve debating and advertising to sway voters, elections determine who holds office",
        "The quick brown fox jumped over the lazy dog. Politics is the practice and theory of government.",
        "Politics is the practice and theory of government",
        # Add more test sentences as needed
    ]
    
    # Create visualizations for each sentence
    for i, sentence in enumerate(test_sentences, 1):
        # Create bar plot
        fig1 = visualizer.visualize_sentence(sentence)
        fig1.savefig(f'sentence_{i}_bars.png')
        plt.close(fig1)
        
        
        # Print the predictions
        predictions = visualizer.trainer.predict(sentence)
        print(f"\nPredictions for: {sentence}")
        print("Word\t\tDelay")
        print("-" * 30)
        for word, delay in predictions:
            print(f"{word:<15} {delay:.3f}s")
        print("\n")

if __name__ == "__main__":
    test_sentences()