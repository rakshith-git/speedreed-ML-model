# main.py
import spacy
import json
import os

def get_text_input() -> str:
    """
    Reads a large amount of text input from a user.
    """
    print("Paste your text input below (press Enter twice to finish):")
    lines = []
    while True:
        line = input()
        if line == "EOF":
            break
        lines.append(line)
    return "\n".join(lines)

def get_output_filename() -> str:
    """
    Prompts the user to enter a file name for saving the data.
    """
    while True:
        filename = input("Enter the filename to save the output JSON (e.g., output.json): ").strip()
        if filename:
            # Ensure it has the .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            return filename
        else:
            print("Filename cannot be empty. Try again.")
            
def process_text_to_json(input_text, output_file):
    """
    Processes text using spaCy to split into sentences and words, allows manual multiplier tagging,
    and saves the resulting data into a JSON file.
    
    :param input_text: str, the large amount of text to process
    :param output_file: str, the path to save the resulting JSON data
    """
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(input_text)

    # Mapping for multipliers
    key_to_multiplier = {
        "j": 0.1, "k": 0.2, "l": 0.3, ";": 0.4,  # Positive adjustments
        "f": -0.1, "d": -0.2, "s": -0.3, "a": -0.4,  # Negative adjustments
        "g": 0.0,"h": 0.0  # No adjustment
    }

    data = []

    print("\n--- Begin Sentence Processing ---")
    for sent in doc.sents:
        sentence_data = {"sentence": sent.text, "multipliers": []}
        print(f"\nSentence: {sent.text}")
        print("Enter a string of characters for adjustment (j, k, l, ; for positive | f, d, s, a for negative | g,h for no adjustment): ")
        adjustment_string = input().strip() 
        word_array = sent.text.split()
        if len(word_array) != len(adjustment_string):
            print("Error not same length")
            adjustment_string = input().strip()         
        for i in range(len(word_array)):
            word_multiplier = 1.0
            char=adjustment_string[i]
            if char in key_to_multiplier:
                word_multiplier += key_to_multiplier[char]
                print(word_array[i],char,key_to_multiplier[char])
            sentence_data["multipliers"].append(word_multiplier)
        print(sentence_data)
        data.append(sentence_data)

    print("\n--- Sentence Processing Complete ---")

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nData saved to {output_file}")

# Import functions from helper.py
if __name__ == "__main__":
    input_text = get_text_input()
    output_file = get_output_filename()

    process_text_to_json(input_text, output_file)
