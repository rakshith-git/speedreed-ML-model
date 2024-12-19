import textstat

# Sample text
sample_text = """
"As we navigate the treacherous waters of historical interpretation, it becomes clear that the past is a prism refracting multiple perspectives, each casting its own shadow on the canvas of human experience."
"""

# Calculate Flesch Reading Ease score
flesch_score = textstat.flesch_reading_ease(sample_text)

print(f"Sample Text:")
print(sample_text)
print("\nFlesch Reading Ease Score:", flesch_score)

# Interpretation guide
if flesch_score >= 90:
    print("Very Easy")
elif 80 <= flesch_score < 90:
    print("Easy")
elif 70 <= flesch_score < 80:
    print("Fairly Easy")
elif 60 <= flesch_score < 70:
    print("Standard")
elif 50 <= flesch_score < 60:
    print("Fairly Difficult")
elif 30 <= flesch_score < 50:
    print("Difficult")
else:
    print("Very Confusing")

print("\nAdditional statistics:")
print(f"Number of words: {textstat.lexicon_count(sample_text)}")
print(f"Number of sentences: {textstat.sentence_count(sample_text)}")
print(f"Number of characters: {textstat.char_count(sample_text)}")
