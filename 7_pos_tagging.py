import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def pos_tagging(words_list):
    """
    Performs POS tagging on a list of words.
    """
    return pos_tag(words_list)

def find_pos(word, tagged_list):
    """
    Finds the POS tag for a given word.
    """
    for w, pos in tagged_list:
        if w.lower() == word.lower():
            return pos
    return "POS not found"

def main():
    """
    Main function to demonstrate POS tagging and finding POS for a given word.
    """
    # Sample list of words
    words_list = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    
    # Perform POS tagging
    tagged_words = pos_tagging(words_list)
    
    # Display POS tagging results
    print("POS tagging results:")
    for word, pos in tagged_words:
        print(f"{word}: {pos}")
    
    # Take input from user
    search_word = input("\nEnter a word to find its POS: ")
    pos_result = find_pos(search_word, tagged_words)
    print(f"POS for '{search_word}': {pos_result}")

if __name__ == "__main__":
    main()

