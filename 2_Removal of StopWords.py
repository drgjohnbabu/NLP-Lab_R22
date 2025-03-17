import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

def remove_stopwords(text):
    """
    Removes stop words from the tokenized text.

    Args:
    text (str): The input text to be processed.

    Returns:
    list: A list of words after removing stop words.
    """
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Get English stop words
    stop_words = set(stopwords.words('english'))
    
    # Remove stop words
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    return filtered_tokens

def main():
    """
    Main function to demonstrate stop word removal.
    """
    # Sample text for demonstration
    text = """This is a simple example to demonstrate how stop word removal works in text processing. 
    It removes commonly used words like 'is', 'a', 'to', and so on."""
    
    print("Original Text:")
    print(text)
    
    # Perform stop word removal
    filtered_words = remove_stopwords(text)
    
    # Display results
    print("\nText After Stop Word Removal:")
    print(filtered_words)

if __name__ == "__main__":
    main()
