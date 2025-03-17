import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.corpus import wordnet
from nltk import pos_tag
from collections import Counter

# Download required resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(word):
    """
    Converts POS tag to format compatible with WordNet Lemmatizer.
    """
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN if not found

def morphological_analysis(sentence):
    """
    Performs morphological analysis on a sentence using lemmatization.
    """
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return " ".join(lemmatized_words)

def generate_ngrams(text, n):
    """
    Generates n-grams from a given text.
    """
    tokens = word_tokenize(text)
    return list(ngrams(tokens, n))

def laplace_smoothing(ngrams_list, vocab):
    """
    Implements Laplace (add-one) smoothing for n-grams.
    """
    ngram_counts = Counter(ngrams_list)
    total_ngrams = sum(ngram_counts.values())
    vocab_size = len(vocab)
    smoothed_probs = {ngram: (ngram_counts.get(ngram, 0) + 1) / (total_ngrams + vocab_size) for ngram in vocab}
    return smoothed_probs, ngram_counts

def main():
    """
    Main function to demonstrate morphological analysis, n-gram generation, and n-gram smoothing.
    """
    sentence = "The running dogs are barking loudly."
    lemmatized_sentence = morphological_analysis(sentence)
    print(f"Original sentence: {sentence}")
    print(f"Morphological analysis (lemmatized): {lemmatized_sentence}")
    
    text = "The quick brown fox jumps over the lazy dog"
    n = 3
    trigrams_list = generate_ngrams(text, n)
    print(f"\nGenerated {n}-grams:")
    print(trigrams_list)
    
    vocab = set(trigrams_list)
    vocab.add(("blue", "fox", "jumps"))
    vocab.add(("lazy", "cat", "sleeps"))
    
    print("\nN-gram probabilities before smoothing:")
    ngram_counts = Counter(trigrams_list)
    total_ngrams = sum(ngram_counts.values())
    for ngram in vocab:
        prob = ngram_counts.get(ngram, 0) / total_ngrams if total_ngrams > 0 else 0
        print(f"{ngram}: {prob:.4f}")
    
    smoothed_probs, _ = laplace_smoothing(trigrams_list, vocab)
    print("\nN-gram probabilities after Laplace smoothing:")
    for ngram, prob in smoothed_probs.items():
        print(f"{ngram}: {prob:.4f}")

if __name__ == "__main__":
    main()