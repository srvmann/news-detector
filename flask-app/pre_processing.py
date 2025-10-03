import re
import string
from nltk.corpus import stopwords
import nltk



def preprocess_text(text):
    """
    Cleans and preprocesses a single string of text.

    Steps include:
    - Converting to lowercase.
    - Removing HTML tags, URLs, and emails.
    - Removing punctuation.
    - Removing non-ASCII characters.
    - Tokenizing and removing stopwords.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The cleaned and preprocessed text.
    """
    # Ensure it's a string and convert to lowercase
    text = str(text).lower()
    
    # Remove HTML, URLs, and emails
    text = re.sub(r"<.*?>|http\S+|www\.\S+|\S+@\S+", "", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()
    
    # Tokenize and remove stopwords
    words = text.split()
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join back into a single string
    return " ".join(filtered_words)
