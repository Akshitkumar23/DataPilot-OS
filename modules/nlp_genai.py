"""
NLP and GenAI module for DataPilot OS.
Provides functionality for text cleaning, sentiment analysis, TF-IDF keyword extraction,
and generating NLP pipeline code.
"""

import logging
import re
from typing import List

import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLPGenAIEngine:
    """
    Engine for NLP and Generative AI tasks.
    """

    def __init__(self):
        """Initializes the NLPGenAIEngine."""
        pass

    def clean_text_column(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Cleans a text column in a DataFrame (lowercasing, stopword removal, special character removal).

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_col (str): The name of the text column to clean.

        Returns:
            pd.DataFrame: The DataFrame with a new cleaned text column '{text_col}_cleaned'.
        """
        if text_col not in df.columns:
            logger.error(f"Column '{text_col}' not found in DataFrame.")
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        logger.info(f"Cleaning text column: {text_col}")
        
        try:
            # Simple stopword list for basic cleaning without NLTK dependency
            stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])

            def clean_text(text: str) -> str:
                if not isinstance(text, str):
                    return ""
                # Lowercase
                text = text.lower()
                # Remove special characters and digits
                text = re.sub(r'[^a-z\s]', '', text)
                # Remove stopwords
                words = text.split()
                words = [w for w in words if w not in stopwords]
                return " ".join(words)

            df_out = df.copy()
            df_out[f"{text_col}_cleaned"] = df_out[text_col].apply(clean_text)
            logger.info("Text cleaning completed.")
            return df_out
        except Exception as e:
            logger.error(f"Error cleaning text column: {e}")
            raise

    def analyze_sentiment_polarity(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """
        Calculates sentiment polarity (-1.0 to +1.0) and adds a sentiment label column (Positive, Neutral, Negative).

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_col (str): The name of the text column to analyze.

        Returns:
            pd.DataFrame: The DataFrame with new columns '{text_col}_polarity' and '{text_col}_sentiment'.
        """
        if text_col not in df.columns:
            logger.error(f"Column '{text_col}' not found in DataFrame.")
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        logger.info(f"Analyzing sentiment for column: {text_col}")

        try:
            df_out = df.copy()
            
            def get_polarity(text: str) -> float:
                if not isinstance(text, str):
                    return 0.0
                return TextBlob(text).sentiment.polarity

            def get_sentiment_label(polarity: float) -> str:
                if polarity > 0.1:
                    return 'Positive'
                elif polarity < -0.1:
                    return 'Negative'
                else:
                    return 'Neutral'

            polarity_col = f"{text_col}_polarity"
            sentiment_col = f"{text_col}_sentiment"
            
            df_out[polarity_col] = df_out[text_col].apply(get_polarity)
            df_out[sentiment_col] = df_out[polarity_col].apply(get_sentiment_label)
            
            logger.info("Sentiment analysis completed.")
            return df_out
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            raise

    def extract_keywords_tfidf(self, df: pd.DataFrame, text_col: str, top_n: int = 10) -> List[str]:
        """
        Extracts top TF-IDF keywords from a text column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_col (str): The name of the text column.
            top_n (int): The number of top keywords to extract. Defaults to 10.

        Returns:
            List[str]: A list of the top `top_n` keywords.
        """
        if text_col not in df.columns:
            logger.error(f"Column '{text_col}' not found in DataFrame.")
            raise ValueError(f"Column '{text_col}' not found in DataFrame.")

        logger.info(f"Extracting top {top_n} keywords using TF-IDF from column: {text_col}")
        
        try:
            # Drop na and convert to string
            texts = df[text_col].dropna().astype(str).tolist()
            if not texts:
                logger.warning("No valid text data found for TF-IDF.")
                return []

            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Sum tfidf scores for each word across all documents
            sum_words = tfidf_matrix.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            
            top_keywords = [word for word, freq in words_freq[:top_n]]
            logger.info("Keyword extraction completed.")
            return top_keywords
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            raise

    def generate_nlp_pipeline_code(self, df: pd.DataFrame, text_col: str) -> str:
        """
        Generates complete Python code snippet for Sentiment + TF-IDF visualization.

        Args:
            df (pd.DataFrame): The input DataFrame (used for context if needed).
            text_col (str): The name of the text column.

        Returns:
            str: The generated Python code snippet.
        """
        logger.info(f"Generating NLP pipeline code for column: {text_col}")
        
        code = f'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Assuming 'df' is your DataFrame and '{text_col}' is the text column.
# Replace 'your_data.csv' with your actual data source if loading from file.
# df = pd.read_csv('your_data.csv')

# 1. Sentiment Analysis
def get_sentiment(text):
    if not isinstance(text, str):
        return 0.0
    return TextBlob(text).sentiment.polarity

def get_label(polarity):
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    return 'Neutral'

# Calculate sentiment
print("Calculating sentiment...")
df['Sentiment_Polarity'] = df['{text_col}'].apply(get_sentiment)
df['Sentiment_Label'] = df['Sentiment_Polarity'].apply(get_label)

# Plot Sentiment Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Sentiment_Label', palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# 2. TF-IDF Keyword Extraction
print("Extracting keywords...")
texts = df['{text_col}'].dropna().astype(str).tolist()
if texts:
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Calculate average TF-IDF score for each word
    avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    word_scores = list(zip(vectorizer.get_feature_names_out(), avg_scores))
    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)[:10]
    
    words, scores = zip(*word_scores)
    
    # Plot Top Keywords
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(scores), y=list(words), palette='mako')
    plt.title('Top 10 Keywords by TF-IDF Score')
    plt.xlabel('Average TF-IDF Score')
    plt.ylabel('Keyword')
    plt.show()
else:
    print("No valid text data to extract keywords.")
'''
        return code.strip()


# Global Singleton Instance
nlp_engine = NLPGenAIEngine()
