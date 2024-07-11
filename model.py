import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the classifier for Emotion
with open('emotion_classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f)

# Load the vectorizer for Emotion
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the classifier for Sentiment
with open('sentiment_classifier.pkl', 'rb') as f:
    loaded_clf_sent = pickle.load(f)

# Load the vectorizer for Sentiment
with open('vectorizer_sent.pkl', 'rb') as f:
    vectorizer_sent = pickle.load(f)

# Load the classifier for Sentiment
with open('rating_classifier.pkl', 'rb') as f:
    loaded_clf_rating = pickle.load(f)

# Load the vectorizer for Sentiment
with open('vectorizer_rating.pkl', 'rb') as f:
    vectorizer_rating = pickle.load(f)

# Load the classifier for Rating 
with open('rating_classifier.pkl', 'rb') as f:
    loaded_clf_rating = pickle.load(f)

# Load the vectorizer for Rating
with open('vectorizer_rating.pkl', 'rb') as f:
    vectorizer_rating = pickle.load(f)


stopwords = set([
    "akin", "aking", "ako", "alin", "am", "amin", "aming", "ang", "ano", "anumang",
    "apat", "at", "atin", "ating", "ay", "bababa", "bakit", "bawat", "bilang", "dahil",
    "dalawa", "dapat", "din", "dito", "doon", "gagawin", "gayunman", "ginagawa", "ginawa",
    "ginawang", "gumawa", "habang", "hanggang", "iba", "ibaba", "ibabaw", "ikaw", "ilagay",
    "ilalim", "ilan", "inyong", "isa", "isang", "itaas", "ito", "iyo", "iyon", "iyong", "ka",
    "kahit", "kailangan", "kami", "kanila", "kanilang", "kanino", "kanya", "kanyang", "kapag",
    "kapwa", "karamihan", "katiyakan", "katulad", "kaya", "kaysa", "ko", "kong", "kulang",
    "kumuha", "kung", "laban", "lahat", "lamang", "likod", "lima", "maaari", "maaaring", "maging",
    "mahusay", "makita", "marami", "marapat", "masyado", "may", "mayroon", "mga", "minsan", "mismo",
    "mula", "muli", "na", "nabanggit", "naging", "nagkaroon", "nais", "nakita", "namin", "napaka",
    "narito", "nasaan", "ng", "ngayon", "ni", "nila", "nilang", "nito", "niya", "niyang", "noon",
    "o", "pa", "paano", "pababa", "paggawa", "pagitan", "pagkakaroon", "pagkatapos", "palabas",
    "pamamagitan", "panahon", "pangalawa", "para", "paraan", "pareho", "pataas", "pero", "pumunta",
    "pumupunta", "sa", "saan", "sabi", "sabihin", "sarili", "si", "sila", "sino", "siya", "tatlo",
    "tayo", "tulad", "tungkol", "una"
])

# Function for Text Preprocessing
def cleaned_data(text):    
    # Tokenize
    tokens = re.split('\W+', text)
    
    # Remove stopwords
    text_clean = [word for word in tokens if word.lower() not in stopwords]
    
    # Join the words back into a string
    cleaned_text = ' '.join(text_clean)
    
    return cleaned_text

# Function for Emotion Recognition
def classify_cleaned(text):
    cleaned_text = cleaned_data(text)
    # Transform the text using the loaded vectorizer
    text_vector = vectorizer.transform([cleaned_text])
    return loaded_clf.predict(text_vector)

# Function for Sentiment Recognition
def classify_sent(text):
    cleaned_text = cleaned_data(text)
    # Transform the text using the loaded vectorizer
    text_vector = vectorizer_sent.transform([cleaned_text])
    return loaded_clf_sent.predict(text_vector)


# Function for Rating Classification
def classify_rating(text):
    cleaned_text = cleaned_data(text)
    # Transform the text using the loaded vectorizer
    text_vector = vectorizer_rating.transform([cleaned_text])
    return loaded_clf_rating.predict(text_vector)