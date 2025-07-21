import string
import re

from data_io import load_data
import config

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def count_suspicious_bigrams(text):
    return sum(1 for bigram in config.suspicious_bigrams if bigram in text)

def preprocessing():
    spam_data = load_data('data/spam.csv')
    # Selecting relevant columns and combining them into a single message column
    columns_to_join = ['v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
    existing_columns = [col for col in columns_to_join if col in spam_data.columns]
    spam_data['full_message'] = spam_data[existing_columns].apply(
        lambda row: ' '.join(row.dropna().astype(str)), axis=1
    )
    spam_data = spam_data[['v1', 'full_message']].rename(columns={'v1': 'label', 'full_message': 'message'})
    
    # Clearing the text
    spam_data['cleaned'] = spam_data['message'].apply(clean_text)
    
    # Adding the feature of number of words, rather than number of letters
    spam_data['num_words'] = spam_data['message'].apply(lambda x: len(x.split()))

    # Preparing a list of suspicious bigrams
    suspicious_bigrams = config.suspicious_bigrams
    # numerical feature - 
    # Tree-based models handle numeric values ​​well and do not require normalization, 
    # so numeric rather than binary is preferable.
    spam_data['count_suspicious_bigrams'] = spam_data['cleaned'].apply(count_suspicious_bigrams)

    # Adding a feature for the presence of specific keywords
    keywords = config.keywords
    spam_data['has_keyword'] = spam_data['cleaned'].apply(
        lambda x: any(word in x for word in keywords)
    ).astype(int)

    # new feature - number of uppercase words
    spam_data['num_uppercase'] = spam_data['message'].apply(lambda x: sum(1 for w in x.split() if w.isupper()))

    # number or special characters 
    spam_data['num_exclamations'] = spam_data['message'].apply(lambda x: x.count('!'))
    spam_data['num_digits'] = spam_data['message'].apply(lambda x: len(re.findall(r'\d+', x)))

    # Presence of question/command words
    question_words = config.question_words
    spam_data['has_question_word'] = spam_data['cleaned'].apply(lambda x: any(w in x.split() for w in question_words)).astype(int)
    spam_data['has_question_mark'] = spam_data['message'].apply(lambda x: '?' in x).astype(int)

    # Does the message end with a word like "stop" (common in spam)
    spam_data['ends_with_stop'] = spam_data['cleaned'].apply(lambda x: x.strip().endswith('stop')).astype(int)
    
    # Does the message contain a phone number
    spam_data['has_phone_number'] = spam_data['message'].str.contains(r'\b(?:\+?\d{1,3})?[ -.]?\(?\d{2,4}\)?[ -.]?\d{3,4}[ -.]?\d{4}\b').astype(int)
    
    # Does the message contain a marketing intro
    spam_data['has_marketing_intro'] = spam_data['cleaned'].str[:30].apply(
        lambda x: any(word in x for word in ['congratulations', 'limited', 'offer', 'exclusive'])
    ).astype(int)
    return spam_data