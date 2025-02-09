import json
import re
from hazm import Normalizer, Lemmatizer, stopwords_list
from bs4 import BeautifulSoup


normalizer = Normalizer()
lemmatizer = Lemmatizer()
stop_words = set(stopwords_list()) 


common_phrases = [
    "تمامی حقوق محفوظ است", "مطالب پیشنهادی", "کپی‌رایت", "پشتیبانی", "تماس با ما", "ورود به سایت"
]

def clean_text(text):
    
    text = BeautifulSoup(text, "html.parser").get_text()

    
    text = re.sub(r"\s+", " ", text)  
    text = re.sub(r"[^آ-ی\s]", "", text)  

    text = normalizer.normalize(text)


    for phrase in common_phrases:
        text = text.replace(phrase, "")

    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

 
    cleaned_text = " ".join(words).strip()

    return cleaned_text if len(cleaned_text.split()) > 5 else None  

with open("znu_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)


cleaned_data = []
unique_texts = set()

for doc in data:
    cleaned_text = clean_text(doc["text"])
    if cleaned_text and cleaned_text not in unique_texts:
        unique_texts.add(cleaned_text)
        cleaned_data.append({"url": doc["url"], "text": cleaned_text})


with open("clean_data.json", "w", encoding="utf-8") as file:
    json.dump(cleaned_data, file, ensure_ascii=False, indent=4)

