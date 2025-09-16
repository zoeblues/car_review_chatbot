# First part: sentiment analysis with classification llm 

import pandas as pd
from transformers import logging, pipeline

logging.set_verbosity(logging.WARNING)

sentiment_pipeline = pipeline("sentiment-analysis")
data = pd.read_csv('car_reviews.csv', sep=';')

predicted_labels = sentiment_pipeline(data['Review'].tolist())
print(predicted_labels)

# Convert output to {0,1} format
predictions = [1 if r['label'].upper() == 'POSITIVE' else 0 for r in predicted_labels]
true_labels = [1 if str(label).upper() == 'POSITIVE' else 0 for label in data['Class']]

# Evaluation of the classification accuracy and F1 score of predictions
from sklearn.metrics import accuracy_score, f1_score

accuracy_result = accuracy_score(true_labels, predictions)
f1_result = f1_score(true_labels, predictions, pos_label=1)

print(f"Predictions (1 is positive, 0 is negative): {predictions}")
print(f"Accuracy: {accuracy_result:.2f}")
print(f"F1 Score: {f1_result:.2f}")



# Second part: eng-spanish translation llm
import pandas as pd
from transformers import pipeline
import nltk
from nltk.translate.bleu_score import sentence_bleu

first_review = data['Review'][0]
first_two_sentences = '.'.join(first_review.split('.')[:2]).strip() + '.'

translator = pipeline("translation_en_to_es", model = "Helsinki-NLP/opus-mt-en-es")
translated_review = translator(first_two_sentences, clean_up_tokenization_spaces=True)[0]['translation_text']

# Evaluate the model with the BLEU SCORE
from transformers import pipeline
import evaluate

with open("reference_translations.txt", encoding='utf-8') as file:
    reference_text = file.read().strip()

bleu = evaluate.load("bleu")

bleu_score = bleu.compute(predictions=[translated_review], references=[[reference_text]])

print("BLEU score dictionary:", bleu_score)



# Third part: Q&A LLM

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/minilm-uncased-squad2"

context = data['Review'][1]
question = "What did he like about the brand?"

#Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': question,
    'context': context
}
answer_text = nlp(QA_input)

# Extract only the answer text
answer = answer_text['answer']
print("Predicted answer:", answer)

# Summarize the last review in the dataset
summarizer = pipeline(task="summarization", model= "facebook/bart-large-cnn")
last_review = data['Review'].iloc[-1]

summary_output = summarizer(
    last_review,
    max_length=55,   
    min_length=50
)

summarized_text = summary_output[0]['summary_text']
print(summarized_text)