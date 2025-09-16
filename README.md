# Car Review LLM 

## Project overview:
This project was developed as part of a DataCamp assignment simulating a real-world scenario. My task was to prototype a chatbot-like application that leverages Large Language Models (LLMs) to handle diverse customer inquiries.

The prototype demonstrates how LLMs can support:
- Sentiment classification of customer reviews
- Translation into another language with BLEU evaluation
- Extractive question answering 
- Summarization of long customer reviews

## Project structure:
```
├── car_reviews.csv              # Dataset with 5 car reviews
├── reference_translations.txt   # Reference translations for BLEU scoring
├── notebook.ipynb               # Implementation of all tasks
├── README.md                   
```

## Tech Stack
- Python
- Hugging Face Transformers
- NLTK for BLEU scoring
- Scikit-learn for evaluation metrics

## Real-World Impact
This prototype shows how LLMs can assist customer support teams in:
- Understanding customer sentiment
- Serving international audiences with translations
- Answering product-specific queries
- Condensing long customer feedback into digestible insights
