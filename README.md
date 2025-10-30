Emotion Detection System Using Large Language Models
====================================================

This project implements a text-based emotion detection system using fine-tuned transformer models (BERT, DistilBERT, XLNet). It classifies emotions from social media text and deploys the best-performing model via a Streamlit-based API.

Features
--------
- Fine-tuned LLMs for emotion classification
- Supports multiple emotion categories (joy, sadness, anger, fear, love, surprise)
- Streamlit-based web interface for real-time predictions
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Bias and interpretability analysis

Technologies Used
-----------------
- Python 3.10+
- Transformers (Hugging Face)
- PyTorch or TensorFlow
- Scikit-learn
- Pandas, NumPy
- Streamlit

Project Structure
-----------------
emotion_detection/
│
├── data/                  - Emotion dataset (Dair.ai Emotion)
├── models/                - Saved fine-tuned models
├── notebooks/             - EDA and model training notebooks
├── src/
│   ├── preprocess.py      - Text preprocessing functions
│   ├── train.py           - Model training and fine-tuning
│   ├── evaluate.py        - Evaluation metrics
│   ├── deploy.py          - Streamlit app
│
├── requirements.txt       - Python dependencies
├── README.txt             - Project documentation

Installation
------------
1. Clone the repository:
   git clone https://github.com/your-username/emotion-detection-llm.git
   cd emotion-detection-llm

2. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

Usage
-----
Train the model:
   python src/train.py --model distilbert --dataset data/emotion.csv

Evaluate the model:
   python src/evaluate.py --model distilbert

Run the Streamlit app:
   streamlit run src/deploy.py

Results
-------
- DistilBERT: ~89.5% Accuracy, F1-score
- BERT: ~88.9% Accuracy
- XLNet: ~85.6% Accuracy

Ethical Considerations
----------------------
This project includes bias analysis and interpretability tools to ensure responsible AI deployment. Refer to the documentation for more details.

License
-------
This project is licensed under the MIT License.
