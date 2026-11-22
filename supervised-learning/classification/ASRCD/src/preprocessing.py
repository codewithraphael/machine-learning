from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from config import (
    RANDOM_STATE,
    MAX_FEATURES,
    MAX_ITER
)

def build_classifier():

    '''
    build a TF-IDF and logistic regression pipeline

    TF-IDF:
        text feature extraction (converts text into numerical features)

    LogisticRegression:
        learn patterns and predicts the target class

    Pipeline:
        combine preprocessing and model training

    Returns:
        pipeline  

    '''

    pipe = Pipeline([
        (
            'tfidf',
            TfidfVectorizer(
                lowercase=True, 
                strip_accents='unicode', 
                ngram_range=(1, 2), 
                min_df=1, 
                max_features=MAX_FEATURES,
                sublinear_tf=True
            )
        ),

        (
            'classifier',
            LogisticRegression(
                max_iter=MAX_ITER,
                class_weight='balanced',
                random_state=RANDOM_STATE
            )
        )
    ])

    return pipe