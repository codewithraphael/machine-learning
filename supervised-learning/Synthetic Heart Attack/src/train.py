from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier



def train_models(preprocessor, X_train, y_train):
    models = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced',
        random_state=42
    ),
    'Randomforest Classifier': RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MultiLayer Perceptron Classifier': MLPClassifier(
        hidden_layer_sizes=10,
        random_state=42
    ),
    'Xgboost Classifier': XGBClassifier() 
}
    
    trained_models = {}

    for name, model in models.items():
       pipe = Pipeline(
           steps=[
               ('preprocessor', preprocessor),
               ('model', model)
           ]
       )
       pipe.fit(X_train, y_train)

       trained_models[name] = pipe
    
    return name, trained_models, pipe