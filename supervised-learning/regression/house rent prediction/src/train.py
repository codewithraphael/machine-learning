from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from xgboost import XGBRegressor

from config import TARGET_COL, RANDOM_STATE

def split_features(data):
    x = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)

    return x_train, x_test, y_train, y_test



def train_models(x_train, y_train, preprocessor):

    models = {
        'LINEAR REGRESSION': LinearRegression(),
        'RIDGE REGRESSION': Ridge(random_state = RANDOM_STATE),
        'LASSO REGRESSION': Lasso(random_state = RANDOM_STATE),
        'ELSATIC NET REGRESSION': ElasticNet(random_state = RANDOM_STATE),
        'RANDOM FOREST REGRESSOR': RandomForestRegressor(random_state = RANDOM_STATE),
        'GRADIENT BOOSTING REGRESSOR': GradientBoostingRegressor(random_state = RANDOM_STATE),
        'DECISION TREE REGRESSOR': DecisionTreeRegressor(random_state = RANDOM_STATE),
        'EXTRA TREE REGRESSOR': ExtraTreeRegressor(random_state = RANDOM_STATE),
        'KNEIGHORS REGRESSOR': KNeighborsRegressor(n_neighbors=5),
        'SUPPORT VECTOR REGRESSOR': SVR(),
        'MULTILAYER PERCEPTRON': MLPRegressor(random_state = RANDOM_STATE),
        'XGBOOST REGRESSOR': XGBRegressor(random_state = RANDOM_STATE)

    }

    trained_models = {}

    for name, model in models.items():
        pipe = Pipeline(
            [
                (
                    'preprocessor',
                    preprocessor
                ),
                (
                    'model',
                    model
                )
            ]
        )

        pipe.fit(x_train, y_train)
        trained_models[name] = pipe

    return trained_models
