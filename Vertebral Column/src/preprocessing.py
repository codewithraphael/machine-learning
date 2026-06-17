
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess(data):
    '''
    Data Splitting and Feature Scaling
    
    '''

    X = data.drop(['is_abnormal'], axis=1)
    y = data['is_abnormal']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

