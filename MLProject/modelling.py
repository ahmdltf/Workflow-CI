import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

### Load dataset
df = pd.read_csv(
    'MLProject/dataset_preprocessing.csv'
)

### Split feature dan target
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']

### Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

### Enable autolog
mlflow.sklearn.autolog()

### Start run
with mlflow.start_run():

    ### Training model
    model = LinearRegression()
    model.fit(X_train, y_train)

    ### Prediction
    y_pred = model.predict(X_test)

    ### Evaluation
    mse = mean_squared_error(y_test, y_pred)

    print('MSE:', mse)
