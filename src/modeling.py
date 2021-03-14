import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/indice_de_emprego.csv")

X, y = df["preco"], df["fruta"]
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=305)
model = LogisticRegression()
model.fit(x_train.values.reshape(-1, 1), y_train)
preds = model.predict(x_test.values.reshape(-1, 1))

print(
    confusion_matrix(
        y_test,
        preds,
    )
)
