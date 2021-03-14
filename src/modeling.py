import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/indice_de_emprego.csv")

X, y = df["preco"], df["fruta"]
x_train, x_test, y_train, y_test = train_test_split(df, random_state=305)
model = LogisticRegression()
model.fit(x_train, y_train)
model.predict(x_test)