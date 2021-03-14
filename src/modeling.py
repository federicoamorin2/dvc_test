import pickle
from pathlib import Path

import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import yaml

PARAMS = yaml.safe_load(open("params.yaml"))
app = typer.Typer()


@app.command()
def split_data(raw_db: Path) -> None:
    df = pd.read_csv(raw_db)
    params = PARAMS["split"]
    X, y = df.drop("target", axis=1), df["target"]
    train_df, test_df, y_train, y_test = train_test_split(
        X, y, random_state=305, test_size=params["test_size"]
    )
    train_df["target"] = y_train
    test_df["target"] = y_test
    train_df.to_csv("data/training.csv")
    test_df.to_csv("data/test.csv")


@app.command()
def find_hyper_params(train_df_path, hyper_param_space) -> None:
    pass


@app.command()
def make_model(train_df_path: Path, model_params_path: Path, output_path: Path) -> None:
    """Recieves dataframe and model parameters, then trains model and persists
    it.

    Args:
        train_df_path (Path): Location of train dataframe.
        model_params_path (Path): Location of post hp-tunning parameters file.
        output_path (Path): Path to place final model in.
    """
    # Read train dataset and exctract features and target.
    df = pd.read_csv(train_df_path)
    x_train, y_train = df.drop("target", axis=1), df["target"]

    # Load best parameters of hp-tunning.
    pass

    # Instanciate model with best hyperparams found.
    model = LogisticRegression(multi_class="ovr")
    model = model.fit(x_train, y_train)

    # Persist model...
    with open(output_path, "wb") as fd:
        pickle.dump(model, fd, pickle.HIGHEST_PROTOCOL)


@app.command()
def predict(model_path: Path, test_df_path: Path, predict_path: Path) -> None:
    """Recieves trained model and test dataset and persists predictions.

    Args:
        model_path (Path): Path to where trained model is.
        test_df_path (Path): Location of test dataframe.
        predict_path (Path): Path to store predictions.
    """
    # Read test dataset.
    x_test = pd.read_csv(test_df_path).drop("target", axis=1)

    # Read trained model and predict
    with open(model_path, "rb") as fd:
        model = pickle.load(fd)
    preds = model.predict_proba(x_test)
    print(preds)
    # Persist predictions...
    with open(predict_path, "wb") as fd:
        pickle.dump(preds, fd, pickle.HIGHEST_PROTOCOL)


@app.command()
def evaluate(predict_file: Path, test_df_path: Path) -> None:
    """Compute and persist model metrics.

    Args:
        predict_file (Path): File where predicts are stored.
        test_df_path (Path): Location of test dataframe.
        thresh (float): Classification threshold to use in experiment.
    """

    params = PARAMS["evaluate"]
    # Load predicted and true test values.
    with open(predict_file, "rb") as fd:
        predict = pickle.load(fd)
    y_test = pd.read_csv(test_df_path).loc[:, "target"]
    y_test_dummies = pd.get_dummies(y_test)

    # Set threshhold and manipulate prediction array to leave it in the right
    # format.
    binary_preds = (predict > params["thresh"]).astype(int)
    final_preds = (
        binary_preds[:, 0] * 0 + binary_preds[:, 1] * 1 + binary_preds[:, 2] * 2
    )

    # Compute metrics and persist
    print(confusion_matrix(y_test, final_preds))
    print(classification_report(y_test_dummies, binary_preds))


if __name__ == "__main__":
    app()
