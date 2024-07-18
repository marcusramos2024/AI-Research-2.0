import os
import pandas as pd
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

class Protein:
    def __init__(self):
        self.labelTemp = None
        self.id = None
        self.sequence = None
        self.growthTemp = None
        self.lysate = None
        self.cell = None

def read_embedding(directory, file_name, layer):
    embedding = torch.load(os.path.join(directory, file_name))
    return embedding['mean_representations'][layer]

def prepare_proteins(dataframe):
    proteins = []
    for index, row in dataframe.iterrows():
        protein = Protein()
        protein.id = row["Protein"]
        protein.sequence = row["sequence"]
        protein.growthTemp = row["growth_temp"]
        protein.lysate = row["lysate"]
        protein.cell = row["cell"]
        protein.labelTemp = row["label_tm"]
        proteins.append(protein)
    return proteins

def train_model(model_name):
    train_df = pd.read_csv("Datasets/Training.csv")
    embeddings_directory = "Embeddings/" + model_name

    train_proteins = prepare_proteins(train_df)

    file = os.listdir(embeddings_directory)[0]
    layers = torch.load(embeddings_directory + "/" + file)
    num_layers = len(layers['mean_representations'])

    models = {}

    os.makedirs(f"Models/{model_name}" + "/2.0", exist_ok=True)
    for layer in range(1, num_layers + 1):
        print(f"Training layer: {layer}")
        xs_train = []
        ys_train = []

        for p in train_proteins:
            x = read_embedding(embeddings_directory, p.id + ".pt", layer)
            # x = np.append(x, p.growthTemp) 
            # x = np.append(x, p.lysate)
            # x = np.append(x, p.cell)
            y = p.labelTemp
            xs_train.append(x)
            ys_train.append(y)

        xs_train = np.array(xs_train)
        ys_train = np.array(ys_train)

        model = LinearRegression()
        model.fit(xs_train, ys_train)
        model_path = os.path.join(f"Models/{model_name}" + "/2.0/", f"{layer}.joblib")
        joblib.dump(model, model_path)

        models[layer] = model

    return models

def test_models(model_name, version):
    test_df = pd.read_csv("Datasets/Testing.csv")
    embeddings_directory = "Embeddings/" + model_name
    os.makedirs(embeddings_directory, exist_ok=True)

    test_proteins = prepare_proteins(test_df)

    file = os.listdir(embeddings_directory)[0]
    layers = torch.load(embeddings_directory + "/" + file)
    num_layers = len(layers['mean_representations'])

    r2_list = []
    pcc_list = []
    mae_list = []
    mse_list = []
    rmse_list = []

    for layer in range(1, num_layers + 1):
        print(f"Testing layer: {layer}")
        xs_test = []
        ys_test = []

        for p in test_proteins:
            x = read_embedding(embeddings_directory, p.id + ".pt", layer)
            # if version > 1:
            #     x = np.append(x, p.growthTemp) 
            #     x = np.append(x, p.lysate)
            #     x = np.append(x, p.cell)
            y = p.labelTemp
            xs_test.append(x)
            ys_test.append(y)

        xs_test = np.array(xs_test)
        ys_test = np.array(ys_test)
        os.makedirs(f"Models/{model_name}/{version}", exist_ok=True)
        model_path = f"Models/{model_name}/{version}/{layer}.joblib"
        model = joblib.load(model_path)

        y_pred = model.predict(xs_test)

        output_data = {
            "ID": [],
            "Prediction": [],
            "Actual": [],
            "Difference": [],

        }

        for i in range(len(y_pred)):
            prediction = y_pred[i]
            actual = ys_test[i]
            dif = abs(actual - prediction)

            output_data["ID"].append(test_proteins[i].id)
            output_data["Prediction"].append(prediction)
            output_data["Actual"].append(actual)
            output_data["Difference"].append(dif)


        df = pd.DataFrame(output_data)
        os.makedirs(f'Results/{model_name}/{version}/Overview', exist_ok=True)
        os.makedirs(f'Results/{model_name}/{version}/Layers', exist_ok=True)
        df.to_csv(f'Results/{model_name}/{version}/Layers/{layer}.csv', index=False)
        
        r2 = r2_score(ys_test, y_pred)
        pcc, _ = pearsonr(ys_test, y_pred)
        mae = mean_absolute_error(ys_test, y_pred)
        mse = mean_squared_error(ys_test, y_pred)
        rmse = np.sqrt(mse)

        r2_list.append(r2)
        pcc_list.append(pcc)
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)

    data = {
        "Layer": list(range(1, len(mse_list) + 1)),
        "R2" : r2_list,
        "PCC" : pcc_list,
        "MAE" : mae_list,
        "MSE" : mse_list,
        "RMSE" : rmse_list,
    }

    df = pd.DataFrame(data)
    os.makedirs(f'Results/{model_name}/{version}/Layers', exist_ok=True)
    df.to_csv(f'Results/{model_name}/{version}/Layers.csv', index=False)


# train_model("esm2_t33_650M_UR50D")
test_models("esm2_t33_650M_UR50D", 2.0)