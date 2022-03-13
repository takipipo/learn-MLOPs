import pickle
import numpy as np
from fastapi import FastAPI
from app.data_format import Wine, BatchWine
from typing import List

app = FastAPI(title="Predicting Wine Class")


@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    with open("./app/wine.pkl", "rb") as model_file:
        global model
        model = pickle.load(model_file)


@app.post("/predict_single")
def predict_single(wine: Wine):
    data_point = np.array(
        [
            [
                wine.alcohol,
                wine.malic_acid,
                wine.ash,
                wine.alcalinity_of_ash,
                wine.magnesium,
                wine.total_phenols,
                wine.flavanoids,
                wine.nonflavanoid_phenols,
                wine.proanthocyanins,
                wine.color_intensity,
                wine.hue,
                wine.od280_od315_of_diluted_wines,
                wine.proline,
            ]
        ]
    )
    pred = model.predict(data_point).tolist()[0]
    print(pred)
    return {"Prediction": pred}

@app.post("/predict_batch")
def predict_batch(batch_wine: BatchWine):
    batch_wine = batch_wine.batch_of_wine
    batch_wine = np.array(batch_wine)
    batch_pred = model.predict(batch_wine).tolist()

    return {"Prediction": batch_pred}

    


