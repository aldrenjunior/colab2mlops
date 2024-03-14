# from typing import Union
import sys
import os
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
import pandas as pd
import joblib
import wandb

from source.api.pipeline import FeatureSelector, CategoricalTransformer, NumericalTransformer

# global variables (module)
setattr(sys.modules["__main__"], "FeatureSelector", FeatureSelector)
setattr(sys.modules["__main__"], "CategoricalTransformer", CategoricalTransformer)
setattr(sys.modules["__main__"], "NumericalTransformer", NumericalTransformer)

# Acessa a chave da API do Weights & Biases (wandb) da variável de ambiente
wandb_api_key = os.environ.get('WANDB_KEY_ID')

# Inicializa o Weights & Biases (wandb) com a chave da API
wandb.login(key=wandb_api_key)

# name of the model artifact
artifact_model_name = "decision_tree/model_export:latest"

# initiate the wandb project
run = wandb.init(project="decision_tree", job_type="api")

# create the api
app = FastAPI()


# declare request example data using pydantic a person in our dataset has the following attributes
class Person(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 72,
                "workclass": 'Self-emp-inc',
                "fnlwgt": 473748,
                "education": 'Some-college',
                "education_num": 10,
                "marital_status": 'Married-civ-spouse',
                "occupation": 'Exec-managerial',
                "relationship": 'Husband',
                "race": 'White',
                "sex": 'Male',
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 25,
                "native_country": 'United-States'
            }
        }


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <p><span style="font-size:28px"><strong>Api</strong></span></p>"""\
    """<p><span style="font-size:20px">In this project, we will apply the skills """\
        """acquired in the Deploying a Scalable ML Pipeline in Production course to develop """\
        """a classification model on publicly available"""\
        """<a href="http://archive.ics.uci.edu/ml/datasets/Adult"> Census Bureau data</a>.</span></p>"""\
        """<p> Types to /docs for acess documentation </p>"""


# run the model inference and use a Person data structure via POST to the API.
@app.post("/predict")
async def get_inference(person: Person):
    # Download inference artifact
    model_export_path = run.use_artifact(artifact_model_name).file()
    pipe = joblib.load(model_export_path)

    # .dict() -> .model_dump()
    df = pd.DataFrame([person.model_dump()])

    # Predict test data
    predict = pipe.predict(df)

    return "low income <=50K" if predict[0] <= 0.5 else "high income >50K"
