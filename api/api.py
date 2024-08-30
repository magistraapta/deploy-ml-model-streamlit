from fastapi import FastAPI
import joblib
app = FastAPI()

@app.post("/predict/decision-tree")
async def dt_predict(data: list[float]):
    model = joblib.load("model/decision_tree.pkl")

    prediction = model.predict([data])[0]

    return prediction