from fastapi import FastAPI
import joblib
app = FastAPI()

@app.post("/predict/decision_tree")
async def dt_predict(data: list[float]):
    model = joblib.load("model/decision_tree.pkl")

    prediction = model.predict([data])[0]

    return {"result": prediction}

@app.post("/predict/KNN")
async def dt_predict(data: list[float]):
    model = joblib.load("model/KNN.pkl")

    prediction = model.predict([data])[0]

    return prediction
@app.get("/")
def root():
    return {"message": "hello"}