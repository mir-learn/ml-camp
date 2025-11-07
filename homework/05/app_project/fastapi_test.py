# import pickle
# from fastapi import FastAPI
# from fastapi import requests

# app = FastAPI('courses')

# input_file = '/home/mirodov/ml-camp/homework/05/app_project/pipeline_v1.bin'
# with open(input_file, 'rb') as f_in: # very important to use 'rb' here, it means read-binary
#     dict_vectorizer, model = pickle.load(f_in)



# @app.route("/predict", methods=["POST"])
# def predict(course):
#   X = dict_vectorizer.transform([course])
#   y_pred = model.predict_proba(X)[0,1]

#   return y_pred
import pickle
from fastapi import FastAPI, Request

app = FastAPI()

input_file = 'pipeline_v2.bin'
with open(input_file, 'rb') as f_in:
    dict_vectorizer, model = pickle.load(f_in)

@app.post("/predict")
async def predict(request: Request):
    course = await request.json()
    X = dict_vectorizer.transform([course])
    y_pred = model.predict_proba(X)[0, 1]

    return {"prediction": float(y_pred)}



#docker run -it --rm -p 8000:8000 courses_predict:latest