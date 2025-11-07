# def main():
#     print("Hello from app-project!")


# if __name__ == "__main__":
#     main()


import pickle
from fastapi import FastAPI

app = FastAPI()

input_file = '/home/mirodov/ml-camp/homework/05/app_project/pipeline_v1.bin'
with open(input_file, 'rb') as f_in: # very important to use 'rb' here, it means read-binary
    dict_vectorizer, model = pickle.load(f_in)


input = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X = dict_vectorizer.transform([input])
y_pred = model.predict_proba(X)[0,1]

print(f"Predicted probability: {y_pred:.3f}")