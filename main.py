import io

import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File

from code import predict

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to crop-disease-prediction"}


@app.post("/predict")
async def upload_image_to_predict(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        uploaded_image = Image.open(io.BytesIO(contents))
        prediction = predict(uploaded_image)

    except Exception as e:
        print(e)
        return {"message": "There was an error uploading the file"}

    finally:
        file.file.close()

    return prediction


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)
