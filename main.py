
# 1. Library imports
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# 2. Create the app object
app = FastAPI()

MODEL = tf.keras.models.load_model("potatoes.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# 4. Expose the prediction functionality, make a prediction from the passed
#    image data and return the predicted with the confidence
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)

#uvicorn main:app --reload
