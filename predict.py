import random
from PIL import Image
from datasets import load_dataset
from ultralytics import YOLO

# Loading the dataset
dataset = load_dataset("Kili/plastic_in_river")

# Testing the model against a random image from test dataset
rand_img = random.randint(0, 100)
img = dataset["test"][rand_img]["image"]

# Loading the best model
model = YOLO("runs/detect/train/weights/best.pt")

res = model.predict(img)

# Plotting the bboxes on the image
res = res[0].plot(line_width=1)
# Converting from BGR to RGB
res = res[:, :, ::-1]

# Saving the image in png
res = Image.fromarray(res)
res.save("output.png")
