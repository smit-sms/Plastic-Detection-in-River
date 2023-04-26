from PIL import Image
from datasets import load_dataset
from ultralytics import YOLO

# Loading the dataset
# dataset = load_dataset("Kili/plastic_in_river")

# img = dataset["test"][0]["image"]

img = './datasets/images/validation/10.png'

# Loading the best model
model = YOLO("runs/detect/train/weights/best.pt")

res = model.predict(img)
print(res)

# Plotting the bboxes on the image
res = res[0].plot(line_width=1)
# Converting from BGR to RGB
res = res[:, :, ::-1]

# Saving the image in png
res = Image.fromarray(res)
res.save("output.png")
