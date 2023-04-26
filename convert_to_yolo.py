import os
import logging
from datasets import load_dataset
logging.basicConfig(level=logging.INFO)
"""
datasets/
    - images
        - train
            - 0.png
            - 1.png
            ...
        - validation
    - labels
        - train
            - 0.txt
            - 1.txt
            ...
        - validation

    0.txt
    0 x_n y_n w_n h_n
    1 x_n y_n w_n h_n
    ...
"""
def dump_images_and_labels(data, split: str) -> None:
    '''
    Function to dump the images and labels into the
    respective folders for training and validation
    '''
    data = data[split]
    logging.info(f"Dumping images and labels for {split}")
    for i, example in enumerate(data):
        image = example["image"]
        labels = example["litter"]["label"]
        bboxes = example["litter"]["bbox"]

        targets = []
        for label, box in zip(labels, bboxes):
            targets.append(f"{label} {box[0]} {box[1]} {box[2]} {box[3]}")

        with open(f"datasets/labels/{split}/{i}.txt", "w") as f:
            for target in targets:
                f.write(target + "\n")

        # Saving the image to respective folder
        image.save(f"datasets/images/{split}/{i}.png")

if __name__ == "__main__":
    # loading the dataset from huggingface
    dataset = load_dataset("Kili/plastic_in_river")
    logging.info("Completed loading the dataset.")

    # Making respective dirs
    os.makedirs("datasets/images/train", exist_ok=True)
    os.makedirs("datasets/images/validation", exist_ok=True)

    os.makedirs("datasets/labels/train", exist_ok=True)
    os.makedirs("datasets/labels/validation", exist_ok=True)

    dump_images_and_labels(dataset, "train")
    dump_images_and_labels(dataset, "validation")

    logging.info("Completed segregation of images into train and validation.")
