# Plastic-Detection-in-River

Plastic pollution in marine environnement is a global threat. It threatens marine species health, human health, food security, costal tourism. More than 350 million tons of plastic are produced every year and it is estimated that more than 15 million tons end up in the world Ocean. The plastic then degrades over time into micro-plastic and both macro and microplastic have serious environmental impacts. Rivers are the very first source of plastic in Oceans. It is estimated that more than 80% of river plastic comes from only 1000 rivers. Organizations like the The Ocean Cleanup are investing resources to address the ocean plastic pollution at the root cause by cleaning rivers.

Inorder to tackle the above problem, this is a streamlit app that detects whether there is plastic in river or not using Deep Learning techniques like Object Detection and state-of-the-art Object Detection Models like "You only look once (YOLO)".

This app uses the YOLOv8m Pre-trained Object Detection model to train on the custom dataset. This dataset contains photos of rivers on which there may be waste. The waste items are annotated through bounding boxes, and are assigned to one of the 4 following categories: Plastic Bags, Plastic Bottles, Other Plastic Waste and No Plastic waste. Note that some photos may not contain any waste.

## Dataset

The dataset is included in the file ```convert_to_yolo.py``` which downloads the datasets and converts them into YOLO usable format. Alternatively, you can also download the dataset from [here](https://huggingface.co/datasets/Kili/plastic_in_river). The dataset is split into ***```train```*** and ***```validation```*** with ***3407*** and ***425*** images respectively

The downloaded dataset after conversion to YOLO Format is in the following structure:
```
datasets
├─ images
│  ├─ train
│  │  ├─ 0.png
│  │  ├─ 1.png
│  │  └─ ...
│  └─ validation
│     ├─ 0.png
│     ├─ 1.png
│     └─ ...
└─ labels
   ├─ train
   │  ├─ 0.txt
   │  ├─ 1.txt
   │  └─ ...
   └─ validation
      ├─ 0.txt
      ├─ 1.txt
      └─ ...
```

The text file for labels contains the data in the following format:
```
0 x_n y_n w_n h_n
1 x_n y_n w_n h_n
...
```

## Preview
![screenshot](/Screenshots/Example-screenshot.png)
