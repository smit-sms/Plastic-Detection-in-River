import os
import shutil
import time
from collections import Counter
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(layout="wide", page_title="Plastic in River")
st.write("# Detect whether there is plastic in river or not")

@st.cache_data
def load_model():
    '''
    Method used to Load the YOLO model
    '''
    # Loading the best model
    model_path = "runs/detect/train/weights/best.pt"
    return YOLO(model_path)


def get_predictions(model, image) -> Image:
    '''
    Method used to get predictions from the model for the image passed
    '''
    res = model.predict(image, save_txt=True)
    # Plotting the bboxes on the image
    res = res[0].plot(line_width=1)
    # Converting from BGR to RGB
    res = res[:, :, ::-1]

    # Converting the image in png
    res = Image.fromarray(res)
    return res

def get_pred_labels() -> dict:
    '''
    Method to get the predicted labels from text file
    '''
    LABELS = {
        0: 'PLASTIC_BAG', 
        1: 'PLASTIC_BOTTLE', 
        2: 'OTHER_PLASTIC_WASTE', 
        3: 'NOT_PLASTIC_WASTE'
    }
    results = []
    # Reading the predicted labels.txt file
    results_file_path = "runs/detect/predict/labels.txt"
    with open(results_file_path, 'r') as f:
        lines = f.readlines()

    results.extend(LABELS[int(line[0])] for line in lines)
    count_labels = dict(Counter(results))
    # Removing the labels.txt file
    os.remove(results_file_path)

    return count_labels

with st.sidebar:
    st.title("Plastic in River")
    st.sidebar.write("Try uploading an image to predict whether there are any plastics \
         (bags/bottles/other plastic items) in the image or not.")
    with st.form("my_form"):
        if 'model' not in st.session_state:
            with st.spinner('Loading the model, please wait...'):
                # Loading and saving the model to session state
                model = load_model()
                st.session_state['model'] = model
                success_msg = st.success('Successfully loaded the model!')

            time.sleep(2)
            success_msg.empty()

        uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
        submitted = st.form_submit_button("Predict")

if not submitted or not uploaded_image:
    # Stopping the execution if no image is uploaded
    st.stop()
else:
    try:
        # Converting the streamlit image to PIL Image format
        image = Image.open(uploaded_image)
        predicted_image = get_predictions(st.session_state['model'], image)
        predicted_labels = get_pred_labels()
        tab = '&ensp;'
        predictions = f',{tab}'.join(f'{key} : {str(value)}' for key,value in predicted_labels.items())
        st.info(f"PREDICTIONS{tab} → {tab} {predictions}")
        st.image(predicted_image)
    except FileNotFoundError as e:
        del_dir = 'runs/detect/'
        for fname in os.listdir(del_dir):
            if fname.startswith("predict"):
                shutil.rmtree(os.path.join(del_dir, fname))
        st.warning("Something went wrong. Please reload the app.")
