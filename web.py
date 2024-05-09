import numpy as np
import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('last.pt')


# Define a function to process the image
def process_image(input_image):
    # Perform prediction
    np_image = np.array(input_image)
    results = model.predict(source=np_image, show=False, save=True, show_labels=True, show_conf=True, conf=0.3,
                            save_txt=False, save_crop=False, line_width=2)

    # Extract the annotated image from the result
    annotated_image = results

    # Return the annotated image
    return annotated_image


# Streamlit UI
st.title("Spruce Infected Tree Detection")
st.subheader("A Project by Christo")
st.write("Timely detection of newly infested trees is important for minimizing economic losses"
         "and effectively planning forest management activities to stop or at least slow outbreaks"
         "in Sweden forests")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Process and display the image
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    processed_image = process_image(image)
    st.image(processed_image, caption='Processed Image', use_column_width=True)  # Display processed image
else:
    st.write("Predicted image not found.")


# Define the base folder path
base_path = "runs/segment/"

# Find all subdirectories (predicted folders) in the base pathy
subdirs = [subdir for subdir in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, subdir))]
# Sort the subdirectories by converting them to integers first
subdirs = sorted(subdirs, key=lambda x: int(x.split("predict")[-1]) if x.startswith("predict") and x.split("predict")[-1].isdigit() else -1)

# If there are subdirectories (predicted folders), get the latest image path
if subdirs:
    latest_subdir = subdirs[-1]
    latest_image_path = os.path.join(base_path, latest_subdir, "image0.jpg")

    # Check if the image file exists
    if os.path.exists(latest_image_path):
        # Display the image
        st.image(latest_image_path, caption='Predicted Image', use_column_width=True)
    else:
        st.write("Predicted image not found.")
else:
    st.write("No predicted folders found.")
