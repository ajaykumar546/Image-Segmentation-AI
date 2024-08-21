import streamlit as st
import cv2
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

# Load the necessary files and models
mapped_data_file = '/data/mapped_data.json'
annotated_image_path = '/data/annotated_image.jpg'
summary_csv = '/data/object_summary_table.csv'

# Load mapped data
with open(mapped_data_file, 'r') as f:
    data_mapping = json.load(f)

# Streamlit app layout
st.title("Image Segmentation and Object Summary")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load uploaded image with OpenCV
    file_bytes = uploaded_file.read()
    original_image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Step 2: Segmentation Display
    st.header("Original Image with Segmented Objects")
    
    # Show annotated image
    st.image(annotated_image_path, caption="Annotated Image", use_column_width=True)

    # Step 3: Object Details
    st.header("Object Details")
    
    # Display object details
    for obj in data_mapping['objects']:
        st.subheader(f"Object ID: {obj['object_id']}")
        st.image(obj['object_file'], caption=f"Object ID: {obj['object_id']}")
        st.text(f"Bounding Box: {obj['bounding_box']}")
        st.text(f"Extracted Text: {obj['extracted_text']}")
        st.text(f"Summary: {obj['summary']}")

    # Step 4: Final Output
    st.header("Final Output: Annotated Image and Summary Table")
    
    # Show final annotated image
    st.image(annotated_image_path, caption="Final Annotated Image", use_column_width=True)
    
    # Show summary table as a pandas DataFrame
    summary_df = pd.read_csv(summary_csv)
    st.dataframe(summary_df)