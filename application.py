import streamlit as st
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the path to your service account credentials JSON file
credentials_path = 'path/to/your/credentials.json'

# Create a Vision client and authenticate with the credentials
client = vision.ImageAnnotatorClient.from_service_account_json(credentials_path)

def main():
    st.header("XELP OCR - Document AI")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        content = uploaded_file.read()
        image = types.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations

        if texts:
            st.subheader("Extracted Text:")
            for text in texts:
                st.write(text.description)

            # You can also display the bounding boxes of the detected text
            # if text.bounding_poly:
            #     st.image(image)
            #     st.write("Bounding box vertices:")
            #     for vertex in text.bounding_poly.vertices:
            #         st.write(f"- ({vertex.x},{vertex.y})")
        else:
            st.write("No text detected in the image.")

if __name__ == '__main__':
    main()
