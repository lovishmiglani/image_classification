import streamlit as st
from streamlit_option_menu import option_menu
import base64
import json 
import requests 
import streamlit as st 
from streamlit_lottie import st_lottie
import io
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras.applications import VGG16
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Add background image
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_base64_of_bin_file("/Users/lovishmiglani/Desktop/img_cls/bg5.jpg")
page_bg_img = f'''
<style>
[data-testid="stAppViewContainer"] {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
}}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

side_bar_img = get_base64_of_bin_file("/Users/lovishmiglani/Desktop/img_cls/bg.jpg")
side_bar_bg_img = f'''
<style>
[data-testid="stSidebarContent"] {{
background-image: url("data:image/png;base64,{side_bar_img}");
background-size: cover;
}}
</style>
'''
st.markdown(side_bar_bg_img, unsafe_allow_html=True)

# Content text contrast 
text_color = '''
<style>
.stApp {
color: white;
}
</style>
'''
st.markdown(text_color, unsafe_allow_html=True)

# App content padding
padding = 0
st.markdown(f'''<style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem; 
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
        font-size: 20px;

    }} </style>''', unsafe_allow_html=True)
# now lest add some Custom CSS to make sidebar menu and improve styling 
st.markdown('<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>', unsafe_allow_html=True)

padding = 100
st.markdown(f'''<style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style>''', unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    choose = option_menu("Main Menu", ["Home", "Image Classification", "About Project"], 
        icons=['house', 'image', 'book'], menu_icon="cast", default_index=0)
# Center page title
# Center and style page title
page_title_style = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@900&display=swap');
        [data-testid="StyledLinkIconContainer"] {{
        color: #185664cf;
        font-family: 'Kanit', sans-serif;
        line-height: 1em;
        font-size: 58px;
        font-weight: bold;
        text-align: center;
        text-shadow: 0 1px 0  rgb(196, 180, 84),
                    0 2px 0 rgb(196, 180, 84),  
                    0 3px 0 rgb(196, 180, 84),
                    0 4px 0 rgb(196, 180, 84),
                    0 5px 0 rgb(214,142,142),
                    0 6px 1px rgba(0,0,0,.1),
                    0 0 5px rgba(0,0,0,.1),
                    0 1px 3px rgba(0,0,0,.3),
                    0 3px 5px rgba(0,0,0,.2),
                    0 5px 10px rgba(0,0,0,.25),
                    0 10px 10px rgba(0,0,0,.2), 
                     0 20px 20px rgba(0,0,0,.15);
        }}
    </style>
"""
st.markdown(page_title_style, unsafe_allow_html=True)
st.title("Image Classification Web App")

page_home_style = f"""
    <style>
        [data-testid="stMarkdownContainer"] {{
        color: #ffe495;
        font-family: 'Georgia', sans-serif;
        text-align: center;
        }}
    </style>
"""
st.markdown(page_home_style, unsafe_allow_html=True)

header_style = f"""
    <style>
        [data-testid="stHeader"] {{
        position: fixed;
        top: 0px;
        left: 0px;
        right: 0px;
        height: 0.875rem;
        background: rgb(0 0 0);
        /* outline: none; */
        z-index: 999990;
        display: block;
        }}
    </style>
"""
st.markdown(header_style, unsafe_allow_html=True)
# Home page content
if choose == "Home":
    st.write("Welcome to the image classification web app with VGG16 by ImageNet dataset!") 
    file_ = open("/Users/lovishmiglani/Desktop/img_cls/gif1.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )


# Image classification page  
elif choose == "Image Classification":
    st.header("Classify Images")
    upload_file = st.file_uploader("Choose an image...", type=["jpg", "png", "webl"])
    uploader_style = """
    <style>
    div[data-testid="stFileUploader"] {
        background-image: linear-gradient(to bottom, #185664cf, #90dbb4); 
        border-radius: 5px;
        box-shadow: 2px 2px 4px #888; 
        padding: 8px;
    }
    div[data-baseweb="file-uploader"] {
        align-items: center;
        justify-content: center;   
    } 
    label[data-baseweb="file-uploader-text"] {
        margin: 0 !important;
    }
    </style>
    """
    st.markdown(uploader_style, unsafe_allow_html=True)
    def read_image (uploaded_file):
    # Convert the file to an OpenCv compatible format
        file_bytes = np.asarray(bytearray(uploaded_file.getbuffer()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        return image

    def classify_image(image_path):
        st.write("[INFO] loading and preprocessing image....")
        image = image_utils.load_img(image_path, target_size=(224, 224))
        image = image_utils.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        st.write("[INFO] loading network...")
        model = VGG16(weights="imagenet")
        url = "https://lottie.host/5bc15346-ea85-4e0e-bb0b-a591d4f1fdd6/U60wGTVp4k.json"  # Correct Lottie URL

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            lottie_json = response.json()
            st_lottie(lottie_json, width=650, height=300)
            resize_animation = """<style> 
            div[data-testid="lottie"] {
                width: 650px !important;
                height: 300px !important;
            }
            </style>"""
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Lottie animation: {e}")
        response = requests.get(url)
        print(response.content)
        st.write(" [INFO] classifying image")
        preds = model.predict(image)
        P = decode_predictions(preds)
        st.write("Top predictins:")
        for i, (imagenetID, label, prob) in enumerate(P[0]):
            st.write(f"{i+1}. {label} - {prob*100:.2f}%")
            if i == 2:
                break
        orig = read_image(image_path)
        orig= cv2.resize(orig, (224, 224))
        (imagenetID, Label, prob) = P[0][0]
        cv2.putText(orig, "Label: {}: {:.2f}%".format(Label, prob * 100),
                 (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 250, 0), 1)
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        st.image(orig_rgb, caption = "Classification", use_column_width = True)
        #st.image(orig_rgb, caption = "Classification", use_column_width = True)
    if upload_file is not None:
        classify_image(upload_file)
elif choose == "About Project":
        st.write("Image classification is the process of categorizing and labeling groups of pixels or vectors within an image based on specific rules. The categorization law can be devised using one or more spectral or textural characteristics. Two general methods of classification are 'supervised' and 'unsupervised'!") 
        st.write("Unsupervised classification method is a fully automated process without the use of training data. Using a suitable algorithm, the specified characteristics of an image is detected systematically during the image processing stage. The classification methods used in here are ‘image clustering’ or ‘pattern recognition’. Two frequent algorithms used are called ‘ISODATA’ and ‘K-mean’.Supervised classification method is the process of visually selecting samples (training data) within the image and assigning them to pre-selected categories (i.e., roads, buildings, water body, vegetation, etc.) in order to create statistical measures to be applied to the entire image. ‘maximum likelihood’ and ‘minimum distance’ are two common methods to categorize the entire image using the training data. For example, ‘maximum likelihood’ classification uses the statistical characteristics of the data where the mean and standard deviation values of each spectral and textural indices of the image are computed first. Then, considering a normal distribution for the pixels in each class and using some classical statistics and probabilistic relationships, the likelihood of each pixel to belong to individual classes is computed. Finally, the pixels are labeled to a class of features that show the highest likelihood.")
        file_ = open("/Users/lovishmiglani/Desktop/img_cls/gif.gif", "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()

        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )
# Footer    
st.markdown('''<script>function setPadding() {
    var el = document.getElementsByTagName("section")[0];
    var style = window.getComputedStyle(el);
    var paddingTop = style.getPropertyValue('padding-top');
    document.querySelectorAll(".element").forEach(el => {
        el.style.paddingTop = paddingTop;
    })
}
setPadding();
</script>''', unsafe_allow_html=True)
