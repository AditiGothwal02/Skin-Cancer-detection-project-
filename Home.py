# Import section
import sklearn
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import time
from joblib import load
import cv2
import matplotlib.pyplot as plt


# load the model
def load_model_cnn():
    model = load_model('best_model.keras')
    return model

# main page
def main():
    st.set_page_config(page_title="Skin Cancer Detector", layout="wide")
    pages = {
        "Home": home_page,
        "Detection Form": detection_form_page,
        "FAQs": FAQ_pages
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()


def home_page():
    # Open an image
    image_1 = Image.open("Image1.jpg")
    image1 = image_1.resize((200, 200))

    image_2 = Image.open("Image2.jpg")
    image2 = image_2.resize((200, 200))

    image_6 = Image.open("Image6.jpeg")
    image6 = image_6.resize((200, 200))

    image_3 = Image.open("Image3.jpeg")
    image3 = image_3.resize((200, 200))
    st.title("Welcome to Skin Cancer Detector")



    # Custom CSS for Styling
    st.markdown(
        """
        <style>
        .main-container {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2);
        }
        .title {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            color: white;
        }
        .subtitle {
            text-align: center;
            font-size: 1.5rem;
            color: white;
        }
        .section {
            margin-top: 2rem;
            background: rgba(255, 255, 255, 0.8);
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main Container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Title and Subtitle
    st.markdown('<div class="title">Welcome to Skin Cancer Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Empowering early detection with AI-driven insights</div>',
                unsafe_allow_html=True)


    # Animated Text Effect
    progress_text = "Analyzing skin health..."
    progress_bar = st.progress(0)
    for percent in range(100):
        time.sleep(0.02)
        progress_bar.progress(percent + 1)
    st.success("System Ready! You may begin diagnosis")

    # About Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("🔬 About Our Technology")
    st.write(
        "Our deep learning model analyzes skin lesions and provides an accurate risk assessment for potential skin cancer cases. "
        "Upload an image and get an AI-powered analysis instantly."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Features Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("🚀 Key Features")
    st.write("✔️ AI-based Skin Cancer Detection")
    st.write("✔️ Instant Analysis and Risk Assessment")
    st.write("✔️ Secure and Confidential Diagnosis")
    st.write("✔️ Easy-to-use Web Interface")
    st.markdown('</div>', unsafe_allow_html=True)

    # Call to Action
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("📷 Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image of your skin lesion", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    
    col1, col2, col3, col4 = st.columns(4)
    #st.image("Image1.jpg", caption="Skin Cancer", width=340)
    #st.image("Image2.jpg", caption="Skin Cancer", width=340)

    with col1:
        st.image(image1, caption="Skin Cancer", use_container_width=True, width=30)

    with col2:
        st.image(image2, caption="Skin Cancer",use_container_width=True, width=30)

    with col3:
        st.image(image6, caption="Skin Cancer", use_container_width=True, width=30)
    with col4:
        st.image(image3, caption="Skin Cancer ",use_container_width=True, width=30,)

    st.header("Facts you Should know about skin cancer!")
    st.write("""
    - Skin cancer is the most common cancer in the United States.
    - One in five Americans will develop skin cancer by the age of 70.
    - More than two people die of skin cancer in the U.S. every hour.
    - Having 5 or more sunburns doubles your risk for melanoma.
    """)
    st.image("Image4.jpeg", caption="Skin Cancer Awareness", width=500)
    st.image("Image5.jpeg", caption="Skin Cancer Awareness", width=500)

    st.header("Cases, Impacts, and Cures")
    st.write("""
    - Melanoma accounts for only about 1% of skin cancers but causes a large majority of skin cancer deaths.
    - There are three major types of skin cancer: basal cell carcinoma, squamous cell carcinoma, and melanoma.
    - Early detection and treatment are crucial for the best outcomes.
    - Treatments include surgery, radiation therapy, chemotherapy, photo dynamic therapy, and biologic therapy.
    """)

    if st.button("Detect Your Skin Cancer"):
        st.session_state.page = "Detection Form"
        detection_form_page()


def to_save_file(name, gender, age, skin_type, skin_color, skin_diseases, email):
    with open("Data.txt","a") as file:
        file.write(f"name: {name}, gender: {gender}, age: {age} \n skin type: {skin_type}, skin color: {skin_color}"
                   f"\n previous skin disease : {skin_diseases} \n email:{email} ")






def load_and_prepare_image(uploaded_image):
    """
    Load and preprocess the image to match the input format for the model.
    """
    # Read image using OpenCV
    global image
    if uploaded_image is not None:
        # Reset the file pointer to the beginning in case it's been read before
        uploaded_image.seek(0)

        # Read the image bytes and convert to numpy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)

        # Decode the image as a color image
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image you uploaded is  not found.")

    # Resize to 28x28
    img = cv2.resize(image, (28, 28))

    # Normalize using same method as training data
    img = (img - np.mean(img)) / np.std(img)

    # Reshape to match input shape (1, 28, 28, 3)
    img = img.reshape(1, 28, 28, 3)

    return img


# Class names from the dataset
class_names = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
def class_of(var):
    st.subheader("🔍 Diagnosis Result")
    st.success(f"Hello {st.session_state.name}, based on our analysis, the predicted skin condition is:")

    if var == 'nv':
        st.markdown("### 🧪: **Your skin cancer type is Melanocytic nevi.**")
    elif var == 'mel':
        st.markdown(" ###🧪: **Your skin cancer type is Melanoma**")
    elif var == 'bkl':
        st.markdown("###🧪: **Your skin cancer type is Benign keratosis-like lesions**")
    elif var == 'bcc':
        st.markdown("###🧪 : Your skin cancer type is Basal cell carcinoma**")
    elif var == 'akiec':
        st.markdown(" ###🧪 : Your skin cancer type is Actinic keratoses and intraepithelial carcinoma (Bowen's disease)**")
    elif var == 'vasc':
        st.markdown("### 🧪: **Your skin cancer type is Vascular lesions**")
    elif var == 'df':
        st.markdown("###🧪 :**Your skin cancer type is Dermatofibroma**")
    else:
        st.markdown("###🧪 :**You don't have a skin cancer! Relax and chill, just chill chill just chill** ")





def predict_image_class(uploaded_image, model):
    """
    Predicts the class of the image at the given path.
    """
    img = load_and_prepare_image(uploaded_image)

    # Predict
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]

    # Show image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {predicted_class_name}")
    plt.axis('off')
    plt.show()
    return predicted_class_name

def detection_form_page():
    global uploaded_image
    st.title("Skin Cancer Detection Form")

    name = st.text_input("Name")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.text_input("Age")
    skin_type = st.selectbox("Skin Type", ["Simple", "Dry", "Oily", "Sensitive", "Combination"])
    skin_color = st.selectbox("Skin Color",
                              ["White", "Indian White", "Brown", "Indian Black", "Black", "Asian White", "Asian Dark"])
    skin_diseases = st.text_area("Skin Diseases (if any)")
    other_skin_issues = st.text_area("Other Skin Issues (if any)")
    email = st.text_input("Email Address")

    uploaded_image = st.file_uploader("Upload your image of skin cancer only", type=["jpg", "jpeg", "png"],
                                      help="Please upload a clear image of cancer-prone part of skin.")

    if st.button("Submit"):
        if not name or not age or not email or not uploaded_image:
            st.error("Please fill in all required fields.")
        else:
            to_save_file(name, gender, age, skin_type, skin_color, skin_diseases, email)
            st.session_state.name = name
            st.session_state.gender = gender
            st.session_state.age = age
            st.session_state.skin_type = skin_type
            st.session_state.skin_color = skin_color
            st.session_state.skin_diseases = skin_diseases
            st.session_state.other_skin_issues = other_skin_issues
            st.session_state.email = email
            st.session_state.uploaded_image = uploaded_image
            st.session_state.page = "Results"
            results_page()


def results_page():
    global input_data
    st.title("🩺 Skin Disease Detection Results")

    # Create two columns
    col1, col2 = st.columns([1, 2])

    # Left column: user details
    with col1:
        st.subheader("👤 Patient Information")
        st.markdown(f"**Name:** {st.session_state.name}")
        st.markdown(f"**Gender:** {st.session_state.gender}")
        st.markdown(f"**Age:** {st.session_state.age}")
        st.markdown(f"**Skin Type:** {st.session_state.skin_type}")
        st.markdown(f"**Skin Color:** {st.session_state.skin_color}")
        st.markdown(f"**Known Skin Conditions:** {st.session_state.skin_diseases}")
        st.markdown(f"**Other Concerns:** {st.session_state.other_skin_issues}")
        st.markdown(f"**Email:** {st.session_state.email}")

    # Right column: image and prediction
    with col2:
        st.subheader("📸 Uploaded Image")
        st.image(st.session_state.uploaded_image, caption="Analyzed Skin Area", use_column_width=True)


        # for set 2 : if accuracy goes good we may try it
        model2 = load_model_cnn()
        prediction2 = predict_image_class(st.session_state.uploaded_image, model2)
        st.write("Here is prediction")
        class_of(prediction2)

def FAQ_pages():
    import streamlit as st

    st.title("🩺 Skin Cancer - Frequently Asked Questions")

    # Skin Cancer FAQs
    if 'faq_data' not in st.session_state:
        st.session_state.faq_data = [
            {
                "question": "What is skin cancer?",
                "answer": "Skin cancer is the uncontrolled growth of abnormal skin cells, often caused by damage from UV radiation from the sun or tanning beds."
            },
            {
                "question": "What are the types of skin cancer?",
                "answer": "The main types include Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), and Melanoma. Melanoma is the most serious type."
            },
            {
                "question": "What are the warning signs of skin cancer?",
                "answer": "Look for new growths, sores that don’t heal, changes in moles (asymmetry, border, color, diameter, evolving), or any unusual skin changes."
            },
            {
                "question": "How can I check my skin for cancer?",
                "answer": "Regular self-examinations in good lighting and full-length mirrors can help spot unusual changes. Consult a dermatologist for a professional exam."
            },
            {
                "question": "Is skin cancer curable?",
                "answer": "Yes, when detected early, most skin cancers are highly treatable, especially BCC and SCC. Melanoma can also be treated if caught early."
            },
            {
                "question": "How is skin cancer diagnosed?",
                "answer": "Doctors may examine the skin visually and confirm with a biopsy – a sample of skin tissue examined under a microscope."
            },
            {
                "question": "Can skin cancer be prevented?",
                "answer": "Yes, by limiting sun exposure, using sunscreen, avoiding tanning beds, and wearing protective clothing, you can significantly reduce the risk."
            },
            {
                "question": "What is the ABCDE rule for melanoma?",
                "answer": "A - Asymmetry, B - Border irregularity, C - Color variation, D - Diameter >6mm, E - Evolving size, shape, or color."
            },
            {
                "question": "Are people with dark skin at risk of skin cancer?",
                "answer": "Yes. While it's less common, skin cancer can occur in people of all skin tones. It may be diagnosed later, leading to worse outcomes."
            },
            {
                "question": "What are common treatments for skin cancer?",
                "answer": "Treatments include surgical removal, cryotherapy, topical medications, radiation, and in some cases, immunotherapy or chemotherapy."
            },
        ]

    # --- Search Function ---
    search_query = st.text_input("🔍 Search Skin Cancer FAQ", "")

    def filter_faqs(query):
        if not query:
            return st.session_state.faq_data
        return [faq for faq in st.session_state.faq_data if
                query.lower() in faq["question"].lower() or query.lower() in faq["answer"].lower()]

    filtered_faqs = filter_faqs(search_query)

    # --- Display FAQs ---
    st.markdown("## 📄 FAQ List")
    if filtered_faqs:
        for i, faq in enumerate(filtered_faqs, start=1):
            with st.expander(f"❓ {faq['question']}", expanded=False):
                st.write(faq['answer'])
    else:
        st.warning("No FAQs found matching your search.")

    # --- Add New FAQ ---
    st.markdown("---")
    st.markdown("## ➕ Add a New Skin Cancer Question")

    with st.form("add_faq_form"):
        new_question = st.text_input("Enter your question")
        new_answer = st.text_area("Enter the answer")
        submit_button = st.form_submit_button("Add FAQ")

        if submit_button:
            if new_question and new_answer:
                st.session_state.faq_data.append({"question": new_question, "answer": new_answer})
                st.success("New FAQ added successfully!")
            else:
                st.error("Please fill in both question and answer fields.")

    # Footer
    st.markdown("---")
    st.markdown("💡 Stay informed. Early detection saves lives.")


if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    main()
