import streamlit as st
import tensorflow as tf
import numpy as np



#tensorflow
def model_prediction(test_image):
    model_path = "trained_model.h5"
    # Load the trained model
    model = tf.keras.models.load_model(model_path)
    
    # Load and resize the image to the target size
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    
    # Convert the image to a NumPy array and normalize pixel values
    input_arr = tf.keras.preprocessing.image.img_to_array(image)  # Normalize to [0, 1]
    
    # Add a batch dimension to the input array
    try:
        input_arr = np.expand_dims(input_arr, axis=0)
        print(f"Shape after expanding dimensions: {input_arr.shape}")  # Debugging log
    except Exception as e:
        print(f"Error: {e}")

    
    # Make a prediction
    predictions = model.predict(input_arr)
    
    # Return the index of the highest probability
    return np.argmax(predictions)






# sidebar
st.sidebar.title("Dashbar")
app_mode = st.sidebar.selectbox("Select Page",["Home", "About Project", "Prediction"])

# main page
if(app_mode=="Home"):
    st.header("Fruits & Vegetable Recogniton System")
    image_path = "veg_image.jpg"
    st.image(image_path)

# about project
elif(app_mode=="About Project"):
    st.header("About Project")
    st.subheader("About the Dataset")
    st.text("This dataset encompasses images of various fruits and vegetables, providing a diverse collection for image recognition tasks. The included food items are:")
    st.code("Fruits: Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango")
    st.code("Vegetables: Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant")
    st.subheader("Content")
    st.text("The dataset is organized into three main folders:")
    st.text("1. Train: Contains 100 images per category")
    st.text("2. Test: Contains 10 images per category.")
    st.text("3. Validation : Contains 10 images per category")

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image= st.file_uploader("Upload an image")
    if test_image is not None:
        # Show the uploaded image
        st.image(test_image, caption="Uploaded Image", use_column_width=True)
        
        # Prediction button
        if st.button("Predict"):
            st.balloons()
            st.write("Our Prediction:")
            result_index = model_prediction(test_image)
            
            # Reading labels
            with open("label.txt") as f:
                content = f.readlines()
            label = [line.strip() for line in content]
            
            # Display the prediction result
            st.success(f"Model is predicting it's a {label[result_index]}")
    else:
        st.warning("Please upload an image to proceed.")
    
