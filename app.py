import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable


@register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@register_keras_serializable()
def bce_dice_loss(y_true, y_pred, smooth=1):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_coefficient(y_true, y_pred, smooth)
    return bce + (1 - dice)


@st.cache_resource
def load_segmentation_model():
    model = load_model("best_model.keras", custom_objects={
        "dice_coefficient": dice_coefficient,
        "bce_dice_loss": bce_dice_loss
    })
    return model

model = load_segmentation_model()


st.title("Ultrasound Nerve Segmentation")
st.markdown("Upload a `.tif`, `.jpg`, or `.png` image to get a predicted nerve mask.")

uploaded_file = st.file_uploader("Choose an image", type=["tif", "tiff", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        
        img = Image.open(uploaded_file).convert('L')  
        original_img = img.copy()
        img = img.resize((128, 128))
        img_arr = np.array(img) / 255.0
        img_arr = img_arr.reshape(1, 128, 128, 1)

        # Predict
        pred = model.predict(img_arr)[0]
        pred_mask = (pred > 0.5).astype(np.uint8).squeeze() * 255

        # Display
        st.subheader("Original Image")
        st.image(original_img, caption="Uploaded Image", use_column_width=True)

        st.subheader("Predicted Mask")
        st.image(pred_mask, caption="Predicted Segmentation", use_column_width=True, clamp=True)

    except Exception as e:
        st.error(f"Error: {e}")
