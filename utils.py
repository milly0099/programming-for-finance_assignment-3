import streamlit as st
import base64
import requests
from io import BytesIO
from PIL import Image
import pandas as pd
import numpy as np

def apply_money_heist_theme():
    """Apply Money Heist themed styling to the app"""
    # Define the custom Money Heist styling
    st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: #f0f0f0;
    }
    .stButton button {
        background-color: #e50914;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #b2070f;
        color: white;
    }
    h1, h2, h3 {
        color: #e50914;
        font-family: 'Arial Black', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: #f0f0f0;
    }
    .css-145kmo2 {
        font-family: 'Arial', sans-serif;
    }
    .css-1d391kg {
        background-color: #e50914;
    }
    .stProgress > div > div {
        background-color: #e50914;
    }
    </style>
    """, unsafe_allow_html=True)

def get_mask_image():
    """Get the Salvador Dalí mask image from Money Heist"""
    # Use one of the masks from the provided URLs
    mask_url = "https://pixabay.com/get/g922d4f239d427232ec1db1c27c974745fafb107e63bae34842f82da0148f14da9013592d21af96a9a7d2a5ed30123b7f8e3923c94f4a523297f61c9fb0f27fca_1280.jpg"
    
    try:
        response = requests.get(mask_url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception:
        # Fallback to a simple text message if image can't be loaded
        return "Salvador Dalí Mask - Money Heist Symbol"

def get_welcome_gif():
    """Get a Money Heist themed welcome GIF"""
    # Since we can't include actual GIFs, using one of the provided money heist images
    url = "https://pixabay.com/get/g7080de81b8bcbfcd63eaabb1a946590acc03a427c91f8983d74fcced89ff999424439165fd51c423a5f563830b46cd0dfc1c7d2caf0882be52b3eb68b278dae0_1280.jpg"
    
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception:
        # Fallback to a simple text message if image can't be loaded
        return "Money Heist Welcome Image"

def get_completion_gif():
    """Get a Money Heist themed completion GIF"""
    # Since we can't include actual GIFs, using one of the provided stock market images
    url = "https://pixabay.com/get/g1f630c8003bfa752b330d9ea61ac4f82bd6a4454e486974322dd3cb255c0d48675f075d62622b7b8126cea20505fd249680506c3e093a30609a7638975d832eb_1280.jpg"
    
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception:
        # Fallback to a simple text message if image can't be loaded
        return "Heist Successful Image"

def display_notification(type, message):
    """Display a styled notification message"""
    if type == "success":
        st.success(f"✅ {message}")
    elif type == "error":
        st.error(f"❌ {message}")
    elif type == "warning":
        st.warning(f"⚠️ {message}")
    elif type == "info":
        st.info(f"ℹ️ {message}")

def show_image_with_caption(url, caption):
    """Display an image with a caption"""
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption=caption, use_column_width=True)
    except Exception:
        st.write(f"*{caption}*")
