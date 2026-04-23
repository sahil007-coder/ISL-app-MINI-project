import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("ISL Gesture Recognition")

st.write("Live Camera Feed Below")

webrtc_streamer(key="isl-camera")