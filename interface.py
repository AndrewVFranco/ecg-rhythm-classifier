import streamlit as st
import numpy as np
import tensorflow as tf
import json

st.set_page_config(layout="wide", page_title="ECG Rhythm Classifier")

# Load model and samples
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/optimized_ecg_model.keras")

NSR_sample = np.load("frontend_files/NSR_sample.npy")
AFIB_sample = np.load("frontend_files/AFIB_sample.npy")
AVB_sample = np.load("frontend_files/AVB_sample.npy")

model = load_model()

# Initialize session state
if 'selected_rhythm' not in st.session_state:
    st.session_state.selected_rhythm = 'Normal Sinus Rhythm'

# Start the page layout
st.title("ECG Rhythm Classifier")

# Change session state based on the selected button
if st.session_state.selected_rhythm == 'Normal Sinus Rhythm':
    current_sample = NSR_sample
elif st.session_state.selected_rhythm == 'A-Fib':
    current_sample = AFIB_sample
elif st.session_state.selected_rhythm == '1st Degree AVB':
    current_sample = AVB_sample

# Format the sample and make a prediction
formatted_sample = np.expand_dims(current_sample, axis=0)
prediction = model.predict(formatted_sample)

# Label the prediction and get the confidence score
prediction_label = np.argmax(prediction)
confidence = prediction[0][prediction_label] * 100
labels = {0: '1st Degree AVB', 1: 'A-Fib', 2: 'Normal Sinus Rhythm'}
predicted_class = labels[prediction_label]

# Print rhythm classification
classcol, confcol = st.columns(2)
with classcol:
    st.metric("Rhythm Classification" ,predicted_class)
with confcol:
    st.metric("Confidence", confidence , format="%.2f%%")

# Format the sample for display
sample_for_display = current_sample.tolist()
sample_for_display = json.dumps(sample_for_display)

# Start html content for smooth sample scrolling with a confidence bar
html_content = f""" 
<html>
<body style="background:#050508; margin:0; padding:1rem; font-family:'Courier New', monospace; display:flex; justify-content:center;">
<div style="background:#050508; border:1px solid #ffffff15; border-radius:4px; padding:1rem; width:100%; height:190px; overflow:hidden;">
    <svg id="ecg" width="100%" viewBox="0 0 800 560" style="display:block; padding:0px;">
    </svg>
</div>
<script>
const svg = document.getElementById('ecg');
const w = 800, h = 150;

// Vertical grid lines
for (let i = 0; i < 20; i++) {{
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', i * (w/20));
    line.setAttribute('y1', 0);
    line.setAttribute('x2', i * (w/20));
    line.setAttribute('y2', h);
    line.setAttribute('stroke', '#ffffff08');
    line.setAttribute('stroke-width', '1');
    svg.appendChild(line);
}}

// Horizontal grid lines
for (let i = 0; i < 8; i++) {{
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', 0);
    line.setAttribute('y1', i * (h/8));
    line.setAttribute('x2', w);
    line.setAttribute('y2', i * (h/8));
    line.setAttribute('stroke', '#ffffff08');
    line.setAttribute('stroke-width', '1');
    svg.appendChild(line);
}}

const signal = {sample_for_display};
const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
svg.appendChild(pathEl);

let offset = 0;
const windowSize = 2500;

function draw() {{
    let d = '';
    for (let i = 0; i < windowSize; i++) {{
        const idx = (offset + i) % signal.length;
        const x = (i / windowSize) * w;
        const y = h/2 + 5 - signal[idx][0] * (h/20);
        d += (i === 0 ? 'M ' : 'L ') + x + ',' + y + ' ';
    }}
    pathEl.setAttribute('d', d);
    pathEl.setAttribute('fill', 'none');
    pathEl.setAttribute('stroke', '#ff3333');
    pathEl.setAttribute('stroke-width', '1.5');
    offset = (offset + 1) % signal.length;
}}

setInterval(draw, 2);
</script>
</body>
</html>
"""

st.components.v1.html(html_content, height=260)

# Create columns and buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button('Normal Sinus Rhythm sample', use_container_width=True):
        st.session_state.selected_rhythm = "Normal Sinus Rhythm"
        st.rerun()

with col2:
    if st.button('A-Fib sample', use_container_width=True):
        st.session_state.selected_rhythm = "A-Fib"
        st.rerun()

with col3:
    if st.button('1st Degree AVB sample', use_container_width=True):
        st.session_state.selected_rhythm = "1st Degree AVB"
        st.rerun()
