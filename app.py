import streamlit as st
from Dazl_Compile import compile_from_code_string
import matplotlib.pyplot as plt
import time
from io import StringIO

# Page setup must be FIRST
st.set_page_config(page_title="DAZL Graphical Compiler", layout="centered", page_icon="🧠")

# Now add animated background CSS
st.markdown("""
<style>
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.stApp {
    background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    min-height: 100vh;
}

/* Keep content readable */
.main .block-container {
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 10px;
    padding: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Your existing custom CSS
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
.fade-in {
    animation: fadeIn 0.6s ease-in-out;
}
.success-box {
    border-radius: 10px;
    padding: 15px;
    background-color: #e6f7e6;
    border-left: 4px solid #2e7d32;
    margin-top: 20px;
    font-weight: bold;
    color: #2e7d32;
}
.error-box {
    border-radius: 10px;
    padding: 15px;
    background-color: #ffebee;
    border-left: 4px solid #c62828;
    margin-top: 20px;
    font-weight: bold;
    color: #c62828;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<h1 class="fade-in">🧠 DAZL Graphical Language Compiler</h1>
<p class="fade-in">Visualize your algorithms with DAZL's intuitive syntax</p>
""", unsafe_allow_html=True)

# Default starter code
default_code = '''# Example DAZL Code
SET R = 5
CIRCLE 10 10 R
SHOW
'''

code_input = st.text_area("✏️ Write your DAZL code here:", default_code, height=300)

# Compile and run button
if st.button("🚀 Compile & Visualize", use_container_width=True):
    with st.spinner("Compiling your DAZL code..."):
        time.sleep(0.5)
        progress_bar = st.progress(0)
        for percent in range(100):
            time.sleep(0.005)
            progress_bar.progress(percent + 1)

        try:
            output = compile_from_code_string(code_input)

            if output.strip():
                st.markdown('<div class="success-box">✅ Code executed successfully!</div>', unsafe_allow_html=True)
                with st.expander("📋 Execution Output", expanded=True):
                    st.code(output, language="text")
            else:
                st.markdown('<div class="success-box">✅ Code executed successfully (no console output).</div>', unsafe_allow_html=True)

        except Exception as e:
            st.markdown(
                f'<div class="error-box">❌ Error during execution:<br>{str(e)}</div>',
                unsafe_allow_html=True
            )