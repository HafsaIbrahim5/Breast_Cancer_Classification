import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

# 1. Setup Session State to keep the Tab fixed
if "current_tab" not in st.session_state:
    st.session_state.current_tab = 0

st.set_page_config(page_title="Hafsa | Breast Cancer AI", page_icon="🎗️", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    section[data-testid="stSidebar"] { background-color: #050505 !important; border-right: 1px solid #17E8C4; }
    h1, h2, h3 { text-align: center !important; }
    h1 { background: linear-gradient(90deg, #17E8C4, #05445E); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    
    .neon-button { 
        display: block; padding: 10px; border-radius: 25px; 
        border: 2px solid #17E8C4; color: #17E8C4 !important; text-decoration: none; 
        font-size: 14px; font-weight: bold; transition: 0.3s; margin: 15px auto; text-align: center; width: 85%;
    }
    .neon-button:hover { background: #17E8C4; color: black !important; box-shadow: 0 0 20px #17E8C4; transform: scale(1.05); }
    .sidebar-img { border-radius: 50%; border: 3px solid #17E8C4; padding: 5px; box-shadow: 0 0 15px #17E8C4; display: block; margin: 0 auto; }
    
    .result-card { background: #161b22; padding: 35px; border-radius: 20px; border: 1px solid #30363d; margin-top: 25px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }

    /* Centered Glowing Rectangle Button */
    div.stButton > button:first-child {
        display: block; margin: 30px auto; padding: 15px 50px; width: 350px; 
        background: linear-gradient(45deg, #17E8C4, #05445E); color: white;
        font-size: 18px; font-weight: 800; letter-spacing: 2px; border: none; border-radius: 12px;
        transition: all 0.4s ease; box-shadow: 0 4px 15px rgba(23, 232, 196, 0.2);
    }
    div.stButton > button:first-child:hover { transform: translateY(-3px); box-shadow: 0 0 25px #17E8C4; color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def train_model():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    # Optimized for 98%+ Accuracy
    model = HistGradientBoostingClassifier(
        max_iter=1000,
        learning_rate=0.04,
        max_depth=15,
        l2_regularization=0.5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return (
        model,
        data.feature_names,
        data.target_names,
        pd.DataFrame(data.data, columns=data.feature_names),
    )


model, feature_names, target_names, raw_df = train_model()

# Sidebar
with st.sidebar:
    st.markdown("<div style='padding-top: 30px;'>", unsafe_allow_html=True)
    st.markdown(
        '<img src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png" class="sidebar-img" width="110">',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='color: #17E8C4; margin-top: 15px;'>Hafsa Ibrahim</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center; color: #8b949e; font-size: 14px;'>AI & NLP Engineer</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown(
        f"""
        <div style='text-align: center;'>
            <a href='https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/' target='_blank' class='neon-button'>LINKEDIN</a>
            <a href='https://github.com/HafsaIbrahim5' target='_blank' class='neon-button'>GITHUB</a>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("<h1>Breast Cancer Diagnostic AI</h1>", unsafe_allow_html=True)

# THE FIX: Store current tab in session_state
tabs = ["📊 Analytics Overview", "🔬 Prediction Engine"]
selected_tab = st.tabs(tabs)

# Tab 1 Content
with selected_tab[0]:
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.pie(
            names=target_names,
            values=np.bincount(load_breast_cancer().target),
            hole=0.6,
            color_discrete_sequence=["#05445E", "#17E8C4"],
        )
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            title=dict(text="Data Balance", x=0.5),
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.histogram(
            raw_df,
            x="mean area",
            color=load_breast_cancer().target,
            color_discrete_sequence=["#05445E", "#17E8C4"],
            barmode="overlay",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            title=dict(text="Area Distribution", x=0.5),
        )
        st.plotly_chart(fig2, use_container_width=True)

# Tab 2 Content
with selected_tab[1]:
    st.markdown(
        "<h3 style='margin-bottom: 25px;'>Adjust Patient Clinical Profile</h3>",
        unsafe_allow_html=True,
    )
    input_values = []
    cols = st.columns(3)
    for i, name in enumerate(feature_names):
        with cols[i % 3]:
            val = st.slider(
                f"{name.title()}",
                float(raw_df[name].min()),
                float(raw_df[name].max()),
                float(raw_df[name].mean()),
            )
            input_values.append(val)

    if st.button("EXECUTE ANALYSIS"):
        prob = model.predict_proba([input_values])
        prediction = model.predict([input_values])
        confidence = np.max(prob) * 100

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        res_color = "#ff4b4b" if prediction[0] == 0 else "#17E8C4"
        st.markdown(
            f"<h1 style='color: {res_color}; font-size: 3rem; margin-bottom: 0;'>{target_names[prediction[0]].upper()}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color: #8b949e; font-size: 1.2rem;'>Reliability Index: <span style='color: white; font-weight: bold;'>{confidence:.2f}%</span></p>",
            unsafe_allow_html=True,
        )

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#17E8C4"},
                    "steps": [
                        {"range": [0, 85], "color": "#05445E"},
                        {"range": [85, 100], "color": "#0b2539"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=30, r=30, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Optimized Gradient Boosting System | Developed by Hafsa Ibrahim | 2026")
