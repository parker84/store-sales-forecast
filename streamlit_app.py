import streamlit as st


# --- PAGE SETUP ---
forecasts_page = st.Page(
    "views/forecasts.py",
    title="Forecasts",
    icon="🔮",
    default=True,
)
train_page = st.Page(
    "views/eda_train.py",
    title="Training Data",
    icon="🚊",
)
make_dataset_page = st.Page(
    "views/make_dataset.py",
    title="Make Datasets",
    icon="📦",
)

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Forecasts": [forecasts_page],
        "EDA": [train_page],
        "Preprocessing": [
            make_dataset_page,
        ],
    }
)


# --- SHARED ON ALL PAGES ---
st.logo("assets/TorCrime_logo.png")
st.sidebar.markdown("Made with ❤️ by [Brydon](https://brydon.streamlit.app/)")


# --- RUN NAVIGATION ---
pg.run()