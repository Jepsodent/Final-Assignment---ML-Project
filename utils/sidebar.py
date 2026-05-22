import streamlit as st

def render_sidebar(active_page: str | None = None) -> None:
    # 1. Menyuntikkan style CSS kustom untuk mempercantik Sidebar
    st.markdown(
        """
        <style>


        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        /* Styling judul di sidebar */
        .sidebar-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: #ff7f50;
            margin-bottom: 0.5rem;
            text-align: center;
            letter-spacing: 1px;
            text-shadow: 0px 4px 10px rgba(255, 127, 80, 0.3);
        }

        .sidebar-subtitle {
            color: #94a3b8;
            font-size: 0.85rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .sidebar-divider {
            margin: 1rem 0;
            border-bottom: 1px solid rgba(255, 127, 80, 0.2);
        }

        .sidebar-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.75rem;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        #Judul
        st.markdown('<div class="sidebar-title">🔥 Calories AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">ML Pipeline Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<span class="sidebar-label">Pipeline Menu</span>', unsafe_allow_html=True)

        menu_items = [
            {"label": " Home", "page": "app.py", "id": "Home"},
            {"label": " Dataset Overview", "page": "pages/1_Dataset.py", "id": "Dataset"},
            {"label": " Exploratory Data (EDA)", "page": "pages/2_EDA.py", "id": "EDA"},
            {"label": " Preprocessing", "page": "pages/3_Preprocessing.py", "id": "Preprocessing"},
            {"label": " Model Training & Evaluation", "page": "pages/4_Training-Evaluation.py", "id": "TrainingnEvaluation"},
            {"label": " Interactive Demo", "page": "pages/6_Demo.py", "id": "Demo"},
        ]
        #Label
        for item in menu_items:
            is_active = (active_page == item["id"])
            button_label = f"👉 {item['label']}" if is_active else item["label"]
            
            if st.button(button_label, width="stretch", key=f"nav_{item['id'].lower()}"):
                st.switch_page(item["page"])

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.caption("Developed with ❤️ for Machine Learning Final Project")
