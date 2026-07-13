"""
Application Entry Point & Router (Controller Layer)
----------------------------------------------------
This file is the main entry point of the Streamlit application. It serves as the
system controller, managing global settings, layout styling, and page navigation routing.

Design Rationale:
- Unified Controller: By centralizing settings (like page config) and routing in main.py, 
  we ensure a single source of truth for the application's lifecycle.
- st.Page and st.navigation: We utilize the modern Streamlit page routing framework. 
  Instead of linking raw script files, we point to self-contained page functions imported 
  directly from our presentation layer (in ui/). This modular approach enables cleaner code, 
  shared import namespaces, and avoids disk script loading overhead.
"""

import streamlit as st
from ui.home import show_home
from ui.regression_ui import show_regression_workspace
from ui.classification_ui import show_classification_workspace

# 1. Global Page Layout Configurations
# Set config must be called as the very first Streamlit command on page load!
st.set_page_config(
    page_title="Enterprise AutoML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom global styles for general dashboard elements (theme safe variables)
st.markdown("""
<style>
    /* Styling headers & cards globally */
    .section-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# 2. Define Page Routing Links
# st.Page encapsulates a callable page rendering function, supplying titles and sidebar icons.
home_page = st.Page(
    show_home, 
    title="Overview & Home", 
    icon="🏠", 
    default=True
)

regression_page = st.Page(
    show_regression_workspace, 
    title="Regression Workspace", 
    icon="📈"
)

classification_page = st.Page(
    show_classification_workspace, 
    title="Classification Workspace", 
    icon="🎯"
)

# 3. Create Navigation Tree
# We organize our pages into sections in the sidebar: "Introduction" and "ML Benchmarks"
pg = st.navigation({
    "Introduction": [home_page],
    "ML Benchmarks": [regression_page, classification_page]
})

# 4. Run Page Router
# Executes the rendering function corresponding to the page currently selected in the sidebar
pg.run()
