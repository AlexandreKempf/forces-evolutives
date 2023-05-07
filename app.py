import streamlit as st
import pages.hardy_weinberg
import pages.modelisation
import pages.statistique
import pages.exemples


def main():

    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{max-width: 80%;}}

    </style>
    """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.selectbox(
        "Choisis une page",
        ["Hardy-Weinberg", "Test Statistique", "Modélisation", "Exemples"],
    )

    if page == "Hardy-Weinberg":
        pages.hardy_weinberg.generate_page()

    elif page == "Test Statistique":
        pages.statistique.generate_page()

    elif page == "Modélisation":
        pages.modelisation.generate_page()

    elif page == "Exemples":
        pages.exemples.generate_page()


if __name__ == "__main__":

    main()
