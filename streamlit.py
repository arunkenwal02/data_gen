import streamlit as st 
import requests
import os

downloads_dir = os.path.join(os.getcwd(), "downloads")
os.makedirs(downloads_dir, exist_ok=True)

st.header("Test Data Generation for Credit Strategy")

git_url = st.text_input("Git Repository URL")

button = st.button('Submit')

if button:
    if not git_url.strip():
        st.error("Please enter a valid Git repository URL")
    else:
        st.info("Processing repository...")
        response = requests.post(
        "http://127.0.0.1:8000/create",
        json={"git_repo": git_url}
    )
        if response.status_code == 200:
            st.success("Excel file created successfully!")

            st.download_button(
                label="Download Excel",
                data=response.content,
                file_name="dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error(f"API error: {response.status_code}")