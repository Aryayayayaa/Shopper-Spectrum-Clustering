# Shopper-Spectrum-Clustering
This project, "Aura AI," is a Python-based retail analytics dashboard. It uses K-Means clustering for customer segmentation and an item-based collaborative filtering model for product recommendations. The Streamlit app provides a user-friendly interface for data-driven decision-making in marketing and sales. It is an end-to-end data science solution designed to provide actionable insights from retail transactional data. It features a comprehensive analytics pipeline, machine learning models for customer segmentation and product recommendations, and a user-friendly web dashboard built with Streamlit.

The primary goal of this project is to empower businesses to make data-driven decisions on marketing, sales, and customer engagement by transforming raw data into strategic intelligence.

-----

### Project Overview

The project is divided into several key stages:

1.  **Data Preprocessing & EDA:** The initial phase involved cleaning a raw transactional dataset, handling missing values, removing anomalies, and engineering key features like `TotalPrice`. A thorough Exploratory Data Analysis (EDA) was performed to uncover trends in sales, transaction volume, product popularity, and geographical distribution.

2.  **Customer Segmentation:** Using the K-Means clustering algorithm, the customer base was segmented based on their Recency, Frequency, and Monetary (RFM) values. This segmentation allows for targeted marketing strategies and a deeper understanding of customer behavior.

3.  **Product Recommendation System:** An Item-based Collaborative Filtering model was developed to provide personalized product recommendations. The model computes the cosine similarity between products to identify which items are most likely to be purchased together, facilitating cross-selling.

4.  **Dashboard Deployment:** All models and insights were integrated into an interactive dashboard using Streamlit. This application serves as a central hub for business users to access real-time predictions and visualizations, making the project's output accessible and practical.

-----

### Key Features

  * **Interactive Dashboard:** A responsive Streamlit application for real-time interaction with the models.
  * **Customer Segmentation:** Predicts a customer's segment based on their RFM values and provides detailed profiles for each cluster.
  * **Product Recommendations:** Recommends similar products for any given item, based on co-purchase patterns.
  * **Visual Analytics:** Features a range of visualizations, including:
      * Top-selling products by quantity and sales value.
      * Sales and transaction trends over time (monthly, daily).
      * Geographical distribution of sales.
  * **Reproducible Code:** All data cleaning, EDA, modeling, and dashboard code is well-documented, allowing for easy reproducibility.

-----

### Getting Started

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```
    git clone [Your Repository URL]
    cd [Your Repository Folder]
    ```

2.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**

    ```
    streamlit run app.py
    ```

    (Note: You will need the required data files in the correct path for the app to function correctly.)

-----

### Technologies Used

  * Python
  * Pandas, NumPy
  * Scikit-learn
  * Streamlit
  * Matplotlib, Seaborn
