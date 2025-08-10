import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# --- Custom Styling ---
@st.cache_data
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Background image not found at {image_path}. Please ensure it's in the same directory.")
        return None

BACKGROUND_IMAGE_PATH = 'background.jpg'

base64_image = get_base64_image(BACKGROUND_IMAGE_PATH)

if base64_image:
    # Determine MIME type based on file extension
    if BACKGROUND_IMAGE_PATH.lower().endswith(".png"):
        mime_type = "image/png"
    elif BACKGROUND_IMAGE_PATH.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    else:
        mime_type = "image/webp" # Default or add more types as needed

    background_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="st-"] {{
        font-family: 'Inter', sans-serif;
        color: #FFFFFF; /* White text for dark background */
    }}

    /* Background image and darker overlay for better text readability */
    .stApp {{
        background-image: url("data:{mime_type};base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.6);               /* Darker overlay for readability */
        z-index: -1;
    }}

    /* Main content area styling with darker transparent background */
    .main .block-container {{
        background-color: rgba(0, 0, 0, 0.75);              /* Darker transparent background for content boxes */
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);     /* Light shadow for contrast */
        margin-top: 20px;
        margin-bottom: 20px;
        color: #FFFFFF;                                     /* Ensure text inside is white */
    }}

    /* Sidebar styling with solid dark background - MORE ROBUST SELECTOR */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: #1a1a1a !important;               /* Solid very dark grey/nearly black with !important */
        border-radius: 15px; /* Keep consistent rounding */
        box-shadow: 0 4px 8px rgba(255, 255, 255, 0.1);
        padding: 1rem;
        color: #FFFFFF; 
    }}

    /* Sidebar Navigation (radio buttons) styling */
    .st-emotion-cache-1q1u406 {{                    /* This class often targets the selected radio button item */
        background-color: #FFD700 !important;       /* Bright yellow for active item */
        color: #000000 !important;                  /* Black text on active yellow item */
        border-radius: 5px;
        padding: 5px 10px;
        margin-bottom: 5px;
    }}
    .st-emotion-cache-1kv5xkv > label {{            /* Targets the label of all radio button items */
        color: #FFFFFF !important; 
        margin-bottom: 5px;
        padding: 5px 10px;
        border-radius: 5px;
    }}
    .st-emotion-cache-1kv5xkv > label:hover {{          /* Hover effect for radio buttons */
        background-color: rgba(255, 255, 255, 0.1) !important;
    }}


    /* Header styling in a contrasting yellow */
    h1, h2, h3, h4, h5, h6 {{
        color: #FFD700;                             /* Bright yellow header to complement background */
        font-weight: 700;
    }}

    /* Input fields (text and number) with darker background and larger, brighter text */
    .stTextInput input, .stNumberInput input, input[type="text"], input[type="number"], textarea {{
        background-color: #000000 !important; 
        color: #FFFFFF !important; 
        font-size: 1.2em !important; 
        border-radius: 8px;
        border: 1px solid #666 !important;                          /* Border for visibility */
        padding: 10px !important;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.7) !important; /* More prominent inner shadow */
    }}

    /* Adjust text color for the placeholder inside input fields */
    .stTextInput input::placeholder, .stNumberInput input::placeholder, input::placeholder {{
        color: rgba(255, 255, 255, 0.7) !important; /* Slightly transparent white for placeholder */
    }}


    /* Button styling (e.g., Predict Cluster, Get Recommendations) with dark background and white text */
    /* Targeting by data-testid for higher specificity on buttons in forms */
    div[data-testid="stFormSubmitButton"] > button,
    .stButton > button,
    .stDownloadButton > button {{
        background-color: #1a1a1a !important; /* Very dark grey/nearly black button background */
        color: #FFFFFF !important; /* White text on button */
        border-radius: 8px !important;
        border: 1px solid #444 !important; /* Darker border */
        padding: 10px 20px !important;
        font-weight: 600 !important;
        transition: background-color 0.3s ease, transform 0.2s ease !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4) !important; /* Darker shadow */
    }}
    div[data-testid="stFormSubmitButton"] > button:hover,
    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        background-color: #333333 !important; /* Slightly lighter dark grey on hover */
        transform: translateY(-2px) !important;
    }}


    /* Success/Error messages with adjusted colors for dark theme */
    .st-emotion-cache-1c7y2qn {{ /* Success box */
        background-color: rgba(76, 175, 80, 0.3); /* Transparent dark green */
        border-left: 5px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
        color: #FFFFFF; /* White text for messages */
    }}
    .st-emotion-cache-1y4y20p {{ /* Error box */
        background-color: rgba(244, 67, 54, 0.3); /* Transparent dark red */
        border-left: 5px solid #F44336;
        border-radius: 8px;
        padding: 10px;
        color: #FFFFFF; /* White text for messages */
    }}

    /* List styling for recommendations with darker background */
    ul {{
        list-style-type: none;
        padding: 0;
    }}
    li {{
        background-color: rgba(0, 0, 0, 0.8); /* Darker transparent background for list items */
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(255, 255, 255, 0.05); /* Subtle light shadow */
        color: #EEE; /* Lighter text for list items */
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)
else:
    st.warning("Background image not loaded. Using default Streamlit styling. Check console for FileNotFoundError.")


# --- Load Models and Data ---
@st.cache_resource # Cache resource to load once
def load_assets():
    try:
        kmeans_model = joblib.load('kmeans_model.joblib')
        scaler = joblib.load('scaler.joblib')
        item_similarity_df = joblib.load('item_similarity_df.joblib')
        rfm_data_with_clusters = pd.read_csv('rfm_data_with_clusters.csv')
        rfm_cluster_profiles = pd.read_csv('rfm_cluster_profiles.csv', index_col=0)

        # Define the cluster label map - IMPORTANT: Adjust this based on your Colab analysis!
        # You MUST update this dictionary based on your actual cluster_profiles output
        # and your manual interpretation from the Colab notebook.
        cluster_id_to_label = {
            1: "High-Value/Loyal Customer",
            2: "At-Risk/Mid-Value Customer",
            0: "Recent/Low-Frequency Customer",
            3: "Churned/Lost Customer"
        }
        # Add the labels to the cluster profiles DataFrame for easy lookup
        rfm_cluster_profiles['Segment Label'] = rfm_cluster_profiles.index.map(cluster_id_to_label)


        return kmeans_model, scaler, item_similarity_df, rfm_data_with_clusters, rfm_cluster_profiles, cluster_id_to_label
    except FileNotFoundError as e:
        st.error(f"Error loading required files: {e}. Please ensure all .joblib and .csv files are in the same directory as app.py.")
        st.stop() # Stop the app if files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred during asset loading: {e}")
        st.stop()

kmeans_model, scaler, item_similarity_df, rfm_data_with_clusters, rfm_cluster_profiles, cluster_id_to_label = load_assets()

# --- Helper Function for Recommendations ---
def get_top_similar_products(product_name, item_similarity_df, n=5):
    """
    Returns the top N most similar products to a given product name.
    """
    if product_name not in item_similarity_df.index:
        return [], f"Product '{product_name}' not found in our database. Please check the spelling or try another product."

    similar_scores = item_similarity_df[product_name].sort_values(ascending=False)
    similar_scores = similar_scores.drop(product_name, errors='ignore') # Remove itself, ignore error if already dropped

    top_n_similar = similar_scores.head(n)
    return list(top_n_similar.items()), "" # Return list of (product, score) tuples and empty error message

# --- Streamlit App Layout ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Customer Segmentation", "Product Recommendation"])

# --- Home Page ---
if page == "Home":
    st.title("Welcome to the: Aura AI - Retail Analytics Dashboard!")
    # Content box for main introduction
    st.markdown("""
        <div style="background-color: rgba(0, 0, 0, 0.6); padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(255,255,255,0.1);">
            <p style="font-size: 1.1em; line-height: 1.6; color: #FFFFFF;">
                This interactive dashboard provides powerful insights into your retail business,
                helping you understand customer behavior and optimize sales strategies.
                Leveraging advanced data analysis and machine learning, we offer:
            </p>
            <ul>
                <li><b>Customer Segmentation:</b> Understand different customer groups based on their purchasing habits (Recency, Frequency, Monetary value). Predict which segment a new or existing customer belongs to.</li>
                <li><b>Product Recommendations:</b> Get personalized product suggestions based on collaborative filtering, helping customers discover items they'll love.</li>
            </ul>
            <p style="font-size: 1.1em; line-height: 1.6; color: #FFFFFF;">
                Navigate through the modules using the sidebar on the left to explore these features.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---") # This will now be a light gray line against the dark background
    st.subheader("How it Works:")
    
    st.markdown("""
    <div style="background-color: rgba(0, 0, 0, 0.6); padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(255,255,255,0.1);">
        <p style="color: #FFFFFF;">
            <ul>    
                <li> <b>Data Preprocessing:</b> Raw transaction data is cleaned and prepared.
                <li> <b>Feature Engineering:</b> Key metrics like TotalPrice and InvoiceDate are transformed.
                <li> <b>RFM Analysis:</b> Customers are characterized by their Recency (how recently they purchased), Frequency (how often), and Monetary value (how much they spent).
                <li> <b>Clustering:</b> K-Means algorithm groups customers into distinct segments based on their RFM scores.
                <li> <b>Collaborative Filtering:</b> Item-based similarity is computed from purchase history to recommend similar products.
            </ul> 
        </p>
    </div>
    """, unsafe_allow_html=True)


# --- Customer Segmentation Module ---
elif page == "Customer Segmentation":
    st.title("Customer Segmentation")
    st.write("Enter a customer's RFM values to predict their segment.")

    with st.form("customer_segmentation_form"):
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, help="Number of days since the customer's last purchase.")
        frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5, help="Total number of unique transactions made by the customer.")
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=100.0, format="%.2f", help="Total amount of money spent by the customer.")

        predict_button = st.form_submit_button("Predict Cluster")

    if predict_button:
        # Create a DataFrame for the new customer's RFM values
        new_customer_rfm = pd.DataFrame([[recency, frequency, monetary]],
                                        columns=['Recency', 'Frequency', 'Monetary'])

        # Apply log transformation (must be consistent with training)
        # Use np.log1p for Recency, Frequency, Monetary to handle potential zeros
        new_customer_rfm_log = new_customer_rfm.copy()
        new_customer_rfm_log['Recency'] = np.log1p(new_customer_rfm_log['Recency'])
        new_customer_rfm_log['Frequency'] = np.log1p(new_customer_rfm_log['Frequency'])
        new_customer_rfm_log['Monetary'] = np.log1p(new_customer_rfm_log['Monetary'])

        # Scale the new customer's RFM values using the loaded scaler
        new_customer_scaled = scaler.transform(new_customer_rfm_log)

        # Predict the cluster
        predicted_cluster_id = kmeans_model.predict(new_customer_scaled)[0]
        predicted_segment_label = cluster_id_to_label.get(predicted_cluster_id, "Unknown Segment")

        st.success(f"### Predicted Cluster: {predicted_cluster_id}")
        st.write(f"This customer belongs to the **{predicted_segment_label}** segment.")

        # Display characteristics of the predicted cluster
        if predicted_cluster_id in rfm_cluster_profiles.index:
            st.subheader(f"Characteristics of {predicted_segment_label} Segment:")
            segment_info = rfm_cluster_profiles.loc[predicted_cluster_id]
            st.markdown(f"""
            - **Average Recency:** {segment_info['AvgRecency']:.2f} days
            - **Average Frequency:** {segment_info['AvgFrequency']:.2f} purchases
            - **Average Monetary:** £{segment_info['AvgMonetary']:.2f}
            - **Represents:** {segment_info['PercentageOfCustomers']:.2f}% of all customers
            """)
        else:
            st.info("No detailed profile available for this cluster ID.")

# --- Product Recommendation Module ---
elif page == "Product Recommendation":
    st.title("Product Recommender")
    st.write("Enter a product name to get similar product recommendations.")

    with st.form("product_recommendation_form"):
        product_name_input = st.text_input("Enter Product Name", help="e.g., REGENCY CAKESTAND 3 TIER")
        recommend_button = st.form_submit_button("Get Recommendations")

    if recommend_button:
        if product_name_input:
            # Convert to upper for consistency with item_similarity_df index
            recommendations, error_message = get_top_similar_products(product_name_input.strip().upper(), item_similarity_df, n=5)
            if recommendations:
                st.subheader("Recommended Products:")
                for product, score in recommendations:
                    st.markdown(f"-{product} (Similarity: {score:.4f})")
            else:
                st.warning(error_message or "No recommendations found for this product. It might be a new or very rare item.")
        else:
            st.warning("Please enter a product name.")