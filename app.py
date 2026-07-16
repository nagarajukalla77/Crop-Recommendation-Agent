import streamlit as st
import numpy as np
import joblib
import crop_info

from logger import save_record, load_history


# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="AI Crop Recommendation Agent",
    page_icon="🌱",
    layout="centered"
)


# ---------------------------------
# Load Model and Encoder
# ---------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/best_crop_model.pkl"
    )

    encoder = joblib.load(
        "models/label_encoder.pkl"
    )

    return model, encoder


model, encoder = load_model()



# ---------------------------------
# Title
# ---------------------------------

st.title("🌱 AI Crop Recommendation Agent")

st.write(
    """
    An intelligent farming assistant that recommends
    the best crop based on soil nutrients and climate conditions.
    """
)


st.divider()



# ---------------------------------
# Input Section
# ---------------------------------

st.subheader("🌾 Enter Farm Details")


N = st.number_input(
    "Nitrogen (N)",
    min_value=0,
    max_value=200,
    value=90
)


P = st.number_input(
    "Phosphorus (P)",
    min_value=0,
    max_value=200,
    value=42
)


K = st.number_input(
    "Potassium (K)",
    min_value=0,
    max_value=200,
    value=43
)


temperature = st.number_input(
    "Temperature (°C)",
    value=25.0
)


humidity = st.number_input(
    "Humidity (%)",
    value=80.0
)


ph = st.number_input(
    "Soil pH",
    value=6.5
)


rainfall = st.number_input(
    "Rainfall (mm)",
    value=200.0
)



st.divider()



# ---------------------------------
# Prediction
# ---------------------------------

if st.button("🌱 Recommend Crop"):


    input_data = np.array(
        [[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]]
    )


    prediction = model.predict(
        input_data
    )


    crop = encoder.inverse_transform(
        prediction
    )[0]


    crop = crop.lower()



    # Display Prediction

    st.success(
        f"Recommended Crop: {crop.upper()}"
    )



    # Save Record

    save_record(
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall,
        crop
    )


    st.info(
        "✅ Prediction saved successfully!"
    )



    st.divider()



    # ---------------------------------
    # Crop Information
    # ---------------------------------

    st.subheader("🌾 Crop Information")


    if hasattr(crop_info, "crop_details"):


        details = crop_info.crop_details


        if crop in details:


            st.write(
                "### Description"
            )

            st.write(
                details[crop]["description"]
            )


            st.write(
                "### Growing Conditions"
            )

            st.write(
                details[crop]["conditions"]
            )


            st.write(
                "### Fertilizer Recommendation"
            )

            st.write(
                details[crop]["fertilizer"]
            )


            st.write(
                "### Farming Advice"
            )

            st.write(
                details[crop]["advice"]
            )


        else:


            st.warning(
                "Crop information is not available."
            )


    else:


        st.warning(
            "crop_details not found in crop_info.py"
        )




# ---------------------------------
# Prediction History
# ---------------------------------

st.divider()


st.subheader("📋 Previous Prediction History")


history = load_history()


if not history.empty:


    st.dataframe(
        history,
        use_container_width=True
    )


else:


    st.write(
        "No prediction records available."
    )



# ---------------------------------
# Footer
# ---------------------------------

st.divider()


st.caption(
    "AI Crop Recommendation Agent | Machine Learning + Streamlit 🌱"
)