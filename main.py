import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# --- Module 1: Data Loading ---
@st.cache_data
def load_data():
    """Loads all data files and the trained model."""
    try:
        df = pd.read_csv('datasets/Training.csv')
        sym_des = pd.read_csv('datasets/Symptom-severity.csv')
        precaution = pd.read_csv('datasets/precautions_df.csv')
        workout = pd.read_csv('datasets/workout_df.csv')
        description = pd.read_csv('datasets/description.csv')
        medication = pd.read_csv('datasets/medications.csv')
        diet = pd.read_csv('datasets/diets.csv')

        # Load the trained model.
        # Note: You may see an InconsistentVersionWarning here.
        # To fix it, ensure your scikit-learn version matches the
        # version used to train and save the model (e.g., pip install scikit-learn==<version>).
        with open('datasets/svc.pkl', 'rb') as f:
            svc = pickle.load(f)

        le = LabelEncoder()
        le.fit(df['prognosis'])

        # Get the feature names directly from the training data columns.
        # This is the most crucial step to ensure alignment.
        symptom_columns = df.drop('prognosis', axis=1).columns.tolist()

        return df, sym_des, precaution, workout, description, medication, diet, svc, le, symptom_columns
    except FileNotFoundError as e:
        st.error(f"Missing data file: {e}. Please ensure all files are in the 'datasets/' directory.")
        st.stop()


# --- Module 2: Model Prediction Logic (Corrected) ---
def get_predicted_values(user_symptoms, symptom_columns, svc, le):
    """Predicts a disease based on user-selected symptoms."""
    # Initialize a zero vector with the correct number of features
    input_vector = np.zeros(len(symptom_columns))

    # Fill the vector with 1s for the selected symptoms
    symptom_to_index = {symptom: i for i, symptom in enumerate(symptom_columns)}
    for symptom in user_symptoms:
        symptom_clean = symptom.strip().replace(" ", "_").lower()
        if symptom_clean in symptom_to_index:
            input_vector[symptom_to_index[symptom_clean]] = 1

    # To fix the 'X does not have valid feature names' warning,
    # create a DataFrame from the input vector with the correct feature names.
    input_df = pd.DataFrame([input_vector], columns=symptom_columns)
    
    # Pass the DataFrame to the predict method.
    prediction = svc.predict(input_df)
    
    # Inverse transform the prediction to get the disease name
    predicted_disease = le.inverse_transform(prediction)
    return predicted_disease[0]


# --- Module 3: Main UI Layout and App Logic (Corrected) ---
def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Disease Prediction App", layout="wide")
    
    # Load all the data and the model
    df, sym_des, precaution, workout, description, medication, diet, svc, le, symptom_columns = load_data()
    
    # --- Title and Description ---
    st.title("Symptom-Based Disease Prediction")
    st.write("Enter your symptoms below to get a predicted disease and related information.")
    st.markdown("---")
    
    # --- UI for Symptom Input ---
    available_symptoms = [col.replace('_', ' ').title() for col in symptom_columns]

    user_symptoms_raw = st.multiselect(
        "Select your symptoms:",
        options=available_symptoms,
        help="You can select multiple symptoms from the list."
    )
    
    user_symptoms_processed = [s.replace(' ', '_').lower() for s in user_symptoms_raw]

    if st.button("Predict Disease"):
        if not user_symptoms_processed:
            st.warning("Please select at least one symptom.")
        else:
            predicted_disease = get_predicted_values(user_symptoms_processed, symptom_columns, svc, le)
            
            # --- Display Prediction and other info ---
            st.markdown("### Prediction Result")
            st.success(f"**Predicted Disease:** {predicted_disease}")
            st.markdown("---")
            st.markdown("### Related Information")
            
            with st.expander("Description"):
                try:
                    disease_description = description[description['Disease'] == predicted_disease]['Description'].iloc[0]
                    st.write(disease_description)
                except IndexError:
                    st.warning("No description found for this disease.")
            
            with st.expander("Precautions"):
                try:
                    disease_precautions = precaution[precaution['Disease'] == predicted_disease].iloc[0, 2:].tolist()
                    for item in disease_precautions:
                        if pd.notna(item):
                            st.write(f"- {item}")
                except IndexError:
                    st.warning("No precautions found for this disease.")
            
            with st.expander("Medications"):
                try:
                    disease_medications = medication[medication['Disease'] == predicted_disease]['Medication'].iloc[0]
                    st.write(disease_medications)
                except IndexError:
                    st.warning("No medication information found for this disease.")

            with st.expander("Diet Recommendations"):
                try:
                    disease_diet = diet[diet['Disease'] == predicted_disease]['Diet'].iloc[0]
                    st.write(disease_diet)
                except IndexError:
                    st.warning("No diet recommendations found for this disease.")
            
            with st.expander("Workout Plans"):
                try:
                    workout_plans = workout[workout['disease'] == predicted_disease]['workout'].tolist()
                    for item in workout_plans:
                        st.write(f"- {item}")
                except IndexError:
                    st.warning("No workout plans found for this disease.")
    
    # --- HERE IS THE FIX for the AttributeError ---
    if st.button("Clear Selections"):
        st.rerun()

if __name__ == "__main__":
    main()