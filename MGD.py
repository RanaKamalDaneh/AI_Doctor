import streamlit as st
import numpy as np
import pandas as pd
import pickle
from IPython.display import Audio


# Load datasets
sym_des = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\symptoms_disease.csv")
description = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\description.csv")
medications = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\medications.csv')
diets = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\diets.csv")

# Load model
svc = pickle.load(open(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\mgp.sav', 'rb'))

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    med = medications[medications['Disease'] == dis]['Medication']
    die = diets[diets['Disease'] == dis]['Diet']
    return desc.values[0], med.values.tolist(), die.values.tolist()

# Dictionary mappings
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Function for model prediction
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]



# Function to play audio (without visible controls)
def play_audio():
        audio_file = open(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\Doctor.mp3',
                          'rb')
        audio_bytes = audio_file.read()
        st.sidebar.audio(audio_bytes, format='audio/mp3', start_time=0)


# Streamlit app
def main():
    st.title('Medical Diagnosis App')

    # Display an image at the top
    st.image(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\AI_Doctor.png", caption='Hi i am Doctor AI how can i assist you ?')

    play_audio()


    # Sidebar for user input (modify this section)
    st.sidebar.header('Enter Symptoms')
    symptom_1 = st.sidebar.text_input('Enter symptom 1:')
    symptom_2 = st.sidebar.text_input('Enter symptom 2:')
    symptom_3 = st.sidebar.text_input('Enter symptom 3:')
    symptom_4 = st.sidebar.text_input('Enter symptom 4:')

    # Collect the symptoms in a list
    user_symptoms = [symptom_1.strip(), symptom_2.strip(), symptom_3.strip(), symptom_4.strip()]

    # Filter out empty strings
    user_symptoms = [s for s in user_symptoms if s]

    if st.sidebar.button('Predict'):
        if user_symptoms:
            # Predict disease based on symptoms
            disease = get_predicted_value(user_symptoms)
            st.subheader(f"**Predicted Disease: {disease}**")

            # Show description, medications, and diet
            description, medications, diet = helper(disease)
            st.write(f"**Description:** {description}")
            st.write(f"**Medications:** {', '.join(medications)}")
            st.write(f"**Diet Recommendations:** {', '.join(diet)}")
        else:
            st.write("Please enter at least one symptom.")

if __name__ == '__main__':
    main()






















































########  الاساسي
# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
#
# # Load datasets
# sym_des = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\symptoms_disease.csv")
# description = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\description.csv")
# medications = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\medications.csv')
# diets = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\diets.csv")
#
# # Load model
# svc = pickle.load(open(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\mgp.sav', 'rb'))
#
# # Helper function
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     med = medications[medications['Disease'] == dis]['Medication']
#     die = diets[diets['Disease'] == dis]['Diet']
#     return desc.values[0], med.values.tolist(), die.values.tolist()
#
# # Dictionary mappings
# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
#
# # Function for model prediction
# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     for item in patient_symptoms:
#         input_vector[symptoms_dict[item]] = 1
#     return diseases_list[svc.predict([input_vector])[0]]
#
# # Streamlit app
# def main():
#     st.title('Medical Diagnosis App')
#
#     # Display an image at the top
#     st.image(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\AI_Doctor.png", caption='Hi i am Doctor AI how can i assist you ?')
#
#     # Sidebar for user input (modify this section)
#     st.sidebar.header('Enter Symptoms')
#     symptom_1 = st.sidebar.text_input('Enter symptom 1:')
#     symptom_2 = st.sidebar.text_input('Enter symptom 2:')
#     symptom_3 = st.sidebar.text_input('Enter symptom 3:')
#     symptom_4 = st.sidebar.text_input('Enter symptom 4:')
#
#     # Collect the symptoms in a list
#     user_symptoms = [symptom_1.strip(), symptom_2.strip(), symptom_3.strip(), symptom_4.strip()]
#
#     # Filter out empty strings
#     user_symptoms = [s for s in user_symptoms if s]
#
#     if st.sidebar.button('Predict'):
#         if user_symptoms:
#             # Predict disease based on symptoms
#             disease = get_predicted_value(user_symptoms)
#             st.subheader(f"**Predicted Disease: {disease}**")
#
#             # Show description, medications, and diet
#             description, medications, diet = helper(disease)
#             st.write(f"**Description:** {description}")
#             st.write(f"**Medications:** {', '.join(medications)}")
#             st.write(f"**Diet Recommendations:** {', '.join(diet)}")
#         else:
#             st.write("Please enter at least one symptom.")
#
# if __name__ == '__main__':
#     main()





############################################ with one search bar

# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
#
# # Load datasets
# sym_des = pd.read_csv(r"D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\dataset\symptoms_disease.csv")
# description = pd.read_csv(r"D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\dataset\description.csv")
# medications = pd.read_csv(r'D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\dataset\medications.csv')
# diets = pd.read_csv(r"D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\dataset\diets.csv")
#
# # Load model
# svc = pickle.load(open(r'D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\mgp.sav', 'rb'))
#
# # Helper function
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     med = medications[medications['Disease'] == dis]['Medication']
#     die = diets[diets['Disease'] == dis]['Diet']
#     return desc.values[0], med.values.tolist(), die.values.tolist()
#
# # Dictionary mappings
# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
#
# # Function for model prediction
# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     for item in patient_symptoms:
#         input_vector[symptoms_dict[item]] = 1
#     return diseases_list[svc.predict([input_vector])[0]]
#
# # Streamlit app
# def main():
#     st.title('Medical Diagnosis App')
#
#     # Display an image at the top
#     st.image(r"D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\AI_Doctor.png", caption='Medical Diagnosis')
#
#     # Sidebar for user input
#     st.sidebar.header('Enter Symptoms')
#     symptoms = st.sidebar.text_input('Enter symptoms separated by commas')
#
#     if st.sidebar.button('Predict'):
#         if symptoms.strip():
#             user_symptoms = [s.strip() for s in symptoms.split(',')]
#             predicted_disease = get_predicted_value(user_symptoms)
#             dis_des, medications, rec_diet = helper(predicted_disease)
#
#             st.subheader(f'Predicted Disease: {predicted_disease}')
#             st.write(f'Description: {dis_des}')
#             st.write(f'Medications: {medications}')
#             st.write(f'Recommended Diet: {rec_diet}')
#         else:
#             st.warning('Please enter symptoms.')
#
#
#
#
# if __name__ == '__main__':
#     main()




#################################################################



# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
#
# # Load datasets
# sym_des = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\symptoms_disease.csv")
# description = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\description.csv")
# medications = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\medications.csv')
# diets = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\diets.csv")
#
# # Load model
# svc = pickle.load(open(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\mgp.sav', 'rb'))
#
# # Helper function
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     med = medications[medications['Disease'] == dis]['Medication']
#     die = diets[diets['Disease'] == dis]['Diet']
#     return desc.values[0], med.values.tolist(), die.values.tolist()
#
# # Dictionary mappings
# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
#
# # Function for model prediction
# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     for item in patient_symptoms:
#         input_vector[symptoms_dict[item]] = 1
#     return diseases_list[svc.predict([input_vector])[0]]
#
# # Function to play the audio
# def play_audio():
#     # ضع مسار ملف الصوت هنا
#     audio_file = r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\Doctor.mp3"
#     st.audio(audio_file, format="audio/mp3")
#
# # Main Streamlit app
# def main():
#     st.title('Medical Disease Prediction Platform')
#     st.image(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\AI_Doctor.png", caption='Hi i am Doctor AI how can i assist you ?')
#
#     # Get user input for symptoms
#     patient_symptoms = st.multiselect('Choose the symptoms', list(symptoms_dict.keys()))
#
#     # Check if symptoms are provided
#     if len(patient_symptoms) > 4:
#         disease = get_predicted_value(patient_symptoms)
#         st.write(f'The predicted disease is {disease}')
#         des, med, die = helper(disease)
#         st.write('Description:', des)
#         st.write('Medications:', ', '.join(med))
#         st.write('Diet:', ', '.join(die))
#
#         # Play audio button
#         if st.button('Play Doctor\'s Voice'):
#             play_audio()
#
# if __name__ == '__main__':
#     main()




###################################

# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
# from IPython.display import Audio
#
# # Load datasets
# sym_des = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\symptoms_disease.csv")
# description = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\description.csv")
# medications = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\medications.csv')
# diets = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\dataset\diets.csv")
#
# # Load model
# svc = pickle.load(open(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\mgp.sav', 'rb'))
#
#
# # Helper function
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     med = medications[medications['Disease'] == dis]['Medication']
#     die = diets[diets['Disease'] == dis]['Diet']
#     return desc.values[0], med.values.tolist(), die.values.tolist()
#
#
# # Dictionary mappings
# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4,
#                  'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9,
#                  'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13,
#                  'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18,
#                  'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22,
#                  'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27,
#                  'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
#                  'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
#                  'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
#                  'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
#                  'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
#                  'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54,
#                  'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
#                  'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
#                  'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
#                  'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70,
#                  'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
#                  'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
#                  'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
#                  'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
#                  'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
#                  'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
#                  'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
#                  'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100,
#                  'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103,
#                  'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
#                  'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110,
#                  'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113,
#                  'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
#                  'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
#                  'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
#                  'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
#                  'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
#                  33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
#                  23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
#                  28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
#                  19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
#                  36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)',
#                  18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
#                  25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis',
#                  0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection',
#                  35: 'Psoriasis', 27: 'Impetigo'}
#
#
# # Function for model prediction
# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     for item in patient_symptoms:
#         input_vector[symptoms_dict[item]] = 1
#     return diseases_list[svc.predict([input_vector])[0]]
#
#
# # Function to play audio (without visible controls)
# def play_audio():
#     audio_file = open(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\Doctor.mp3',
#                       'rb')
#     audio_bytes = audio_file.read()
#
#     st.audio(audio_bytes, format='audio/mp3', start_time=0)
#
#
# # Streamlit app
# def main():
#     st.title('Medical Guidance Platform')
#     st.subheader('Welcome to the medical disease diagnosis platform.')
#
#     # Call audio function
#     play_audio()
#
#     # Input symptoms from user
#     options = st.multiselect('Select your symptoms', list(symptoms_dict.keys()))
#
#     if st.button('Predict Disease'):
#         prediction = get_predicted_value(options)
#         st.success(f"You may have: {prediction}")
#         desc, meds, diet = helper(prediction)
#         st.write(f"**Description:** {desc}")
#         st.write(f"**Medications:** {', '.join(meds)}")
#         st.write(f"**Diet Recommendations:** {', '.join(diet)}")
#
#
# if __name__ == '__main__':
#     main()
#


