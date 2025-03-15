import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from keras.models import load_model

# تحميل نموذج التنبؤ بالأعراض
file_name_symptoms = r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MCD\MCD_C.sav'
with open(file_name_symptoms, 'rb') as file:
    svc = pickle.load(file)

# تحميل نموذج تصنيف الصور
model_path = r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MCD\cancer_classification_model.h5'
model = load_model(model_path)

symptoms_dict = {
    'persistent_headaches': 0,
    'unexplained_nausea': 1,
    'unexplained_vomiting': 2,
    'difficulty_with_balance': 3,
    'difficulty_concentrating': 4,
    'memory_loss': 5,
    'seizures': 6,
    'personality_or_behavior_changes': 7,
    'weakness_on_one_side_of_the_body': 8,
    'numbness_on_one_side_of_the_body': 9,
    'painless_lump_in_the_breast': 10,
    'painless_lump_underarm': 11,
    'changes_in_the_size_of_the_breast': 12,
    'changes_in_the_shape_of_the_breast': 13,
    'unusual_nipple_discharge': 14,
    'dimpling_skin': 15,
    'redness_skin': 16,
    'pain_in_the_breast': 17,
    'nipple_pain': 18,
    'inward_turning_of_the_nipple': 19,
    'swelling': 20,
    'hematuria': 21,
    'pain_in_the_side': 22,
    'pain_in_the_back': 23,
    'lump_in_the_side': 24,
    'lump_in_the_abdomen': 25,
    'unexplained_weight_loss': 26,
    'loss_of_appetite': 27,
    'persistent_fatigue': 28,
    'unexplained_fever': 29,
    'anemia': 30,
    'swelling_in_the_ankles': 31,
    'swelling_in_legs': 32,
    'appearance_of_a_new_mole': 33,
    'change_in_an_existing_mole': 34,
    'asymmetrical_mole': 35,
    'irregular_borders_of_the_mole': 36,
    'change_in_color_of_the_mole': 37,
    'increase_in_the_size_of_the_mole': 38,
    'itching_in_the_mole': 39,
    'bleeding_in_the_mole': 40,
    'pain_in_the_mole': 41,
    'peeling_of_the_skin_on_the_mole': 42
}

diseases_list = {
    0: 'Brain Cancer',
    1: 'Breast Cancer',
    2: 'Kidney Cancer',
    3: 'Skin Cancer'
}

label_mapping = {
    0: 'No Brain Tumor',
    1: 'Glioma Tumor',
    2: 'Meningioma Tumor',
    3: 'Pituitary Tumor',
    4: 'Benign Breast Tumor',
    5: 'Malignant Breast Tumor',
    6: 'Kidney Cancer',
    7: 'Normal Kidney'
}

#  لتنبؤ المرض بناءً على الأعراض
def get_predicted_value(patient_symptoms):
    try:
        feature_names = svc.feature_names_in_
    except AttributeError:
        st.error("Model does not have 'feature_names_in_' attribute. Ensure the model was trained with feature names.")
        return

    input_vector = np.zeros(len(feature_names))

    for item in patient_symptoms:
        normalized_item = item.strip().lower().replace(' ', '_')
        if normalized_item in symptoms_dict:
            index = symptoms_dict[normalized_item]
            if index < len(input_vector):
                input_vector[index] = 1

    input_df = pd.DataFrame([input_vector], columns=feature_names)
    predicted_index = svc.predict(input_df)[0]
    return diseases_list[predicted_index]

# الصورة للتنبؤ
def preprocess_image(img, img_type):
    image = Image.open(img)

    if img_type in ['brain', 'kidney']:
        image = image.convert('L')
        image = image.resize((128, 128))
        image = np.array(image)
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)
    elif img_type == 'breast':
        image = image.convert('RGB')
        image = image.resize((128, 128))
        image = np.array(image)

    image = image / 255.0
    input_img = np.expand_dims(image, axis=0)
    return input_img

#  للحصول على نتيجة التنبؤ من الصور
def get_result(img, img_type):
    input_img = preprocess_image(img, img_type)
    prediction = model.predict({'brain_input': input_img, 'breast_input': input_img, 'kidney_input': input_img})
    class_index = np.argmax(prediction, axis=1)[0]
    probability = prediction[0][class_index]
    return class_index, probability

#  للحصول على اسم الفئة
def get_class_name(class_no):
    return label_mapping[class_no]


# Function to play audio (without visible controls)
def play_audio():
        audio_file = open(r'C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\Doctor.mp3',
                          'rb')
        audio_bytes = audio_file.read()
        st.sidebar.audio(audio_bytes, format='audio/mp3', start_time=0)



#  Streamlit
def main():
    st.title('Medical Diagnosis App')

    # Display an image at the top
    st.image(r"C:\Users\DELL\OneDrive\Desktop\Pro\Medical Guidance Platform\MGD\AI_Doctor.png", caption='Hi i am Doctor AI how can i assist you ?')

    play_audio()


    #  لتخزين المعلومات
    if 'predicted_disease' not in st.session_state:
        st.session_state.predicted_disease = None

    #  إدخال الأعراض
    st.sidebar.header('Enter Symptoms')
    symptom_1 = st.sidebar.text_input('Enter symptom 1:')
    symptom_2 = st.sidebar.text_input('Enter symptom 2:')
    symptom_3 = st.sidebar.text_input('Enter symptom 3:')
    symptom_4 = st.sidebar.text_input('Enter symptom 4:')

    if st.sidebar.button('Predict'):
        symptoms = [symptom_1, symptom_2, symptom_3, symptom_4]
        symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms if s]
        if symptoms:
            predicted_disease = get_predicted_value(symptoms)
            if predicted_disease:
                st.session_state.predicted_disease = predicted_disease
                st.subheader(f'Predicted Disease based on Symptoms: {predicted_disease}')
                st.write("Now, upload an MRI or medical image to classify its cancer type and likelihood.")
            else:
                st.warning('Unable to predict disease based on symptoms.')
        else:
            st.warning('Please enter at least one symptom.')

    #  الصورة
    if st.session_state.predicted_disease is not None:
        uploaded_file = st.file_uploader("Choose an MRI or medical image...", type=["jpg", "jpeg", "png"])
        image_type = st.selectbox("Select the image type:", ('brain', 'breast', 'kidney'))

        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write("")

            if st.button('Predict Image'):
                st.write("Classifying...")
                result, probability = get_result(uploaded_file, image_type)
                prediction = get_class_name(result)

                st.write(f'Result: {prediction}')
                st.write(f'Probability: {probability * 100:.2f}%')

                if result > 0:
                    if result in [1, 2, 3]:
                        st.write(f'Type of brain tumor: {prediction}')
                    elif result in [4, 5]:
                        st.write(f'Type of breast tumor: {prediction}')
                    elif result == 6:
                        st.write(f'Type of kidney condition: {prediction}')
                else:
                    st.write("No cancer detected in the image.")

if __name__ == '__main__':
    main()































########## ZaRo
# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
# from PIL import Image
# from keras.models import load_model
#
# # تحميل نموذج التنبؤ بالأعراض
# file_name_symptoms = 'D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\MCD\MCD_C.sav'
# with open(file_name_symptoms, 'rb') as file:
#     svc = pickle.load(file)
#
# # تحميل نموذج تصنيف الصور
# model_path = r'D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\MCD\cancer_classification_model.h5'
# model = load_model(model_path)
#
#
# symptoms_dict = {
#     'persistent_headaches': 0,
#     'unexplained_nausea': 1,
#     'unexplained_vomiting': 2,
#     'difficulty_with_balance': 3,
#     'difficulty_concentrating': 4,
#     'memory_loss': 5,
#     'seizures': 6,
#     'personality_or_behavior_changes': 7,
#     'weakness_on_one_side_of_the_body': 8,
#     'numbness_on_one_side_of_the_body': 9,
#     'painless_lump_in_the_breast': 10,
#     'painless_lump_underarm': 11,
#     'changes_in_the_size_of_the_breast': 12,
#     'changes_in_the_shape_of_the_breast': 13,
#     'unusual_nipple_discharge': 14,
#     'dimpling_skin': 15,
#     'redness_skin': 16,
#     'pain_in_the_breast': 17,
#     'nipple_pain': 18,
#     'inward_turning_of_the_nipple': 19,
#     'swelling': 20,
#     'hematuria': 21,
#     'pain_in_the_side': 22,
#     'pain_in_the_back': 23,
#     'lump_in_the_side': 24,
#     'lump_in_the_abdomen': 25,
#     'unexplained_weight_loss': 26,
#     'loss_of_appetite': 27,
#     'persistent_fatigue': 28,
#     'unexplained_fever': 29,
#     'anemia': 30,
#     'swelling_in_the_ankles': 31,
#     'swelling_in_legs': 32,
#     'appearance_of_a_new_mole': 33,
#     'change_in_an_existing_mole': 34,
#     'asymmetrical_mole': 35,
#     'irregular_borders_of_the_mole': 36,
#     'change_in_color_of_the_mole': 37,
#     'increase_in_the_size_of_the_mole': 38,
#     'itching_in_the_mole': 39,
#     'bleeding_in_the_mole': 40,
#     'pain_in_the_mole': 41,
#     'peeling_of_the_skin_on_the_mole': 42
# }
#
# diseases_list = {
#     0: 'Brain Cancer',
#     1: 'Breast Cancer',
#     2: 'Kidney Cancer',
#     3: 'Skin Cancer'
# }
#
# label_mapping = {
#     0: 'No Brain Tumor',
#     1: 'Glioma Tumor',
#     2: 'Meningioma Tumor',
#     3: 'Pituitary Tumor',
#     4: 'Benign Breast Tumor',
#     5: 'Malignant Breast Tumor',
#     6: 'Kidney Cancer',
#     7: 'Normal Kidney'
# }
#
# #  لتنبؤ المرض بناءً على الأعراض
# def get_predicted_value(patient_symptoms):
#     try:
#         feature_names = svc.feature_names_in_
#     except AttributeError:
#         st.error("Model does not have 'feature_names_in_' attribute. Ensure the model was trained with feature names.")
#         return
#
#     input_vector = np.zeros(len(feature_names))
#
#     for item in patient_symptoms:
#         normalized_item = item.strip().lower().replace(' ', '_')
#         if normalized_item in symptoms_dict:
#             index = symptoms_dict[normalized_item]
#             if index < len(input_vector):
#                 input_vector[index] = 1
#
#     input_df = pd.DataFrame([input_vector], columns=feature_names)
#     predicted_index = svc.predict(input_df)[0]
#     return diseases_list[predicted_index]
#
# # الصورة للتنبؤ
# def preprocess_image(img, img_type):
#     image = Image.open(img)
#
#     if img_type in ['brain', 'kidney']:
#         image = image.convert('L')
#         image = image.resize((128, 128))
#         image = np.array(image)
#         image = np.expand_dims(image, axis=-1)
#         image = np.repeat(image, 3, axis=-1)
#     elif img_type == 'breast':
#         image = image.convert('RGB')
#         image = image.resize((128, 128))
#         image = np.array(image)
#
#     image = image / 255.0
#     input_img = np.expand_dims(image, axis=0)
#     return input_img
#
# #  للحصول على نتيجة التنبؤ من الصور
# def get_result(img, img_type):
#     input_img = preprocess_image(img, img_type)
#     prediction = model.predict({'brain_input': input_img, 'breast_input': input_img, 'kidney_input': input_img})
#     class_index = np.argmax(prediction, axis=1)[0]
#     probability = prediction[0][class_index]
#     return class_index, probability
#
# #  للحصول على اسم الفئة
# def get_class_name(class_no):
#     return label_mapping[class_no]
#
# #  Streamlit
# def main():
#     st.title(' Medical Cancer Diagnosis & Image Classification App')
#
#     #  لتخزين المعلومات
#     if 'predicted_disease' not in st.session_state:
#         st.session_state.predicted_disease = None
#
#     #  إدخال الأعراض
#     st.sidebar.header('Enter Symptoms')
#     symptoms = st.sidebar.text_input('Enter symptoms separated by commas')
#
#     if st.sidebar.button('Predict Symptoms'):
#         if symptoms.strip():
#             user_symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms.split(',')]
#             predicted_disease = get_predicted_value(user_symptoms)
#             if predicted_disease:
#                 st.session_state.predicted_disease = predicted_disease
#                 st.subheader(f'Predicted Disease based on Symptoms: {predicted_disease}')
#                 st.write("Now, upload an MRI or medical image to classify its cancer type and likelihood.")
#             else:
#                 st.warning('Unable to predict disease based on symptoms.')
#         else:
#             st.warning('Please enter symptoms.')
#
#     # قسم تحميل الصورة
#     if st.session_state.predicted_disease is not None:
#         uploaded_file = st.file_uploader("Choose an MRI or medical image...", type=["jpg", "jpeg", "png"])
#         image_type = st.selectbox("Select the image type:", ('brain', 'breast', 'kidney'))
#
#         if uploaded_file is not None:
#             st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
#             st.write("")
#
#             if st.button('Predict Image'):
#                 st.write("Classifying...")
#                 result, probability = get_result(uploaded_file, image_type)
#                 prediction = get_class_name(result)
#
#                 st.write(f'Result: {prediction}')
#                 st.write(f'Probability: {probability * 100:.2f}%')
#
#                 if result > 0:
#                     if result in [1, 2, 3]:
#                         st.write(f'Type of brain tumor: {prediction}')
#                     elif result in [4, 5]:
#                         st.write(f'Type of breast tumor: {prediction}')
#                     elif result == 6:
#                         st.write(f'Type of kidney condition: {prediction}')
#                 else:
#                     st.write("No cancer detected in the image.")
#
# if __name__ == '__main__':
#     main()






#############################################################The final code

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
# svc = pickle.load(open(r'D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\mgp.sav', 'rb'))
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
# if __name__ == '__main__':
#     main()
#
#

############################################################# the code that we will use
















########################################################################3



#
# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
# from PIL import Image
# from keras.models import load_model
#
# # Load datasets
# sym_des = pd.read_csv(r"D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\dataset\symptoms_disease.csv")
# description = pd.read_csv(r"D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\dataset\description.csv")
# medications = pd.read_csv(r'D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\dataset\medications.csv')
# diets = pd.read_csv(r"D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\dataset\diets.csv")
#
# # Load models
# svc_path = r'D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\mgp.sav'
# model_path = r'D:\one drive\OneDrive\Desktop\Pro\Cancers\brain&breast\cancer_classification_model.h5'
#
# svc = pickle.load(open(svc_path, 'rb'))
# model = load_model(model_path)
#
# # Helper function for disease info
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     med = medications[medications['Disease'] == dis]['Medication']
#     die = diets[diets['Disease'] == dis]['Diet']
#     return desc.values[0], med.values.tolist(), die.values.tolist()
#
# # Dictionaries for symptoms and diseases
# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
#
#
#
#
# # Function for predicting disease from symptoms
# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     for item in patient_symptoms:
#         input_vector[symptoms_dict[item]] = 1
#     return diseases_list[svc.predict([input_vector])[0]]
#
# # Preprocess image for model input
# def preprocess_image(img, img_type):
#     image = Image.open(img)
#     if img_type in ['brain', 'kidney']:
#         image = image.convert('L')
#         image = image.resize((128, 128))
#         image = np.array(image)
#         image = np.expand_dims(image, axis=-1)
#         image = np.repeat(image, 3, axis=-1)
#     elif img_type == 'breast':
#         image = image.convert('RGB')
#         image = image.resize((128, 128))
#         image = np.array(image)
#     image = image / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image
#
# # Predict cancer type from image
# def classify_cancer_image(img, img_type):
#     processed_image = preprocess_image(img, img_type)
#     prediction = model.predict(processed_image)
#     predicted_label = np.argmax(prediction)
#     return label_mapping[predicted_label]
#
# # Streamlit app
# def main():
#     st.title('Medical Guidance Platform')
#
#     # Sidebar for disease prediction from symptoms
#     st.sidebar.header('Enter Symptoms')
#     symptoms = st.sidebar.text_input('Enter symptoms separated by commas')
#
#     if st.sidebar.button('Predict Disease'):
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
#     # Image upload for cancer type classification
#     st.header('Upload Image for Cancer Classification')
#     uploaded_image = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
#     img_type = st.selectbox('Select Image Type', ['brain', 'breast', 'kidney'])
#
#     if st.button('Classify Cancer Image'):
#         if uploaded_image:
#             result = classify_cancer_image(uploaded_image, img_type)
#             st.subheader(f'Predicted Cancer Type: {result}')
#         else:
#             st.warning('Please upload an image.')
#
# if __name__ == '__main__':
#     main()
#












# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
#
# # Load model
# svc = pickle.load(open(r"D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\MCD_C.sav", 'rb'))
#
# # Dictionary mappings
# symptoms_dict = {
#     'persistent_headaches': 0,
#     'unexplained_nausea': 1,
#     'unexplained_vomiting': 2,
#     'difficulty_with_balance': 3,
#     'difficulty_concentrating': 4,
#     'memory_loss': 5,
#     'seizures': 6,
#     'personality_or_behavior_changes': 7,
#     'weakness_on_one_side_of_the_body': 8,
#     'numbness_on_one_side_of_the_body': 9,
#     'painless_lump_in_the_breast': 10,
#     'painless_lump_underarm': 11,
#     'changes_in_the_size_of_the_breast': 12,
#     'changes_in_the_shape_of_the_breast': 13,
#     'unusual_nipple_discharge': 14,
#     'dimpling_skin': 15,
#     'redness_skin': 16,
#     'pain_in_the_breast': 17,
#     'nipple_pain': 18,
#     'inward_turning_of_the_nipple': 19,
#     'swelling': 20,
#     'hematuria': 21,
#     'pain_in_the_side': 22,
#     'pain_in_the_back': 23,
#     'lump_in_the_side': 24,
#     'lump_in_the_abdomen': 25,
#     'unexplained_weight_loss': 26,
#     'loss_of_appetite': 27,
#     'persistent_fatigue': 28,
#     'unexplained_fever': 29,
#     'anemia': 30,
#     'swelling_in_the_ankles': 31,
#     'swelling_in_legs': 32,
#     'appearance_of_a_new_mole': 33,
#     'change_in_an_existing_mole': 34,
#     'asymmetrical_mole': 35,
#     'irregular_borders_of_the_mole': 36,
#     'change_in_color_of_the_mole': 37,
#     'increase_in_the_size_of_the_mole': 38,
#     'itching_in_the_mole': 39,
#     'bleeding_in_the_mole': 40,
#     'pain_in_the_mole': 41,
#     'peeling_of_the_skin_on_the_mole': 42
# }
# diseases_list = {
#     0: 'Brain Cancer',
#     1: 'Breast Cancer',
#     2: 'Kidney Cancer',
#     3: 'Skin Cancer'
# }
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
#     # Sidebar for user input
#     st.sidebar.header('Enter Symptoms')
#     symptoms = st.sidebar.text_input('Enter symptoms separated by commas')
#
#     if st.sidebar.button('Predict'):
#         if symptoms.strip():
#             user_symptoms = [s.strip() for s in symptoms.split(',')]
#             predicted_disease = get_predicted_value(user_symptoms)
#
#             st.subheader(f'Predicted Disease: {predicted_disease}')
#         else:
#             st.warning('Please enter symptoms.')
#
# if __name__ == '__main__':
#     main()

########################################################################
# import streamlit as st
# import numpy as np
# import pickle
#
# # Load model
# svc = pickle.load(open(r"D:\one drive\OneDrive\Desktop\Pro\Medical Guidance Platform\MCD_C.sav", 'rb'))
#
# # Dictionary mappings
# symptoms_dict = {
#     'persistent_headaches': 0,
#     'unexplained_nausea': 1,
#     'unexplained_vomiting': 2,
#     'difficulty_with_balance': 3,
#     'difficulty_concentrating': 4,
#     'memory_loss': 5,
#     'seizures': 6,
#     'personality_or_behavior_changes': 7,
#     'weakness_on_one_side_of_the_body': 8,
#     'numbness_on_one_side_of_the_body': 9,
#     'painless_lump_in_the_breast': 10,
#     'painless_lump_underarm': 11,
#     'changes_in_the_size_of_the_breast': 12,
#     'changes_in_the_shape_of_the_breast': 13,
#     'unusual_nipple_discharge': 14,
#     'dimpling_skin': 15,
#     'redness_skin': 16,
#     'pain_in_the_breast': 17,
#     'nipple_pain': 18,
#     'inward_turning_of_the_nipple': 19,
#     'swelling': 20,
#     'hematuria': 21,
#     'pain_in_the_side': 22,
#     'pain_in_the_back': 23,
#     'lump_in_the_side': 24,
#     'lump_in_the_abdomen': 25,
#     'unexplained_weight_loss': 26,
#     'loss_of_appetite': 27,
#     'persistent_fatigue': 28,
#     'unexplained_fever': 29,
#     'anemia': 30,
#     'swelling_in_the_ankles': 31,
#     'swelling_in_legs': 32,
#     'appearance_of_a_new_mole': 33,
#     'change_in_an_existing_mole': 34,
#     'asymmetrical_mole': 35,
#     'irregular_borders_of_the_mole': 36,
#     'change_in_color_of_the_mole': 37,
#     'increase_in_the_size_of_the_mole': 38,
#     'itching_in_the_mole': 39,
#     'bleeding_in_the_mole': 40,
#     'pain_in_the_mole': 41,
#     'peeling_of_the_skin_on_the_mole': 42
# }
# diseases_list = {
#     0: 'Brain Cancer',
#     1: 'Breast Cancer',
#     2: 'Kidney Cancer',
#     3: 'Skin Cancer'
# }
#
#
# # Function for model prediction
# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     unknown_symptoms = []
#     for item in patient_symptoms:
#         if item in symptoms_dict:
#             input_vector[symptoms_dict[item]] = 1
#         else:
#             unknown_symptoms.append(item)
#
#     if unknown_symptoms:
#         st.warning(f'Unknown symptoms: {", ".join(unknown_symptoms)}')
#         return None
#     return diseases_list[svc.predict([input_vector])[0]]
#
#
# # Streamlit app
# def main():
#     st.title('Medical Diagnosis App')
#
#     # Sidebar for user input
#     st.sidebar.header('Enter Symptoms')
#     symptoms = st.sidebar.text_input('Enter symptoms separated by commas')
#
#     if st.sidebar.button('Predict'):
#         if symptoms.strip():
#             user_symptoms = [s.strip() for s in symptoms.split(',')]
#             predicted_disease = get_predicted_value(user_symptoms)
#
#             if predicted_disease:
#                 st.subheader(f'Predicted Disease: {predicted_disease}')
#         else:
#             st.warning('Please enter symptoms.')
#
#
# if __name__ == '__main__':
#     main()


############################################################

#
# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
#
# # Load the trained model
# file_name = 'MCD_C.sav'
# with open(file_name, 'rb') as file:
#     svc = pickle.load(file)
#
# # Retrieve feature names used during training (if available)
# try:
#     feature_names = svc.feature_names_in_
# except AttributeError:
#     # Handle cases where feature_names_in_ is not available
#     st.error("Model does not have 'feature_names_in_' attribute. Ensure the model was trained with feature names.")
#     feature_names = list(symptoms_dict.keys())
#
# # Define the symptoms_dict and diseases_list
# symptoms_dict = {
#     'persistent_headaches': 0,
#     'unexplained_nausea': 1,
#     'unexplained_vomiting': 2,
#     'difficulty_with_balance': 3,
#     'difficulty_concentrating': 4,
#     'memory_loss': 5,
#     'seizures': 6,
#     'personality_or_behavior_changes': 7,
#     'weakness_on_one_side_of_the_body': 8,
#     'numbness_on_one_side_of_the_body': 9,
#     'painless_lump_in_the_breast': 10,
#     'painless_lump_underarm': 11,
#     'changes_in_the_size_of_the_breast': 12,
#     'changes_in_the_shape_of_the_breast': 13,
#     'unusual_nipple_discharge': 14,
#     'dimpling_skin': 15,
#     'redness_skin': 16,
#     'pain_in_the_breast': 17,
#     'nipple_pain': 18,
#     'inward_turning_of_the_nipple': 19,
#     'swelling': 20,
#     'hematuria': 21,
#     'pain_in_the_side': 22,
#     'pain_in_the_back': 23,
#     'lump_in_the_side': 24,
#     'lump_in_the_abdomen': 25,
#     'unexplained_weight_loss': 26,
#     'loss_of_appetite': 27,
#     'persistent_fatigue': 28,
#     'unexplained_fever': 29,
#     'anemia': 30,
#     'swelling_in_the_ankles': 31,
#     'swelling_in_legs': 32,
#     'appearance_of_a_new_mole': 33,
#     'change_in_an_existing_mole': 34,
#     'asymmetrical_mole': 35,
#     'irregular_borders_of_the_mole': 36,
#     'change_in_color_of_the_mole': 37,
#     'increase_in_the_size_of_the_mole': 38,
#     'itching_in_the_mole': 39,
#     'bleeding_in_the_mole': 40,
#     'pain_in_the_mole': 41,
#     'peeling_of_the_skin_on_the_mole': 42
# }
# diseases_list = {
#     0: 'Brain Cancer',
#     1: 'Breast Cancer',
#     2: 'Kidney Cancer',
#     3: 'Skin Cancer'
# }
#
#
# # Function to predict disease based on symptoms
# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(feature_names))
#
#     for item in patient_symptoms:
#         normalized_item = item.strip().lower().replace(' ', '_')
#         if normalized_item in symptoms_dict:
#             index = symptoms_dict[normalized_item]
#             if index < len(input_vector):
#                 input_vector[index] = 1
#
#     input_df = pd.DataFrame([input_vector], columns=feature_names)
#     predicted_index = svc.predict(input_df)[0]
#     return diseases_list[predicted_index]
#
#
# # Streamlit app
# def main():
#     st.title('Medical Diagnosis & Cancer Classification App ')
#
#     # Sidebar for user input
#     st.sidebar.header('Enter Symptoms')
#     symptoms = st.sidebar.text_input('Enter symptoms separated by commas')
#
#     if st.sidebar.button('Predict'):
#         if symptoms.strip():
#             user_symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms.split(',')]
#             predicted_disease = get_predicted_value(user_symptoms)
#
#             st.subheader(f'Predicted Disease: {predicted_disease}')
#         else:
#             st.warning('Please enter symptoms.')
#
#
# if __name__ == '__main__':
#     main()
#
#
#
# import os
# import numpy as np
# from PIL import Image
# from keras.models import load_model
# import streamlit as st
#
# # تحميل النموذج المدمج
# model = load_model(r'D:\one drive\OneDrive\Desktop\Pro\Cancers\brain&breast\cancer_classification_model.h5')
#
# # عنوان التطبيق
# # st.title('Cancer Classification Application')
# st.text('Upload an MRI or medical image to classify its cancer type and likelihood.')
#
#
# # الدالة لتحديد اسم الفئة
# def get_class_name(class_no, label_mapping):
#     return label_mapping[class_no]
#
#
# # الدالة لتحضير الصورة للتنبؤ
# def preprocess_image(img, img_type):
#     image = Image.open(img)
#
#     # تحويل الصورة للتدرج الرمادي ثم إلى RGB
#     if img_type == 'brain' or img_type == 'kidney':  # الصور التي تكون في البداية بتدرج الرمادي
#         image = image.convert('L')
#         image = image.resize((128, 128))
#         image = np.array(image)
#         image = np.expand_dims(image, axis=-1)
#         image = np.repeat(image, 3, axis=-1)  # تحويل الصورة من التدرج الرمادي إلى RGB
#     elif img_type == 'breast':  # الصور التي تكون بالفعل RGB
#         image = image.convert('RGB')
#         image = image.resize((128, 128))
#         image = np.array(image)
#
#     image = image / 255.0  # تطبيع الصورة
#     input_img = np.expand_dims(image, axis=0)
#     return input_img
#
#
# # الدالة للحصول على نتيجة التنبؤ
# def get_result(img, img_type, label_mapping):
#     input_img = preprocess_image(img, img_type)
#     prediction = model.predict({'brain_input': input_img, 'breast_input': input_img, 'kidney_input': input_img})
#     class_index = np.argmax(prediction, axis=1)[0]
#     probability = prediction[0][class_index]
#     return class_index, probability
#
#
# # تحديد التسميات للفئات المختلفة بناءً على النموذج
# label_mapping = {
#     0: 'No Brain Tumor',  # لا يوجد ورم في الدماغ
#     1: 'Glioma Tumor',  # ورم الدبقيات (نوع من سرطان الدماغ)
#     2: 'Meningioma Tumor',  # ورم السحايا (نوع من سرطان الدماغ)
#     3: 'Pituitary Tumor',  # ورم الغدة النخامية (نوع من سرطان الدماغ)
#     4: 'Benign Breast Tumor',  # ورم الثدي الحميد
#     5: 'Malignant Breast Tumor',  # ورم الثدي الخبيث
#     6: 'Kidney Cancer',  # سرطان الكلى
#     7: 'Normal Kidney'  # كلى سليمة
# }
#
# # تحميل الصورة من واجهة المستخدم
# uploaded_file = st.file_uploader("Choose an MRI or medical image...", type=["jpg", "jpeg", "png"])
#
# # تحديد نوع الصورة (دماغ، ثدي، كلى)
# image_type = st.selectbox("Select the image type:", ('brain', 'breast', 'kidney'))
#
# if uploaded_file is not None:
#     # عرض الصورة
#     st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#
#     if st.button('Predict'):
#         st.write("Classifying...")
#         # الحصول على نتيجة التنبؤ
#         result, probability = get_result(uploaded_file, image_type, label_mapping)
#         prediction = get_class_name(result, label_mapping)
#         st.write(f'Result: {prediction}')
#         st.write(f'Probability: {probability * 100:.2f}%')
#
#         # تقديم تفاصيل إضافية بناءً على نوع السرطان
#         if result > 0:  # إذا كانت النتيجة غير "No Brain Tumor" أو "Normal Kidney"
#             if result in [1, 2, 3]:  # الأنواع المتعلقة بالدماغ
#                 st.write("Detected a brain tumor.")
#                 st.write(f'Type of brain tumor: {prediction}')
#             elif result in [4, 5]:  # الأنواع المتعلقة بالثدي
#                 st.write("Detected a breast tumor.")
#                 st.write(f'Type of breast tumor: {prediction}')
#             elif result == 6:  # سرطان الكلى
#                 st.write("Detected a kidney cancer.")
#                 st.write(f'Type of kidney condition: {prediction}')
#         else:
#             st.write("No cancer detected in the image.")
