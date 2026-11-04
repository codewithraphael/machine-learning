def clean_data(data):

    data.rename(
        columns={
            'Pregnancies' : 'pregnancies',
            'Glucose' : 'glucose',
            'BloodPressure' : 'blood_pressure',
            'SkinThickness' : 'skin_thickness',
            'Insulin' : 'insulin',
            'BMI' : 'bmi',
            'DiabetesPedigreeFunction' : 'diabetes_pedigree_function',
            'Age' : 'age',
            'Outcome' : 'outcome' 
        },
        inplace=True
    )

    return data