def clean_data(data):

    data['class'] = (data['class'] == 'g').astype(int)

    print(data)