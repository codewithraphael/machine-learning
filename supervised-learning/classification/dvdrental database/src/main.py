from config import DATA_DIR
from data_loader import load_data
from data_validation import validate_data







def main():

    data = load_data(DATA_DIR)
    validate_data(data)



if __name__ == '__main__':
    main()