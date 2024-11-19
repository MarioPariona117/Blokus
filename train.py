import argparse

def train_model(model_name, data_path):
    # Placeholder for training logic
    print(f"Training {model_name} model with data from {data_path}")

def main():
    parser = argparse.ArgumentParser(description='Train a Blokus agent.')
    parser.add_argument('--model', type=str, required=True, help='The name of the model to use')
    parser.add_argument('--data', type=str, required=True, help='The path to the training data')

    args = parser.parse_args()

    train_model(args.model, args.data)

if __name__ == '__main__':
    main()