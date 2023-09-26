import pandas as pd

RES_PATH = 'results/output/'


def save_predictions(test: pd.DataFrame, filename: str) -> None:
    """
    Save the 'id' and 'prediction' columns of the test DataFrame to a CSV file.
    
    Parameters:
        test (pd.DataFrame): The DataFrame containing the predictions.
        filename (str): The name of the file where the predictions will be saved.
    """
    
    # Select the 'id' and 'prediction' columns and apply the lambda function to 'prediction' column
    model = test[['id', 'prediction']].copy()
    model['prediction'] = model['prediction'].apply(lambda x: max(0, x))
    
    # Save the resulting DataFrame to a CSV file
    model.to_csv(f'{RES_PATH}{filename}.csv', index=False)
    
    # Display the first few rows of the saved DataFrame
    print(model.head())