import pandas as pd

RES_PATH = 'results/output/'


def save_predictions(test: pd.DataFrame, filename: str) -> None:
    """
    Save the 'id' and 'prediction' columns of the test DataFrame to a CSV file.
    
    Parameters:
        test (pd.DataFrame): A 1D DataFrame containing only the predictions.
        filename (str): The name of the file where the predictions will be saved.
    """
    model = pd.DataFrame()
    
    model["prediction"] = test
    model['id'] = model.index

    model['prediction'] = model['prediction'].apply(lambda x: max(0, x))
    
    # Reorder the columns to ensure 'id' comes before 'prediction'
    model = model[['id', 'prediction']]
    

    # Save the resulting DataFrame to a CSV file
    model.to_csv(f'{RES_PATH}{filename}.csv', index=False)
    
    # Display the first few rows of the saved DataFrame
    print(model.head())