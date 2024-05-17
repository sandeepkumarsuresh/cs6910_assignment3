import wandb
import pandas as pd

def wandb_plot(path_to_csv,title,prediction_type):
    """
    Uploads a CSV file to Weights & Biases for visualization.

    Args:
        path_to_csv (str): The path to the CSV file to be uploaded.
        title (str): The title of the plot or visualization.
        prediction_type (str): The type of prediction or data represented by the CSV file.

    Example:
        To upload a CSV file for visualization:

        wandb_plot("data.csv", "Prediction Plot", "prediction")
        
    """
    df = pd.read_csv(path_to_csv)
    table = wandb.Table(dataframe=df)

    table_artifact = wandb.Artifact(title,type="dataset")
    table_artifact.add(table,prediction_type)
    table_artifact.add_file(path_to_csv)
    
    run = wandb.init(project='dl_ass3')
    print("inside this")

    run.log({title: table})

    run.log_artifact(table_artifact)

if __name__ == '__main__':
    
    vanilla_correct_pred = 'Predictions/Vannila_correct_predictions.csv'
    vanilla_incorrect_pred = 'Predictions/Vannilla_incorrect_predictions.csv'
    attn_correct_pred = 'Predictions/Attn_correct_predictions.csv'
    attn_incorrect_pred = 'Predictions/Attn_incorrect_predictions.csv'


    wandb_plot(vanilla_correct_pred,"Vannila_Correct_Prediction", "Correct")
    wandb_plot(vanilla_incorrect_pred,"Vannila_InCorrect_Prediction", "InCorrect")
    wandb_plot(attn_correct_pred,"Attention_Correct_Prediction", "Correct")
    wandb_plot(attn_incorrect_pred,"Attention_InCorrect_Prediction", "InCorrect")







