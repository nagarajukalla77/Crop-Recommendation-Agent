import pandas as pd
import os
from datetime import datetime


HISTORY_FILE = "prediction_history.csv"


def save_record(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    crop
):

    data = {

        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "Nitrogen": N,
        "Phosphorus": P,
        "Potassium": K,

        "Temperature": temperature,
        "Humidity": humidity,
        "pH": ph,
        "Rainfall": rainfall,

        "Recommended Crop": crop

    }


    new_record = pd.DataFrame([data])


    if os.path.exists(HISTORY_FILE):

        old_records = pd.read_csv(HISTORY_FILE)

        updated_records = pd.concat(
            [old_records, new_record],
            ignore_index=True
        )

    else:

        updated_records = new_record


    updated_records.to_csv(
        HISTORY_FILE,
        index=False
    )



def load_history():

    if os.path.exists(HISTORY_FILE):

        return pd.read_csv(HISTORY_FILE)

    return pd.DataFrame()