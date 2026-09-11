# production.py
import numpy as np
import pandas as pd
import csv
import joblib

#Input: a string csv_path holding the path to an excel file
#Output: A dataframe with the excel file loaded into it
def load_data(csv_path):
    with open(csv_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if "CATCHER_ID" in line or "PITCHCALL" in line:
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Header row not found.")

    df = pd.read_csv(csv_path,
                     skiprows=start_idx,
                     quoting=csv.QUOTE_ALL,
                     na_values=["NULL","null","NaN","nan",""],
                     skip_blank_lines=True)

    df = df.rename(columns=lambda x: x.strip())
    df.dropna(how="all", inplace=True)
    return df


# Load data for predictions
df = load_data("new_data.csv")

# Load model file 
model = joblib.load("model.pkl")

# Build CS_Probability for actual strike indicator
df["CS_Probability"] = np.where(df["PITCHCALL"] == "StrikeCalled", 1, 0)

# Select same features used when training
X = df[['PLATELOCHEIGHT', 'PLATELOCSIDE', 'BOT_ZONE', 'TOP_ZONE']]

# Predict strike probability
pred_probs = model.predict_proba(X)[:, 1]

# Pitch-level output
pitch_output = pd.DataFrame({
    "PITCH_ID": df["PITCH_ID"],
    "IS_STRIKE": df["CS_Probability"],
    "CS_PROB": np.round(pred_probs, 3)
})

pitch_output.to_csv("pitch_level_predictions.csv", index=False)


# ---- Catcher Framing Value Output ----
df["PRED_PROB"] = pred_probs

pitch_counts = df.groupby(["CATCHER_ID", "GAME_YEAR"])["PITCHCALL"].count()
strike_calls = df.groupby(["CATCHER_ID", "GAME_YEAR"])["CS_Probability"].sum()
added_calls = df.groupby(["CATCHER_ID", "GAME_YEAR"])["PRED_PROB"].sum()

summary_df = pd.DataFrame({
    "OPPORTUNITIES": pitch_counts,
    "ACTUAL_CALLED_STRIKES": strike_calls,
    "CALLED_STRIKES_ADDED": added_calls
})

summary_df["CALLED_STRIKES_ADDED_PER_100"] = (
    summary_df["CALLED_STRIKES_ADDED"] / summary_df["OPPORTUNITIES"] * 100
).round(3)

summary_df.reset_index().to_csv("new_output.csv", index=False)

print("Saved pitch_level_predictions.csv and new_output.csv")
