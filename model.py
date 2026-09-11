import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import csv
import random

# Set random seed for reproducibility, or use a random seed
#rand = 44
rand = random.randint(1, 10000)

'''
#Imports for tuning and evaluation 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

'''





####################################################
# PLOT FUNCTIONS USED FOR OPTIONAL FEATURE ANALYSIS
####################################################

def plot_feature_importance(model1, model2, X, y, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    tree_importance_sorted_idx = np.argsort(model1.feature_importances_)
    tree_indices = np.arange(0, len(model1.feature_importances_)) + 0.5
    ax1.barh(tree_indices, model1.feature_importances_[tree_importance_sorted_idx], height=0.7, color='#B2D7D0')
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(np.array(feature_names)[tree_importance_sorted_idx], fontsize=12)
    ax1.set_ylim((0, len(model1.feature_importances_)))
    ax1.set_xlabel("Impurity Based Feature Importance", fontsize=16)
    ax1.set_title("Single Tree", fontsize=18)

    forest_importance_sorted_idx = np.argsort(model2.feature_importances_)
    forest_indices = np.arange(0, len(model2.feature_importances_)) + 0.5
    difference = model2.feature_importances_ - model1.feature_importances_
    difference = difference[forest_importance_sorted_idx]

    ax2.barh(forest_indices, model2.feature_importances_[forest_importance_sorted_idx], height=0.7, color='#EFAEA4')
    for index, value in enumerate(model2.feature_importances_[forest_importance_sorted_idx]):
        ax2.text(value, index + 0.3, f" {str(round(difference[index],3))}", fontsize=14)

    ax2.set_yticks(forest_indices)
    ax2.set_yticklabels(np.array(feature_names)[forest_importance_sorted_idx], fontsize=12)
    ax2.set_ylim((0, len(model2.feature_importances_)))
    ax2.set_xlabel("Impurity Based Feature Importance", fontsize=16)
    ax2.set_title("Random Forest", fontsize=18)

    fig.suptitle(name, fontsize=20, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_permute_importance(result1, result2, X, y, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    tree_importance_sorted_idx = np.argsort(result1.importances_mean)
    tree_indices = np.arange(0, len(result1.importances_mean)) + 0.5
    ax1.barh(tree_indices, result1.importances_mean[tree_importance_sorted_idx], height=0.7, color='#B2D7D0')
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(np.array(feature_names)[tree_importance_sorted_idx], fontsize=12)
    ax1.set_ylim((0, len(result1.importances_mean)))
    ax1.set_xlabel("Permutation Feature Importance", fontsize=16)
    ax1.set_title("Single Tree", fontsize=18)

    tree_importance_sorted_idx2 = np.argsort(result2.importances_mean)
    tree_indices2 = np.arange(0, len(result2.importances_mean)) + 0.5
    difference = result2['importances_mean'] - result1['importances_mean']
    difference = difference[tree_importance_sorted_idx]

    ax2.barh(tree_indices2, result2.importances_mean[tree_importance_sorted_idx2], height=0.7, color='#EFAEA4')
    for index, value in enumerate(result2.importances_mean[tree_importance_sorted_idx2]):
        ax2.text(value, index+0.3, f" {str(round(difference[index],3))}", fontsize=14)

    ax2.set_yticks(tree_indices2)
    ax2.set_yticklabels(np.array(feature_names)[tree_importance_sorted_idx2], fontsize=12)
    ax2.set_ylim((0, len(result2.importances_mean)))
    ax2.set_xlabel("Permutation Feature Importance", fontsize=16)
    ax2.set_title("Random Forest", fontsize=18)

    fig.suptitle(name, fontsize=20, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

######################################################


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
        raise ValueError("Column headers not found in file.")

    df = pd.read_csv(csv_path,
                     skiprows=start_idx,
                     quoting=csv.QUOTE_ALL,
                     na_values=["NULL","null","NaN","nan",""],
                     skip_blank_lines=True)

    df = df.rename(columns=lambda x: x.strip())
    df.dropna(how="all", inplace=True)
    return df

# Load provided dataset for training
df = load_data("ML_TAKES_ENCODED.csv")

# Target label
df["CS_Probability"] = np.where(df["PITCHCALL"] == "StrikeCalled", 1, 0)

# Feature selection 
X = df[['PLATELOCHEIGHT', 'PLATELOCSIDE', 'BOT_ZONE', 'TOP_ZONE']]
y = df["CS_Probability"]

'''
#############################
# HYPERPARAMETER TUNING
#############################

print("Running Grid Search for optimal parameters...")

param_grid = {
    "n_estimators": [200, 500, 700, 1000, 1200],
    "max_depth": [10, 15, 20, 25, None]
}

rf_model = RandomForestClassifier(random_state=44)

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X, y)

print("\nBest Parameters Found:")
print(grid_search.best_params_)
print("Best AUC Score:", round(grid_search.best_score_, 3))

optimal_params = grid_search.best_params_
model = RandomForestClassifier(
    random_state=44,
    n_estimators=optimal_params["n_estimators"],
    max_depth=optimal_params["max_depth"]
)

##################################################


'''


#Optimal model parameters found via Grid Search
model = RandomForestClassifier(
    random_state=rand,
    max_depth=10,
    n_estimators=1200
)

model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")



'''
#############################
# MODEL PERFORMANCE TESTING & IMPORTANCE 
#############################


# Example split format before evaluating:
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=44)
pred_probs = model.predict_proba(X_val)[:,1]

auc = roc_auc_score(y_val, pred_probs)
brier = brier_score_loss(y_val, pred_probs)

print("AUC:", round(auc,3))
print("Brier Score:", round(brier,3))

tree_model = DecisionTreeClassifier(max_depth=10, random_state=44)
tree_model.fit(X_train, y_train)

feature_names = X.columns

plot_feature_importance(tree_model, model, X, y, "Feature Importance Comparison")

result_tree = permutation_importance(tree_model, X_val, y_val, n_repeats=10, random_state=44)
result_rf = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=44)

plot_permute_importance(result_tree, result_rf, X, y, "Permutation Importance Comparison")


#############################################################

'''


