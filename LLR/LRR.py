# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from argparse import ArgumentParser

# %%
parser = ArgumentParser()
parser.add_argument("--target", type=str, required=True, help="target to be predicted")
args = parser.parse_args()
target = args.target

# %%
df1 = pd.read_feather("/home/haotsung/test_juseless/HCPFCSchaefer400x17.feather")
df2 = pd.read_csv("/home/haotsung/test_juseless/unrestricted.csv", sep=";")
all_targets = [
    "PicSeq_Unadj",
    "CardSort_Unadj",
    "Flanker_Unadj",
    "PMAT24_A_CR",
    "ReadEng_Unadj",
    "PicVocab_Unadj",
    "ProcSpeed_Unadj",
    "SCPT_SEN",
    "SCPT_SPEC",
    "IWRD_TOT",
    "ListSort_Unadj",
    "Endurance_Unadj",
    "Dexterity_Unadj",
    "Strength_Unadj",
    "Odor_Unadj",
    "PainInterf_Tscore",
    "Taste_Unadj",
    "Emotion_Task_Face_Acc",
    "Language_Task_Math_Avg_Difficulty_Level",
    "Language_Task_Story_Avg_Difficulty_Level",
    "Relational_Task_Acc",
    "Social_Task_Perc_Random",
    "Social_Task_Perc_TOM",
    "WM_Task_Acc",
    "NEOFAC_A",
    "NEOFAC_O",
    "NEOFAC_C",
    "NEOFAC_N",
    "NEOFAC_E", 
    "ER40_CR",
    "ER40ANG",
    "ER40FEAR",
    "ER40HAP",
    "ER40NOE",
    "ER40SAD",
    "AngAffect_Unadj",
    "AngHostil_Unadj",
    "AngAggr_Unadj",
    "FearAffect_Unadj",
    "FearSomat_Unadj",
    "Sadness_Unadj",
    "LifeSatisf_Unadj",
    "MeanPurp_Unadj",
    "PosAffect_Unadj",
    "Friendship_Unadj",
    "Loneliness_Unadj",
    "PercHostil_Unadj",
    "PercReject_Unadj",
    "EmotSupp_Unadj",
    "InstruSupp_Unadj",
    "PercStress_Unadj",
    "SelfEff_Unadj",
]


# %%


# Ignore ROI1==ROI2
columns_with_tilde = [col for col in df1.columns if '~' in col]
columns_to_keep = ['phase_encoding'] + ['subject'] + [col for col in columns_with_tilde if col.split('~')[0] != col.split('~')[1]]
df1 = df1[columns_to_keep]

# Calculate average across REST1/LR, REST1/RL, REST2/LR, REST2/RL
grouped = df1.groupby("subject", as_index=False)
columns_to_average = [x for x in df1.columns if x not in ["task", "phase_encoding","subject"]]
averaged_df1 = grouped[columns_to_average].mean()
# %%
# Select WM_task in 'unresticted.csv'
df2["Subject"] = df2["Subject"].astype(str)
if target == "PCA":
    selected_targets = all_targets
else:
    selected_targets = [target]
# %%
selected_columns = ["Subject"] + all_targets
df2_selected = df2[selected_columns]

# Merge them together
merged_df = pd.merge(averaged_df1, df2_selected, left_on="subject", right_on="Subject")


# Clean malformed data
def clean_number(number_str):
    if isinstance(number_str, str):
        parts = number_str.split(".")
        if len(parts) > 1:
            integer_part = "".join(parts[:-1])
            decimal_part = parts[-1]
            cleaned_number = integer_part + "." + decimal_part
        else:
            cleaned_number = number_str
    else:
        cleaned_number = number_str

    return cleaned_number


# %%
merged_df[all_targets] = merged_df[all_targets].applymap(clean_number)
# %%
# Remove missing value and non-numeric value in 'merged_df'

for col in all_targets:
    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")

merged_df_clean = merged_df.dropna(subset=all_targets)

# Step 1: Seperate features (X) and targets (y)

X = merged_df_clean.iloc[:, 1:80203]
y = merged_df_clean[selected_targets]

# Step 2: train/test split by 5-fold cross validation
r2_scores = []
k = 5
kf = KFold(n_splits=k)
# Hyperparmeter Tuning
param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0]}

for train_index, test_index in kf.split(X):
    print("e")
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Step 3: Standardize features (important for regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if target == "PCA":
        # Step 4 (optional): Train a PCA model for targets
        pca = PCA(n_components=1)
        y_train = pca.fit_transform(y_train)
        y_test = pca.transform(y_test)

    # Step 5.1: Train a Ridge Regression Model for Data without PCA (single target)
    models = []
    grid_search = GridSearchCV(estimator=Ridge(), param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    models.append(grid_search.best_estimator_)

    y_pred = grid_search.predict(X_test)

    r_squared = r2_score(y_test, y_pred)
    r2_scores.append(r_squared)

mean_r2 = np.mean(r2_scores)
print(f"Mean R2 for {target} = {mean_r2}")

results_df = pd.DataFrame(
    {
        "R2": r2_scores,
        "folds": [i for i in range(k)],
        "target": [target for i in range(k)],
    }
)
results_df.to_csv(f"/home/haotsung/test_juseless/results/scores_{target}.csv")

