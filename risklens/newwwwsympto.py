import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss
)
from sklearn.impute import SimpleImputer
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# ================= LOAD DATA =================
df = pd.read_excel(r"C:/Users/Mansha/OneDrive/Desktop/risklens/diseasefinalset.xlsx")
recs = pd.read_excel(r"C:/Users/Mansha/OneDrive/Desktop/risklens/recommendationsdoc_precise_detailed.xlsx")

diseases = [
    "Diabetes","HeartDisease",
    "CKD","Asthma","Dyslipidemia","Anemia"
]

disease_features = {
    "Diabetes":["SugarLevel","FrequentUrination","ExcessiveThirst","FamilyHistoryDiabetes","Fatigue"],
    "HeartDisease":["ChestPain","BloodPressure","Smoking","FamilyHistoryHeart","Alcohol"],
    "CKD":["SwellingAnkles","FrequentUrination","BloodPressure"],
    "Asthma":["Wheezing","Breathlessness","Cough","Smoking"],
    "Dyslipidemia":["DietQuality","PhysicalActivity","Smoking","Alcohol"],
    "Anemia":["PaleSkin","Fatigue","WeightLoss","Dizziness"]
}

# ================= ORDINAL FEATURES =================
ORDINAL_FEATURES = {
    "BloodPressure": ["Normal", "Elevated", "High"],
    "StressLevel": ["Low", "Medium", "High"],
    "DietQuality": ["Poor", "Average", "Good"],
    "PhysicalActivity": ["Low", "Moderate", "High"],
    "SaltIntake": ["Low", "Medium", "High"]
}

STATE_NAMES = {
    "BloodPressure": [0,1,2],
    "StressLevel": [0,1,2],
    "DietQuality": [0,1,2],
    "PhysicalActivity": [0,1,2],
    "SaltIntake": [0,1,2]
}

# ================= ENCODING =================
encoders = {}

for col in df.columns:
    if col in ORDINAL_FEATURES:
        mapping = {v: i for i, v in enumerate(ORDINAL_FEATURES[col])}
        df[col] = df[col].map(mapping)
        encoders[col] = mapping
    elif df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# ================= AGE GROUP =================
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0,30,45,60,120],
    labels=["Young","Adult","Middle","Senior"]
)

le_age = LabelEncoder()
df["AgeGroup"] = le_age.fit_transform(df["AgeGroup"].astype(str))
encoders["AgeGroup"] = le_age

# ================= HANDLE MISSING VALUES =================
imputer = SimpleImputer(strategy="most_frequent")
df[df.columns] = imputer.fit_transform(df)

# ================= OVERALL SYSTEM EVALUATION =================
print("\n========= OVERALL SCREENING PERFORMANCE =========")

all_y_true, all_y_pred, all_risk_probs = [], [], []
nb_models = {}

for d in diseases:
    X = df[disease_features[d]]
    y = df[d]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    nb = CategoricalNB()
    nb.fit(X_train, y_train)
    nb_models[d] = nb

    probs = nb.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)

    all_y_true.extend(y_test.tolist())
    all_y_pred.extend(preds.tolist())
    all_risk_probs.extend(probs.tolist())

print(f"Overall Accuracy  : {accuracy_score(all_y_true, all_y_pred):.3f}")
print(f"Overall Precision : {precision_score(all_y_true, all_y_pred, zero_division=0):.3f}")
print(f"Overall Recall    : {recall_score(all_y_true, all_y_pred, zero_division=0):.3f}")
print(f"Overall F1 Score  : {f1_score(all_y_true, all_y_pred, zero_division=0):.3f}")

print("\n========= RISK CALIBRATION =========")
print(f"Brier Score (↓ better): {brier_score_loss(all_y_true, all_risk_probs):.4f}")

# ================= BAYESIAN NETWORK =================
bn_models, bn_infer = {}, {}

for d in diseases:
    feats = ["AgeGroup"] + disease_features[d]
    data = df[feats + [d]]

    edges = [(f, d) for f in feats]

    state_names = {}
    for col in feats + [d]:
        if col in STATE_NAMES:
            state_names[col] = STATE_NAMES[col]
        else:
            state_names[col] = sorted(data[col].unique().tolist())

    model = DiscreteBayesianNetwork(edges)
    model.fit(
        data,
        estimator=MaximumLikelihoodEstimator,
        state_names=state_names
    )

    bn_models[d] = model
    bn_infer[d] = VariableElimination(model)

# ================= USER INPUT =================
print("\n======= HEALTH INTAKE FORM =======")

user = {}
age = int(input("Enter Age (number): "))
age_group = "Young" if age<=30 else "Adult" if age<=45 else "Middle" if age<=60 else "Senior"
user["AgeGroup"] = encoders["AgeGroup"].transform([age_group])[0]

all_features = sorted(set(sum(disease_features.values(), [])))

for col in all_features:
    opts = ORDINAL_FEATURES[col] if col in ORDINAL_FEATURES else encoders[col].classes_

    print(f"\n{col} options:")
    for o in opts:
        print(" -", o)

    while True:
        v = input(f"Enter {col}: ").strip().title()
        if v in opts:
            user[col] = encoders[col][v] if col in ORDINAL_FEATURES else encoders[col].transform([v])[0]
            break
        print("Invalid input. Choose from above.")

# ================= HEALTH RISK REPORT =================
print("\n========= HEALTH RISK REPORT =========")

for d in diseases:
    X_nb = pd.DataFrame([{f: user[f] for f in disease_features[d]}])
    p_nb = nb_models[d].predict_proba(X_nb)[0][1]

    evidence = {f: int(user[f]) for f in disease_features[d]}
    evidence["AgeGroup"] = int(user["AgeGroup"])

    q = bn_infer[d].query([d], evidence=evidence)
    p_bn = q.values[1]

    risk = round((0.6*p_nb + 0.4*p_bn)*100, 2)

    band = (
        "0-20" if risk<=20 else
        "21-40" if risk<=40 else
        "41-60" if risk<=60 else
        "61-80" if risk<=80 else
        "81-100"
    )

    advice = recs[
        (recs.Disease == d) &
        (recs.RiskRangePercent == band)
    ]["DoctorRecommendation"].values[0]

    print(f"\n{d} Risk: {risk}%")
    print("Doctor Recommendation:", advice)
    print("Why this risk?")

    base_q = bn_infer[d].query(
        [d],
        evidence={"AgeGroup": int(user["AgeGroup"])}
    )
    base_p = base_q.values[1]

    for f in disease_features[d]:
        temp_q = bn_infer[d].query(
            [d],
            evidence={"AgeGroup": int(user["AgeGroup"]), f: int(user[f])}
        )
        delta = round((temp_q.values[1] - base_p)*100, 2)
        direction = "increased" if delta > 0 else "reduced"
        print(f"  {f} {direction} risk by {abs(delta)}%")
