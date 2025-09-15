import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind

# Set matplotlib backend for non-interactive plotting
plt.switch_backend('Agg')

print("Starting comprehensive bias and fairness analysis...")

# --- PARAMETERS ---
DATAFILE = "medteach.csv"
OUTPREFIX = "bias_report"
RANDOM_SEED = 42
# choose outcome column: mbi_ex (emotional exhaustion). We'll binarize at median.
OUTCOME_COL = "mbi_ex"

# --- LOAD ---
print("Loading data...")
df = pd.read_csv(DATAFILE)
print("Raw rows:", len(df))
print("Columns:", df.columns.tolist())

# --- CLEAN / SELECT COLUMNS ---
print("Cleaning data...")
# Ensure key columns exist
for c in ["sex", "glang", OUTCOME_COL]:
    if c not in df.columns:
        raise SystemExit(f"Missing column: {c}")

# Drop rows with missing essential data
df = df[[ "id" ] + [ "age", "year", "sex", "glang", OUTCOME_COL ]].copy()
df = df.dropna(subset=[OUTCOME_COL, "sex", "glang"])
print("After dropping missing:", len(df))

# Recode sex (dataset uses numeric codes per codebook: 1=Man;2=Woman;3=Non-binary)
def recode_sex(x):
    try:
        xi = int(x)
        if xi == 1: return "man"
        if xi == 2: return "woman"
        if xi == 3: return "non-binary"
    except:
        pass
    return str(x).lower()

df["sex_cat"] = df["sex"].apply(recode_sex)

# Simplify glang: keep top 3 languages, group rest as 'other'
top_langs = df["glang"].value_counts().nlargest(3).index.tolist()
df["glang_cat"] = df["glang"].apply(lambda x: x if x in top_langs else "other")

# Create binary outcome: high_burnout
threshold = df[OUTCOME_COL].median()
df["high_burnout"] = (df[OUTCOME_COL] >= threshold).astype(int)
print("Threshold (median) for high_burnout:", threshold)

# --- Descriptive counts ---
print("\nCounts by sex:")
print(df["sex_cat"].value_counts())
print("\nCounts by glang:")
print(df["glang_cat"].value_counts())

print("Creating visualizations...")
# Save bar plots
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="sex_cat", order=df["sex_cat"].value_counts().index)
plt.title("Count by sex")
plt.tight_layout()
plt.savefig(f"{OUTPREFIX}_count_sex.png", dpi=150)
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="glang_cat", order=df["glang_cat"].value_counts().index)
plt.title("Count by mother tongue (top3 + other)")
plt.tight_layout()
plt.savefig(f"{OUTPREFIX}_count_glang.png", dpi=150)
plt.close()

# Boxplot of mbi_ex by sex
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x="sex_cat", y=OUTCOME_COL)
plt.title(f"{OUTCOME_COL} by sex")
plt.tight_layout()
plt.savefig(f"{OUTPREFIX}_box_mbi_sex.png", dpi=150)
plt.close()

print("Visualizations saved!")

# --- Statistical test example: sex groups (man vs woman) ---
g_man = df[df["sex_cat"]=="man"][OUTCOME_COL].dropna()
g_woman = df[df["sex_cat"]=="woman"][OUTCOME_COL].dropna()
if len(g_man)>1 and len(g_woman)>1:
    tstat, pval = ttest_ind(g_man, g_woman, equal_var=False)
    print(f"\nT-test mbi_ex man vs woman: t={tstat:.3f}, p={pval:.3f}, mean(man)={g_man.mean():.2f}, mean(woman)={g_woman.mean():.2f}")

print("Building predictive model...")
# --- Simple predictive model ---
# Features: age, year, sex_cat (one-hot), glang_cat (one-hot)
model_df = df.dropna(subset=["age", "year"])
X = model_df[["age","year"]].copy()
X = pd.concat([X,
               pd.get_dummies(model_df["sex_cat"], prefix="sex"),
               pd.get_dummies(model_df["glang_cat"], prefix="lang")],
              axis=1)
y = model_df["high_burnout"]

# scale numeric features
scaler = StandardScaler()
X[["age","year"]] = scaler.fit_transform(X[["age","year"]])

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(X, y, model_df, test_size=0.3, random_state=RANDOM_SEED, stratify=y)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
yprob = clf.predict_proba(X_test)[:,1]
yhat = (yprob >= 0.5).astype(int)

print("\nOverall AUC:", round(roc_auc_score(y_test, yprob), 3))
print("Overall precision/recall:", precision_score(y_test, yhat), recall_score(y_test, yhat))

# Function to compute per-group metrics
def group_metrics(df_subset, y_true, y_prob, y_pred):
    auc = roc_auc_score(y_true, y_prob) if len(pd.unique(y_true))>1 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn)>0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp)>0 else np.nan
    prec = precision_score(y_true, y_pred) if (tp+fp)>0 else np.nan
    rec = recall_score(y_true, y_pred) if (tp+fn)>0 else np.nan
    return {"n": len(y_true), "auc": auc, "precision": prec, "recall": rec, "fpr": fpr, "fnr": fnr}

print("Analyzing fairness by demographic groups...")
# Compute metrics by sex_cat
print("\nPer-group metrics by sex (test set):")
test_with_preds = df_test.copy()
test_with_preds["y_true"] = y_test.values
test_with_preds["y_prob"] = yprob
test_with_preds["y_pred"] = yhat

for g in test_with_preds["sex_cat"].unique():
    sel = test_with_preds["sex_cat"]==g
    if sel.sum() < 5:
        print(f"{g}: too few samples ({sel.sum()})")
        continue
    gm = group_metrics(test_with_preds[sel], test_with_preds[sel]["y_true"], test_with_preds[sel]["y_prob"], test_with_preds[sel]["y_pred"])
    print(f"{g}: {gm}")

# Compute metrics by glang_cat
print("\nPer-group metrics by language (test set):")
for g in test_with_preds["glang_cat"].unique():
    sel = test_with_preds["glang_cat"]==g
    if sel.sum() < 5:
        print(f"{g}: too few samples ({sel.sum()})")
        continue
    gm = group_metrics(test_with_preds[sel], test_with_preds[sel]["y_true"], test_with_preds[sel]["y_prob"], test_with_preds[sel]["y_pred"])
    print(f"{g}: {gm}")

print("Applying bias mitigation technique...")
# --- Simple reweighing mitigation: weight training samples inverse to group prevalence of glang_cat ---
group = df_train["glang_cat"]
group_counts = group.value_counts().to_dict()
total = len(group)
weights = df_train["glang_cat"].apply(lambda g: total / (group_counts.get(g,1) * len(group_counts)))
Xw = X_train.copy()
yw = y_train.copy()
# Fit weighted logistic
clf_w = LogisticRegression(max_iter=200)
clf_w.fit(Xw, yw, sample_weight=weights)
yprob_w = clf_w.predict_proba(X_test)[:,1]
yhat_w = (yprob_w >= 0.5).astype(int)

print("\nAfter reweighing by language - Overall AUC:", round(roc_auc_score(y_test, yprob_w), 3))
print("Overall precision/recall (weighted model):", precision_score(y_test, yhat_w), recall_score(y_test, yhat_w))

# Compare FNR by language before/after
print("\nFNR by language before -> after:")
for g in test_with_preds["glang_cat"].unique():
    sel = test_with_preds["glang_cat"]==g
    if sel.sum() < 5:
        print(f"{g}: too few samples ({sel.sum()})")
        continue
    gm_before = group_metrics(test_with_preds[sel], test_with_preds[sel]["y_true"], test_with_preds[sel]["y_prob"], test_with_preds[sel]["y_pred"])
    # for after, compute predictions from clf_w on the same rows
    X_sel = X_test.loc[test_with_preds[sel].index]
    yprob_sel_after = clf_w.predict_proba(X_sel)[:,1]
    yhat_sel_after = (yprob_sel_after >= 0.5).astype(int)
    gm_after = group_metrics(test_with_preds[sel], test_with_preds[sel]["y_true"], yprob_sel_after, yhat_sel_after)
    print(f"{g}: FNR {gm_before['fnr']:.3f} -> {gm_after['fnr']:.3f}, n={sel.sum()}")

print("Saving results...")
# Save a simple table
metrics_rows = []
for g in test_with_preds["glang_cat"].unique():
    sel = test_with_preds["glang_cat"]==g
    if sel.sum() < 5:
        continue
    X_sel = X_test.loc[test_with_preds[sel].index]
    yprob_sel_after = clf_w.predict_proba(X_sel)[:,1]
    yhat_sel_after = (yprob_sel_after >= 0.5).astype(int)
    gm_before = group_metrics(test_with_preds[sel], test_with_preds[sel]["y_true"], test_with_preds[sel]["y_prob"], test_with_preds[sel]["y_pred"])
    gm_after = group_metrics(test_with_preds[sel], test_with_preds[sel]["y_true"], yprob_sel_after, yhat_sel_after)
    metrics_rows.append({
        "group": g,
        "n": int(gm_before["n"]),
        "fnr_before": gm_before["fnr"],
        "fnr_after": gm_after["fnr"],
        "auc_before": gm_before["auc"],
        "auc_after": gm_after["auc"]
    })
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(f"{OUTPREFIX}_group_metrics_glang.csv", index=False)
print(f"\nSaved group metrics to {OUTPREFIX}_group_metrics_glang.csv")

print("\nBias and fairness analysis complete!")
print("Generated files:")
print("- bias_report_count_sex.png: Count distribution by sex")
print("- bias_report_count_glang.png: Count distribution by language")  
print("- bias_report_box_mbi_sex.png: Emotional exhaustion by sex")
print("- bias_report_group_metrics_glang.csv: Fairness metrics before/after mitigation")

print("\nCreating additional slide visualizations...")

# Slide 4: FNR before mitigation
groups = metrics_df["group"]
x_pos = np.arange(len(groups))

plt.figure(figsize=(6,4))
plt.bar(x_pos, metrics_df["fnr_before"], color="steelblue")
plt.ylabel("False Negative Rate (FNR)")
plt.title("FNR by Group (Before Mitigation)")
plt.xticks(x_pos, groups, rotation=45)
plt.tight_layout()
plt.savefig("slide4_fnr_before.png", dpi=150)
plt.close()
print("- slide4_fnr_before.png: FNR by group before mitigation")

# Slide 5: FNR before vs after
width = 0.35

plt.figure(figsize=(6,4))
plt.bar(x_pos - width/2, metrics_df["fnr_before"], width, label="Before", color="skyblue")
plt.bar(x_pos + width/2, metrics_df["fnr_after"], width, label="After", color="orange")

plt.ylabel("False Negative Rate (FNR)")
plt.title("FNR by Group: Before vs After Reweighing")
plt.xticks(x_pos, groups, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("slide5_fnr_before_after.png", dpi=150)
plt.close()
print("- slide5_fnr_before_after.png: FNR comparison before vs after mitigation")

print("\nScript complete. Look for images and CSV output to paste into slides.")
