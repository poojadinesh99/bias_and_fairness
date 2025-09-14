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

print("Starting bias and fairness analysis...")

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
# Quick preview
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

print("Script completed successfully!")
