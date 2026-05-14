import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# date range for all data
start = "2010-01-01"
end = "2024-12-31"

# download price data from yahoo finance
spy = yf.download("SPY", start=start, end=end, progress=False)
ief = yf.download("IEF", start=start, end=end, progress=False)
vix = yf.download("^VIX", start=start, end=end, progress=False)

# grab closing prices
prices = pd.DataFrame()
prices["spy"] = spy["Close"]
prices["ief"] = ief["Close"]
prices["vix"] = vix["Close"]

# pull macro data from FRED api
def get_fred(series_id, start, end):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "observation_start": start,
        "observation_end": end,
        "api_key": "YOUR_FRED_API_KEY_HERE",
        "file_type": "json",
    }
    r = requests.get(url, params=params)
    obs = r.json()["observations"]
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.columns = [series_id]
    return df

t10 = get_fred("DGS10", start, end)
t2 = get_fred("DGS2", start, end)
unrate = get_fred("UNRATE", start, end)

macro = pd.DataFrame()
macro["t10"] = t10["DGS10"]
macro["t2"] = t2["DGS2"]
macro["spread"] = macro["t10"] - macro["t2"]  # yield curve spread, negative = bad sign
macro["unrate"] = unrate["UNRATE"]

# merge everything together, ffill because unemployment is monthly
df = prices.join(macro, how="left").ffill()

# calculate daily returns
df["spy_ret"] = df["spy"].pct_change()
df["ief_ret"] = df["ief"].pct_change()

# build features
df["vol20"] = df["spy_ret"].rolling(20).std() * np.sqrt(252)
df["vol60"] = df["spy_ret"].rolling(60).std() * np.sqrt(252)
df["vix_chg5"] = df["vix"].pct_change(5)
df["mom20"] = df["spy"].pct_change(20)
df["corr60"] = df["spy_ret"].rolling(60).corr(df["ief_ret"])

# create target variable
# top 25% of forward 20 day vol = high vol regime (label 1), rest = calm (label 0)
future_vol = df["spy_ret"].rolling(20).std().shift(-20) * np.sqrt(252)
threshold = future_vol.quantile(0.75)
df["target"] = (future_vol > threshold).astype(int)

feat_cols = ["vix", "vol20", "vol60", "vix_chg5", "mom20", "corr60", "spread", "unrate"]
data = df[feat_cols + ["target", "spy_ret", "ief_ret"]].dropna()

# save datasets
df.to_csv("full_data.csv")
data.to_csv("model_data.csv")

print("rows:", len(data))
print("class balance:")
print(data["target"].value_counts(normalize=True))

# train/test split by time (no shuffling, this is time series data)
split = int(len(data) * 0.7)
train = data.iloc[:split]
test = data.iloc[split:]

X_train = train[feat_cols]
y_train = train["target"]
X_test = test[feat_cols]
y_test = test["target"]

# scale for logistic regression and svm
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)

print("\nlogistic regression")
print("accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# random forest
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("random forest")
print("accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# svm with rbf kernel
svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm.fit(X_train_s, y_train)
svm_pred = svm.predict(X_test_s)

print("svm (rbf kernel)")
print("accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))


# plot classification reports for all 3 models as heatmaps
def plot_classification_report(y_true, y_pred, model_name, ax):
    report = classification_report(y_true, y_pred, output_dict=True)
    rows = ["0 (calm)", "1 (volatile)", "macro avg", "weighted avg"]
    cols = ["precision", "recall", "f1-score"]
    keys = ["0", "1", "macro avg", "weighted avg"]
    table = np.array([[report[k][c] for c in cols] for k in keys])
    ax.imshow(table, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title(model_name)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
plot_classification_report(y_test, lr_pred,  "Logistic Regression", axes[0])
plot_classification_report(y_test, rf_pred,  "Random Forest",       axes[1])
plot_classification_report(y_test, svm_pred, "SVM (RBF kernel)",    axes[2])
plt.suptitle("classification reports", fontsize=13)
plt.tight_layout()
plt.savefig("classification_reports.png", dpi=120)
plt.show()


# feature importance for each model
# logistic regression - use absolute value of coefficients
lr_imp = pd.Series(np.abs(lr.coef_[0]), index=feat_cols)

# random forest - built in feature importances
rf_imp = pd.Series(rf.feature_importances_, index=feat_cols)

# svm - permutation importance (rbf kernel doesnt give coefficients directly)
perm = permutation_importance(svm, X_test_s, y_test, n_repeats=10, random_state=42)
svm_imp = pd.Series(perm.importances_mean, index=feat_cols)

print("\nfeature importance (random forest):")
print(rf_imp.sort_values(ascending=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
lr_imp.sort_values(ascending=False).plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("logistic regression (|coef|)")
axes[0].set_xlabel("feature")
axes[0].set_ylabel("importance")
axes[0].tick_params(axis="x", rotation=45)
rf_imp.sort_values(ascending=False).plot(kind="bar", ax=axes[1], color="seagreen")
axes[1].set_title("random forest (built-in)")
axes[1].set_xlabel("feature")
axes[1].set_ylabel("importance")
axes[1].tick_params(axis="x", rotation=45)
svm_imp.sort_values(ascending=False).plot(kind="bar", ax=axes[2], color="indianred")
axes[2].set_title("svm (permutation importance)")
axes[2].set_xlabel("feature")
axes[2].set_ylabel("importance")
axes[2].tick_params(axis="x", rotation=45)
plt.suptitle("feature importance for predicting market volatility", fontsize=13)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.show()


# portfolio strategy comparison
# rule: if model predicts 1 (volatile) -> go 30/70, else stay 60/40
test = test.copy()
test["lr_pred"] = lr_pred
test["rf_pred"] = rf_pred
test["svm_pred"] = svm_pred

def strategy_returns(pred_col):
    return np.where(
        test[pred_col] == 1,
        0.3 * test["spy_ret"] + 0.7 * test["ief_ret"],
        0.6 * test["spy_ret"] + 0.4 * test["ief_ret"],
    )

test["base_ret"] = 0.6 * test["spy_ret"] + 0.4 * test["ief_ret"]
test["lr_ret"] = strategy_returns("lr_pred")
test["rf_ret"] = strategy_returns("rf_pred")
test["svm_ret"] = strategy_returns("svm_pred")

# cumulative returns
test["base_cum"] = (1 + test["base_ret"]).cumprod()
test["lr_cum"] = (1 + test["lr_ret"]).cumprod()
test["rf_cum"] = (1 + test["rf_ret"]).cumprod()
test["svm_cum"] = (1 + test["svm_ret"]).cumprod()

def max_drawdown(cum):
    peak = cum.cummax()
    dd = cum / peak - 1
    return dd.min()

print("\nstrategy results")
for name, ret_col, cum_col in [
    ("baseline 60/40",      "base_ret", "base_cum"),
    ("logistic regression", "lr_ret",   "lr_cum"),
    ("random forest",       "rf_ret",   "rf_cum"),
    ("svm",                 "svm_ret",  "svm_cum"),
]:
    vol = test[ret_col].std() * np.sqrt(252)
    dd = max_drawdown(test[cum_col])
    final = test[cum_col].iloc[-1]
    print(f"{name}: annual vol={round(vol,4)}, max drawdown={round(dd,4)}, final value=${round(final,4)}")

# plot results
plt.figure(figsize=(11, 5))
plt.plot(test.index, test["base_cum"], label="60/40 baseline")
plt.plot(test.index, test["lr_cum"],   label="logistic regression")
plt.plot(test.index, test["rf_cum"],   label="random forest")
plt.plot(test.index, test["svm_cum"],  label="svm (rbf)")
plt.legend()
plt.title("cumulative returns on test set")
plt.xlabel("date")
plt.ylabel("growth of $1")
plt.tight_layout()
plt.savefig("returns.png", dpi=120)
plt.show()