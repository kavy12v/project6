"""
=============================================================
  WINE QUALITY PREDICTION - Complete ML Project
  Slash Mark Internship | Task 7
=============================================================
  Dataset  : winequality-red.csv (1599 rows, 12 columns)
  Target   : quality (converted to binary: good / not good)
  Models   : Logistic Regression, Random Forest, SVM, KNN,
             Decision Tree, Gradient Boosting
=============================================================
"""

# ─────────────────────────────────────────────
# STEP 1 : Import Libraries
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # save plots to file (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, roc_auc_score, roc_curve)

print("=" * 60)
print("   WINE QUALITY PREDICTION PROJECT")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 2 : Load Dataset
# ─────────────────────────────────────────────
print("\n📂 STEP 2: Loading Dataset...")
df = pd.read_csv('winequality-red.csv')
print(f"   ✅ Dataset loaded! Shape: {df.shape}")
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ─────────────────────────────────────────────
# STEP 3 : Understand the Data
# ─────────────────────────────────────────────
print("\n📊 STEP 3: Understanding the Data...")
print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Data Types & Non-null Counts ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe().round(2))

# ─────────────────────────────────────────────
# STEP 4 : Check Missing Values
# ─────────────────────────────────────────────
print("\n🔍 STEP 4: Checking Missing Values...")
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("   ✅ No missing values found! Dataset is clean.")
else:
    print(f"   ⚠️  Found {missing.sum()} missing values. Filling with column median...")
    df.fillna(df.median(), inplace=True)

# ─────────────────────────────────────────────
# STEP 5 : Explore the Data (EDA)
# ─────────────────────────────────────────────
print("\n📈 STEP 5: Exploratory Data Analysis (EDA)...")

# 5a. Quality distribution
plt.figure(figsize=(8, 5))
df['quality'].value_counts().sort_index().plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Distribution of Wine Quality Scores', fontsize=14)
plt.xlabel('Quality Score')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plot_01_quality_distribution.png', dpi=120)
plt.close()
print("   ✅ Saved: plot_01_quality_distribution.png")

# 5b. Correlation heatmap
plt.figure(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('plot_02_correlation_heatmap.png', dpi=120)
plt.close()
print("   ✅ Saved: plot_02_correlation_heatmap.png")

# 5c. Boxplots for key features vs quality
key_features = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, feat in zip(axes.flatten(), key_features):
    df.boxplot(column=feat, by='quality', ax=ax)
    ax.set_title(f'{feat} vs Quality')
    ax.set_xlabel('Quality Score')
plt.suptitle('')
plt.tight_layout()
plt.savefig('plot_03_feature_boxplots.png', dpi=120)
plt.close()
print("   ✅ Saved: plot_03_feature_boxplots.png")

# ─────────────────────────────────────────────
# STEP 6 : Feature Engineering — Create Binary Target
# ─────────────────────────────────────────────
print("\n⚙️  STEP 6: Feature Engineering...")
# Quality >= 7 → "Good" (1),  Quality < 7 → "Not Good" (0)
df['good_quality'] = (df['quality'] >= 7).astype(int)

good  = df['good_quality'].sum()
total = len(df)
print(f"   Good Quality wines  (score ≥ 7): {good}  ({good/total*100:.1f}%)")
print(f"   Not Good wines      (score < 7): {total - good}  ({(total-good)/total*100:.1f}%)")

# 6a. Class balance pie chart
plt.figure(figsize=(6, 6))
plt.pie([good, total - good],
        labels=['Good Quality', 'Not Good Quality'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'],
        startangle=140,
        explode=[0.05, 0])
plt.title('Proportion of Good vs Not-Good Wines', fontsize=13)
plt.tight_layout()
plt.savefig('plot_04_class_balance.png', dpi=120)
plt.close()
print("   ✅ Saved: plot_04_class_balance.png")

# ─────────────────────────────────────────────
# STEP 7 : Feature Selection
# ─────────────────────────────────────────────
print("\n🎯 STEP 7: Feature Selection...")
X = df.drop(columns=['quality', 'good_quality'])   # 11 features
y = df['good_quality']                              # binary target
print(f"   Features (X): {X.columns.tolist()}")
print(f"   Target   (y): good_quality  (0 = not good, 1 = good)")

# ─────────────────────────────────────────────
# STEP 8 : Split Data & Scale Features
# ─────────────────────────────────────────────
print("\n✂️  STEP 8: Train-Test Split & Feature Scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("   ✅ Features scaled with StandardScaler")

# ─────────────────────────────────────────────
# STEP 9 : Train Multiple Models
# ─────────────────────────────────────────────
print("\n🤖 STEP 9: Training & Evaluating Models...")
print("-" * 60)

models = {
    'Logistic Regression'    : LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'          : DecisionTreeClassifier(random_state=42),
    'Random Forest'          : RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM'                    : SVC(probability=True, random_state=42),
    'K-Nearest Neighbors'    : KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting'      : GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    # Predict
    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)[:, 1]
    # Metrics
    acc        = accuracy_score(y_test, y_pred)
    auc        = roc_auc_score(y_test, y_prob)
    cv_scores  = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    results[name] = {
        'model'    : model,
        'accuracy' : acc,
        'auc'      : auc,
        'cv_mean'  : cv_scores.mean(),
        'cv_std'   : cv_scores.std(),
        'y_pred'   : y_pred,
        'y_prob'   : y_prob,
    }
    print(f"\n  ➤  {name}")
    print(f"      Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"      ROC-AUC Score : {auc:.4f}")
    print(f"      5-Fold CV     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# STEP 10 : Model Comparison
# ─────────────────────────────────────────────
print("\n📊 STEP 10: Model Comparison...")

names      = list(results.keys())
accuracies = [results[n]['accuracy'] for n in names]
auc_scores = [results[n]['auc']      for n in names]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy bar chart
bars = axes[0].barh(names, [a*100 for a in accuracies],
                    color='steelblue', edgecolor='black')
axes[0].set_xlabel('Accuracy (%)')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_xlim(50, 100)
for bar, acc in zip(bars, accuracies):
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{acc*100:.2f}%', va='center', fontsize=10)

# ROC-AUC bar chart
bars2 = axes[1].barh(names, auc_scores, color='coral', edgecolor='black')
axes[1].set_xlabel('ROC-AUC Score')
axes[1].set_title('Model ROC-AUC Comparison')
axes[1].set_xlim(0.5, 1.0)
for bar, auc in zip(bars2, auc_scores):
    axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{auc:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('plot_05_model_comparison.png', dpi=120)
plt.close()
print("   ✅ Saved: plot_05_model_comparison.png")

# ─────────────────────────────────────────────
# STEP 11 : Best Model — Detailed Analysis
# ─────────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]['accuracy'])
best       = results[best_name]
print(f"\n🏆 STEP 11: Best Model → {best_name}  (Accuracy: {best['accuracy']*100:.2f}%)")

print("\n--- Classification Report ---")
print(classification_report(y_test, best['y_pred'],
                             target_names=['Not Good (0)', 'Good (1)']))

# 11a. Confusion Matrix
cm = confusion_matrix(y_test, best['y_pred'])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Good', 'Good'],
            yticklabels=['Not Good', 'Good'])
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plot_06_confusion_matrix.png', dpi=120)
plt.close()
print("   ✅ Saved: plot_06_confusion_matrix.png")

# 11b. ROC Curve (all models)
plt.figure(figsize=(9, 7))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models')
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('plot_07_roc_curves.png', dpi=120)
plt.close()
print("   ✅ Saved: plot_07_roc_curves.png")

# ─────────────────────────────────────────────
# STEP 12 : Feature Importance (Random Forest)
# ─────────────────────────────────────────────
print("\n🌲 STEP 12: Feature Importance (Random Forest)...")
rf_model     = results['Random Forest']['model']
importances  = rf_model.feature_importances_
feat_names   = X.columns.tolist()

feat_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True)

plt.figure(figsize=(9, 6))
plt.barh(feat_df['Feature'], feat_df['Importance'],
         color='mediumseagreen', edgecolor='black')
plt.xlabel('Importance Score')
plt.title('Feature Importance — Random Forest')
plt.tight_layout()
plt.savefig('plot_08_feature_importance.png', dpi=120)
plt.close()
print("   ✅ Saved: plot_08_feature_importance.png")

print("\n   Top 3 most important features:")
for _, row in feat_df.sort_values('Importance', ascending=False).head(3).iterrows():
    print(f"   • {row['Feature']:25s} → {row['Importance']:.4f}")

# ─────────────────────────────────────────────
# STEP 13 : Predict on a New Wine Sample
# ─────────────────────────────────────────────
print("\n🍷 STEP 13: Predicting Quality of a New Wine Sample...")
# [fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
#  free SO2, total SO2, density, pH, sulphates, alcohol]
new_wine = np.array([[7.4, 0.70, 0.00, 1.9, 0.076,
                       11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]])
new_wine_scaled = scaler.transform(new_wine)
pred  = results['Random Forest']['model'].predict(new_wine_scaled)[0]
prob  = results['Random Forest']['model'].predict_proba(new_wine_scaled)[0]

label = "GOOD Quality 🍾" if pred == 1 else "NOT Good Quality 🚫"
print(f"   Prediction     : {label}")
print(f"   Probability    : Good={prob[1]:.2%}  |  Not Good={prob[0]:.2%}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("   ✅  PROJECT COMPLETE! — Final Summary")
print("=" * 60)
print(f"\n{'Model':<28} {'Accuracy':>10} {'ROC-AUC':>10} {'CV Score':>12}")
print("-" * 62)
for n, r in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
    print(f"  {n:<26} {r['accuracy']*100:>9.2f}%  {r['auc']:>9.4f}  "
          f"{r['cv_mean']:>9.4f}±{r['cv_std']:.4f}")
print("-" * 62)
print(f"\n  🏆 Best Model : {best_name}")
print(f"     Accuracy   : {best['accuracy']*100:.2f}%")
print(f"     ROC-AUC    : {best['auc']:.4f}")
print(f"\n  📁 Plots saved in project folder:")
plots = [
    "plot_01_quality_distribution.png",
    "plot_02_correlation_heatmap.png",
    "plot_03_feature_boxplots.png",
    "plot_04_class_balance.png",
    "plot_05_model_comparison.png",
    "plot_06_confusion_matrix.png",
    "plot_07_roc_curves.png",
    "plot_08_feature_importance.png",
]
for p in plots:
    print(f"     • {p}")
print("\n  🎓 Learning Outcomes achieved:")
print("     ✅ Feature Analysis & EDA")
print("     ✅ Classification Models (6 algorithms)")
print("     ✅ Model Evaluation (Accuracy, AUC, CV)")
print("     ✅ Feature Importance Analysis")
print("=" * 60)
print("  CHEERS! 🍷  Your Wine Quality ML Project is Done!")
print("=" * 60)

