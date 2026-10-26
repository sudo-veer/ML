# Machine Learning - Practical & Oral Exam Prep

Short, tested Python code for all 6 ML experiments, written so you can **type it from memory during the exam**
(no internet, no pre-downloaded datasets). Every file builds its own small dataset, runs on its own,
and has a one-line explanation comment at the top.

> Made by students, for students. Not an official university resource. Always follow your own
> examiner's instructions and syllabus.

---

## Experiments

| # | File | Topic | Algorithm | Key metrics |
|---|------|-------|-----------|-------------|
| 1 | `exp1_boston_linear_regression.py` | Boston Housing | Linear Regression | MAE, MSE, RMSE, R2 |
| 2 | `exp2_titanic_logistic_regression.py` | Titanic Survival | Logistic Regression | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 3 | `exp3_adult_decision_tree.py` | Adult Census Income | Decision Tree | Same as above + feature importance + tree plot |
| 4 | `exp4_adult_random_forest.py` | Adult Census Income | Random Forest | Same as above + feature importance |
| 5 | `exp5_adult_boosting.py` | Adult Census Income | Gradient Boosting (AdaBoost note below) | Same as above + feature importance |
| 6 | `exp6_wholesale_hierarchical_clustering.py` | Wholesale Customers | Hierarchical Clustering (Ward, Euclidean) | Dendrogram, Silhouette Score |

Why "regression" for Titanic? Survived is 0/1, so the right technique is **Logistic** Regression.

---

## How to run

```bash
pip install numpy pandas matplotlib scikit-learn scipy
python exp1_boston_linear_regression.py
```

Or open any file in VS Code / Jupyter / Colab and run it. Tested with Python 3 and scikit-learn 1.8.
Plots open in a window, so use `plt.show()` as written.

---

## How to make your own dataset (exam hint)

The exam says to use **your own dataset**. The pattern is always the same:

1. **Features:** create 3-6 columns with `np.random` (numbers) or `np.random.choice` (categories).
2. **Target:** build it from the features with a formula + noise, so the model has something to learn.
3. **Mess it up a little:** add some missing values (`np.nan`) and one text column, to show cleaning.
4. **Size:** 100-500 rows is enough.

| Experiment | Dataset recipe |
|---|---|
| 1 Regression | 3 random numeric columns (`RM`, `LSTAT`, `TAX`); `MEDV = formula + noise` |
| 2 Logistic | `Pclass`, `Sex`, `Age` (some NaN), `Fare`; `Survived` from a score (female and better class = higher survival) |
| 3-5 Adult | `age`, `education_num`, `hours_per_week`, `sex`, `marital`; `income` (`>50K`/`<=50K`) from a score that rises with education, hours, married |
| 6 Clustering | Stack 3 blocks of `np.random.normal` with different means (small / medium / large spenders) |

**If you are given a real CSV:** delete the dataset lines and use `df = pd.read_csv("file.csv")`.

---

## Exam-day checklist

- Oral exam starts **9:15 a.m. sharp**; practical at **12:30 p.m.** (check your own notice for the date and time)
- College **ID card** and **formal attire** are compulsory
- **Signed practical journal** and a **pen**
- No internet in the practical: practise typing the code without looking
- Prepare the **complete syllabus** for the oral, not only the 6 experiments

---

## Tips for the practical

1. **Always set a seed:** `np.random.seed(1)` gives the same data and results every run.
2. **Order of work:** imports -> data -> clean -> split -> fit -> predict -> metrics -> plot.
3. **Type the imports first.** Most exam errors are a missing import or a typo in a name.
4. **Fill missing values before** `train_test_split` is fine for a lab, but say you would do it after the split in real projects (to avoid data leakage).
5. **Convert text to numbers:** `map({...})` for 2 categories, `pd.get_dummies(..., drop_first=True)` for more.
6. **Use `stratify=y`** for classification so train and test keep the same class ratio.
7. **Never scale or fit on the test set.** Fit on train, only predict on test.
8. **Linear vs Logistic:** continuous target = Linear, 0/1 target = Logistic.
9. **Clustering needs scaling:** use `StandardScaler` before distance-based methods.
10. **Read your output out loud** to the examiner: "R2 is 0.91, so the model explains 91% of the variance."
11. **Run the code once fully before the exam** so you know what each output looks like.
12. If a library is missing and there is no internet, tell the examiner. Do not waste time trying to install it.
13. **AdaBoost instead of Gradient Boosting?** Change only the import and the model line:
    `from sklearn.ensemble import AdaBoostClassifier` and `AdaBoostClassifier(n_estimators=100, random_state=1)`.
14. **Different examiner twist?** Common ones: change `max_depth`, `n_estimators`, `test_size`, `n_clusters`, or the linkage. Know where each one is in the code.

---

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Library not installed | `pip install scikit-learn scipy` (needs internet) |
| `ValueError: could not convert string to float` | Text column not encoded | Use `map` or `get_dummies` |
| `ValueError: Input contains NaN` | Missing values | `df["col"].fillna(df["col"].median())` |
| `ConvergenceWarning` (Logistic) | Not enough iterations | `LogisticRegression(max_iter=1000)` |
| Plot does not appear | `plt.show()` missing | Add `plt.show()` |
| `KeyError` on a column | Typo or case mismatch | Check `df.columns` |
| Shape mismatch | `X` and `y` out of sync | Print `X.shape`, `y.shape` |

---

## Oral exam quick revision

**Regression and classification**
- **Linear Regression:** fits a straight line `y = b0 + b1x1 + ...` by minimising squared error.
- **R2:** share of variance explained by the model. Adjusted R2 penalises useless features.
- **MAE vs MSE vs RMSE:** MAE = average absolute error; MSE squares errors (punishes big ones); RMSE = square root of MSE, in the same units as y.
- **Logistic Regression:** uses the sigmoid to give a probability between 0 and 1; threshold 0.5 gives the class.
- **Confusion matrix:** TN, FP, FN, TP. Precision = TP/(TP+FP). Recall = TP/(TP+FN). F1 = harmonic mean of the two.
- **Accuracy can mislead** when classes are imbalanced.

**Tree-based models**
- **Decision Tree:** splits on the feature that gives the most information gain (entropy) or lowest Gini impurity.
- **Overfitting:** a deep tree memorises training data. Fix with `max_depth`, `min_samples_leaf` (pruning).
- **Random Forest:** many trees on random samples and random features (bagging), then majority vote. Lowers variance.
- **Boosting:** trees built one after another, each fixing the previous errors. Lowers bias.
- **Bagging vs Boosting:** bagging trains in parallel; boosting trains in sequence.
- **AdaBoost vs Gradient Boosting:** AdaBoost reweights misclassified samples; Gradient Boosting fits the residual errors.

**Clustering**
- **Hierarchical clustering** is unsupervised. Agglomerative = start with each point alone and merge the closest pair.
- **Dendrogram:** a tree of merges. Cut it where the vertical gap is largest to pick the number of clusters.
- **Linkage:** single (nearest points, can chain), complete (farthest points, compact), average (in between), Ward (minimise variance, Euclidean only).
- **Distance metrics:** Euclidean, Manhattan, Cosine.
- **Silhouette score:** from -1 to 1; closer to 1 means well-separated clusters.
- **Why scale first?** Large-valued features would otherwise dominate the distance.

**General**
- **Train/test split:** test data is only for final evaluation.
- **Cross-validation:** repeated splits to get a more reliable score.
- **Supervised vs unsupervised:** with labels vs without labels.
- **Overfitting vs underfitting:** high train score with low test score = overfit; both low = underfit.

---

## Contributing

Found a bug or a better explanation? Open an issue or a pull request. Please keep the code short and simple,
since it has to be typed by hand in an exam.

## Disclaimer

The datasets are synthetic and only for practice. Results will differ from the real Boston, Titanic, Adult and
Wholesale datasets. Use this to understand the workflow, not as a substitute for studying the theory.
---
If the examiner gives a real dataset, delete the dataset-creation lines. Add df = pd.read_csv("file.csv") and keep everything below it.
Experiment 6 uses scipy only, so it avoids the sklearn metric= versus affinity= version problem.
Results from my test run: Exp 1 R² ≈ 0.91, Exp 2 accuracy ≈ 0.78, Exp 3 ≈ 0.85, Exp 4 ≈ 0.80, Exp 5 ≈ 0.79, Exp 6 silhouette ≈ 0.68. Your numbers will match exactly if you type the code as given, including np.random.seed(1).
---
