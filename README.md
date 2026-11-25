# Exoplanet Classification with the Kepler Telescope

Master 2 Data Science project, ISIMA Mahdia.

This project is based on the repository of Nicolas Taddei: https://github.com/nicolastaddei/kepler-exoplanet-classification
The data comes from the NASA Exoplanet Archive (*Cumulative KOI* table).

## Live Application

**Try it online:** https://kepler-exoplanet-classification.streamlit.app

The application is hosted on Streamlit Community Cloud and is available at any time. It runs directly from this repository.

---

## The Problem

The Kepler space telescope observed about 150,000 stars and detected repeated drops in brightness. Each drop may come from a **real planet** passing in front of its star, or from a **false signal** (a double star whose two members eclipse each other, an instrument defect, etc.).

Checking each signal with a follow-up telescope is slow and expensive. This project trains a machine learning model to **tell real planets apart from false signals**, using the measurements made for each signal.

## The Data

- **Source**: NASA Exoplanet Archive, *Cumulative KOI* table (`data/raw/koi_cumulative.csv`)
- **Size**: 9,564 signals and 141 columns
- **Verdicts**: 2,748 confirmed planets, 4,839 false positives, 1,977 candidates without a verdict

Columns that already contain the verdict (NASA's score, false-positive flags, planet name, etc.) are **removed from the start**. Without this step, the model would get an artificially high score that does not hold on a new signal. This is called **data leakage**.

After cleaning, the model uses **51 features**: planet radius, transit depth, orbital period, stellar properties, detection quality, and more.

## The Approach

1. **Cleaning**: removal of leakage columns, identifiers, uncertainty columns and empty columns.
2. **Splitting**: 80% of the signals for training, 20% for testing (stratified, to keep the same share of planets in both parts).
3. **Model comparison** with 5-fold cross-validation:

| Model | F1 (cross-validation) |
|---|---|
| Logistic Regression | 0.909 ± 0.007 |
| Random Forest | 0.958 ± 0.004 |
| Gradient Boosting | 0.966 ± 0.007 |

4. **Improvements tested**:
   - **PCA** (reducing the number of features): it lowers the score for every model. It is not kept, because the features are physical measurements used one by one.
   - **Tuning** (8 settings tried): the gain (0.002) is smaller than the variation between folds. The default model is kept.
   - **Calibration**: it turns the model's score into a reliable probability. The final model is calibrated with the sigmoid method.
   - **Cost-based decision threshold**: a missed planet is assumed to cost 10 times more than a false alarm. The optimal threshold is then 0.10 instead of 0.5.

5. **Final evaluation** on the test set (1,518 signals never seen during training, used only once).

## Results

Final model (calibrated gradient boosting, threshold 0.5):

| Metric | Value |
|---|---|
| F1 | 0.978 |
| Accuracy | 98.4% |
| Errors | 24 out of 1,518 (14 missed planets, 10 false alarms) |

With the cost-based threshold (0.10), the model misses only 7 planets but produces 52 false alarms. The right threshold depends on what a wasted follow-up costs, which should be decided with astronomers.

**What the model relies on**: the most important feature is the planet radius. Two light-position measurements (centroids), used by NASA's own vetting, also matter a lot. Without them, the F1 drops from 0.966 to 0.927. The model therefore partly benefits from verification work already done.

**Candidates**: the model ranked the 1,977 candidates without a verdict. 322 of them reach a probability of 0.90 or more. This ranking is meant to **prioritise review by astronomers**, not to announce discoveries.

## The Streamlit Application

The web interface lets you explore the project:

- **Summary**: performance on the test set, with a slider to adjust the threshold.
- **Test a signal**: enter the measurements of a signal and get its probability of being a planet.
- **Analyse a file**: upload a CSV with the NASA table columns and download the predictions.
- **NASA candidates**: ranking of the candidates.
- **Influence of settings**: effect of PCA, calibration, feature importance, and retraining with some features removed.

## Project Structure

```
kepler-exoplanet-classification/
├── app.py                      Streamlit application
├── requirements.txt            Libraries and versions (used by Streamlit Cloud)
├── requirements_streamlit.txt  Streamlit only, for running the app locally
├── .streamlit/config.toml      Blue and white theme
├── data/raw/koi_cumulative.csv NASA table
├── models/
│   ├── kepler_calibrated_model.joblib   Final model
│   └── decision_threshold.json          Decision threshold
└── notebooks/
    └── kepler_complete_project.ipynb    Full notebook: from raw data to conclusions
```

## Running Locally

If you want to run the application on your own computer, from the project folder:

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
pip install -r requirements.txt
streamlit run app.py
```

The application opens at http://localhost:8501.

To retrain the model, open `notebooks/kepler_complete_project.ipynb` and run all cells.

## Limitations

- The range between 1 and 3 Earth radii remains ambiguous: most errors are concentrated there.
- The model relies on NASA verification measurements: it is a pre-screening tool, not an autonomous discovery system.
- The 10:1 cost ratio is an assumption that should be validated.
- Only one dataset (Kepler) is used: the model has not been tested on another mission such as TESS.
- Candidates cannot be validated without new observations.

## Future Work

- Test the model on TESS data.
- Have the cost ratio validated by astronomers.
- Add a per-signal explanation (for example with SHAP).

## License

MIT. See the [LICENSE](LICENSE) file.