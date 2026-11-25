"""Interface Streamlit du projet Kepler : classification d'exoplanètes.

Lancer depuis le dossier du projet :
    streamlit run app.py
"""
import json
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- chemins et constantes
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "raw" / "koi_cumulative.csv"
MODEL_PATH = ROOT / "models" / "kepler_calibrated_model.joblib"
THRESHOLD_PATH = ROOT / "models" / "decision_threshold.json"
RANDOM_STATE = 42

LEAKAGE_COLS = ["koi_score", "koi_pdisposition", "koi_fpflag_nt", "koi_fpflag_ss",
                "koi_fpflag_co", "koi_fpflag_ec", "koi_vet_stat", "koi_vet_date",
                "koi_disp_prov", "koi_comment"]
ID_COLS = ["rowid", "kepid", "kepoi_name", "kepler_name"]
TEXT_COLS = ["koi_fittype", "koi_limbdark_mod", "koi_parm_prov", "koi_tce_delivname", "koi_quarters",
             "koi_trans_mod", "koi_datalink_dvr", "koi_datalink_dvs", "koi_sparprov"]
CONSTANT_COLS = ["koi_eccen", "koi_ldm_coeff3", "koi_ldm_coeff4"]

TOP_FEATURES = {
    "koi_prad": "Rayon de la planète (rayons terrestres)",
    "koi_dikco_msky": "Écart de position de la lumière (mesure de vérification)",
    "koi_max_mult_ev": "Force de détection du signal (MES)",
    "koi_count": "Nombre de signaux détectés sur l'étoile",
    "koi_model_snr": "Rapport signal sur bruit du transit",
    "koi_dor": "Distance planète-étoile, relative au rayon de l'étoile",
    "koi_ror": "Rapport rayon planète / rayon étoile",
    "koi_smet": "Métallicité de l'étoile",
    "koi_depth": "Profondeur du transit (ppm)",
    "koi_impact": "Paramètre d'impact (0 = centre, 1 = bord)",
}

BLUE = "#1E3A8A"
NAVY = "#0B1F44"
LIGHT = "#E8F0FE"

st.set_page_config(page_title="Kepler · Classification d'exoplanètes", page_icon="🪐", layout="wide")
st.markdown(f"""
<style>
  h1, h2, h3 {{ color: {NAVY}; }}
  .stTabs [aria-selected="true"] {{ color: {BLUE} !important; border-bottom-color: {BLUE} !important; }}
  div[data-testid="stMetric"] {{ background: {LIGHT}; border-radius: 10px; padding: 10px 14px;
                                 border-left: 5px solid {BLUE}; }}
  .stButton > button[kind="primary"] {{ background: {BLUE}; border-color: {BLUE}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- chargement
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_threshold_info():
    if THRESHOLD_PATH.exists():
        info = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        return float(info.get("threshold", 0.5)), float(info.get("cost_ratio_fn_to_fp", 10))
    return 0.5, 10.0


@st.cache_data(show_spinner="Lecture du fichier NASA…")
def load_raw():
    return pd.read_csv(RAW_PATH, comment="#")


def clean_labelled(raw):
    """Reproduit le nettoyage du notebook : retire fuite, identifiants, incertitudes, colonnes vides."""
    df = raw.drop(columns=LEAKAGE_COLS + ID_COLS)
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
    df = df.drop(columns=[c for c in df.columns if "_err" in c])
    df = df.drop(columns=df.columns[df.isna().all()].tolist())
    y = df["koi_disposition"].map({"FALSE POSITIVE": 0, "CONFIRMED": 1})
    X = df.drop(columns=["koi_disposition"] + TEXT_COLS + CONSTANT_COLS)
    return X, y


@st.cache_data(show_spinner="Préparation des données…")
def load_splits():
    X, y = clean_labelled(load_raw())
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    return X_tr, X_te, y_tr, y_te


def feature_names(model):
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else list(load_splits()[0].columns)


@st.cache_resource(show_spinner="Calcul des prédictions sur le jeu de test…")
def test_probabilities():
    _, X_te, _, y_te = load_splits()
    proba = model.predict_proba(X_te[FEATURES])[:, 1]
    return proba, y_te.values


def predict(model_, df):
    return model_.predict_proba(df[FEATURES])[:, 1]


def verdict(p, threshold):
    return "Planète probable (à suivre)" if p >= threshold else "Faux signal probable"


# ---------------------------------------------------------------- fonctions des variantes
def base_pipeline(estimator, pca=None):
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if pca is not None:
        steps.append(PCA(n_components=pca, random_state=RANDOM_STATE))
    steps.append(estimator)
    return make_pipeline(*steps)


CALIBRATION_METHODS = {"Aucune": None, "Sigmoïde": "sigmoid", "Isotonique": "isotonic"}


@st.cache_resource(show_spinner="Entraînement de la variante…")
def fit_variant(model_name, pca_value, calibration_label, removed):
    """Entraîne une variante sur le jeu d'entraînement uniquement. Résultats mis en cache."""
    X_tr, _, y_tr, _ = load_splits()
    cols = [c for c in X_tr.columns if c not in removed]
    if model_name == "Random Forest":
        est = RandomForestClassifier(random_state=RANDOM_STATE)
    else:
        est = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    pipe = base_pipeline(est, pca=pca_value)
    method = CALIBRATION_METHODS[calibration_label]
    fitted = CalibratedClassifierCV(estimator=pipe, method=method, cv=5) if method else pipe
    fitted.fit(X_tr[cols], y_tr)
    return fitted, cols


def evaluate(model_, cols, threshold=0.5):
    _, X_te, _, y_te = load_splits()
    p = model_.predict_proba(X_te[cols])[:, 1]
    pred = (p >= threshold).astype(int)
    return {
        "F1": f1_score(y_te, pred),
        "Accuracy": accuracy_score(y_te, pred),
        "Erreurs": int(np.sum(pred != y_te.values)),
        "Brier (↓)": brier_score_loss(y_te, p),
        "Log-loss (↓)": log_loss(y_te, np.clip(p, 1e-6, 1 - 1e-6)),
        "Valeurs distinctes": int(len(np.unique(np.round(p, 6)))),
    }, p


@st.cache_data(show_spinner=False)
def pca_components(variance):
    X_tr, _, _, _ = load_splits()
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_tr))
    return int(PCA(n_components=variance, random_state=RANDOM_STATE).fit(Z).n_components_)


@st.cache_resource(show_spinner="Calcul de l'importance des variables…")
def permutation_table():
    _, X_te, _, y_te = load_splits()
    r = permutation_importance(model, X_te[FEATURES], y_te, scoring="f1",
                               n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    return (pd.DataFrame({"importance": r.importances_mean, "ecart_type": r.importances_std}, index=FEATURES)
              .sort_values("importance", ascending=False))


def pca_scatter_figure():
    X_tr, _, y_tr, _ = load_splits()
    Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_tr))
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z)
    P = pca2.transform(Z)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for label, color, name in [(0, "#94a3b8", "Faux signal"), (1, BLUE, "Planète")]:
        mask = y_tr.values == label
        ax.scatter(P[mask, 0], P[mask, 1], s=6, alpha=0.45, c=color, label=name)
    ax.set_xlabel(f"Composante 1 ({pca2.explained_variance_ratio_[0]:.0%} de l'information)")
    ax.set_ylabel(f"Composante 2 ({pca2.explained_variance_ratio_[1]:.0%})")
    ax.set_title("Deux composantes : les classes se chevauchent", color=NAVY)
    ax.legend(markerscale=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- vérifications
model = load_model()
DEFAULT_THRESHOLD, COST_RATIO = load_threshold_info()

st.title("🪐 Classification d'exoplanètes avec le télescope Kepler")
st.caption("Un modèle qui distingue les vraies planètes des faux signaux, à partir des mesures de chaque signal.")

if model is None:
    st.error(f"Modèle introuvable : {MODEL_PATH.relative_to(ROOT)}. Exécute d'abord le notebook principal.")
    st.stop()
if not RAW_PATH.exists():
    st.error(f"Données introuvables : {RAW_PATH.relative_to(ROOT)}. Place le fichier koi_cumulative.csv dans data/raw/.")
    st.stop()

X_tr, X_te, y_tr, y_te = load_splits()
FEATURES = feature_names(model)
MEDIANS = X_tr[FEATURES].median()
P_LOW = X_tr[FEATURES].quantile(0.01)
P_HIGH = X_tr[FEATURES].quantile(0.99)

# ---------------------------------------------------------------- barre latérale
st.sidebar.header("Réglages")
threshold = st.sidebar.slider(
    "Seuil de décision", 0.01, 0.99, float(DEFAULT_THRESHOLD), 0.01,
    help="Probabilité à partir de laquelle un signal est déclaré planète. "
         "Plus le seuil baisse, plus on signale de signaux, et moins on rate de planètes.")
st.sidebar.caption(f"Seuil d'origine : {DEFAULT_THRESHOLD:.2f}, choisi pour un coût de {COST_RATIO:.0f} "
                   "par planète manquée, contre 1 par fausse alerte.")
st.sidebar.markdown("---")
st.sidebar.caption("Modèle : gradient boosting calibré (sigmoid). Données : NASA, table Cumulative KOI.")

tab_resume, tab_signal, tab_csv, tab_candidats, tab_influence = st.tabs(
    ["📊 Résumé", "🔭 Tester un signal", "📁 Analyser un fichier", "🌌 Candidats NASA",
     "⚙️ Influence des réglages"])

# ---------------------------------------------------------------- onglet 1 : résumé
with tab_resume:
    st.subheader("Performance sur le jeu de test (1 518 signaux jamais vus)")
    st.write("Les chiffres ci-dessous sont calculés en direct avec le modèle sauvegardé. "
             "Le seuil se règle dans la barre latérale.")

    proba, y_true = test_probabilities()

    def metrics_at(t):
        pred = (proba >= t).astype(int)
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        return {
            "F1": f1_score(y_true, pred), "Accuracy": accuracy_score(y_true, pred),
            "Précision": precision_score(y_true, pred), "Rappel": recall_score(y_true, pred),
            "Planètes manquées": fn, "Fausses alertes": fp,
            "Coût total": fp + COST_RATIO * fn,
        }, pred

    m, pred = metrics_at(threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1", f"{m['F1']:.3f}")
    c2.metric("Accuracy", f"{m['Accuracy']:.1%}")
    c3.metric("Planètes manquées", m["Planètes manquées"])
    c4.metric("Fausses alertes", m["Fausses alertes"])

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**Matrice de confusion (seuil {threshold:.2f})**")
        cm = confusion_matrix(y_true, pred)
        st.dataframe(pd.DataFrame(cm, index=["Réel : faux signal", "Réel : planète"],
                                  columns=["Prédit : faux signal", "Prédit : planète"]), width="stretch")
        st.caption(f"Coût total avec un ratio {COST_RATIO:.0f}:1 : {m['Coût total']}")
    with right:
        st.markdown("**Comparaison de deux seuils**")
        m05, _ = metrics_at(0.5)
        mo, _ = metrics_at(DEFAULT_THRESHOLD)
        comp = pd.DataFrame({
            "Seuil 0,50": [m05["F1"], m05["Précision"], m05["Rappel"], m05["Planètes manquées"], m05["Fausses alertes"], m05["Coût total"]],
            f"Seuil {DEFAULT_THRESHOLD:.2f} (coût)": [mo["F1"], mo["Précision"], mo["Rappel"], mo["Planètes manquées"], mo["Fausses alertes"], mo["Coût total"]],
        }, index=["F1", "Précision", "Rappel", "Planètes manquées", "Fausses alertes", "Coût total"])
        st.dataframe(comp.style.format("{:.3f}", subset=pd.IndexSlice[["F1", "Précision", "Rappel"], :]),
                     width="stretch")

    st.info("Un seuil bas rate moins de planètes, mais génère davantage de fausses alertes. "
            "Le bon compromis dépend de ce que coûte un suivi inutile, à fixer avec les astronomes.")

# ---------------------------------------------------------------- onglet 2 : un seul signal
with tab_signal:
    st.subheader("Évaluer un signal")
    st.write("Règle les variables principales. Les autres sont préremplies avec la valeur médiane des données d'entraînement.")

    with st.form("signal_form"):
        cols = st.columns(2)
        values = {}
        for i, (name, help_text) in enumerate(TOP_FEATURES.items()):
            lo, hi = float(P_LOW[name]), float(P_HIGH[name])
            default = float(np.clip(MEDIANS[name], lo, hi))
            step = (hi - lo) / 200 if hi > lo else 0.01
            with cols[i % 2]:
                values[name] = st.slider(name, lo, hi, default, step=step, help=help_text)
        with st.expander("Toutes les autres variables (valeurs médianes par défaut)"):
            for name in FEATURES:
                if name in values:
                    continue
                values[name] = st.number_input(name, value=float(MEDIANS[name]), format="%.4g")
        submitted = st.form_submit_button("Prédire", type="primary")

    if submitted:
        row = pd.DataFrame([{f: values[f] for f in FEATURES}])
        p = float(predict(model, row)[0])
        a, b = st.columns([1, 2])
        a.metric("Probabilité d'être une planète", f"{p:.1%}")
        b.markdown(f"### {verdict(p, threshold)}")
        b.progress(min(max(p, 0.0), 1.0))
        st.caption("Estimation statistique à partir de signaux déjà connus. Elle ne remplace pas la vérification par des astronomes.")

# ---------------------------------------------------------------- onglet 3 : fichier CSV
with tab_csv:
    st.subheader("Analyser un fichier de signaux")
    st.write("Charge un CSV contenant les mêmes colonnes que la table NASA (les colonnes en trop sont ignorées). "
             "Si la colonne `koi_disposition` est présente, les résultats sont comparés au verdict.")
    up = st.file_uploader("Fichier CSV", type=["csv"])
    if up is not None:
        try:
            data = pd.read_csv(BytesIO(up.getvalue()), comment="#")
        except Exception as e:
            st.error(f"Lecture impossible : {e}")
            data = None
        if data is not None:
            missing = [f for f in FEATURES if f not in data.columns]
            if missing:
                st.error(f"Colonnes manquantes ({len(missing)}) : {', '.join(missing[:15])}"
                         + (" …" if len(missing) > 15 else ""))
            else:
                res = data.copy()
                res["probabilite_planete"] = predict(model, data)
                res["verdict"] = np.where(res["probabilite_planete"] >= threshold,
                                          "Planète probable", "Faux signal probable")
                res = res.sort_values("probabilite_planete", ascending=False)
                n_flag = int((res["probabilite_planete"] >= threshold).sum())
                st.metric("Signaux signalés (planètes probables)", f"{n_flag} / {len(res)}")

                if "koi_disposition" in res.columns:
                    lab = res[res["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])]
                    if len(lab):
                        yt = lab["koi_disposition"].map({"FALSE POSITIVE": 0, "CONFIRMED": 1}).values
                        yp = (lab["probabilite_planete"] >= threshold).astype(int).values
                        a, b, c = st.columns(3)
                        a.metric("F1 (signaux étiquetés)", f"{f1_score(yt, yp):.3f}")
                        b.metric("Accuracy", f"{accuracy_score(yt, yp):.1%}")
                        c.metric("Erreurs", int((yt != yp).sum()))

                show_cols = [c for c in ["kepoi_name", "koi_disposition", "probabilite_planete", "verdict"] if c in res.columns]
                show_cols += [c for c in FEATURES[:3] if c not in show_cols]
                st.dataframe(res[show_cols].head(500), width="stretch")
                st.download_button("Télécharger les résultats (CSV)",
                                   res.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions_kepler.csv", mime="text/csv")

# ---------------------------------------------------------------- onglet 4 : candidats
with tab_candidats:
    st.subheader("Classer les 1 977 candidats NASA")
    st.warning("Les candidats n'ont pas de verdict officiel. Ce classement sert à prioriser une revue par les astronomes, "
               "pas à annoncer des découvertes.")
    raw = load_raw()
    cand = raw[raw["koi_disposition"] == "CANDIDATE"].copy()
    cand["probabilite_planete"] = predict(model, cand)
    cand = cand.sort_values("probabilite_planete", ascending=False)

    a, b, c = st.columns(3)
    a.metric("Candidats", len(cand))
    b.metric("Au-dessus de 0,90", int((cand["probabilite_planete"] >= 0.90).sum()))
    c.metric(f"Au-dessus du seuil ({threshold:.2f})", int((cand["probabilite_planete"] >= threshold).sum()))

    top_n = st.slider("Nombre de candidats à afficher", 5, 50, 10)
    table = cand[["kepoi_name", "koi_prad", "koi_period", "probabilite_planete"]].head(top_n)
    st.dataframe(table.style.format({"koi_prad": "{:.2f}", "koi_period": "{:.1f}", "probabilite_planete": "{:.4f}"}),
                 width="stretch")
    st.download_button("Télécharger le classement complet (CSV)",
                       cand[["kepoi_name", "koi_prad", "koi_period", "probabilite_planete"]].to_csv(index=False).encode("utf-8"),
                       file_name="classement_candidats.csv", mime="text/csv")

# ---------------------------------------------------------------- onglet 5 : influence des réglages
with tab_influence:
    st.subheader("Comment les réglages changent les résultats")
    st.caption("Ces tests servent à expliquer le projet. Chaque variante est entraînée sur le jeu d'entraînement, "
               "puis évaluée sur le jeu de test. Le modèle final de l'application n'est pas modifié. "
               "Les calculs se lancent avec un bouton, et sont ensuite gardés en mémoire.")

    # ---- 1. PCA
    st.markdown("### 1. La PCA : réduire le nombre de variables")
    st.write("La PCA remplace les 51 variables par de nouvelles variables, plus courtes, qui gardent une part donnée de l'information. "
             "Plus on en garde, plus on reste proche des données d'origine.")
    variance = st.slider("Part de l'information conservée", 0.80, 0.99, 0.95, 0.01, key="pca_var")
    if st.button("Calculer l'effet de la PCA", key="btn_pca", type="primary"):
        nb = pca_components(variance)
        sans_m, _ = evaluate(*fit_variant("Gradient boosting", None, "Aucune", ()))
        avec_m, _ = evaluate(*fit_variant("Gradient boosting", variance, "Aucune", ()))
        st.metric("Variables après PCA", f"{nb} au lieu de {len(FEATURES)}")
        table_pca = pd.DataFrame([sans_m, avec_m], index=["Sans PCA", f"PCA ({variance:.0%})"])
        st.dataframe(table_pca.style.format({"F1": "{:.3f}", "Accuracy": "{:.3f}", "Brier (↓)": "{:.4f}",
                                             "Log-loss (↓)": "{:.3f}"}), width="stretch")
        left, right = st.columns([1, 1])
        with left:
            st.pyplot(pca_scatter_figure())
        with right:
            delta = avec_m["F1"] - sans_m["F1"]
            if delta < -0.005:
                st.warning(f"Le F1 baisse de {abs(delta):.3f} avec la PCA. Les variables sont des mesures physiques "
                           "utilisées une par une : la PCA les mélange et perd de l'information utile.")
            else:
                st.success(f"Le F1 change très peu ({delta:+.3f}). La PCA ne dégrade pas le modèle ici.")
            st.caption("Essaie une part plus faible (par exemple 0,80) : le nombre de variables baisse, "
                       "et le F1 baisse généralement avec.")

    st.markdown("---")

    # ---- 2. Calibration
    st.markdown("### 2. La calibration : transformer un score en probabilité")
    st.write("Un modèle à arbres donne un score. La calibration le corrige pour qu'un score de 0,7 corresponde "
             "à 70 % de vraies planètes dans ce groupe de signaux.")
    c1, c2 = st.columns(2)
    model_choice = c1.radio("Modèle", ["Gradient boosting", "Random Forest"], horizontal=True, key="cal_model")
    method_choice = c2.radio("Méthode de calibration", ["Sigmoïde", "Isotonique"], horizontal=True, key="cal_method")
    if st.button("Calculer l'effet de la calibration", key="btn_cal", type="primary"):
        brut_m, cols_b = fit_variant(model_choice, None, "Aucune", ())
        cal_m, cols_c = fit_variant(model_choice, None, method_choice, ())
        brut_metrics, p_brut = evaluate(brut_m, cols_b)
        cal_metrics, p_cal = evaluate(cal_m, cols_c)
        table_cal = pd.DataFrame([brut_metrics, cal_metrics],
                                 index=[f"{model_choice} brut", f"{model_choice} calibré ({method_choice.lower()})"])
        st.dataframe(table_cal.style.format({"F1": "{:.3f}", "Accuracy": "{:.3f}", "Brier (↓)": "{:.4f}",
                                             "Log-loss (↓)": "{:.3f}"}), width="stretch")

        _, _, _, y_te_ = load_splits()
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "k--", label="calibration parfaite")
        for name, p, color in [("brut", p_brut, "#94a3b8"), (f"calibré ({method_choice.lower()})", p_cal, BLUE)]:
            fr, mp = calibration_curve(y_te_, p, n_bins=10, strategy="uniform")
            ax.plot(mp, fr, marker="o", color=color, label=name)
        ax.set_xlabel("Probabilité annoncée")
        ax.set_ylabel("Fréquence réelle de planètes")
        ax.set_title("Courbe de fiabilité (jeu de test)", color=NAVY)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        left, right = st.columns([1, 1])
        with left:
            st.pyplot(fig)
        with right:
            st.info(f"Le Brier passe de {brut_metrics['Brier (↓)']:.4f} à {cal_metrics['Brier (↓)']:.4f}, "
                    f"et les valeurs distinctes de {brut_metrics['Valeurs distinctes']} à {cal_metrics['Valeurs distinctes']}. "
                    "Plus il y a de valeurs distinctes, plus le classement est précis.")
            if model_choice == "Random Forest":
                st.caption("La Random Forest brute donne peu de valeurs distinctes : beaucoup de signaux ont le même score. "
                           "La calibration corrige cela, mais elle ne suffit pas à la rattraper complètement.")

    st.markdown("---")

    # ---- 3. Importance des variables
    st.markdown("### 3. L'importance des variables")
    st.write("On brouille une variable à la fois sur le jeu de test, et on mesure la baisse du F1. "
             "Plus la baisse est grande, plus le modèle s'appuie sur cette variable.")
    top_n = st.slider("Nombre de variables affichées", 5, 20, 10, key="imp_n")
    if st.button("Calculer l'importance des variables", key="btn_imp", type="primary"):
        imp = permutation_table()
        top = imp.head(top_n)[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.42 * top_n + 1.5))
        ax.barh(top.index, top["importance"], xerr=top["ecart_type"], color=BLUE, ecolor=NAVY, capsize=3)
        ax.set_xlabel("Baisse du F1 quand la variable est brouillée")
        ax.set_title("Variables les plus importantes", color=NAVY)
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        left, right = st.columns([1, 1])
        with left:
            st.pyplot(fig)
        with right:
            st.dataframe(imp.head(top_n).style.format({"importance": "{:.4f}", "ecart_type": "{:.4f}"}),
                         width="stretch")
            st.caption("La petite barre noire est l'écart-type entre les répétitions : plus elle est courte, plus la mesure est stable.")

    st.markdown("#### Et si on retire des variables ?")
    st.write("Choisis des variables à retirer, puis réentraîne le modèle sans elles. "
             "Cela montre si le modèle peut se passer de ces informations.")
    removed = st.multiselect("Variables à retirer", FEATURES, key="removed")
    if st.button("Réentraîner sans ces variables", key="btn_rm", type="primary", disabled=not removed):
        base_m, base_cols = fit_variant("Gradient boosting", None, "Aucune", ())
        red_m, red_cols = fit_variant("Gradient boosting", None, "Aucune", tuple(sorted(removed)))
        base_metrics, _ = evaluate(base_m, base_cols)
        red_metrics, _ = evaluate(red_m, red_cols)
        table_rm = pd.DataFrame([base_metrics, red_metrics],
                                index=["Toutes les variables", f"Sans {len(removed)} variable(s)"])
        st.dataframe(table_rm.style.format({"F1": "{:.3f}", "Accuracy": "{:.3f}", "Brier (↓)": "{:.4f}",
                                            "Log-loss (↓)": "{:.3f}"}), width="stretch")
        delta = red_metrics["F1"] - base_metrics["F1"]
        if delta < -0.01:
            st.warning(f"Le F1 baisse de {abs(delta):.3f}. Les variables retirées apportent une information utile.")
        else:
            st.success(f"Le F1 change peu ({delta:+.3f}). Le modèle se passe bien de ces variables.")
