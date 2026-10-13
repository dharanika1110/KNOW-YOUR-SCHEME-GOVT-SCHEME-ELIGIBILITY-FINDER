"""
Know Your Scheme - core engine (no Streamlit code here, so it can be unit-tested).

  1. Data loading + cleaning          (myScheme dataset: 3,397 schemes, NFHS-5 district indicators)
  2. Rule-based eligibility engine + profile-matching score + TF-IDF text relevance
  3. Eligibility classifier (Logistic Regression vs Random Forest)  -> Module 2 of the report
  4. District analytics (K-Means + Random Forest regression on NFHS-5)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
SEED = 42

# --------------------------------------------------------------------------- #
# Constants shared with the UI
# --------------------------------------------------------------------------- #
OCCUPATIONS = [
    "Student", "Farmer", "Laborer / Worker", "Entrepreneur / MSME", "Artisan / Craftsman",
    "Fisherman", "Unemployed / Job seeker", "Salaried employee", "Homemaker", "Other",
]
# user occupation -> occupation tag used in the scheme data
OCC_TO_TAG = {
    "Student": "Student", "Farmer": "Farmer", "Laborer / Worker": "Laborer/Worker",
    "Entrepreneur / MSME": "Entrepreneur/MSME", "Artisan / Craftsman": "Artisan/Craftsman",
    "Fisherman": "Fisherman", "Unemployed / Job seeker": "Unemployed",
}
OCC_TAGS = ["Student", "Laborer/Worker", "Entrepreneur/MSME", "Farmer", "Person with Disability",
            "Widow", "Senior Citizen", "Artisan/Craftsman", "Unemployed", "Fisherman"]
SCHEME_CATS = ["SC", "ST", "OBC", "EWS", "Minority"]
CATEGORIES = ["General", "OBC", "SC", "ST"]
GENDERS = ["Female", "Male", "Other"]
NO_STATE = ("", "Unknown", "Not specified", "All India")

# NFHS indicator -> text query used to find the schemes that address that gap
INDICATOR_QUERY = {
    "Clean cooking fuel": "LPG gas connection clean cooking fuel",
    "Health insurance cover": "health insurance cover hospitalisation treatment",
    "Institutional births": "maternity benefit pregnant women institutional delivery",
    "Institutional births (public)": "maternity benefit pregnant women institutional delivery",
    "Women literate": "girl child education literacy scholarship",
    "Women 10+ yrs schooling": "girl child secondary education scholarship",
    "Female ever attended school": "girl child education school enrolment",
    "Improved sanitation": "toilet sanitation construction subsidy",
    "Improved drinking water": "drinking water supply connection",
    "Electricity": "electricity connection solar power subsidy",
}
STATE_ALIASES = {
    "andaman and nicobar island": "Andaman and Nicobar Islands",
    "dadra and nagar haveli and daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
}


def canon_state(name: str) -> str:
    n = str(name).replace("&", "and").strip()
    return STATE_ALIASES.get(n.lower(), n)


# --------------------------------------------------------------------------- #
# 1. Data loading & cleaning
# --------------------------------------------------------------------------- #
def _find(*names: str) -> Path:
    for n in names:
        for folder in (BASE / "data", BASE):
            p = folder / n
            if p.exists():
                return p
    raise FileNotFoundError(f"None of {names} found in /data or the app folder.")


def _clean_text(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str).str.replace("\ufeff", "", regex=False)
            .str.replace(r"[ \t]+", " ", regex=True).str.strip())


def _split(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).apply(lambda v: [x.strip() for x in v.split(",") if x.strip()])


def load_schemes() -> pd.DataFrame:
    """Load + clean the myScheme dataset (3,397 Central & State schemes)."""
    df = pd.read_csv(_find("schemes_processed.csv", "schemes.csv"))
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates(subset="slug").reset_index(drop=True)
    for c in ["scheme_name", "details", "benefits", "eligibility", "application", "documents", "tags",
              "state", "level", "primary_category", "gender", "caste_category", "occupation_tags"]:
        df[c] = _clean_text(df[c]) if c in df else ""
    df["state"] = df["state"].replace({"Unknown": "Not specified"})
    df["category_tag"] = df["primary_category"].replace("", "Other")
    df["gender"] = df["gender"].replace({"Any": "All", "": "All"})

    # numeric sanity checks (age / income are key eligibility criteria)
    df["min_age"] = pd.to_numeric(df["min_age"], errors="coerce")
    df["max_age"] = pd.to_numeric(df["max_age"], errors="coerce")
    df["max_annual_income"] = pd.to_numeric(df["max_income"], errors="coerce")
    df.loc[df["max_annual_income"] < 1000, "max_annual_income"] = np.nan        # invalid / typo values
    bad = df["min_age"].notna() & df["max_age"].notna() & (df["max_age"] < df["min_age"])
    df.loc[bad, "max_age"] = np.nan
    df["min_age"] = df["min_age"].fillna(0).astype(int)
    df["max_age"] = df["max_age"].fillna(120).astype(int)

    df["caste_list"] = _split(df["caste_category"].replace("General/Any", ""))
    df["occ_list"] = _split(df["occupation_tags"])
    df["bpl_required"] = df["eligibility"].str.contains(r"\bBPL\b|below poverty line", case=False, regex=True)
    df["official_link"] = "https://www.myscheme.gov.in/schemes/" + df["slug"]
    return df


def list_states(schemes: pd.DataFrame) -> list[str]:
    return sorted(s for s in schemes["state"].unique() if s not in NO_STATE)


def load_nfhs() -> pd.DataFrame:
    """Load + clean NFHS-5 district indicators and give columns short readable names."""
    df = pd.read_csv(_find("nfhs_district_wide.csv", "nfhs_district.csv"))
    df.columns = [c.strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if c in ("State", "District"):
            continue
        if "female population" in lc:
            rename[c] = "Female ever attended school"
        elif "clean fuel" in lc:
            rename[c] = "Clean cooking fuel"
        elif "health insurance" in lc:
            rename[c] = "Health insurance cover"
        elif "public facility" in lc:
            rename[c] = "Institutional births (public)"
        elif "institutional births" in lc:
            rename[c] = "Institutional births"
        elif "below age 15" in lc:
            rename[c] = "Population below 15 yrs"
        elif "sanitation" in lc:
            rename[c] = "Improved sanitation"
        elif "drinking-water" in lc or "drinking water" in lc:
            rename[c] = "Improved drinking water"
        elif "electricity" in lc:
            rename[c] = "Electricity"
        elif "women who are literate" in lc:
            rename[c] = "Women literate"
        elif "10 or more years" in lc:
            rename[c] = "Women 10+ yrs schooling"
        else:
            rename[c] = re.sub(r"[\d\(\)%]", "", c).strip()
    df = df.rename(columns=rename)
    df["State"] = df["State"].apply(canon_state)
    ind = [c for c in df.columns if c not in ("State", "District")]
    df[ind] = df[ind].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=ind).drop_duplicates(subset=["State", "District"]).reset_index(drop=True)
    return df


def indicator_cols(nfhs: pd.DataFrame) -> list[str]:
    return [c for c in nfhs.columns if c not in ("State", "District")]


# --------------------------------------------------------------------------- #
# 2. Rule-based eligibility engine + match score
# --------------------------------------------------------------------------- #
def scheme_records(schemes: pd.DataFrame) -> list[dict]:
    return schemes.to_dict("records")


def user_tags(p: dict) -> set[str]:
    """Occupation / situation tags that describe this user (same vocabulary as the scheme data)."""
    t = set()
    if p["occupation"] in OCC_TO_TAG:
        t.add(OCC_TO_TAG[p["occupation"]])
    if p.get("disability"):
        t.add("Person with Disability")
    if p.get("widow"):
        t.add("Widow")
    if p["age"] >= 60:
        t.add("Senior Citizen")
    return t


def user_groups(p: dict) -> set[str]:
    g = {p["category"]}
    if p.get("minority"):
        g.add("Minority")
    if p.get("ews"):
        g.add("EWS")
    return g


def check_scheme(p: dict, s: dict) -> dict:
    """Eligibility, list of unmet criteria and number of 'specific' criteria the user matched."""
    fails, specific = [], 0

    st_ = s["state"]
    if s["level"] == "State" and st_ not in NO_STATE:
        if p["state"] != st_:
            fails.append(f"Only for residents of {st_}")
        else:
            specific += 2                                   # a scheme from the user's own state

    if p["age"] < s["min_age"]:
        fails.append(f"Minimum age is {s['min_age']} (you: {p['age']})")
    elif p["age"] > s["max_age"]:
        fails.append(f"Maximum age is {s['max_age']} (you: {p['age']})")
    elif s["min_age"] > 0 or s["max_age"] < 120:
        specific += 1

    if s["gender"] not in ("", "All"):
        if p["gender"] != s["gender"]:
            fails.append(f"For {s['gender'].lower()} applicants only")
        else:
            specific += 1

    if s["caste_list"]:
        if user_groups(p) & set(s["caste_list"]):
            specific += 1
        else:
            fails.append("Meant for " + " / ".join(s["caste_list"]) + " applicants")

    if s["occ_list"]:
        if user_tags(p) & set(s["occ_list"]):
            specific += 1
        else:
            fails.append("Meant for: " + ", ".join(s["occ_list"]))

    cap = s["max_annual_income"]
    if pd.notna(cap) and p.get("income") is not None:
        if p["income"] > cap:
            fails.append(f"Family income must be ≤ ₹{int(cap):,} (you: ₹{int(p['income']):,})")
        else:
            specific += 1

    if s["bpl_required"]:
        if p.get("bpl", False):
            specific += 1
        else:
            fails.append("Mentions a BPL household requirement")

    return {"eligible": not fails, "fails": fails, "specific": specific}


def build_text_index(schemes: pd.DataFrame):
    """TF-IDF index over scheme text (name, category, tags, details, benefits)."""
    docs = (schemes["scheme_name"] + " " + schemes["scheme_name"] + " " + schemes["category_tag"] + " " +
            schemes["tags"] + " " + schemes["details"].str[:700] + " " + schemes["benefits"].str[:500])
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
    return vec, vec.fit_transform(docs)


def text_scores(index, query: str) -> np.ndarray:
    vec, mat = index
    if not query.strip():
        return np.zeros(mat.shape[0])
    return (mat @ vec.transform([query]).T).toarray().ravel()


def match_score(res: dict, in_interest: bool, sim: float) -> float:
    """Profile-matching score (0-99). Eligible schemes start at 50 and gain points for how
    targeted the scheme is at the user (up to +27), interest area (+10) and text relevance (up to +12)."""
    if not res["eligible"]:
        return 0.0
    score = 50 + min(res["specific"], 6) * 4.5 + (10 if in_interest else 0) + min(sim / 0.30, 1.0) * 12
    return float(min(round(score, 1), 99))


def recommend(p: dict, schemes: pd.DataFrame, interests: list[str] | None = None, query: str = "",
              text_index=None) -> pd.DataFrame:
    sims = text_scores(text_index, query) if text_index is not None else np.zeros(len(schemes))
    interests = set(interests or [])
    rows = []
    for i, s in enumerate(scheme_records(schemes)):
        r = check_scheme(p, s)
        rows.append({"eligible": r["eligible"], "fails": r["fails"], "n_fails": len(r["fails"]),
                     "relevance": float(sims[i]),
                     "match": match_score(r, s["category_tag"] in interests, sims[i])})
    out = pd.concat([schemes.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out.sort_values(["eligible", "match"], ascending=[False, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Eligibility classifier (Logistic Regression vs Random Forest)
# --------------------------------------------------------------------------- #
def synth_profiles(n: int, states: list[str], seed: int = SEED) -> pd.DataFrame:
    """Simulate citizens with realistic marginals (used to train / test the classifier)."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.gamma(5.0, 7.5, n).round().astype(int) + 5, 5, 95)
    income = np.clip(rng.lognormal(12.2, 0.9, n), 0, 3_000_000).round(-3)
    occ = rng.choice(OCCUPATIONS, n, p=[.13, .13, .15, .08, .05, .03, .10, .17, .10, .06])
    occ = np.where((age < 22) & (rng.random(n) < .7), "Student", occ)
    female = rng.random(n) < .49
    return pd.DataFrame({
        "age": age, "income": income,
        "gender": np.where(rng.random(n) < .02, "Other", np.where(female, "Female", "Male")),
        "category": rng.choice(CATEGORIES, n, p=[.30, .40, .20, .10]),
        "occupation": occ,
        "bpl": (income < 150_000) & (rng.random(n) < .6),
        "disability": rng.random(n) < .08,
        "widow": female & (age > 30) & (rng.random(n) < .07),
        "minority": rng.random(n) < .15,
        "ews": (income < 800_000) & (rng.random(n) < .10),
        "state": rng.choice(states, n),
    })


def _profile_matrix(profiles: pd.DataFrame) -> pd.DataFrame:
    P = pd.DataFrame({"age": profiles["age"].astype(float),
                      "income": pd.to_numeric(profiles["income"], errors="coerce").fillna(0).astype(float)})
    for c in ("bpl", "disability", "widow", "minority", "ews"):
        P[c] = profiles[c].astype(float).values
    for g in GENDERS:
        P[f"gender_{g}"] = (profiles["gender"].values == g).astype(float)
    for c in CATEGORIES:
        P[f"cat_{c}"] = (profiles["category"].values == c).astype(float)
    for o in OCCUPATIONS:
        P[f"occ_{o}"] = (profiles["occupation"].values == o).astype(float)
    return P


def _scheme_matrix(schemes: pd.DataFrame) -> pd.DataFrame:
    S = pd.DataFrame({
        "sch_min_age": schemes["min_age"].astype(float),
        "sch_max_age": schemes["max_age"].astype(float),
        "sch_income_cap": schemes["max_annual_income"].fillna(5_000_000).astype(float),
        "sch_has_income_cap": schemes["max_annual_income"].notna().astype(float),
        "sch_female_only": (schemes["gender"] == "Female").astype(float),
        "sch_male_only": (schemes["gender"] == "Male").astype(float),
        "sch_bpl": schemes["bpl_required"].astype(float),
    })
    for c in SCHEME_CATS:
        S[f"sch_cat_{c}"] = schemes["caste_list"].apply(lambda l: float(c in l))
    for o in OCC_TAGS:
        S[f"sch_occ_{o}"] = schemes["occ_list"].apply(lambda l: float(o in l))
    return S


def _scheme_state(schemes: pd.DataFrame) -> np.ndarray:
    """'' for schemes open to every state, otherwise the state name."""
    return np.where((schemes["level"] == "State") & (~schemes["state"].isin(NO_STATE)), schemes["state"], "")


def _pair_features(P: pd.DataFrame, pstate: np.ndarray, S: pd.DataFrame, sstate: np.ndarray,
                   pi: np.ndarray, si: np.ndarray) -> pd.DataFrame:
    X = np.hstack([P.values[pi], S.values[si], ((sstate[si] == "") | (sstate[si] == pstate[pi])).astype(float)[:, None]])
    return pd.DataFrame(X, columns=list(P.columns) + list(S.columns) + ["state_ok"])


def train_eligibility_models(schemes: pd.DataFrame, n_profiles: int = 4000, per_profile: int = 50) -> dict:
    """Train Logistic Regression + Random Forest on (citizen, scheme) pairs labelled by the rule engine.
    The pair set is class-balanced (50% eligible) so accuracy is not inflated by the majority class, and the
    train/test split is by citizen, so the models are tested on people they have never seen."""
    rng = np.random.default_rng(SEED)
    states = list_states(schemes)
    profiles = synth_profiles(n_profiles, states)
    P, S, sstate = _profile_matrix(profiles), _scheme_matrix(schemes), _scheme_state(schemes)
    pstate = profiles["state"].values
    n_s = len(schemes)

    pi = np.repeat(np.arange(n_profiles), per_profile)
    si = rng.integers(0, n_s, pi.size)
    prof, srec = profiles.to_dict("records"), scheme_records(schemes)
    y = np.fromiter((check_scheme(prof[a], srec[b])["eligible"] for a, b in zip(pi, si)), dtype=int, count=pi.size)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n_keep = min(len(pos), len(neg))
    keep = np.sort(np.concatenate([rng.choice(pos, n_keep, replace=False), rng.choice(neg, n_keep, replace=False)]))
    pi, si, y = pi[keep], si[keep], y[keep]
    X = _pair_features(P, pstate, S, sstate, pi, si)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, pi))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=400, random_state=SEED).fit(scaler.transform(Xtr), ytr)
    rf = RandomForestClassifier(n_estimators=60, max_depth=18, min_samples_leaf=2, n_jobs=-1,
                                random_state=SEED).fit(Xtr, ytr)

    def metrics(name, proba):
        pred = (proba >= .5).astype(int)
        fpr, tpr, _ = roc_curve(yte, proba)
        return {"Model": name, "Accuracy": accuracy_score(yte, pred),
                "Precision": precision_score(yte, pred, zero_division=0),
                "Recall": recall_score(yte, pred, zero_division=0), "F1-score": f1_score(yte, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(yte, proba), "cm": confusion_matrix(yte, pred), "fpr": fpr, "tpr": tpr}

    lr_p, rf_p = lr.predict_proba(scaler.transform(Xte))[:, 1], rf.predict_proba(Xte)[:, 1]
    results = [metrics("Logistic Regression", lr_p), metrics("Random Forest", rf_p)]

    cat = schemes["category_tag"].values[si[te]]
    lvl = schemes["level"].values[si[te]]
    pred = (rf_p >= .5).astype(int)
    rows = []
    for grp, vals in (("Category", cat), ("Level", lvl)):
        for v in sorted(set(vals)):
            m = vals == v
            rows.append({"Group": grp, "Segment": v, "Test pairs": int(m.sum()),
                         "Accuracy": accuracy_score(yte[m], pred[m])})
    seg = pd.DataFrame(rows)
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return {"rf": rf, "columns": list(X.columns), "results": results, "segments": seg,
            "importance": imp.head(15), "n_train": len(tr), "n_test": len(te), "n_profiles": n_profiles,
            "n_pairs": len(y), "n_pairs_raw": pi.size, "raw_pos_rate": float(len(pos) / (len(pos) + len(neg)))}


def ml_confidence(bundle: dict, p: dict, schemes: pd.DataFrame) -> np.ndarray:
    """Random-Forest probability of eligibility for one user against every scheme."""
    dfp = pd.DataFrame([{k: p.get(k, False) for k in
                         ["age", "income", "gender", "category", "occupation", "bpl", "disability",
                          "widow", "minority", "ews", "state"]}])
    P, S, sstate = _profile_matrix(dfp), _scheme_matrix(schemes), _scheme_state(schemes)
    n_s = len(schemes)
    X = _pair_features(P, dfp["state"].values, S, sstate, np.zeros(n_s, int), np.arange(n_s))
    return bundle["rf"].predict_proba(X[bundle["columns"]])[:, 1]


# --------------------------------------------------------------------------- #
# 4. District analytics (NFHS-5)
# --------------------------------------------------------------------------- #
def kmeans_sweep(nfhs: pd.DataFrame, k_range=range(2, 9)) -> pd.DataFrame:
    X = StandardScaler().fit_transform(nfhs[indicator_cols(nfhs)])
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def cluster_districts(nfhs: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """K-Means on standardised indicators. Clusters are renamed Tier 1 (most developed) .. Tier k."""
    cols = indicator_cols(nfhs)
    X = StandardScaler().fit_transform(nfhs[cols])
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    z = pd.DataFrame(X, columns=cols)
    # 'Population below 15' is a pressure indicator: higher = less developed, so flip its sign
    dev = z.drop(columns=["Population below 15 yrs"], errors="ignore").mean(axis=1) - \
        (z["Population below 15 yrs"] * 0.5 if "Population below 15 yrs" in z else 0)
    order = pd.Series(dev.values).groupby(km.labels_).mean().sort_values(ascending=False).index
    rename = {old: f"Tier {i + 1}" for i, old in enumerate(order)}
    out = nfhs.copy()
    out["Cluster"] = pd.Series(km.labels_).map(rename).values
    coords = PCA(n_components=2, random_state=SEED).fit_transform(X)
    out["PC1"], out["PC2"] = coords[:, 0], coords[:, 1]
    profile = out.groupby("Cluster")[cols].mean().sort_index()
    profile["Districts"] = out.groupby("Cluster").size()
    return out, profile, float(silhouette_score(X, km.labels_))


def driver_analysis(nfhs: pd.DataFrame, target: str) -> dict:
    """Random Forest regression: which indicators influence `target` the most?"""
    cols = [c for c in indicator_cols(nfhs) if c != target]
    X, y = nfhs[cols], nfhs[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    cv = cross_val_score(RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
                         X, y, cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
    perm = permutation_importance(rf, Xte, yte, n_repeats=15, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"Indicator": cols, "Importance": rf.feature_importances_,
                        "Permutation": perm.importances_mean}).sort_values("Importance", ascending=False)
    return {"r2": r2_score(yte, pred), "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "cv_r2": cv.mean(), "cv_std": cv.std(), "importance": imp,
            "actual": yte.values, "pred": pred, "n_test": len(yte)}


def district_needs(nfhs: pd.DataFrame, state: str, district: str, schemes: pd.DataFrame,
                   text_index=None, top: int = 4) -> pd.DataFrame:
    """Rank the indicators on which a district lags the dataset average the most and suggest the
    best-matching schemes (Central + that state's) for each gap using TF-IDF text similarity."""
    cols = [c for c in indicator_cols(nfhs) if c != "Population below 15 yrs"]
    row = nfhs[(nfhs.State == state) & (nfhs.District == district)]
    if row.empty:
        return pd.DataFrame()
    row = row.iloc[0][cols].astype(float)
    mean, std = nfhs[cols].mean(), nfhs[cols].std().replace(0, 1)
    z = ((row - mean) / std).sort_values().head(top)
    usable = ((schemes["level"] == "Central") | (schemes["state"] == state)).values
    sugg = []
    for ind in z.index:
        q = INDICATOR_QUERY.get(ind)
        if q and text_index is not None:
            sc = text_scores(text_index, q) * usable
            best = np.argsort(-sc)[:2]
            sugg.append("; ".join(schemes["scheme_name"].iloc[b] for b in best if sc[b] > 0) or "—")
        else:
            sugg.append("—")
    return pd.DataFrame({"Indicator": z.index, "District (%)": row[z.index].values.round(1),
                         "Average (%)": mean[z.index].values.round(1), "Gap vs avg (z)": z.values.round(2),
                         "Suggested schemes": sugg})
