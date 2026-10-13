"""
Know Your Scheme - A Government Scheme Eligibility Finder
Streamlit app  |  run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import engine as E

st.set_page_config(page_title="Know Your Scheme", page_icon="🇮🇳", layout="wide")

GREEN, SAFFRON, NAVY = "#138808", "#FF9933", "#0B2E4F"

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(f"""
<style>
.block-container {{ padding-top: 1.4rem; max-width: 1250px; }}
.hero {{ background: linear-gradient(120deg, {NAVY} 0%, #0f5a46 60%, {GREEN} 100%);
        padding: 30px 36px; border-radius: 20px; color: #fff; margin-bottom: 18px;
        border-bottom: 6px solid {SAFFRON}; }}
.hero h1 {{ margin: 0; font-size: 2.3rem; color: #fff; }}
.hero p {{ margin: 6px 0 0 0; font-size: 1.05rem; opacity: .92; }}
.stat-strip {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:18px; }}
.stat {{ background: rgba(255,255,255,.14); border-radius: 12px; padding: 10px 18px; min-width: 130px; }}
.stat b {{ font-size: 1.45rem; display:block; }}
.stat span {{ font-size: .78rem; opacity:.85; text-transform: uppercase; letter-spacing:.05em; }}
.chip {{ display:inline-block; padding: 2px 10px; border-radius: 99px; font-size: .75rem;
        margin-right: 6px; background:#EAF3EE; color:#0f5a46; font-weight:600; }}
.chip.o {{ background:#FFF1E0; color:#B25E00; }}
.chip.b {{ background:#E7EEF8; color:{NAVY}; }}
.badge {{ float:right; padding: 6px 14px; border-radius: 99px; color:#fff; font-weight:700; font-size:.95rem; }}
.hi {{ background:#0b7a2e; }} .mid {{ background:#2e9e4f; }} .lo {{ background:#d98a00; }}
.small {{ color:#5b6770; font-size:.85rem; }}
div[data-testid="stMetric"] {{ background:#F3F7F4; border-radius:14px; padding:12px 16px; border:1px solid #E1EAE4; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers (work across old/new Streamlit width APIs)
# --------------------------------------------------------------------------- #
def show(fig, **kw):
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), font=dict(size=13))
    try:
        st.plotly_chart(fig, width="stretch", **kw)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, **kw)


def table(df, **kw):
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kw)
    except Exception:
        st.dataframe(df, hide_index=True, use_container_width=True, **kw)


@st.cache_data(show_spinner=False)
def get_data():
    return E.load_schemes(), E.load_nfhs()


@st.cache_resource(show_spinner="Training eligibility models (one-time, ~10 s)…")
def get_models():
    return E.train_eligibility_models(E.load_schemes())


@st.cache_resource(show_spinner="Indexing scheme text…")
def get_text_index():
    return E.build_text_index(E.load_schemes())


@st.cache_data(show_spinner=False)
def get_sweep():
    return E.kmeans_sweep(E.load_nfhs())


@st.cache_data(show_spinner=False)
def get_clusters(k):
    return E.cluster_districts(E.load_nfhs(), k)


@st.cache_data(show_spinner="Running Random Forest regression…")
def get_drivers(target):
    return E.driver_analysis(E.load_nfhs(), target)


schemes, nfhs = get_data()
bundle = get_models()
tindex = get_text_index()
STATES = E.list_states(schemes)
rf_res, lr_res = bundle["results"][1], bundle["results"][0]
IND = E.indicator_cols(nfhs)


def pick(name):
    return IND.index(name) if name in IND else 0

# --------------------------------------------------------------------------- #
# Hero + sidebar
# --------------------------------------------------------------------------- #
st.markdown(f"""
<div class="hero">
  <h1>🇮🇳 Know Your Scheme</h1>
  <p>Find every Central &amp; State welfare scheme you are eligible for — in under a minute.</p>
  <div class="stat-strip">
    <div class="stat"><b>{len(schemes):,}</b><span>Schemes</span></div>
    <div class="stat"><b>{(schemes['level'] == 'Central').sum()}</b><span>Central</span></div>
    <div class="stat"><b>{(schemes['level'] == 'State').sum():,}</b><span>State / UT</span></div>
    <div class="stat"><b>{len(nfhs)}</b><span>Districts analysed</span></div>
    <div class="stat"><b>{rf_res['Accuracy']:.1%}</b><span>Model accuracy</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write("A data-driven eligibility finder that matches your profile with government "
             "schemes, ranks them with a **match score**, and uses **NFHS-5** district data "
             "to show where schemes are needed most.")
    st.metric("Random Forest accuracy", f"{rf_res['Accuracy']:.1%}", f"{rf_res['Accuracy'] - lr_res['Accuracy']:+.1%} vs baseline")
    st.caption(f"Measured on {bundle['n_test']:,} unseen (citizen, scheme) pairs.")
    st.divider()
    st.caption("Pipeline: Data → Cleaning → EDA → Eligibility engine → Match score → "
               "ML (Logistic Regression · Random Forest · K-Means) → Deployment")
    st.caption("Information is indicative. Always confirm eligibility on the official scheme portal.")

tab_find, tab_eda, tab_dist, tab_model, tab_about = st.tabs(
    ["🔍 Find Schemes", "📊 Data Insights (EDA)", "🗺️ District Analytics", "🤖 Model Evaluation", "ℹ️ About"])

# --------------------------------------------------------------------------- #
# TAB 1 - Find schemes
# --------------------------------------------------------------------------- #
def why_text(s, p):
    out = []
    if s["level"] == "State" and s["state"] not in E.NO_STATE:
        out.append(f"State scheme for your state ({s['state']})")
    if s["min_age"] > 0 or s["max_age"] < 120:
        hi = f"–{s['max_age']}" if s["max_age"] < 120 else "+"
        out.append(f"Your age ({p['age']}) fits the {s['min_age']}{hi} window")
    if s["gender"] not in ("", "All"):
        out.append(f"Open to {s['gender'].lower()} applicants")
    if s["caste_list"]:
        out.append("Category matches (" + " / ".join(s["caste_list"]) + ")")
    if s["occ_list"]:
        out.append("Occupation / situation matches (" + ", ".join(s["occ_list"]) + ")")
    if pd.notna(s["max_annual_income"]) and p.get("income") is not None:
        out.append(f"Income ₹{p['income']:,} is within the ₹{int(s['max_annual_income']):,} limit")
    if s["bpl_required"]:
        out.append("BPL household requirement met")
    return out or ["No restrictive criteria in the structured data — open to all"]


with tab_find:
    st.subheader("Tell us about yourself")
    with st.form("profile"):
        c1, c2, c3 = st.columns(3)
        state = c1.selectbox("State / UT", STATES, index=STATES.index("Tamil Nadu") if "Tamil Nadu" in STATES else 0)
        age = c2.number_input("Age", 5, 100, 18)
        gender = c3.selectbox("Gender", E.GENDERS)
        c4, c5, c6 = st.columns(3)
        income = c4.number_input("Annual family income (₹)", 0, 10_000_000, 120_000, step=10_000)
        skip_income = c4.checkbox("Prefer not to say (skip income rules)")
        category = c5.selectbox("Social category", E.CATEGORIES, index=1)
        occupation = c6.selectbox("Occupation / situation", E.OCCUPATIONS)
        k1, k2, k3, k4, k5 = st.columns(5)
        bpl = k1.checkbox("BPL household")
        disability = k2.checkbox("Person with disability")
        widow = k3.checkbox("Widow")
        minority = k4.checkbox("Minority community")
        ews = k5.checkbox("EWS")
        interests = st.multiselect("Areas of interest (optional)", sorted(schemes["category_tag"].unique()))
        query = st.text_input("Describe what you need (optional)", placeholder="e.g. scholarship for education")
        submitted = st.form_submit_button("🔎 Find my schemes", type="primary")

    if submitted:
        st.session_state["prof"] = dict(
            age=int(age), gender=gender, category=category, occupation=occupation,
            income=None if skip_income else int(income), bpl=bpl, disability=disability,
            widow=widow, minority=minority, ews=ews, state=state)
        st.session_state["interests"], st.session_state["query"] = interests, query

    p = st.session_state.get("prof")
    if not p:
        st.info("Fill in the form and press **Find my schemes** to see your personalised, ranked list.")
    else:
        res = E.recommend(p, schemes, st.session_state["interests"], st.session_state["query"], tindex)
        conf = E.ml_confidence(bundle, p, schemes)
        res["ml_conf"] = pd.Series(conf, index=schemes["slug"]).reindex(res["slug"]).values
        elig = res[res["eligible"]].copy()
        near = res[(~res["eligible"]) & (res["n_fails"] == 1)]
        agree = float((elig["ml_conf"] >= .5).mean()) if len(elig) else 0.0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Eligible schemes", f"{len(elig):,} / {len(res):,}")
        m2.metric("Best match", f"{elig['match'].max():.0f}%" if len(elig) else "—")
        m3.metric("Your state's schemes", int((elig["level"] == "State").sum()))
        m4.metric("Almost eligible", len(near))
        m5.metric("ML agrees", f"{agree:.0%}")

        if elig.empty:
            st.warning("No scheme matches this profile exactly. Check the 'almost eligible' list below.")
        else:
            f1, f2, f3, f4 = st.columns([3, 1.5, 1.5, 1.5])
            cats = f1.multiselect("Filter by category", sorted(elig["category_tag"].unique()))
            lvl = f2.selectbox("Level", ["All", "Central", "State"])
            min_match = f3.slider("Minimum match %", 0, 99, 0)
            top_n = f4.slider("Show top", 10, 100, 25, step=5)
            view = elig[(elig["match"] >= min_match)]
            if cats:
                view = view[view["category_tag"].isin(cats)]
            if lvl != "All":
                view = view[view["level"] == lvl]
            view = view.head(top_n)

            left, right = st.columns([3, 1.4])
            with left:
                st.caption(f"Showing {len(view)} of {len(elig):,} eligible schemes, ranked by match score.")
                for i, (_, s) in enumerate(view.iterrows(), 1):
                    cls = "hi" if s["match"] >= 80 else "mid" if s["match"] >= 65 else "lo"
                    with st.container(border=True):
                        where = s["state"] if s["level"] == "State" else "All India"
                        st.markdown(
                            f"<span class='badge {cls}'>{s['match']:.0f}% match</span>"
                            f"**{i}. {s['scheme_name']}**<br>"
                            f"<span class='chip'>{s['category_tag']}</span>"
                            f"<span class='chip o'>{s['level']} · {where}</span>"
                            f"<span class='chip b'>AI confidence {s['ml_conf']:.0%}</span>",
                            unsafe_allow_html=True)
                        with st.expander("View details"):
                            st.markdown("**Why you match**")
                            for w in why_text(s, p):
                                st.markdown(f"- ✅ {w}")
                            if s["benefits"]:
                                st.markdown(f"**Benefits:** {s['benefits'][:700]}")
                            if s["eligibility"]:
                                st.markdown(f"**Official eligibility text:** {s['eligibility'][:900]}")
                            if s["documents"]:
                                st.markdown(f"**Documents required:** {s['documents'][:400]}")
                            if s["application"]:
                                st.markdown(f"**How to apply:** {s['application'][:500]}")
                            st.markdown(f"[Open on myScheme ↗]({s['official_link']})")
            with right:
                cat_counts = view["category_tag"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Schemes"]
                show(px.pie(cat_counts, names="Category", values="Schemes", hole=.55,
                            title="Your matches by category",
                            color_discrete_sequence=px.colors.qualitative.Safe))
                out = elig[["scheme_name", "category_tag", "level", "state", "match", "benefits", "eligibility",
                            "documents", "official_link"]]
                st.download_button(f"⬇️ Download all {len(elig):,} matches (CSV)", out.to_csv(index=False).encode(),
                                   "my_schemes.csv", "text/csv")
                st.caption("The structured filters cover age, gender, income, category, occupation and state. "
                           "Always read the official eligibility text before applying.")

        if len(near):
            with st.expander(f"🟡 Almost eligible — {len(near):,} scheme(s) missing just one condition"):
                for _, s in near.head(15).iterrows():
                    st.markdown(f"**{s['scheme_name']}** — {s['fails'][0]}  \n"
                                f"<span class='small'>{s['benefits'][:160]}</span>", unsafe_allow_html=True)

        if p["state"] in set(nfhs["State"]):
            with st.expander("📍 Know your district (NFHS-5 insight)"):
                dist = st.selectbox("District", sorted(nfhs[nfhs["State"] == p["state"]]["District"]))
                need = E.district_needs(nfhs, p["state"], dist, schemes, tindex)
                st.caption("Indicators where this district lags the dataset average the most, with the "
                           "best-matching Central / state schemes for each gap.")
                table(need)

# --------------------------------------------------------------------------- #
# TAB 2 - EDA
# --------------------------------------------------------------------------- #
with tab_eda:
    st.subheader("Scheme dataset")
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Total schemes", f"{len(schemes):,}")
    e2.metric("Central", int((schemes["level"] == "Central").sum()))
    e3.metric("State / UT", f"{int((schemes['level'] == 'State').sum()):,}")
    e4.metric("Categories", schemes["category_tag"].nunique())
    e5.metric("States / UTs covered", len(STATES))

    a, b = st.columns(2)
    with a:
        cc = schemes["category_tag"].value_counts().reset_index()
        cc.columns = ["Category", "Schemes"]
        show(px.bar(cc, x="Category", y="Schemes", title="Schemes by category", color="Schemes",
                    color_continuous_scale="Greens").update_coloraxes(showscale=False)
             .update_layout(xaxis_tickangle=-40))
    with b:
        sc = schemes["state"].replace({"Not specified": "Unspecified"}).value_counts().reset_index()
        sc.columns = ["State / UT", "Schemes"]
        show(px.bar(sc, x="State / UT", y="Schemes", title="Schemes by state / UT (All India = Central)",
                    color="Schemes", color_continuous_scale="Oranges").update_coloraxes(showscale=False)
             .update_layout(xaxis_tickangle=-60))

    a, b = st.columns(2)
    with a:
        crit = pd.Series({
            "Income limit": schemes["max_annual_income"].notna().mean(),
            "Age limit (min or max)": ((schemes["min_age"] > 0) | (schemes["max_age"] < 120)).mean(),
            "Category-restricted": schemes["caste_list"].apply(bool).mean(),
            "Occupation-tagged": schemes["occ_list"].apply(bool).mean(),
            "Gender-specific": (schemes["gender"] != "All").mean(),
            "Mentions BPL": schemes["bpl_required"].mean(),
        }).sort_values().reset_index()
        crit.columns = ["Criterion", "Share"]
        show(px.bar(crit, x="Share", y="Criterion", orientation="h",
                    title="Data completeness: share of schemes with each structured criterion",
                    color="Share", color_continuous_scale="Greens").update_layout(xaxis_tickformat=".0%")
             .update_coloraxes(showscale=False))
    with b:
        tags = schemes["occ_list"].explode().dropna().value_counts().reset_index()
        tags.columns = ["Beneficiary group", "Schemes"]
        show(px.bar(tags, x="Schemes", y="Beneficiary group", orientation="h", title="Beneficiary groups targeted",
                    color_discrete_sequence=[GREEN]).update_layout(yaxis=dict(autorange="reversed")))

    a, b = st.columns(2)
    with a:
        inc = schemes["max_annual_income"].dropna()
        show(px.histogram(inc, x="max_annual_income", nbins=30, title=f"Income limits (₹) — {len(inc)} schemes",
                          color_discrete_sequence=[SAFFRON]).update_layout(xaxis_title="Annual income limit"))
    with b:
        lim = schemes[(schemes["min_age"] > 0) | (schemes["max_age"] < 120)]
        show(px.histogram(lim[lim["min_age"] > 0], x="min_age", nbins=25, title="Minimum-age requirements",
                          color_discrete_sequence=[NAVY]).update_layout(xaxis_title="Minimum age"))

    st.subheader("NFHS-5 district dataset")
    n1, n2, n3 = st.columns(3)
    n1.metric("Districts", len(nfhs))
    n2.metric("Indicators", len(IND))
    n3.metric("Data points", f"{len(nfhs) * len(IND):,}")
    a, b = st.columns(2)
    with a:
        ind = st.selectbox("Indicator", IND, index=pick("Women literate"))
        show(px.box(nfhs, x="State", y=ind, color="State", points=False, title=f"{ind} (%) by state")
             .update_layout(showlegend=False, xaxis_tickangle=-45))
    with b:
        show(px.imshow(nfhs[IND].corr(), text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                       color_continuous_scale="RdBu_r", title="Correlation heatmap")
             .update_layout(height=520))
    with st.expander("Browse the raw data"):
        which = st.radio("Dataset", ["Schemes", "NFHS-5 districts"], horizontal=True)
        if which == "Schemes":
            table(schemes[["scheme_name", "level", "state", "category_tag", "min_age", "max_age",
                           "max_annual_income", "gender", "caste_category", "occupation_tags"]])
        else:
            table(nfhs)

# --------------------------------------------------------------------------- #
# TAB 3 - District analytics
# --------------------------------------------------------------------------- #
with tab_dist:
    st.subheader("K-Means district clustering")
    sweep = get_sweep()
    best_k = int(sweep.loc[sweep["Silhouette"].idxmax(), "k"])
    c1, c2 = st.columns([1, 2])
    with c1:
        k = st.slider("Number of clusters (k)", 2, 8, 4)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sweep["k"], y=sweep["Inertia"], name="Inertia", mode="lines+markers",
                                 line_color=NAVY))
        fig.add_trace(go.Scatter(x=sweep["k"], y=sweep["Silhouette"], name="Silhouette", yaxis="y2",
                                 mode="lines+markers", line_color=SAFFRON))
        fig.update_layout(title="Elbow & silhouette", xaxis_title="k", yaxis_title="Inertia",
                          yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
                          legend=dict(orientation="h", y=-0.25))
        show(fig)
        st.caption(f"Highest silhouette in the sweep: k = {best_k}.")
    clustered, profile, sil = get_clusters(k)
    with c2:
        order = sorted(clustered["Cluster"].unique())
        show(px.scatter(clustered, x="PC1", y="PC2", color="Cluster", hover_data=["State", "District"],
                        category_orders={"Cluster": order}, title=f"Districts in PCA space (silhouette = {sil:.2f})",
                        color_discrete_sequence=px.colors.qualitative.Safe))

    prof = profile.drop(columns="Districts")
    z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)
    fig = go.Figure(go.Heatmap(z=z.T.values, x=[f"{i} ({int(profile.loc[i, 'Districts'])})" for i in prof.index],
                               y=prof.columns.tolist(), text=prof.T.round(1).values, texttemplate="%{text}",
                               colorscale="RdYlGn", showscale=False))
    fig.update_layout(title="Cluster profile (mean % per indicator; colour = relative level). Tier 1 = most developed",
                      height=470)
    show(fig)

    st.subheader("What drives an outcome? (Random Forest regression)")
    target = st.selectbox("Outcome to explain", IND, index=pick("Women 10+ yrs schooling"))
    dr = get_drivers(target)
    r1, r2, r3 = st.columns(3)
    r1.metric("R² (test)", f"{dr['r2']:.3f}")
    r2.metric("RMSE (test)", f"{dr['rmse']:.2f} pts")
    r3.metric("5-fold CV R²", f"{dr['cv_r2']:.3f}", f"± {dr['cv_std']:.3f}", delta_color="off")
    a, b = st.columns(2)
    with a:
        imp = dr["importance"].melt(id_vars="Indicator", value_vars=["Importance", "Permutation"], var_name="Method", value_name="Score")
        show(px.bar(imp, x="Score", y="Indicator", color="Method", orientation="h", barmode="group",
                    title=f"Key drivers of '{target}'", color_discrete_sequence=[GREEN, SAFFRON])
             .update_layout(yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=-0.15)))
    with b:
        lo, hi = float(min(dr["actual"].min(), dr["pred"].min())), float(max(dr["actual"].max(), dr["pred"].max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dr["actual"], y=dr["pred"], mode="markers", marker=dict(color=GREEN, opacity=.7),
                                 name="Districts"))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="grey"),
                                 name="Perfect fit"))
        fig.update_layout(title="Actual vs predicted (held-out districts)", xaxis_title="Actual", yaxis_title="Predicted")
        show(fig)

    st.subheader("District explorer")
    d1, d2 = st.columns(2)
    sstate = d1.selectbox("State", sorted(nfhs["State"].unique()), key="ex_state")
    sdist = d2.selectbox("District", sorted(nfhs[nfhs["State"] == sstate]["District"]), key="ex_dist")
    row = clustered[(clustered["State"] == sstate) & (clustered["District"] == sdist)].iloc[0]
    st.markdown(f"**{sdist}, {sstate}** belongs to **{row['Cluster']}** (of {k}).")
    table(E.district_needs(nfhs, sstate, sdist, schemes, tindex, top=5))
    sort_col = "Women literate" if "Women literate" in IND else IND[0]
    worst = clustered[clustered["Cluster"] == order[-1]].sort_values(sort_col).head(15)
    with st.expander(f"Priority districts — lowest tier ({order[-1]}), lowest women's literacy first"):
        table(worst[["State", "District"] + IND[:5]])

# --------------------------------------------------------------------------- #
# TAB 4 - Model evaluation
# --------------------------------------------------------------------------- #
with tab_model:
    st.subheader("Eligibility classifier: Logistic Regression vs Random Forest")
    ok = rf_res["Accuracy"] >= 0.90
    (st.success if ok else st.warning)(
        f"Random Forest test accuracy: **{rf_res['Accuracy']:.2%}**  "
        f"({'meets' if ok else 'below'} the 90% target) — measured live on "
        f"{bundle['n_test']:,} unseen citizen–scheme pairs.")
    m = st.columns(5)
    for col, key in zip(m, ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]):
        col.metric(key, f"{rf_res[key]:.3f}" if key == "ROC-AUC" else f"{rf_res[key]:.1%}",
                   f"{rf_res[key] - lr_res[key]:+.3f} vs LR", delta_color="off")

    comp = pd.DataFrame([{k: v for k, v in r.items() if k in ("Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC")}
                         for r in bundle["results"]])
    a, b = st.columns(2)
    with a:
        show(px.bar(comp.melt("Model", var_name="Metric", value_name="Score"), x="Metric", y="Score", color="Model",
                    barmode="group", title="Model comparison", range_y=[0, 1.05],
                    color_discrete_sequence=[SAFFRON, GREEN]))
    with b:
        fig = go.Figure()
        for r, c in zip(bundle["results"], [SAFFRON, GREEN]):
            fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=f"{r['Model']} (AUC {r['ROC-AUC']:.3f})",
                                     line=dict(color=c, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"), name="Chance"))
        fig.update_layout(title="ROC curves", xaxis_title="False-positive rate", yaxis_title="True-positive rate",
                          legend=dict(orientation="h", y=-0.25))
        show(fig)

    a, b = st.columns(2)
    for col, r in zip((a, b), bundle["results"]):
        with col:
            show(px.imshow(r["cm"], text_auto=True, x=["Not eligible", "Eligible"], y=["Not eligible", "Eligible"],
                           labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Greens",
                           title=f"Confusion matrix — {r['Model']}").update_coloraxes(showscale=False))

    a, b = st.columns(2)
    with a:
        sg = bundle["segments"]
        sg = sg[sg["Group"] == "Category"].sort_values("Accuracy")
        show(px.bar(sg, x="Accuracy", y="Segment", orientation="h", hover_data=["Test pairs"],
                    title="Random Forest accuracy by scheme category", range_x=[0.8, 1.0],
                    color="Accuracy", color_continuous_scale="Greens")
             .update_layout(height=640).update_coloraxes(showscale=False))
    with b:
        imp = bundle["importance"].sort_values().reset_index()
        imp.columns = ["Feature", "Importance"]
        show(px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top features used by the Random Forest",
                    color_discrete_sequence=[GREEN]).update_layout(height=640))

    with st.expander("📝 How this accuracy is measured (read before presenting)"):
        st.markdown(f"""
- **Task:** given a citizen's profile and a scheme's attributes, predict *Eligible / Not eligible*.
- **Data:** {bundle['n_profiles']:,} simulated citizens, each paired with random schemes from the {len(schemes):,}-scheme
  dataset. Only {bundle['raw_pos_rate']:.0%} of random pairs are eligible, so the set was **balanced to 50 / 50**
  ({bundle['n_pairs']:,} pairs) — a model that always says "not eligible" would score only 50%, not 95%.
- **Labels** come from the rule-based eligibility engine (age, gender, income, category, occupation, state, BPL).
- **No leakage:** the train/test split is by *citizen*, so the models are tested on people they never saw.
- **Raw features:** the profile and the scheme's limits are given as inputs. The only pre-computed comparison is
  `state_ok` (does the citizen's state equal the scheme's state?).
- **What it proves:** the ML model learned the eligibility logic and generalises to new profiles. It does not measure how
  well the structured fields reflect the official scheme text (only ~7-10% of schemes have an age or income limit
  recorded), which is why the app also shows the official eligibility text.
- Logistic Regression is linear, so it struggles with interacting conditions; the Random Forest captures them and
  supplies the *AI confidence* shown next to each result.
""")

# --------------------------------------------------------------------------- #
# TAB 5 - About
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("Problem")
    st.write("Scheme information is scattered across many portals and the eligibility rules are buried in long text, "
             "so citizens often do not know which schemes apply to them and benefits go unclaimed.")
    st.subheader("Solution — 7-stage pipeline")
    st.markdown("""
1. **User input** — age, gender, state/UT, category, income, occupation and situation flags
2. **Data preprocessing** — de-duplication, standardised labels, invalid ages/incomes removed, eligibility conditions turned into structured attributes
3. **Exploratory data analysis** — distributions, correlation heat-map, rule prevalence
4. **Eligibility filtering** — rule engine removes schemes whose conditions are not met
5. **Profile matching & ranking** — match score from targeting specificity, interests and free-text relevance
6. **Machine-learning analytics** — Logistic Regression / Random Forest (eligibility), K-Means + Random Forest regression (NFHS-5)
7. **Output & deployment** — Streamlit app, GitHub, Streamlit Community Cloud
""")
    st.subheader("Match score")
    st.write("Eligible schemes start at 50. Up to +27 is added for how *targeted* the scheme is at you (age window, "
             "category, occupation, income cap, BPL, your own state), +10 if it is in your areas of interest, and up to "
             "+12 for TF-IDF text similarity with your description. Maximum 99.")
    st.subheader("Scaling up")
    st.write("Drop an updated scheme CSV with the same columns into `data/schemes_processed.csv` (or an NFHS file with the same "
             "layout into `data/nfhs_district_wide.csv`) and every chart, model and metric recomputes automatically.")
    st.caption("Developed by Dharanika V (24DSC238) · B.Sc. Data Science · The American College, Madurai")
