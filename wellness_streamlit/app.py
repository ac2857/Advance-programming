"""
Digital Wellness Advisor — PEA Team · Advanced Python SP2026

Models integrated:
  wellness_app_outputs.pkl  → RandomForest risk classifier (DF1+DF2)
  student_ridge_model.pkl   → RidgeClassifier exam performance (DF6)

Feedback addressed:
  ✓ Multi-platform entry  (add Instagram + Facebook separately)
  ✓ Exam prediction is optional — opt-in checkbox in profile
  ✓ Explanations + recommendations with dataset citations
  ✓ Trend graphs tab (prof feedback) — unlocks after 3 entries
  ✓ Age and gender inputs in profile
  ✓ Demo data seeding so charts are visible from first visit
"""

import streamlit as st
import pickle
from datetime import date, timedelta
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Digital Wellness Advisor",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("wellness_app_outputs.pkl", "rb") as f:
        rb = pickle.load(f)
    with open("student_ridge_model.pkl", "rb") as f:
        eb = pickle.load(f)
    return rb, eb

risk_bundle, exam_bundle = load_models()

risk_model   = risk_bundle["risk_classifier"]
risk_scaler  = risk_bundle["feature_scaler"]
le_risk      = risk_bundle["risk_label_encoder"]   # high_risk / low_risk / medium_risk
le_gender    = risk_bundle["gender_encoder"]        # female / male
le_platform  = risk_bundle["platform_encoder"]      # facebook…youtube
FEATURE_COLS = risk_bundle["feature_cols"]
# ['age','gender_enc','platform_enc','social_media_time_hrs','sleep_hours',
#  'sm_to_waking_ratio','academic_enc','rel_enc','region_enc']

exam_model  = exam_bundle["model"]    # RidgeClassifier → High / Medium / Low
exam_scaler = exam_bundle["scaler"]
# features: study_hours_per_day, social_media_hours, sleep_hours, mental_health_rating

# ── Encoder look-up tables (verified against pkl inspection) ──────────────────
GENDER_MAP   = {"Female": "female", "Male": "male"}
PLATFORM_MAP = {
    "Facebook":"facebook","Instagram":"instagram","KakaoTalk":"kakaotalk",
    "LINE":"line","LinkedIn":"linkedin","Snapchat":"snapchat","TikTok":"tiktok",
    "Twitter/X":"twitter","VKontakte":"vkontakte","WeChat":"wechat",
    "WhatsApp":"whatsapp","YouTube":"youtube",
}
ACADEMIC_MAP = {"Graduate":0, "High School":1, "Undergraduate":2}
REL_MAP      = {"Complicated":0, "In Relationship":1, "Single":2}
REGION_MAP   = {
    "Africa":0,"East Asia":1,"Europe":2,"Latin America":3,
    "North America":4,"Oceania":5,"Other":6,"South Asia":7,
}
COUNTRY_REGION = {
    "USA":"North America","Canada":"North America","Mexico":"Latin America",
    "Brazil":"Latin America","UK":"Europe","Germany":"Europe","France":"Europe",
    "Italy":"Europe","Spain":"Europe","Netherlands":"Europe","India":"South Asia",
    "Pakistan":"South Asia","Bangladesh":"South Asia","Sri Lanka":"South Asia",
    "China":"East Asia","Japan":"East Asia","South Korea":"East Asia",
    "Australia":"Oceania","New Zealand":"Oceania","Nigeria":"Africa",
    "South Africa":"Africa","Egypt":"Africa","Other":"Other",
}
PLATFORMS_DISPLAY = list(PLATFORM_MAP.keys())
WAKING_HOURS = 16.0
RISK_SCORE_MAP = {"low_risk":20, "medium_risk":55, "high_risk":85}

# ── Session state helpers ─────────────────────────────────────────────────────
HISTORY_KEY = "wellness_history"
PROFILE_KEY = "wellness_profile"

def get_history():
    return st.session_state.get(HISTORY_KEY, [])

def save_entry(entry: dict):
    h = [e for e in get_history() if e["date"] != entry["date"]]
    h.append(entry)
    h.sort(key=lambda e: e["date"])
    st.session_state[HISTORY_KEY] = h

def delete_entry(dt: str):
    st.session_state[HISTORY_KEY] = [e for e in get_history() if e["date"] != dt]

def get_profile():
    return st.session_state.get(PROFILE_KEY, {})

# ── Demo data seeding ─────────────────────────────────────────────────────────
def seed_example_data():
    if st.session_state.get("demo_seeded"):
        return
    today = date.today()
    rng = np.random.default_rng(42)
    platforms_pool = PLATFORMS_DISPLAY
    for i in range(14, 0, -1):
        d = (today - timedelta(days=i)).isoformat()
        social = round(float(rng.uniform(1.5, 5.5)), 1)
        sleep  = round(float(rng.uniform(5.5, 8.5)), 1)
        study  = round(float(rng.uniform(1.0, 5.5)), 1)
        plat   = str(rng.choice(platforms_pool))
        # run the risk model with defaults
        try:
            risk_label, risk_proba = predict_risk(
                age=20, gender_str="Female", primary_platform=plat,
                social_hrs=social, sleep_hrs=sleep,
                academic_str="Undergraduate", rel_str="Single", region_str="North America"
            )
        except Exception:
            risk_label, risk_proba = "medium_risk", {}
        entry = {
            "date": d,
            "platforms": [{"platform": plat, "hours": social}],
            "social_media_time_hrs": social,
            "primary_platform": plat,
            "sleep_hours": sleep,
            "study_hours": study,
            "mental_health_rating": int(rng.integers(4, 10)),
            "conflicts": int(rng.integers(0, 4)),
            "exercise_days": int(rng.integers(0, 6)),
            "dominant_emotion": str(rng.choice(["Happy","Neutral","Anxious","Bored","Sad"])),
            "detox_days": int(rng.integers(0, 3)),
            "risk_label": risk_label,
            "risk_proba": risk_proba,
            "exam_pred": str(rng.choice(["High","Medium","Low"])),
            "is_demo": True,
        }
        save_entry(entry)
    st.session_state["demo_seeded"] = True

# ── Prediction functions ──────────────────────────────────────────────────────
def predict_risk(age, gender_str, primary_platform, social_hrs, sleep_hrs,
                 academic_str, rel_str, region_str):
    g = GENDER_MAP.get(gender_str, "female")
    gender_enc = int(le_gender.transform([g])[0])

    praw = PLATFORM_MAP.get(primary_platform, "instagram")
    if praw not in le_platform.classes_:
        praw = "instagram"
    platform_enc = int(le_platform.transform([praw])[0])

    academic_enc = ACADEMIC_MAP.get(academic_str, 2)
    rel_enc      = REL_MAP.get(rel_str, 2)
    region_enc   = REGION_MAP.get(region_str, 6)
    ratio        = social_hrs / WAKING_HOURS

    X = pd.DataFrame([[age, gender_enc, platform_enc, social_hrs, sleep_hrs,
                        ratio, academic_enc, rel_enc, region_enc]],
                     columns=FEATURE_COLS)
    X_scaled  = risk_scaler.transform(X)
    pred_enc  = risk_model.predict(X_scaled)[0]
    proba     = risk_model.predict_proba(X_scaled)[0]
    label     = le_risk.inverse_transform([pred_enc])[0]
    proba_dict = {le_risk.inverse_transform([i])[0]: round(float(p)*100, 1)
                  for i, p in enumerate(proba)}
    return label, proba_dict

def predict_exam(study_hrs, social_hrs, sleep_hrs, mental_rating):
    X = pd.DataFrame([[study_hrs, social_hrs, sleep_hrs, mental_rating]],
                     columns=["study_hours_per_day","social_media_hours",
                               "sleep_hours","mental_health_rating"])
    X_scaled = exam_scaler.transform(X)
    return exam_model.predict(X_scaled)[0]   # High / Medium / Low

# ── Award helper streaks ──────────────────────────────────────────────────────
def _max_streak_num(entries_sorted, key, threshold, direction="high"):
    streak = maxs = 0
    for e in entries_sorted:
        v = e.get(key) or 0
        cond = (v >= threshold) if direction == "high" else (v < threshold)
        if cond:
            streak += 1; maxs = max(maxs, streak)
        else:
            streak = 0
    return maxs

def _max_streak_risk(entries_sorted):
    streak = maxs = 0
    for e in entries_sorted:
        if e.get("risk_label") == "low_risk":
            streak += 1; maxs = max(maxs, streak)
        else:
            streak = 0
    return maxs

def _max_streak_balanced(entries_sorted):
    streak = maxs = 0
    for e in entries_sorted:
        ok = (e.get("social_media_time_hrs", 99) < 4 and
              (e.get("study_hours") or 0) >= 3 and
              e.get("sleep_hours", 0) >= 7)
        if ok:
            streak += 1; maxs = max(maxs, streak)
        else:
            streak = 0
    return maxs

def _max_streak_emotion(entries_sorted):
    streak = maxs = 0
    for e in entries_sorted:
        if e.get("dominant_emotion") == "Happy":
            streak += 1; maxs = max(maxs, streak)
        else:
            streak = 0
    return maxs

def _consec_days(entries_sorted):
    if len(entries_sorted) < 2:
        return len(entries_sorted)
    streak = maxs = 1
    for i in range(1, len(entries_sorted)):
        a = date.fromisoformat(entries_sorted[i-1]["date"])
        b = date.fromisoformat(entries_sorted[i]["date"])
        if (b - a).days == 1:
            streak += 1; maxs = max(maxs, streak)
        else:
            streak = 1
    return maxs

def check_awards(history_real):
    s = sorted(history_real, key=lambda e: e["date"])
    earned = []
    if any(e.get("detox_days", 0) > 0 for e in s):
        earned.append("detox_starter")
    if any((e.get("study_hours") or 0) >= 4 for e in s):
        earned.append("study_hero")
    if _max_streak_num(s, "social_media_time_hrs", 2, "low") >= 7:
        earned.append("low_screen")
    if _max_streak_num(s, "sleep_hours", 8, "high") >= 5:
        earned.append("sleep_champ")
    if _max_streak_risk(s) >= 7:
        earned.append("wellness_week")
    if _max_streak_balanced(s) >= 3:
        earned.append("balanced_life")
    if _max_streak_emotion(s) >= 5:
        earned.append("no_fomo")
    if _consec_days(s) >= 7:
        earned.append("streak_7")
    return earned

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background:#0A0E1A; color:#F0F4FF; }
  section[data-testid="stSidebar"] { background:#111827; }
  .metric-card {
    background:#1C2537; border:1px solid #2A3550;
    border-radius:14px; padding:18px 20px; text-align:center; height:100%;
  }
  .metric-val  { font-size:2rem; font-weight:900; margin-bottom:4px; }
  .metric-lbl  { font-size:.72rem; color:#6B7A99; letter-spacing:.05em; }
  .badge-low    {background:#0D2E22;color:#34D399;border:1px solid #34D39944;border-radius:20px;padding:3px 12px;font-weight:700;font-size:.82rem;}
  .badge-medium {background:#2E2408;color:#FBBF24;border:1px solid #FBBF2444;border-radius:20px;padding:3px 12px;font-weight:700;font-size:.82rem;}
  .badge-high   {background:#2E0A0A;color:#F87171;border:1px solid #F8717144;border-radius:20px;padding:3px 12px;font-weight:700;font-size:.82rem;}
  .badge-exam-high   {background:#0D2E22;color:#34D399;border:1px solid #34D39944;border-radius:20px;padding:3px 12px;font-weight:700;font-size:.82rem;}
  .badge-exam-medium {background:#2E2408;color:#FBBF24;border:1px solid #FBBF2444;border-radius:20px;padding:3px 12px;font-weight:700;font-size:.82rem;}
  .badge-exam-low    {background:#2E0A0A;color:#F87171;border:1px solid #F8717144;border-radius:20px;padding:3px 12px;font-weight:700;font-size:.82rem;}
  .explain-box {background:#111827;border-left:3px solid #60A5FA;border-radius:0 10px 10px 0;padding:12px 16px;margin:8px 0;font-size:.875rem;color:#94A3B8;line-height:1.7;}
  .rec-box   {background:#0A1A0D;border-left:3px solid #34D399;border-radius:0 10px 10px 0;padding:12px 16px;margin:6px 0;font-size:.875rem;color:#94A3B8;line-height:1.7;}
  .warn-box  {background:#1A1000;border-left:3px solid #FBBF24;border-radius:0 10px 10px 0;padding:12px 16px;margin:6px 0;font-size:.875rem;color:#94A3B8;line-height:1.7;}
  .demo-banner {background:#1A1500;border:1px solid #FBBF2444;border-radius:10px;padding:9px 14px;font-size:.78rem;color:#FBBF24;margin-bottom:10px;}
  #MainMenu, footer { visibility:hidden; }
  .block-container { padding-top:1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-bottom:1.2rem;">
  <div style="font-size:1.7rem;font-weight:900;letter-spacing:-.03em;
    background:linear-gradient(90deg,#34D399,#60A5FA);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    💚 Digital Wellness Advisor
  </div>
  <div style="color:#6B7A99;font-size:.78rem;margin-top:2px;">
    PEA Team · Advanced Python SP2026 · RandomForest + Ridge ML Models
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
TAB_LOG, TAB_DASH, TAB_HISTORY, TAB_GOALS, TAB_AWARDS = st.tabs([
    "✏️ Daily Log", "📊 Dashboard", "📅 History & Trends", "🎯 Goals", "🏆 Awards"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — DAILY LOG
# ══════════════════════════════════════════════════════════════════════════════
with TAB_LOG:
    st.subheader("Log Your Day")
    st.caption("Log today or any past date — all fields feed into the ML models.")

    log_date = st.date_input("Date", value=date.today(), max_value=date.today())
    existing = next((e for e in get_history() if e["date"] == log_date.isoformat()), None)
    if existing:
        st.info(f"✏️ Entry exists for {log_date} — editing it below.")

    # ── Profile ───────────────────────────────────────────────────────────────
    with st.expander("👤 Your Profile  *(fill once — saved across entries)*",
                     expanded=not get_profile()):
        pc1, pc2, pc3 = st.columns(3)
        name      = pc1.text_input("Name",   value=get_profile().get("name",""))
        age       = pc2.number_input("Age",  13, 35, value=int(get_profile().get("age", 20)))
        gender    = pc3.selectbox("Gender",
                       ["Female","Male","Prefer not to say"],
                       index=["Female","Male","Prefer not to say"]
                             .index(get_profile().get("gender","Female")))

        pc4, pc5, pc6 = st.columns(3)
        academic  = pc4.selectbox("Academic Level",
                       ["High School","Undergraduate","Graduate"],
                       index=["High School","Undergraduate","Graduate"]
                             .index(get_profile().get("academic","Undergraduate")))
        rel_status = pc5.selectbox("Relationship Status",
                       ["Single","In Relationship","Complicated"],
                       index=["Single","In Relationship","Complicated"]
                             .index(get_profile().get("rel_status","Single")))
        country   = pc6.selectbox("Country",
                       list(COUNTRY_REGION.keys()),
                       index=list(COUNTRY_REGION.keys())
                             .index(get_profile().get("country","USA")))

        is_student = st.checkbox(
            "🎓 I'm a student and want exam performance predicted",
            value=get_profile().get("is_student", True)
        )
        st.caption("Unchecking this hides the academic section and skips exam prediction.")

        if st.button("💾 Save Profile"):
            st.session_state[PROFILE_KEY] = dict(
                name=name, age=age, gender=gender, academic=academic,
                rel_status=rel_status, country=country, is_student=is_student
            )
            st.success("Profile saved!")

    profile = get_profile()

    # ── Social Media — multi-platform ─────────────────────────────────────────
    st.markdown("#### 📱 Social Media Usage")
    st.caption("Add each platform separately. Use ＋ to add more.")

    plat_key = f"plats_{log_date.isoformat()}"
    if plat_key not in st.session_state:
        if existing and existing.get("platforms"):
            st.session_state[plat_key] = [dict(p) for p in existing["platforms"]]
        else:
            st.session_state[plat_key] = [{"platform":"Instagram","hours":2.0}]

    platlist = st.session_state[plat_key]
    to_remove = None
    for idx, row in enumerate(platlist):
        c1, c2, c3 = st.columns([3, 2, 1])
        platlist[idx]["platform"] = c1.selectbox(
            f"Platform {idx+1}", PLATFORMS_DISPLAY,
            index=PLATFORMS_DISPLAY.index(row["platform"])
                  if row["platform"] in PLATFORMS_DISPLAY else 0,
            key=f"psel_{log_date}_{idx}")
        platlist[idx]["hours"] = c2.number_input(
            "Hours today", 0.0, 24.0, value=float(row["hours"]), step=0.5,
            key=f"phrs_{log_date}_{idx}")
        if c3.button("✕", key=f"pdel_{log_date}_{idx}") and len(platlist) > 1:
            to_remove = idx
    if to_remove is not None:
        platlist.pop(to_remove)
        st.rerun()

    if st.button("＋ Add another platform"):
        platlist.append({"platform":"YouTube","hours":1.0})
        st.rerun()

    total_social    = round(sum(r["hours"] for r in platlist), 1)
    primary_platform = max(platlist, key=lambda r: r["hours"])["platform"]
    st.info(f"**Total social media today: {total_social} hrs** · Primary: {primary_platform}")

    st.divider()

    # ── Lifestyle ─────────────────────────────────────────────────────────────
    st.markdown("#### 😴 Lifestyle")
    lc1, lc2, lc3 = st.columns(3)
    sleep_hrs     = lc1.slider("Sleep Hours", 3.0, 12.0,
                                value=float(existing["sleep_hours"]) if existing else 7.0, step=0.5)
    exercise_days = lc2.slider("Exercise (days/week)", 0, 7,
                                value=int(existing.get("exercise_days",3)) if existing else 3)
    detox_days    = lc3.slider("Detox Days this week", 0, 7,
                                value=int(existing.get("detox_days",0)) if existing else 0)

    lc4, lc5 = st.columns(2)
    conflicts = lc4.slider("Conflicts over social media (0–5)", 0, 5,
                            value=int(existing.get("conflicts",1)) if existing else 1)
    emotion   = lc5.selectbox("Dominant Emotion Today",
                               ["Happy","Neutral","Anxious","Bored","Sad","Angry"],
                               index=["Happy","Neutral","Anxious","Bored","Sad","Angry"]
                                     .index(existing.get("dominant_emotion","Neutral")) if existing else 1)

    # ── Academics (optional) ──────────────────────────────────────────────────
    study_hrs     = None
    mental_rating = None
    if profile.get("is_student", True):
        st.divider()
        st.markdown("#### 📚 Academics")
        st.caption("You opted in for exam performance prediction — fill these in.")
        ac1, ac2 = st.columns(2)
        study_hrs     = ac1.slider("Study Hours Today", 0.0, 12.0,
                                    value=float(existing.get("study_hours",2.0)) if existing else 2.0, step=0.5)
        mental_rating = ac2.slider("Mental Health Rating (1–10)", 1, 10,
                                    value=int(existing.get("mental_health_rating",7)) if existing else 7)

    st.divider()

    # ── Save & run models ────────────────────────────────────────────────────
    if st.button("✅ Save Entry & Analyse", type="primary", use_container_width=True):
        g = profile.get("gender","Female")
        if g == "Prefer not to say":
            g = "Female"
        region = COUNTRY_REGION.get(profile.get("country","USA"), "Other")

        with st.spinner("Running models…"):
            risk_label, risk_proba = predict_risk(
                age              = int(profile.get("age", 20)),
                gender_str       = g,
                primary_platform = primary_platform,
                social_hrs       = total_social,
                sleep_hrs        = sleep_hrs,
                academic_str     = profile.get("academic","Undergraduate"),
                rel_str          = profile.get("rel_status","Single"),
                region_str       = region,
            )
            exam_pred = None
            if profile.get("is_student", True) and study_hrs is not None:
                exam_pred = predict_exam(study_hrs, total_social, sleep_hrs,
                                         mental_rating or 7)

        entry = {
            "date"                : log_date.isoformat(),
            "platforms"           : [dict(p) for p in platlist],
            "social_media_time_hrs": total_social,
            "primary_platform"    : primary_platform,
            "sleep_hours"         : sleep_hrs,
            "exercise_days"       : exercise_days,
            "detox_days"          : detox_days,
            "conflicts"           : conflicts,
            "dominant_emotion"    : emotion,
            "study_hours"         : study_hrs,
            "mental_health_rating": mental_rating,
            "risk_label"          : risk_label,
            "risk_proba"          : risk_proba,
            "exam_pred"           : exam_pred,
            "is_demo"             : False,
        }
        save_entry(entry)
        st.session_state["last_entry"] = entry
        st.success("✅ Saved! Head to the Dashboard tab to see your results.")
        st.balloons()

    # Demo data
    st.divider()
    with st.expander("🧪 Load example data (so you can explore trend charts)"):
        st.markdown('<div class="demo-banner">⚠️ Seeds 14 days of synthetic data to demo the trend charts. Does not affect your real entries.</div>',
                    unsafe_allow_html=True)
        if st.button("Load 14 days of demo data"):
            seed_example_data()
            st.success("Demo data loaded! Check History & Trends.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with TAB_DASH:
    real_entries = [e for e in get_history() if not e.get("is_demo")]
    last_entry   = (st.session_state.get("last_entry") or
                    next((e for e in sorted(real_entries, key=lambda e: e["date"], reverse=True)
                          if e.get("risk_label")), None))

    if not last_entry:
        st.info("👈 Fill in the Daily Log first to see your personalised results here.")
        st.stop()

    profile = get_profile()
    name_disp = (profile.get("name") or "Your") + ("'s" if profile.get("name") else "")

    st.markdown(f"### {name_disp} Wellness Dashboard")
    st.caption(f"Showing entry for **{last_entry['date']}**")

    # ── Top metrics ───────────────────────────────────────────────────────────
    risk_label = last_entry.get("risk_label", "low_risk")
    risk_proba = last_entry.get("risk_proba", {})
    tier_css   = {"low_risk":"low","medium_risk":"medium","high_risk":"high"}[risk_label]
    tier_emoji = {"low_risk":"🟢","medium_risk":"🟡","high_risk":"🔴"}[risk_label]
    tier_name  = {"low_risk":"Low Risk","medium_risk":"Medium Risk","high_risk":"High Risk"}[risk_label]
    tier_color = {"low_risk":"#34D399","medium_risk":"#FBBF24","high_risk":"#F87171"}[risk_label]

    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown(f"""<div class="metric-card">
          <div class="metric-val">{tier_emoji}</div>
          <div style="margin:6px 0"><span class="badge-{tier_css}">{tier_name}</span></div>
          <div class="metric-lbl" style="margin-top:8px">Mental Health Risk<br/>
          Random Forest · DF1+DF2 (n≈5,700)</div>
        </div>""", unsafe_allow_html=True)

    with d2:
        social = last_entry.get("social_media_time_hrs", 0)
        sleep  = last_entry.get("sleep_hours", 0)
        s_color = "#F87171" if social>4 else "#FBBF24" if social>2 else "#34D399"
        sl_color = "#F87171" if sleep<6 else "#FBBF24" if sleep<7 else "#A78BFA"
        st.markdown(f"""<div class="metric-card">
          <div class="metric-val" style="color:{s_color}">{social}h</div>
          <div class="metric-lbl">Social Media Today</div>
          <div style="margin-top:10px" class="metric-val">
            <span style="color:{sl_color};font-size:1.8rem;font-weight:900">{sleep}h</span>
          </div>
          <div class="metric-lbl">Sleep Last Night</div>
        </div>""", unsafe_allow_html=True)

    with d3:
        exam_pred = last_entry.get("exam_pred")
        if exam_pred:
            ex_css   = {"High":"exam-high","Medium":"exam-medium","Low":"exam-low"}[exam_pred]
            ex_emoji = {"High":"📈","Medium":"📊","Low":"📉"}[exam_pred]
            st.markdown(f"""<div class="metric-card">
              <div class="metric-val">{ex_emoji}</div>
              <div style="margin:6px 0"><span class="badge-{ex_css}">{exam_pred} Performance</span></div>
              <div class="metric-lbl" style="margin-top:8px">Exam Performance Tier<br/>
              Ridge Classifier · DF6 (n=1,000)</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="metric-card">
              <div class="metric-val" style="color:#3A4560">—</div>
              <div class="metric-lbl" style="margin-top:8px">Exam prediction off<br/>
              (opt in via Profile)</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Platforms breakdown
    if last_entry.get("platforms") and len(last_entry["platforms"]) > 1:
        st.markdown("**Platform breakdown today:**")
        cols = st.columns(len(last_entry["platforms"]))
        for i, p in enumerate(last_entry["platforms"]):
            cols[i].metric(p["platform"], f"{p['hours']} hrs")

    st.divider()

    # ── Results sub-tabs ──────────────────────────────────────────────────────
    r1, r2, r3 = st.tabs(["🔍 Explanation", "💡 Recommendations", "📊 Trends"])

    # ── Explanation ───────────────────────────────────────────────────────────
    with r1:
        st.markdown("#### What your Mental Health Risk score means")
        explains = {
            "low_risk": (
                "**You're in the Low Risk tier.** Your social media usage, sleep, and lifestyle "
                "are within a healthy range for your profile. Our Random Forest model (trained on "
                "~5,700 students across DF1 and DF2) assigned this tier based on your usage hours, "
                "sleep quality, primary platform, and demographic factors."
            ),
            "medium_risk": (
                "**You're in the Medium Risk tier.** Some habits are raising flags — typically "
                "2–4 hrs of social media, sleep near but below 7 hrs, or moderate conflict frequency. "
                "This is the most actionable tier: small, consistent changes in sleep and screen time "
                "tend to move people into the Low tier within a week."
            ),
            "high_risk": (
                "**You're in the High Risk tier.** Your inputs suggest your current habits may be "
                "significantly affecting your mental health. Across our datasets, heavy users (4+ hrs/day) "
                "showed the highest FOMO scores, lowest mental health indices, and highest anxiety "
                "(DF4: r=−0.95 usage vs mental health; DF2: anxiety spikes above 2 hrs/day)."
            ),
        }
        st.markdown(f'<div class="explain-box">{explains[risk_label]}</div>',
                    unsafe_allow_html=True)

        if risk_proba:
            st.markdown("**Model confidence (probability per tier):**")
            pc1, pc2, pc3 = st.columns(3)
            tier_order = ["low_risk","medium_risk","high_risk"]
            nice_names = {"low_risk":"Low","medium_risk":"Medium","high_risk":"High"}
            for col, k in zip([pc1,pc2,pc3], tier_order):
                col.metric(nice_names[k] + " Risk", f"{risk_proba.get(k,0)}%")

        if exam_pred:
            st.markdown("#### What your Exam Performance prediction means")
            exam_explains = {
                "High": (
                    "**High performance predicted.** Your study hours and sleep are strong inputs "
                    "for the Ridge model. Study time is the dominant predictor (r=+0.825, DF6). "
                    "Students studying 5+ hrs averaged ~90.8 on exams."
                ),
                "Medium": (
                    "**Medium performance predicted.** There's real room to improve. "
                    "The model sees a middling combination of study and lifestyle factors. "
                    "Adding 1 study hour and reducing social media by 30 min can shift this."
                ),
                "Low": (
                    "**Low performance predicted.** The model sees low study hours, high social media, "
                    "or poor sleep. Key fact: students studying under 2 hrs averaged ~47.5 on exams "
                    "vs ~90.8 for those studying 5+ hrs — both groups used ~2.5 hrs/day of social media. "
                    "Study time, not screen time, is the critical differentiator."
                ),
            }
            st.markdown(f'<div class="explain-box">{exam_explains[exam_pred]}</div>',
                        unsafe_allow_html=True)

        st.markdown("#### About the models")
        st.markdown("""<div class="explain-box">
<b>Risk Classifier (Random Forest, DF1+DF2)</b> — trained on students from 110 countries.
Features: age, gender, primary platform, total social media hours, sleep hours,
social media-to-waking-hour ratio, academic level, relationship status, region.<br/><br/>
<b>Exam Performance (Ridge Classifier, DF6)</b> — trained on 1,000 students.
Predicts High / Medium / Low from study hours, social media hours, sleep, and
self-reported mental health rating. Note: social media hours are weakly negative (r=−0.167)
while study hours dominate (r=+0.825). The model captures this nuance.
</div>""", unsafe_allow_html=True)

    # ── Recommendations ───────────────────────────────────────────────────────
    with r2:
        st.markdown("#### Personalised Recommendations")
        st.caption("Based on your entry and findings from our 6 datasets.")

        social    = last_entry.get("social_media_time_hrs", 0)
        sleep     = last_entry.get("sleep_hours", 0)
        study     = last_entry.get("study_hours") or 0
        conflicts = last_entry.get("conflicts", 0)
        emotion   = last_entry.get("dominant_emotion", "Neutral")
        detox     = last_entry.get("detox_days", 0)
        platforms = last_entry.get("platforms", [])

        recs = []

        # Screen time
        if social > 4:
            recs.append(("warn", "📵 Screen Time",
                f"You used **{social} hrs** of social media today. Our analysis (DF3, DF2) found "
                f"happiness > stress only in the **1–4 hr range**. Stress overtakes happiness around "
                f"6 hrs. Try capping at **{max(2, round(social*0.6))} hrs** tomorrow."))
        elif social > 2:
            recs.append(("warn", "📵 Screen Time",
                f"You're at **{social} hrs** — in the manageable zone, but the ideal is under 2 hrs. "
                f"Cutting 30 min per day compounds over a week."))
        else:
            recs.append(("rec", "📵 Screen Time",
                f"**{social} hrs** — you're in the ideal range. Keep it up!"))

        # Multi-platform note
        if len(platforms) > 1:
            platform_str = ", ".join(f"{p['platform']} ({p['hours']}h)" for p in platforms)
            recs.append(("rec", "📱 Multi-Platform Tip",
                f"You used {platform_str}. Research shows the number of platforms matters as much "
                f"as total hours — each platform context-switches your attention. Consider consolidating "
                f"to 1–2 platforms per day."))

        # Sleep
        if sleep < 6:
            recs.append(("warn", "💤 Sleep",
                f"**{sleep} hrs** is critically low. Sleep is the strongest happiness predictor across "
                f"all 6 datasets. DF3 shows daily screen time and sleep quality have r=−0.76 — "
                f"reducing social media at night directly improves sleep."))
        elif sleep < 7:
            recs.append(("warn", "💤 Sleep",
                f"**{sleep} hrs** is slightly below the recommended 7–9. An extra 30–45 min meaningfully "
                f"shifts your mood and stress."))
        else:
            recs.append(("rec", "💤 Sleep",
                f"**{sleep} hrs** — excellent. Sleep is your strongest protective factor."))

        # Study
        if profile.get("is_student", True) and study is not None:
            if study < 2:
                recs.append(("warn", "📚 Study Time",
                    f"**{study} hrs** puts you in the lowest performance tier. Students studying 5+ hrs "
                    f"averaged ~90 vs ~47 for those under 2 hrs (DF6). Even +1 focused hour shifts "
                    f"your exam prediction."))
            elif study >= 4:
                recs.append(("rec", "📚 Study Time",
                    f"**{study} hrs** — strong! Study time is the single most powerful lever for "
                    f"exam performance (r=+0.825, DF6)."))
            else:
                recs.append(("rec", "📚 Study Time",
                    f"**{study} hrs** is solid. The top performance tier tends to appear at 4–5 hrs."))

        # Conflicts
        if conflicts >= 3:
            recs.append(("warn", "⚡ Conflicts",
                f"**{conflicts} conflicts** over social media today. Higher conflict scores strongly "
                f"associate with lower mental health (DF1). Consider whether specific platforms "
                f"or interactions are driving this — our data shows WhatsApp and Snapchat had the "
                f"highest conflict-associated addiction scores."))

        # Emotion
        if emotion in ["Anxious","Bored"]:
            recs.append(("warn", "🧘 Emotional State",
                f"You reported feeling **{emotion.lower()}**. DF5 found users with boredom or anxiety "
                f"as dominant emotions had the highest median daily screen time — social media may be "
                f"extending rather than relieving these feelings. Try replacing one scroll session "
                f"with a 10-min walk or 5-min breathing exercise."))

        # Detox
        if detox == 0:
            recs.append(("warn", "🌿 Digital Detox",
                "No detox days this week. Even **1–2 days** off are associated with noticeably better "
                "mental states vs zero detox (DF3). Pick one evening to go screen-free after 8 pm."))
        elif 4 <= detox <= 6:
            recs.append(("rec", "🌿 Digital Detox",
                f"**{detox} detox days** — you're in the peak happiness zone (4–6 days, DF3). "
                f"This is the sweet spot our data found."))

        for style, title, text in recs:
            box = "rec-box" if style == "rec" else "warn-box"
            st.markdown(f'<div class="{box}"><b>{title}</b><br/>{text}</div>',
                        unsafe_allow_html=True)

    # ── Trends ────────────────────────────────────────────────────────────────
    with r3:
        st.markdown("#### Your Trend Charts")
        all_scored = [e for e in get_history() if e.get("risk_label")]
        all_scored_sorted = sorted(all_scored, key=lambda e: e["date"])

        if len(all_scored) < 3:
            st.info("📈 Log at least 3 entries (or load demo data in the Daily Log tab) "
                    "to unlock trend charts.")
        else:
            dates_tr  = [e["date"][-5:] for e in all_scored_sorted]
            df_tr = pd.DataFrame({
                "Date"       : dates_tr,
                "Social (h)" : [e.get("social_media_time_hrs",0) for e in all_scored_sorted],
                "Sleep (h)"  : [e.get("sleep_hours",0) for e in all_scored_sorted],
                "Study (h)"  : [e.get("study_hours") or 0 for e in all_scored_sorted],
                "Risk Score" : [RISK_SCORE_MAP.get(e.get("risk_label","low_risk"),20)
                                for e in all_scored_sorted],
            }).set_index("Date")

            tc1, tc2 = st.columns(2)
            with tc1:
                st.caption("**Risk Score** (20=Low, 55=Med, 85=High)")
                st.line_chart(df_tr[["Risk Score"]], color=["#F87171"], height=180)
                st.caption("**Social Media Hours**")
                st.line_chart(df_tr[["Social (h)"]], color=["#60A5FA"], height=180)
            with tc2:
                st.caption("**Sleep Hours**")
                st.line_chart(df_tr[["Sleep (h)"]], color=["#A78BFA"], height=180)
                st.caption("**Study Hours**")
                st.line_chart(df_tr[["Study (h)"]], color=["#34D399"], height=180)

            # 7-day summary
            st.divider()
            st.markdown("**7-Day Averages**")
            l7 = all_scored_sorted[-7:]
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Avg Social Media", f"{round(sum(e.get('social_media_time_hrs',0) for e in l7)/len(l7),1)} hrs")
            a2.metric("Avg Sleep",        f"{round(sum(e.get('sleep_hours',0) for e in l7)/len(l7),1)} hrs")
            a3.metric("Avg Study",        f"{round(sum(e.get('study_hours') or 0 for e in l7)/len(l7),1)} hrs")
            a4.metric("Avg Risk Score",   str(round(sum(RISK_SCORE_MAP.get(e.get('risk_label','low_risk'),20) for e in l7)/len(l7))))

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — HISTORY & TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with TAB_HISTORY:
    history = get_history()
    if not history:
        st.info("No entries yet — fill in the Daily Log to get started!")
    else:
        # Full chart at top
        scored_all = sorted([e for e in history if e.get("risk_label")],
                            key=lambda e: e["date"])
        if len(scored_all) >= 3:
            st.subheader("All-time Trends")
            df_all = pd.DataFrame({
                "Date"       : [e["date"][-5:] for e in scored_all],
                "Social (h)" : [e.get("social_media_time_hrs",0) for e in scored_all],
                "Sleep (h)"  : [e.get("sleep_hours",0) for e in scored_all],
                "Study (h)"  : [e.get("study_hours") or 0 for e in scored_all],
                "Risk Score" : [RISK_SCORE_MAP.get(e.get("risk_label","low_risk"),20)
                                for e in scored_all],
            }).set_index("Date")

            metric_choice = st.selectbox("Select metric to chart",
                ["Social (h)","Sleep (h)","Study (h)","Risk Score"])
            color_map = {
                "Social (h)": ["#60A5FA"], "Sleep (h)": ["#A78BFA"],
                "Study (h)":  ["#34D399"], "Risk Score": ["#F87171"],
            }
            st.line_chart(df_all[[metric_choice]], color=color_map[metric_choice], height=220)
            st.divider()

        # Entry list
        st.subheader(f"All Entries ({len(history)} days)")
        for entry in sorted(history, key=lambda e: e["date"], reverse=True):
            demo_tag  = " 🧪" if entry.get("is_demo") else ""
            rl        = entry.get("risk_label","—")
            tier_icon = {"low_risk":"🟢","medium_risk":"🟡","high_risk":"🔴"}.get(rl,"—")
            with st.expander(
                f"**{entry['date']}**{demo_tag}  ·  "
                f"Risk: {tier_icon}  ·  "
                f"Social: {entry.get('social_media_time_hrs','?')}h  ·  "
                f"Sleep: {entry.get('sleep_hours','?')}h"
            ):
                ec1, ec2, ec3, ec4 = st.columns(4)
                ec1.metric("Social Media", f"{entry.get('social_media_time_hrs','?')} hrs")
                ec2.metric("Sleep",        f"{entry.get('sleep_hours','?')} hrs")
                ec3.metric("Study",        f"{entry.get('study_hours') or '—'} hrs")
                ec4.metric("Exam Pred",    entry.get("exam_pred") or "—")

                if entry.get("platforms"):
                    st.markdown("**Platforms:**  " +
                                " · ".join(f"{p['platform']} {p['hours']}h"
                                           for p in entry["platforms"]))

                if not entry.get("is_demo"):
                    if st.button("🗑️ Delete entry", key=f"del_{entry['date']}"):
                        delete_entry(entry["date"])
                        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — GOALS
# ══════════════════════════════════════════════════════════════════════════════
with TAB_GOALS:
    real_h  = [e for e in get_history() if not e.get("is_demo")]
    last_r  = next((e for e in sorted(real_h, key=lambda e:e["date"], reverse=True)), None)
    last7_r = sorted(real_h, key=lambda e:e["date"], reverse=True)[:7]
    profile = get_profile()

    st.subheader("Daily & Weekly Goals")
    st.caption("Targets from our 6-dataset midterm analysis.")

    st.markdown("#### Today's Goals")
    if last_r:
        def goal_prog(label, current, target):
            pct = min(1.0, current / target)
            done = "✅" if pct >= 1.0 else "⏳"
            st.markdown(f"{done} **{label}**: {current} / {target}")
            st.progress(pct)

        goal_prog("Social Media < 4 hrs",  last_r.get("social_media_time_hrs",0), 4)
        goal_prog("Sleep ≥ 7 hrs",         last_r.get("sleep_hours",0), 7)
        goal_prog("Exercise ≥ 3 days/week", last_r.get("exercise_days",0), 3)
        if profile.get("is_student", True):
            goal_prog("Study ≥ 3 hrs", last_r.get("study_hours") or 0, 3)
    else:
        st.info("Log your first real entry to see goal progress.")

    st.divider()
    st.markdown("#### This Week's Goals (last 7 real entries)")
    if last7_r:
        detox_ct = sum(1 for e in last7_r if e.get("detox_days",0)>0)
        low_conf = sum(1 for e in last7_r if e.get("conflicts",10)<=1)
        happy_ct = sum(1 for e in last7_r if e.get("dominant_emotion")=="Happy")
        green_ct = sum(1 for e in last7_r if e.get("risk_label")=="low_risk")

        def week_prog(label, current, target):
            st.markdown(f"**{label}**: {current}/{target}")
            st.progress(min(1.0, current/target))

        week_prog("🌿 Detox Days (goal: 4)",       detox_ct, 4)
        week_prog("🔕 Low Conflict Days (goal: 5)", low_conf, 5)
        week_prog("😊 Happy Days (goal: 5)",        happy_ct, 5)
        week_prog("🟢 Green Risk Days (goal: 5)",   green_ct, 5)
    else:
        st.info("Log real entries to track weekly goals.")

    st.divider()
    st.markdown("#### Why these targets?")
    st.markdown("""<div class="explain-box">
<b>1–4 hrs social media/day:</b> Happiness index consistently exceeds stress in this range (DF3 + DF2 validated).<br/>
<b>4–6 detox days/week:</b> Peak happiness index in our DF3 analysis (n=500).<br/>
<b>7–9 hrs sleep:</b> Strongest single happiness predictor across all 6 datasets. r=−0.76 with daily screen time (DF3).<br/>
<b>3–5 hrs study:</b> Study hours dominate exam prediction (r=+0.825, DF6). Students at 5+ hrs averaged 90.8 vs 47.5 for under 2 hrs.
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — AWARDS
# ══════════════════════════════════════════════════════════════════════════════
with TAB_AWARDS:
    real_h_awards = [e for e in get_history() if not e.get("is_demo")]
    earned        = check_awards(real_h_awards)

    AWARD_DEFS = [
        ("🌿","detox_starter", "Detox Starter",    "Record any detox day"),
        ("📚","study_hero",    "Study Hero",        "Study 4+ hours in a single day"),
        ("📵","low_screen",    "Screen Minimalist", "Under 2 hrs social media 7 days in a row"),
        ("💤","sleep_champ",   "Sleep Champion",    "8+ hours sleep 5 consecutive days"),
        ("⭐","wellness_week", "Wellness Week",     "Low risk score 7 consecutive days"),
        ("⚖️","balanced_life","Balanced Life",     "All daily targets met 3 days in a row"),
        ("🧘","no_fomo",       "FOMO Fighter",      "Log 'Happy' emotion 5 days in a row"),
        ("🔥","streak_7",      "7-Day Streak",      "Log data 7 consecutive days"),
    ]

    st.subheader(f"Achievement Badges — {len(earned)}/{len(AWARD_DEFS)} earned")
    st.progress(len(earned) / len(AWARD_DEFS) if AWARD_DEFS else 0)
    st.caption("Based on your real entries only (not demo data).")

    cols = st.columns(2)
    for i, (icon, key, label, desc) in enumerate(AWARD_DEFS):
        is_earned = key in earned
        with cols[i % 2]:
            bg      = "#1C2537" if is_earned else "#111827"
            opacity = "1"       if is_earned else "0.4"
            c_label = "#34D399" if is_earned else "#6B7A99"
            tick    = "✅ EARNED" if is_earned else "🔒 Locked"
            st.markdown(f"""
<div style="background:{bg};border:1px solid #2A3550;border-radius:12px;
  padding:12px 14px;margin-bottom:10px;opacity:{opacity}">
  <span style="font-size:1.3rem">{icon}</span>
  <b style="color:{c_label};margin-left:8px">{label}</b>
  <div style="color:#6B7A99;font-size:.78rem;margin-top:4px">{desc}</div>
  <div style="font-size:.72rem;margin-top:6px;color:{c_label}">{tick}</div>
</div>""", unsafe_allow_html=True)
