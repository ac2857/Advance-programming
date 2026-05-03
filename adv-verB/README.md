# Digital Wellness Advisor — PEA Team

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files needed
- `app.py` — main app
- `wellness_app_outputs.pkl` — risk classifier bundle (Random Forest + encoders + EDA plots)
- `student_ridge_model.pkl` — exam performance classifier (Ridge)
- `requirements.txt`

## Model details

### Risk Classifier (`wellness_app_outputs.pkl`)
- **Type:** RandomForestClassifier
- **Features (9):** age, gender_enc, platform_enc, social_media_time_hrs, sleep_hours, sm_to_waking_ratio, academic_enc, rel_enc, region_enc
- **Classes:** low_risk, medium_risk, high_risk
- **Training data:** DF1 + DF2 (n=6,862)

### Exam Performance (`student_ridge_model.pkl`)
- **Type:** RidgeClassifier (3-class: High / Medium / Low)
- **Features (4):** study_hours_per_day, social_media_hours, sleep_hours, mental_health_rating
- **Training data:** DF6 (n=1,000)

## Notes
- `scikit-learn==1.6.1` is required — the pickles were trained on this version
- History is saved to `wellness_history.json` in the same directory
