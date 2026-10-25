"""Feature definitions shared by training, prediction, the LLM layer and the app.

Two models are trained:
  * SCREENING model - only features a woman can report herself (no tests needed)
  * CLINICAL model  - screening features + lab / ultrasound values (for comparison)
"""

TARGET = "pcos"

# ---------------------------------------------------------------------------
# Self-reportable features (used by the public-facing screening model)
# kind: "binary" (0/1) or "number"
# ---------------------------------------------------------------------------
SCREENING_FEATURES = {
    "age": {
        "kind": "number", "label": "Age (years)",
        "question": "How old are you?",
        "min": 12, "max": 60, "required": True,
    },
    "bmi": {
        "kind": "number", "label": "BMI",
        "question": "What are your height (in cm) and weight (in kg)? I'll work out your BMI.",
        "min": 12, "max": 60, "required": True,
    },
    "cycle_irregular": {
        "kind": "binary", "label": "Irregular periods",
        "question": "Are your periods regular (roughly every 21-35 days), or irregular / often missed?",
        "required": True,
    },
    "period_length": {
        "kind": "number", "label": "Period length (days of bleeding)",
        "question": "How many days does your period usually last?",
        "min": 0, "max": 15, "required": False,
    },
    "weight_gain": {
        "kind": "binary", "label": "Unexplained weight gain",
        "question": "Have you noticed weight gain that is hard to explain or to lose?",
        "required": True,
    },
    "hair_growth": {
        "kind": "binary", "label": "Excess hair growth (face/body)",
        "question": "Have you noticed extra hair growth on your face, chest, back or stomach?",
        "required": True,
    },
    "skin_darkening": {
        "kind": "binary", "label": "Darkened skin patches",
        "question": "Do you have any dark, velvety patches of skin, for example on your neck or armpits?",
        "required": True,
    },
    "hair_loss": {
        "kind": "binary", "label": "Scalp hair thinning / loss",
        "question": "Have you noticed thinning hair or hair loss on your scalp?",
        "required": True,
    },
    "pimples": {
        "kind": "binary", "label": "Persistent acne",
        "question": "Do you have persistent acne or pimples?",
        "required": True,
    },
    "fast_food": {
        "kind": "binary", "label": "Regular fast food",
        "question": "Do you eat fast food regularly?",
        "required": False,
    },
    "reg_exercise": {
        "kind": "binary", "label": "Regular exercise",
        "question": "Do you exercise regularly?",
        "required": False,
    },
}

# ---------------------------------------------------------------------------
# Extra chat questions (not model inputs): they feed the same NHS/NICE checks that the
# My health tab uses, so the chat can raise warnings beyond PMOS.
# kind: "scale" (0-10), "binary", or "group" (one question covering several yes/no keys)
# ---------------------------------------------------------------------------
EXTRA_FEATURES = {
    "impact": {
        "kind": "scale", "label": "Effect on daily life (0-10)", "min": 0, "max": 10,
        "question": ("On a scale of 0 to 10, how much do your symptoms affect your day? "
                     "0 = no effect at all, 5 = I've had to cut back on some things, "
                     "7 = they stop me doing usual things like work or study, "
                     "10 = I can't get out of bed."),
    },
    "period_pain": {
        "kind": "scale", "label": "Worst period pain (0-10)", "min": 0, "max": 10,
        "question": ("How bad is your period pain at its worst, from 0 (no pain) to "
                     "10 (the worst pain imaginable)?"),
    },
    "checklist": {
        "kind": "group", "label": "Bleeding and pain checklist",
        "keys": ["heavy_bleeding", "pain_sex", "pain_poo", "pain_pee", "bleeding_between",
                 "bleeding_after_sex"],
        "question": ("Have you had any of these? (a) very heavy bleeding: changing a pad or tampon "
                     "every 1 to 2 hours, clots bigger than a 10p coin, or bleeding through to "
                     "clothes; (b) pain during or after sex; (c) pain when pooing during your "
                     "period; (d) pain when peeing during your period; (e) bleeding between "
                     "periods; (f) bleeding after sex. Tell me which letters, or say none."),
    },
    "bloating_persistent": {
        "kind": "binary", "label": "Bloating most days for 3+ weeks",
        "question": "Have you had bloating most days for 3 weeks or more?",
    },
    "hot_flushes": {
        "kind": "binary", "label": "Hot flushes or night sweats", "min_age": 35,
        "question": "Do you get hot flushes or night sweats?",
    },
    "postmeno_bleeding": {
        "kind": "binary", "label": "Bleeding after the menopause", "min_age": 45,
        "question": ("If your periods stopped more than a year ago, have you had any bleeding "
                     "since? You can say yes, no, or doesn't apply."),
    },
}
EXTRA_KEYS = {  # every field the chat can fill for the NHS/NICE checks
    "impact": "Effect of symptoms on daily life, 0-10",
    "period_pain": "Worst period pain, 0-10",
    "heavy_bleeding": "Very heavy bleeding (pad/tampon every 1-2 hours, big clots or flooding)",
    "pain_sex": "Pain during or after sex",
    "pain_poo": "Pain when pooing during a period",
    "pain_pee": "Pain when peeing during a period",
    "bleeding_between": "Bleeding between periods",
    "bleeding_after_sex": "Bleeding after sex",
    "bloating_persistent": "Bloating most days for 3 weeks or more",
    "hot_flushes": "Hot flushes or night sweats",
    "postmeno_bleeding": "Bleeding after periods had stopped for 12+ months",
}

# ---------------------------------------------------------------------------
# Extra lab / ultrasound features (clinical comparison model only)
# ---------------------------------------------------------------------------
CLINICAL_EXTRA_FEATURES = {
    "fsh": "FSH (mIU/mL)",
    "lh": "LH (mIU/mL)",
    "fsh_lh_ratio": "FSH/LH ratio",
    "amh": "AMH (ng/mL)",
    "tsh": "TSH (mIU/L)",
    "prl": "Prolactin (ng/mL)",
    "vit_d3": "Vitamin D3 (ng/mL)",
    "prg": "Progesterone (ng/mL)",
    "rbs": "Random blood sugar (mg/dL)",
    "hb": "Haemoglobin (g/dL)",
    "waist_hip_ratio": "Waist:hip ratio",
    "follicle_l": "Follicle count (left ovary)",
    "follicle_r": "Follicle count (right ovary)",
    "avg_f_size_l": "Average follicle size, left (mm)",
    "avg_f_size_r": "Average follicle size, right (mm)",
    "endometrium": "Endometrium thickness (mm)",
}

SCREENING_FEATURE_LIST = list(SCREENING_FEATURES.keys())
CLINICAL_FEATURE_LIST = SCREENING_FEATURE_LIST + list(CLINICAL_EXTRA_FEATURES.keys())

FEATURE_LABELS = {k: v["label"] for k, v in SCREENING_FEATURES.items()}
FEATURE_LABELS.update(CLINICAL_EXTRA_FEATURES)

# ---------------------------------------------------------------------------
# Map from the Kaggle dataset's column names to ours.
# Keys are "normalised" names: lower-case, letters and digits only.
# (The original file has inconsistent spacing, e.g. " Age (yrs)", "Height(Cm) ".)
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "pcosyn": TARGET,
    "ageyrs": "age",
    "weightkg": "weight_kg",
    "heightcm": "height_cm",
    "bmi": "bmi",
    "cycleri": "cycle_ri",            # 2 = regular, 4 = irregular in the source data
    "cyclelengthdays": "period_length",  # days of bleeding, not the cycle interval
    "weightgainyn": "weight_gain",
    "hairgrowthyn": "hair_growth",
    "skindarkeningyn": "skin_darkening",
    "hairlossyn": "hair_loss",
    "pimplesyn": "pimples",
    "fastfoodyn": "fast_food",
    "regexerciseyn": "reg_exercise",
    "fshmiuml": "fsh",
    "lhmiuml": "lh",
    "fshlh": "fsh_lh_ratio",
    "amhngml": "amh",
    "tshmiul": "tsh",
    "prlngml": "prl",
    "vitd3ngml": "vit_d3",
    "prgngml": "prg",
    "rbsmgdl": "rbs",
    "hbgdl": "hb",
    "waisthipratio": "waist_hip_ratio",
    "folliclenol": "follicle_l",
    "folliclenor": "follicle_r",
    "avgfsizelmm": "avg_f_size_l",
    "avgfsizermm": "avg_f_size_r",
    "endometriummm": "endometrium",
}

DEFAULT_TARGET_SENSITIVITY = 0.85  # screening: prefer catching cases over avoiding false alarms
RANDOM_STATE = 42
