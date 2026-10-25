"""Women's health tracker: stores a user's logs, computes cycle statistics and raises
alerts based on NHS and NICE guidance (pages reviewed October 2026).

Every alert carries:
  - where the rule comes from ("source"), and
  - "How reliable is this sign?" ("evidence"), quoting published studies where they exist.
Only the PMOS pattern check is a model we trained and tested ourselves; everything here is
a transparent rule, so we never claim our own accuracy for it.

The log is a plain dict so it can live in the browser session and be downloaded /
re-uploaded as JSON by the user. Nothing here talks to a server or an AI.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from statistics import median

from .predict import compute_bmi

FLOWS = ["Light", "Medium", "Heavy", "Very heavy (changing pad/tampon every 1-2 hours)"]
VERY_HEAVY = FLOWS[3]

# Symptoms are grouped so the logging form stays readable.
SYMPTOM_GROUPS = {
    "Pain": {
        "pain_stops_activities": "Period pain that stopped normal activities",
        "pelvic_pain": "Pelvic or lower tummy pain outside periods",
        "pain_sex": "Pain during or after sex",
        "pain_poo": "Pain when pooing during a period",
        "pain_pee": "Pain when peeing during a period",
    },
    "Bleeding": {
        "clots": "Blood clots bigger than a 10p coin",
        "flooding": "Bleeding through to clothes or bedding",
        "double_protection": "Needing two period products at once",
        "bleeding_between": "Bleeding between periods",
        "bleeding_after_sex": "Bleeding after sex",
        "postmeno_bleeding": "Any bleeding after periods had stopped for 12+ months",
    },
    "Skin, hair and weight": {
        "acne": "Acne or oily skin",
        "hair_growth": "Extra hair on face, chest, back or stomach",
        "hair_loss": "Hair thinning on the scalp",
        "skin_darkening": "Thick, dark patches of skin (e.g. neck, armpits)",
        "weight_gain": "Weight gain that's hard to explain",
    },
    "Energy and mood": {
        "tiredness": "Feeling very tired",
        "breathless": "Short of breath or heart pounding with everyday activity",
        "low_mood": "Low mood or anxiety",
    },
    "Menopause-type symptoms": {
        "hot_flushes": "Hot flushes",
        "night_sweats": "Night sweats",
        "poor_sleep": "Trouble sleeping",
        "brain_fog": "Brain fog (memory or concentration problems)",
    },
    "Tummy": {
        "bloating": "Bloating or a swollen tummy",
        "full_quickly": "Feeling full quickly or losing appetite",
    },
}
SYMPTOMS = {k: v for group in SYMPTOM_GROUPS.values() for k, v in group.items()}

PMOS_SYMPTOMS = ["hair_growth", "acne", "weight_gain", "tiredness", "skin_darkening", "hair_loss"]
ENDO_SYMPTOMS = ["pain_stops_activities", "pelvic_pain", "pain_sex", "pain_poo", "pain_pee"]
HEAVY_SYMPTOMS = ["clots", "flooding", "double_protection"]
ANAEMIA_SYMPTOMS = ["tiredness", "breathless"]
MENO_SYMPTOMS = ["hot_flushes", "night_sweats", "poor_sleep", "brain_fog"]
VASOMOTOR = ["hot_flushes", "night_sweats"]

SCREENING_OPTIONS = ["Prefer not to say", "Less than 5 years ago", "More than 5 years ago",
                     "Never, or I'm not sure"]

NHS_LINKS = {
    "irregular": "https://www.nhs.uk/symptoms/irregular-periods/",
    "heavy": "https://www.nhs.uk/conditions/heavy-periods/",
    "missed": "https://www.nhs.uk/symptoms/missed-or-late-periods/",
    "pmos": "https://www.nhs.uk/conditions/polyendocrine-metabolic-ovarian-syndrome-pmos/",
    "pain": "https://www.nhs.uk/symptoms/period-pain/",
    "endometriosis": "https://www.nhs.uk/conditions/endometriosis/",
    "bleeding": "https://www.nhs.uk/symptoms/vaginal-bleeding-between-periods-or-after-sex/",
    "postmeno": "https://www.nhs.uk/symptoms/post-menopausal-bleeding/",
    "bloating": "https://www.nhs.uk/symptoms/bloating/",
    "menopause": "https://www.nhs.uk/conditions/menopause-and-perimenopause/symptoms/",
    "screening": "https://www.nhs.uk/tests-and-treatments/cervical-screening/",
    "gp_booking": "https://www.nhs.uk/nhs-services/gps/gp-appointments-and-bookings/",
    "pelvic": "https://www.nhs.uk/symptoms/pelvic-pain/",
    "breathless": "https://www.nhs.uk/symptoms/shortness-of-breath/",
    "mental_health": ("https://www.nhs.uk/nhs-services/mental-health-services/"
                      "where-to-get-urgent-help-for-mental-health/"),
}

NORMAL_CYCLE = (21, 35)       # NHS: irregular if the gap is < 21 or > 35 days
MAX_PERIOD_DAYS = 7           # NHS: see a GP if periods last longer than 7 days
MISSED_DAYS = 90              # NHS: see a GP if you've missed 3 periods in a row (~90 days)
SEVERE_PAIN = 8               # on a 0-10 scale; NHS: urgent advice for severe period pain
STRONG_PAIN = 7               # 7+/10 in 2 or more periods counts as pain affecting daily life
BLOATING_DAYS = 12            # NICE NG12: bloating 12+ times a month -> GP should offer tests
RECENT_PERIODS = 6            # look at roughly the last 6 months of periods
RECENT_DAYS = 180


# ---------------------------------------------------------------------------
# Log structure, import / export
# ---------------------------------------------------------------------------
def empty_log() -> dict:
    return {"profile": {"age": None, "height_cm": None, "weight_kg": None, "menopause": False,
                        "last_screening": SCREENING_OPTIONS[0]},
            "periods": [], "symptoms": []}


def _d(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _span(p: dict, today: date | None = None) -> tuple[date, date]:
    """First and last day of a logged period (an ongoing period runs to today)."""
    start = _d(p["start"])
    end = _d(p["end"]) if p.get("end") else (today or date.today())
    return start, max(start, end)


def overlapping(log: dict, start: date, end: date | None, today: date | None = None) -> list[dict]:
    """Logged periods whose days overlap start..end."""
    new_end = max(start, end or today or date.today())
    return [p for p in log["periods"]
            if _span(p, today)[0] <= new_end and start <= _span(p, today)[1]]


def add_period(log: dict, start: date, end: date | None, flow: str, pain: int | None,
               source: str | None = None) -> list[dict]:
    """Add a period. `pain` can be None when it wasn't recorded (e.g. Apple Health).

    If it has the same first day as, or overlaps, a period already logged, the new entry
    replaces the old one (the newest entry is treated as the correction). Returns the
    periods that were replaced so the app can tell the user."""
    if end is not None and end < start:
        raise ValueError("The end date can't be before the start date.")
    clash = overlapping(log, start, end)
    replaced = [p for p in log["periods"]
                if _d(p["start"]) == start or any(p is q for q in clash)]
    log["periods"] = [p for p in log["periods"] if not any(p is q for q in replaced)]
    entry = {"start": start, "end": end, "flow": flow,
             "pain": int(pain) if pain is not None else None}
    if source:
        entry["source"] = source
    log["periods"].append(entry)
    log["periods"].sort(key=lambda p: _d(p["start"]))
    return replaced


SEVERITY_OPTIONS = ["None", "Mild", "Moderate", "Severe"]
SEVERE = 3
SEVERITY_HELP = ("Mild = noticeable · Moderate = gets in the way of things · "
                 "Severe = stops me doing things or is very hard to cope with")


def add_symptoms(log: dict, day: date, flags: dict, weight_kg: float | None = None,
                 impact: int | None = None, severity: dict | None = None) -> None:
    """`impact`: 0-10 "how much did symptoms affect your day" (optional).
    `severity`: {symptom: 1 mild, 2 moderate, 3 severe} (optional). A rated symptom counts as
    present."""
    sev = {k: int(v) for k, v in (severity or {}).items() if k in SYMPTOMS and v}
    entry = {"date": day, **{k: bool(flags.get(k)) or k in sev for k in SYMPTOMS},
             "weight_kg": weight_kg, "impact": int(impact) if impact is not None else None,
             "severity": sev}
    log["symptoms"] = [s for s in log["symptoms"] if _d(s["date"]) != day]
    log["symptoms"].append(entry)
    log["symptoms"].sort(key=lambda s: _d(s["date"]))


def to_json(log: dict) -> str:
    def enc(o):
        return o.isoformat() if isinstance(o, date) else o
    return json.dumps({"app": "herhealth", "version": 2, **log}, default=enc, indent=2)


def from_json(text: str) -> dict:
    raw = json.loads(text)
    log = empty_log()
    saved = raw.get("profile", {})
    log["profile"].update({k: saved[k] for k in log["profile"] if saved.get(k) is not None})
    for p in raw.get("periods", []):
        add_period(log, _d(p["start"]), _d(p["end"]) if p.get("end") else None,
                   p.get("flow", "Medium"), p.get("pain", 0), p.get("source"))
    for s in raw.get("symptoms", []):
        add_symptoms(log, _d(s["date"]), s, s.get("weight_kg"), s.get("impact"), s.get("severity"))
    if raw.get("wearable"):
        log["wearable"] = raw["wearable"]          # Apple Health import (dates kept as text)
    return log


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def stats(log: dict, today: date) -> dict:
    periods = sorted(log["periods"], key=lambda p: _d(p["start"]))
    starts = [_d(p["start"]) for p in periods]
    cycles = [(starts[i], (starts[i + 1] - starts[i]).days) for i in range(len(starts) - 1)]
    lengths = [(_d(p["end"]) - _d(p["start"])).days + 1 for p in periods if p.get("end")]
    typical = median([c for _, c in cycles]) if cycles else None
    days_since = (today - starts[-1]).days if starts else None
    return {
        "periods": periods,
        "cycles": cycles,                       # (start of cycle, length in days)
        "period_lengths": lengths,
        "typical_cycle": typical,
        "days_since_last": days_since,
        "next_expected": starts[-1] + timedelta(days=round(typical)) if starts and typical else None,
    }


def _recent_symptoms(log: dict, today: date, days: int = RECENT_DAYS) -> list[dict]:
    return [s for s in log["symptoms"] if 0 <= (today - _d(s["date"])).days <= days]


def _logged(recent: list[dict], keys: list[str]) -> list[str]:
    """Labels of the symptoms in `keys` that were logged at least once."""
    return [SYMPTOMS[k] for k in keys if any(s.get(k) for s in recent)]


def _join(items: list[str]) -> str:
    items = [i[0].lower() + i[1:] for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _alert(key, level, title, detail, link, source, evidence) -> dict:
    return {"key": key, "level": level, "title": title, "detail": detail, "link": link,
            "source": source, "evidence": evidence}


# ---------------------------------------------------------------------------
# Alerts (each one maps to explicit NHS / NICE advice)
# ---------------------------------------------------------------------------
def alerts(log: dict, today: date) -> list[dict]:
    st = stats(log, today)
    prof = log.get("profile", {})
    age = prof.get("age")
    out: list[dict] = []
    recent_periods = st["periods"][-RECENT_PERIODS:]
    recent_cycles = [c for _, c in st["cycles"][-3:]]
    recent = _recent_symptoms(log, today)

    if len(st["periods"]) < 2 and not prof.get("menopause"):
        out.append(_alert(
            "keep_logging", "info", "Keep logging",
            "Log at least 2 periods so we can work out your cycle length.", None,
            "HerHealth", "Cycle checks need at least two period start dates."))

    # --- Cycle pattern -----------------------------------------------------------
    irregular = [c for c in recent_cycles if not NORMAL_CYCLE[0] <= c <= NORMAL_CYCLE[1]]
    if irregular:
        if len(recent_cycles) == 1:
            what = f"Your last cycle was {irregular[0]} days"
        else:
            what = (f"{len(irregular)} of your last {len(recent_cycles)} cycles were "
                    f"{', '.join(str(c) for c in irregular)} days")
        extra = (" In your 40s, irregular periods are often an early sign of perimenopause."
                 if age and age >= 40 else "")
        out.append(_alert(
            "irregular", "gp", "Your periods look irregular",
            f"{what}, outside the usual 21 to 35 days. The NHS advises speaking to a GP if your "
            f"periods are irregular.{extra}",
            NHS_LINKS["irregular"], "NHS: irregular periods; NICE PMOS guideline (draft, July 2026)",
            "The 21 to 35 day range is the NHS definition of a regular cycle. Irregular periods "
            "often have harmless causes (stress, weight change, contraception), but they are also "
            "the most common sign of PMOS, which is why NICE's first PMOS guideline (draft, July "
            "2026) suggests checking women with irregular periods. This is guidance, not a test "
            "we have measured ourselves."))

    if st["days_since_last"] is not None and not prof.get("menopause"):
        if st["days_since_last"] > MISSED_DAYS:
            out.append(_alert(
                "missed", "gp", f"No period logged for {st['days_since_last']} days",
                "That's about 3 missed periods. The NHS advises seeing a GP if you've missed your "
                "period 3 times in a row. If there's any chance you could be pregnant, take a "
                "pregnancy test.",
                NHS_LINKS["missed"], "NHS: missed or late periods",
                "NHS rule of thumb. Common causes include pregnancy, stress, weight change, "
                "intense exercise, PMOS and early menopause, which a GP can check."))
        elif st["typical_cycle"] and st["days_since_last"] > st["typical_cycle"] + 7:
            late = st["days_since_last"] - round(st["typical_cycle"])
            out.append(_alert(
                "late", "info", f"Your period is about {late} days later than usual",
                "Late periods are common and often nothing to worry about. If there's any chance "
                "you could be pregnant, take a pregnancy test.",
                NHS_LINKS["missed"], "NHS: missed or late periods",
                "Most people's cycles vary by a few days from month to month."))

    long_ones = [p for p in recent_periods if p.get("end")
                 and (_d(p["end"]) - _d(p["start"])).days + 1 > MAX_PERIOD_DAYS]
    if long_ones:
        out.append(_alert(
            "long_period", "gp", "A period lasted longer than 7 days",
            f"Logged on {', '.join(_d(p['start']).strftime('%d %b') for p in long_ones)}. "
            "The NHS advises seeing a GP if your periods last longer than 7 days.",
            NHS_LINKS["irregular"], "NHS: irregular periods",
            "NHS guidance threshold. Long periods are often linked to heavy bleeding, fibroids "
            "or hormonal changes, which a GP can check."))

    # --- Heavy periods (NHS, NICE NG88) --------------------------------------------
    heavy_signs = []
    if any(p.get("flow") == VERY_HEAVY for p in recent_periods):
        heavy_signs.append("Changing a pad or tampon every 1 to 2 hours")
    heavy_signs += _logged(recent, HEAVY_SYMPTOMS)
    if heavy_signs:
        anaemia = _logged(recent, ANAEMIA_SYMPTOMS)
        detail = (f"You logged: {_join(heavy_signs)}. The NHS counts these as signs of heavy "
                  "periods. The NHS advises seeing a GP if heavy periods are affecting your life.")
        if anaemia:
            detail += (f" You also logged {_join(anaemia)}, which can be signs of iron-deficiency "
                       "anaemia. NICE recommends a blood test (full blood count) for everyone "
                       "with heavy periods.")
        out.append(_alert(
            "heavy", "gp", "Your periods may be heavy", detail, NHS_LINKS["heavy"],
            "NHS: heavy periods; NICE NG88 (heavy menstrual bleeding)",
            "Doctors decide whether periods are heavy mainly by how much they affect your life "
            "(NICE NG88), not by measuring blood. Counting products is only a rough guide: a "
            "pad-and-tampon scoring chart (PBAC) picked up 86% of measured heavy bleeding in its "
            "first small study (Higham, 1990), but later studies varied from 58% to 98%. "
            "In about half of people with heavy periods no underlying cause is found."))

    # --- Pain --------------------------------------------------------------------------
    severe = severe_symptoms(log, today)
    if (any((p.get("pain") or 0) >= SEVERE_PAIN for p in recent_periods)
            or "pain_stops_activities" in severe):
        what = ("You logged pain of 8/10 or more." if any(
            (p.get("pain") or 0) >= SEVERE_PAIN for p in recent_periods)
            else "You rated period pain that stopped normal activities as severe.")
        out.append(_alert(
            "severe_pain", "urgent", "Severe period pain",
            f"{what} If your period pain is severe or worse than usual "
            "and painkillers have not helped, the NHS advises asking for an urgent GP "
            "appointment or getting help from NHS 111. See a GP if pain is stopping you doing "
            "your usual activities.",
            NHS_LINKS["pain"], "NHS: period pain",
            "NHS guidance. Severe pain is not something you should just put up with. It can be "
            "caused by conditions such as endometriosis, adenomyosis or fibroids."))

    endo = _logged(recent, ENDO_SYMPTOMS)
    strong = [p for p in recent_periods if (p.get("pain") or 0) >= STRONG_PAIN]
    if len(strong) >= 2:
        endo.insert(0, f"Period pain of {STRONG_PAIN}/10 or more in {len(strong)} periods")
    if endo:
        out.append(_alert(
            "endometriosis", "gp", "Symptoms doctors link with endometriosis",
            f"You logged: {_join(endo)}. NICE advises doctors to think about endometriosis when "
            "someone has one or more of these. It can take years to diagnose, so it's worth "
            "naming these symptoms to your GP and asking whether endometriosis could be the cause.",
            NHS_LINKS["endometriosis"], "NICE NG73 (endometriosis); NHS: endometriosis",
            "We don't give an endometriosis score, because symptoms alone can't reliably tell who "
            "has it: in a multicentre study of 1,396 women, a 25-question symptom questionnaire "
            "scored an AUC of 0.68 (0.5 is a coin toss, 1.0 is perfect), and a 2026 machine "
            "learning study found much the same (0.67). Diagnosis needs a GP assessment, "
            "usually an ultrasound scan and sometimes keyhole surgery. In the UK it takes "
            "8 years 10 months on average (Endometriosis UK, 2024)."))

    # --- Bleeding that should always be checked ------------------------------------
    if _logged(recent, ["bleeding_between"]):
        out.append(_alert(
            "bleeding_between", "gp", "Bleeding between periods",
            "The NHS advises seeing a GP or going to a sexual health clinic if you have bleeding "
            "between periods.",
            NHS_LINKS["bleeding"], "NHS: bleeding between periods or after sex",
            "It's usually caused by something harmless, such as hormonal contraception, but the "
            "NHS says unusual bleeding should always be checked because it can occasionally be a "
            "sign of something more serious."))

    if _logged(recent, ["bleeding_after_sex"]):
        out.append(_alert(
            "bleeding_after_sex", "gp", "Bleeding after sex",
            "The NHS advises seeing a GP or going to a sexual health clinic if you bleed after sex.",
            NHS_LINKS["bleeding"], "NHS: bleeding between periods or after sex",
            "It's usually caused by something minor, such as an infection or a harmless change "
            "on the cervix, but it can occasionally be a sign of cervical cancer, so the NHS says "
            "it should always be checked."))

    post_bleed = bool(_logged(recent, ["postmeno_bleeding"]))
    if prof.get("menopause") and any((today - _d(p["start"])).days <= RECENT_DAYS
                                     for p in st["periods"]):
        post_bleed = True
    if post_bleed:
        out.append(_alert(
            "postmeno_bleeding", "urgent", "Bleeding after the menopause",
            "The NHS says to see a GP about any bleeding after the menopause, even if it was only "
            "spotting or happened once. Contact your GP surgery and ask for an appointment as "
            "soon as you can.",
            NHS_LINKS["postmeno"], "NHS: post-menopausal bleeding; NICE NG12 (suspected cancer)",
            "It's usually not serious: in a review of 129 studies, about 9 in 100 people with "
            "bleeding after the menopause had womb cancer. But about 9 in 10 womb cancers cause "
            "this bleeding, so getting it checked quickly catches most of them early "
            "(Clarke et al., JAMA Internal Medicine, 2018)."))

    # --- Persistent bloating (NHS, NICE NG12) ----------------------------------------
    month = _recent_symptoms(log, today, days=30)
    bloat_days = sum(1 for s in month if s.get("bloating"))
    if bloat_days >= BLOATING_DAYS:
        also = _logged(month, ["full_quickly", "pelvic_pain"])
        detail = (f"You logged bloating on {bloat_days} days in the last month"
                  + (f", and {_join(also)}" if also else "") + ". The NHS advises seeing a GP if "
                  "you feel bloated regularly, and NICE advises GPs to offer tests when bloating "
                  "happens 12 or more times a month, especially if you're 50 or over.")
        out.append(_alert(
            "bloating", "gp", "Bloating most days", detail, NHS_LINKS["bloating"],
            "NHS: bloating; NICE NG12 (suspected cancer)",
            "Bloating is very common and usually caused by things like diet, constipation or IBS. "
            "Bloating that doesn't go away can occasionally be an early sign of ovarian cancer, "
            "which is why it's worth checking. Symptom checklists like this can't rule it out: "
            "a well-known one spotted 57% of early and 80% of advanced ovarian cancers, while "
            "correctly clearing 87% to 90% of people without it (Goff et al., 2007)."))

    # --- Perimenopause (NHS, NICE NG23) ------------------------------------------------
    meno = _logged(recent, MENO_SYMPTOMS)
    if _logged(recent, VASOMOTOR) or len(meno) >= 2:
        if age is not None and age < 45:
            advice = ("The NHS advises seeing a GP if you think you have menopause symptoms. "
                      "If you're under 45, a GP may do blood tests to check for early menopause "
                      "or other causes, such as thyroid problems.")
        else:
            advice = ("These are common signs of perimenopause, which usually starts in your 40s. "
                      "You don't have to wait for your periods to stop: the NHS advises seeing a "
                      "GP if you think you have menopause symptoms. Treatments, including HRT, "
                      "can help.")
        out.append(_alert(
            "perimenopause", "gp", "Possible perimenopause symptoms",
            f"You logged: {_join(meno)}. {advice}", NHS_LINKS["menopause"],
            "NICE NG23 and quality standard QS143 (menopause); NHS: menopause",
            "NICE says people aged 45 or over with menopause symptoms are diagnosed from their "
            "symptoms alone, without blood tests, because hormone (FSH) levels go up and down "
            "during perimenopause. So there's nothing for an app to 'predict' here: the value "
            "is in tracking your symptoms and taking them to your GP."))

    # --- Cervical screening reminder ---------------------------------------------------
    screening = prof.get("last_screening") or SCREENING_OPTIONS[0]
    if age is not None and 25 <= age <= 64 and screening in SCREENING_OPTIONS[2:]:
        overdue = screening == SCREENING_OPTIONS[2]
        out.append(_alert(
            "screening", "gp" if overdue else "info",
            "You may be overdue for cervical screening" if overdue else "Cervical screening",
            ("Your last cervical screening was more than 5 years ago. " if overdue else "")
            + "Everyone with a cervix aged 25 to 64 is invited every 5 years. Check when you're "
            "due in the NHS App, or contact your GP surgery to book.",
            NHS_LINKS["screening"], "NHS cervical screening programme",
            "Cervical screening checks for HPV, the virus that causes almost all cervical cancers. "
            "Since July 2025 in England, people aged 25 to 49 whose test shows no HPV are invited "
            "every 5 years instead of 3, based on UK National Screening Committee advice. If HPV "
            "is found, you'll be invited more often."))

    # --- Combination our PMOS model looks at ------------------------------------------
    linked = _logged(recent, PMOS_SYMPTOMS)
    if irregular and linked:
        out.append(_alert(
            "pmos", "gp", "Irregular periods with other symptoms",
            f"Alongside irregular periods you logged: {_join(linked)}. The NHS advises seeing a "
            "GP about this combination. Possible causes include polyendocrine metabolic ovarian "
            "syndrome (PMOS, formerly PCOS) and thyroid problems, so a GP can check.",
            NHS_LINKS["pmos"], "NHS: PMOS; NICE PMOS guideline (draft, July 2026)",
            "This is the combination our PMOS pattern check looks at. Its measured accuracy is "
            "shown with the pattern check below. A GP can confirm PMOS with blood tests and "
            "sometimes an ultrasound."))

    # --- Symptoms rated "severe" on the per-symptom scale -------------------------------
    out += severe_symptom_alerts(severe)

    # --- How much symptoms affect daily life (independent of any score) --------------
    rated = [s["impact"] for s in _recent_symptoms(log, today, days=30)
             if s.get("impact") is not None]
    if rated and max(rated) >= IMPACT_GP:
        out.append(impact_alert(max(rated)))

    for a in out:
        a.setdefault("tests", tests_for(a["key"], age))
    return out


# ---------------------------------------------------------------------------
# "How much do your symptoms affect your day?" 0-10 scale (used in all three tabs)
# ---------------------------------------------------------------------------
IMPACT_GP, IMPACT_SEVERE = 7, 9
IMPACT_ANCHORS = {
    0: "no effect on my day",
    3: "noticeable, but I carry on as normal",
    5: "I've had to cut back on some things",
    7: "stopping me doing usual things (work, study, exercise, seeing people)",
    9: "I've had to stay in bed or couldn't manage basic things",
    10: "I can't get out of bed or do anything at all",
}
IMPACT_HELP = ("0 = no effect on my day · 3 = noticeable, but I carry on · 5 = cutting back "
               "on some things · 7 = stopping me doing usual things (work, study, plans) · "
               "10 = I can't get out of bed")


def severe_symptoms(log: dict, today: date, days: int = 30) -> set[str]:
    """Symptoms rated 'Severe' in the last month."""
    return {k for s in _recent_symptoms(log, today, days)
            for k, v in (s.get("severity") or {}).items() if v >= SEVERE}


def worst_severity(log: dict, today: date, key: str, days: int = RECENT_DAYS) -> int | None:
    levels = [(s.get("severity") or {}).get(key) for s in _recent_symptoms(log, today, days)]
    levels = [v for v in levels if v]
    return max(levels) if levels else None


def describe_symptoms(entry: dict) -> str:
    """'Bloating (severe), Feeling very tired' for the log table."""
    sev = entry.get("severity") or {}
    parts = [SYMPTOMS[k] + (f" ({SEVERITY_OPTIONS[sev[k]].lower()})" if sev.get(k) else "")
             for k in SYMPTOMS if entry.get(k)]
    return ", ".join(parts) or "None"


def severe_symptom_alerts(severe: set[str]) -> list[dict]:
    out = []
    if "pelvic_pain" in severe:
        out.append(_alert(
            "severe_pelvic_pain", "urgent", "Severe pelvic pain",
            "You rated pelvic or lower tummy pain as severe. Call 999 or go to A&E if it's severe "
            "and getting worse, hurts when you move or touch the area, or you feel faint, have "
            "heavy bleeding or difficulty breathing. Get help from NHS 111 if you also have pain "
            "peeing, unusual discharge or bleeding, a high temperature, or could be pregnant. "
            "Otherwise, see a GP if it comes and goes or doesn't go away.",
            NHS_LINKS["pelvic"], "NHS: pelvic pain",
            "NHS guidance. Most pelvic pain has a common cause, but a few causes need urgent "
            "treatment, which is why severe pain should be checked quickly."))
    if "breathless" in severe:
        out.append(_alert(
            "severe_breathless", "urgent", "Severe breathlessness or heart pounding",
            "You rated shortness of breath or a pounding heart as severe. Call 999 if you're "
            "struggling to breathe, your chest feels tight or heavy, or your lips or skin are "
            "turning pale, blue or grey. Contact NHS 111 if you're more short of breath than "
            "usual or have heart palpitations. Heavy periods can cause anaemia, which can make "
            "you breathless, so mention your periods too.",
            NHS_LINKS["breathless"], "NHS: shortness of breath",
            "NHS guidance. Breathlessness has many causes, from anaemia to asthma; it should "
            "always be checked."))
    if "low_mood" in severe:
        out.append(_alert(
            "severe_low_mood", "gp", "Severe low mood or anxiety",
            "You rated low mood or anxiety as severe. Please ask for an urgent GP appointment or "
            "get help from NHS 111 (choose the mental health option). If you feel you can't keep "
            "yourself safe, call 999. You can also talk to Samaritans any time on 116 123 (free).",
            NHS_LINKS["mental_health"], "NHS: where to get urgent help for mental health",
            "NHS guidance. Mood can change with hormones (for example around periods or the "
            "menopause), and support and treatment help, so it's worth raising."))
    other = [SYMPTOMS[k] for k in SYMPTOMS
             if k in severe and k not in ("pelvic_pain", "breathless", "low_mood",
                                          "pain_stops_activities")]
    if other:
        out.append(_alert(
            "severe_symptoms", "gp", "Symptoms you rated as severe",
            f"You rated these as severe in the last month: {_join(other)}. The NHS advises seeing "
            "a GP if symptoms are affecting your everyday life, work or relationships. Ask your "
            "GP surgery for an appointment and mention how severe they are.",
            NHS_LINKS["gp_booking"], "NHS: PMOS and period problems ('see a GP if symptoms "
            "affect your everyday life')",
            "This uses your own severity rating, not a test. Severe symptoms deserve a GP's "
            "attention whatever the other checks say."))
    return out


def recent_impact(log: dict, today: date, days: int = 30) -> int | None:
    """Highest 'effect on my day' rating in the last month, if any were given."""
    rated = [s["impact"] for s in _recent_symptoms(log, today, days)
             if s.get("impact") is not None]
    return max(rated) if rated else None


def impact_label(value: int) -> str:
    """Plain-English meaning of a 0-10 impact rating."""
    return IMPACT_ANCHORS[max(k for k in IMPACT_ANCHORS if k <= value)]


def impact_alert(value: int) -> dict:
    severe = value >= IMPACT_SEVERE
    detail = (f"You rated the effect of your symptoms as {value}/10 ({impact_label(value)}). "
              "The NHS advises seeing a GP if symptoms are affecting your everyday life, work or "
              "relationships, whatever any pattern check score says. ")
    detail += ("Contact your GP surgery and ask for an appointment soon: you don't need to wait "
               "for a pattern to build up. " if severe else
               "Ask your GP surgery for an appointment. ")
    detail += ("Get urgent help from NHS 111 (or 999 in an emergency) if you have sudden severe "
               "pain, very heavy bleeding, or feel faint.")
    return _alert(
        "impact", "gp",
        "Your symptoms are stopping you doing everyday things" if severe
        else "Your symptoms are affecting your daily life",
        detail, NHS_LINKS["gp_booking"],
        "NHS: PMOS ('see a GP if symptoms affect your everyday life'); NHS: period pain",
        "This isn't a test score: it's how much your symptoms affect your life, and NHS guidance "
        "treats that as a reason to see a GP on its own. A low PMOS score never means symptoms "
        "this disruptive should be ignored.")


# ---------------------------------------------------------------------------
# Tests a GP might consider, per alert (NHS and NICE). Shown on cards and in the summary.
# ---------------------------------------------------------------------------
def tests_for(key: str, age: int | None = None) -> list[str]:
    pmos = ["Blood tests to check hormone levels and insulin resistance (NHS)",
            "An ultrasound scan of your ovaries, if you're over 18 (NHS)"]
    if key in ("pmos", "irregular"):
        return pmos + ["Other checks your GP thinks are needed to rule out other causes"]
    if key == "missed":
        return ["A pregnancy test first, if there's any chance you could be pregnant",
                "Your GP will decide whether hormone blood tests are needed"]
    if key in ("heavy", "long_period"):
        return ["A full blood count to check for anaemia. NICE recommends this for everyone "
                "with heavy periods (it checks for anaemia, not the cause)",
                "Depending on your symptoms, an examination, ultrasound scan or hysteroscopy "
                "(NICE NG88)"]
    if key == "severe_pain":
        return ["An examination, and possibly an ultrasound scan to look for causes such as "
                "endometriosis or fibroids"]
    if key == "endometriosis":
        return ["No blood test can diagnose endometriosis: NICE advises against using the "
                "CA-125 blood test for this",
                "Ask about a transvaginal (internal) ultrasound scan, which NICE says GPs should "
                "offer, and referral to a gynaecologist if symptoms continue"]
    if key == "perimenopause":
        if age is not None and age >= 45:
            return ["Usually no blood test needed: NICE says perimenopause is diagnosed from "
                    "symptoms if you're 45 or over, because hormone levels go up and down"]
        if age is not None and age < 40:
            return ["FSH hormone blood tests, usually two taken 4 to 6 weeks apart, to check "
                    "for early menopause (premature ovarian insufficiency), per NICE"]
        return ["Your GP may do blood tests if you're under 45 (NHS), for example to check for "
                "early menopause or thyroid problems"]
    if key == "bloating":
        if age is not None and age < 40:
            return ["Your GP will decide which tests are needed. NICE's CA125 blood test advice "
                    "applies mainly to people aged 40 and over"]
        return ["A CA125 blood test, and an ultrasound scan within 2 weeks if needed (NICE, for "
                "people aged 40 and over with persistent symptoms). CA125 is never used on its own"]
    if key == "postmeno_bleeding":
        return ["Not a blood test: your GP will usually refer you to a specialist clinic, "
                "normally within 2 weeks, for a scan and sometimes a hysteroscopy (NHS)"]
    if key in ("bleeding_between", "bleeding_after_sex"):
        return ["An examination, and checking your cervical screening is up to date"]
    return []


# ---------------------------------------------------------------------------
# Alerts from one-off answers (Quick check and chat) instead of a long-term log
# ---------------------------------------------------------------------------
SNAPSHOT_SKIP = {"keep_logging", "late", "missed", "irregular", "pmos"}


def snapshot_alerts(snap: dict, today: date) -> list[dict]:
    """snap keys (all optional): age, menopause, cycle_irregular, period_length, flow, pain,
    impact, flags {symptom: bool}, bloating_persistent, last_screening."""
    log = empty_log()
    log["profile"].update({"age": snap.get("age"), "menopause": bool(snap.get("menopause")),
                           "last_screening": snap.get("last_screening") or SCREENING_OPTIONS[0]})
    length = int(snap.get("period_length") or 0)
    if not snap.get("menopause") and (snap.get("flow") or snap.get("pain") is not None):
        start = today - timedelta(days=14)
        add_period(log, start, start + timedelta(days=max(length, 1) - 1),
                   snap.get("flow") or "Medium", snap.get("pain"))
    flags = dict(snap.get("flags") or {})
    if snap.get("bloating_persistent"):
        for i in range(BLOATING_DAYS):
            add_symptoms(log, today - timedelta(days=2 * i + 1), {"bloating": True})
        flags["bloating"] = True
    add_symptoms(log, today, flags, None, snap.get("impact"), snap.get("severity"))
    out = [a for a in alerts(log, today) if a["key"] not in SNAPSHOT_SKIP]
    if snap.get("menopause") and flags.get("postmeno_bleeding") is False:
        out = [a for a in out if a["key"] != "postmeno_bleeding"]
    age = snap.get("age")
    if snap.get("cycle_irregular") == 1:
        extra = (" In your 40s, irregular periods are often an early sign of perimenopause."
                 if age and age >= 40 else "")
        out.append(_alert(
            "irregular", "gp", "Your periods are irregular",
            "You said your periods are irregular or often missed. The NHS advises speaking to a "
            f"GP if your periods are irregular.{extra}",
            NHS_LINKS["irregular"], "NHS: irregular periods; NICE PMOS guideline (draft, July 2026)",
            "Irregular periods often have harmless causes, but they are also the most common sign "
            "of PMOS, which is why NICE's draft PMOS guideline suggests checking them."))
        linked = [SYMPTOMS[k] for k in PMOS_SYMPTOMS if flags.get(k)]
        if linked:
            out.append(_alert(
                "pmos", "gp", "Irregular periods with other symptoms",
                f"Alongside irregular periods you mentioned: {_join(linked)}. The NHS advises "
                "seeing a GP about this combination. Possible causes include PMOS (formerly PCOS) "
                "and thyroid problems, so a GP can check.",
                NHS_LINKS["pmos"], "NHS: PMOS",
                "This is the combination our PMOS pattern check looks at; its accuracy is shown "
                "with the result."))
    for a in out:
        a["tests"] = tests_for(a["key"], age)
    order = {"urgent": 0, "gp": 1, "info": 2}
    return sorted(out, key=lambda a: order[a["level"]])


# ---------------------------------------------------------------------------
# Follow-up questions for the GP summary, chosen by what was flagged
# ---------------------------------------------------------------------------
FOLLOW_UPS = {
    "heavy": ["On your heaviest day, how many pads or tampons do you use?",
              "Have you missed work, school or plans because of bleeding?"],
    "long_period": ["How many days does your bleeding usually last now, and has that changed?"],
    "endometriosis": ["When in your cycle is the pain worst, and how long does it last?",
                      "Do painkillers such as ibuprofen or paracetamol help?"],
    "severe_pain": ["What have you tried for the pain, and how much did it help?"],
    "irregular": ["When did your periods start becoming irregular?"],
    "pmos": ["Does anyone in your close family have PMOS (PCOS) or type 2 diabetes?"],
    "missed": ["Is there any chance you could be pregnant, and have you done a test?"],
    "perimenopause": ["How often do hot flushes or night sweats happen, and do they wake you?",
                      "Are you using any contraception or HRT at the moment?"],
    "bleeding_between": ["How often has bleeding between periods happened, and when did it start?"],
    "bleeding_after_sex": ["How many times has bleeding after sex happened?"],
    "postmeno_bleeding": ["When did your periods stop, and are you taking HRT?"],
    "bloating": ["Have you also noticed changes in bowel habits, needing to pee more often, "
                 "feeling full quickly, or losing weight without trying?"],
    "impact": ["What have you had to miss or stop doing because of your symptoms recently?"],
    "severe_symptoms": ["Which symptom bothers you most, and when did it become severe?"],
    "severe_pelvic_pain": ["Where exactly is the pelvic pain, and is it constant or does it come "
                           "and go?"],
}


def follow_up_questions(alert_list: list[dict], limit: int = 5) -> list[str]:
    """Personal questions to answer before a GP appointment, most important warnings first."""
    order = {"urgent": 0, "gp": 1, "info": 2}
    qs: list[str] = []
    for a in sorted(alert_list, key=lambda a: order[a["level"]]):
        for q in FOLLOW_UPS.get(a["key"], []):
            if q not in qs:
                qs.append(q)
    return qs[:limit]


# ---------------------------------------------------------------------------
# "How reliable are these checks?" table shown in the app and the README
# ---------------------------------------------------------------------------
CHECKS = [
    ("PMOS pattern check", "Our machine learning model, trained on a public dataset (541 people)",
     "Tested by us (5-fold cross-validation)",
     "Flags about 86 in 100 people with PMOS; about 24 in 100 without it are flagged too"),
    ("Irregular, missed or long periods", "NHS guidance", "Clinical guidance",
     "NHS 'see a GP if' thresholds (21 to 35 days; 3 missed periods; over 7 days)"),
    ("Heavy periods and anaemia", "NHS; NICE NG88", "Published studies",
     "A product-counting chart picked up 58% to 98% of heavy bleeding across studies"),
    ("Endometriosis symptoms", "NICE NG73", "Published studies",
     "Symptoms alone: AUC about 0.68, so we flag rather than score"),
    ("Perimenopause", "NICE NG23 / QS143", "Clinical guidance",
     "Over-45s are diagnosed from symptoms alone; no reliable blood test"),
    ("Bleeding after the menopause", "NHS; NICE NG12", "Review of 129 studies",
     "About 9 in 100 have womb cancer; about 9 in 10 womb cancers cause it"),
    ("Persistent bloating", "NHS; NICE NG12", "Published studies",
     "A symptom checklist spotted 57% of early and 80% of advanced ovarian cancers"),
    ("Bleeding between periods or after sex", "NHS", "Clinical guidance",
     "Usually harmless, but the NHS says it should always be checked"),
    ("Cervical screening reminder", "NHS screening programme", "National programme",
     "Every 5 years from 25 to 64; more often if HPV is found"),
    ("Apple Watch cycle deviation alerts (imported)", "Apple Cycle Tracking", "Apple's own rules",
     "Irregular = 17+ days between shortest and longest cycle in 6 months; shown alongside, "
     "not instead of, the NHS rules"),
    ("Wrist temperature and resting heart rate (imported)", "Apple Watch sensors",
     "Context only", "No NHS or NICE thresholds exist, so we chart them but raise no alerts"),
]


# ---------------------------------------------------------------------------
# Screening-model inputs worked out from the logs (instead of asking questions)
# ---------------------------------------------------------------------------
def screening_answers(log: dict, today: date) -> dict | None:
    st = stats(log, today)
    recent_sym = _recent_symptoms(log, today)
    if len(st["periods"]) < 2 and not recent_sym:
        return None
    cycles = [c for _, c in st["cycles"][-3:]]
    irregular = None
    if cycles:
        irregular = int(any(not NORMAL_CYCLE[0] <= c <= NORMAL_CYCLE[1] for c in cycles))
    if st["days_since_last"] is not None and st["days_since_last"] > MISSED_DAYS:
        irregular = 1
    weight = latest_weight(log)
    flag = (lambda k: int(any(s.get(k) for s in recent_sym))) if recent_sym else (lambda k: None)
    return {
        "age": log["profile"].get("age"),
        "bmi": compute_bmi(log["profile"].get("height_cm"), weight),
        "cycle_irregular": irregular,
        "period_length": median(st["period_lengths"]) if st["period_lengths"] else None,
        "weight_gain": flag("weight_gain"),
        "hair_growth": flag("hair_growth"),
        "skin_darkening": flag("skin_darkening"),
        "hair_loss": flag("hair_loss"),
        "pimples": flag("acne"),
    }


def latest_weight(log: dict) -> float | None:
    """Most recent weight logged with symptoms, otherwise the weight saved in About me."""
    weights = [s["weight_kg"] for s in sorted(log["symptoms"], key=lambda s: _d(s["date"]))
               if s.get("weight_kg")]
    return weights[-1] if weights else log["profile"].get("weight_kg")


def missing_profile(log: dict) -> list[str]:
    """Details the PMOS pattern check needs from the user. We ask rather than guess."""
    prof = log["profile"]
    missing = [label for key, label in (("age", "age"), ("height_cm", "height"))
               if not prof.get(key)]
    if not latest_weight(log):
        missing.append("weight")
    return missing


def pattern_check_notes(log: dict, today: date) -> list[str]:
    """Extra caveats for the PMOS pattern check that depend on the rest of the log."""
    notes = []
    age = log["profile"].get("age")
    if log["profile"].get("menopause"):
        notes.append("You've said your periods have stopped, so this PMOS check doesn't apply "
                     "well to you.")
    elif age and age >= 40 and _logged(_recent_symptoms(log, today), MENO_SYMPTOMS):
        notes.append("Irregular periods in your 40s are often caused by perimenopause, which this "
                     "PMOS check can't tell apart from PMOS. Treat the score with extra caution.")
    return notes


# ---------------------------------------------------------------------------
# Summary to take to a GP appointment
# ---------------------------------------------------------------------------
GP_QUESTIONS = {
    "pmos": "Could these symptoms be linked to PMOS (formerly PCOS), a thyroid problem or "
            "something else? Would blood tests or an ultrasound help?",
    "irregular": "What could be causing my irregular periods?",
    "heavy": "Could I have a blood test (full blood count) to check for anaemia? What "
             "treatments could make my periods lighter?",
    "endometriosis": "Could my symptoms be caused by endometriosis? Could I have an "
                     "ultrasound scan, or be referred to a gynaecologist?",
    "perimenopause": "Could this be perimenopause? What are my treatment options, including HRT?",
    "postmeno_bleeding": "Do I need to be referred to a post-menopausal bleeding clinic?",
    "bloating": "Do I need any tests for my bloating?",
    "screening": "Can I book my cervical screening?",
}


TREATMENTS = ["Painkillers (e.g. ibuprofen, paracetamol)", "Hormonal contraception (pill, coil, "
              "implant)", "Tranexamic acid", "Iron tablets", "HRT", "Heat pads or exercise",
              "Diet or lifestyle changes", "Other"]


def summary_data(log: dict, today: date, alert_list: list[dict], result: dict | None,
                 extras: dict | None = None) -> dict:
    """Everything that goes into the GP summary (used by both the preview and the PDF).
    extras (optional, typed by the user): concern, medicines, tried [list], tried_notes,
    answers [(question, answer)]."""
    extras = extras or {}
    st = stats(log, today)
    prof = log["profile"]
    about = []
    if prof.get("age"):
        about.append(("Age", str(prof["age"])))
    if prof.get("height_cm"):
        about.append(("Height", f"{prof['height_cm']:.0f} cm"))
    weight = latest_weight(log)
    if weight:
        about.append(("Weight", f"{weight:.1f} kg"))
        bmi = compute_bmi(prof.get("height_cm"), weight)
        if bmi:
            about.append(("BMI", f"{bmi:.1f}"))
    if prof.get("menopause"):
        about.append(("Menopause", "Periods stopped 12 or more months ago"))
    if prof.get("last_screening") and prof["last_screening"] != SCREENING_OPTIONS[0]:
        about.append(("Last cervical screening", prof["last_screening"]))
    if log.get("wearable"):
        about.append(("Data source", "Some periods, symptoms and weights imported from "
                      f"{log['wearable'].get('source', 'a wearable')} on "
                      f"{log['wearable'].get('imported_on', 'an earlier date')}"))

    rated = [(s["impact"], _d(s["date"])) for s in _recent_symptoms(log, today, days=30)
             if s.get("impact") is not None]
    impact = max(rated) if rated else None

    flagged = [a for a in alert_list if a["level"] in ("urgent", "gp")]
    tests: list[str] = []
    for a in flagged:
        for t in a.get("tests") or tests_for(a["key"], prof.get("age")):
            if t not in tests:
                tests.append(t)

    periods = []
    for p in st["periods"]:
        length = (_d(p["end"]) - _d(p["start"])).days + 1 if p.get("end") else "ongoing"
        pain = p["pain"] if p.get("pain") is not None else "not recorded"
        periods.append((_d(p["start"]).strftime("%d %b %Y"), str(length), p["flow"].split(" (")[0],
                        str(pain), p.get("source") or "Logged by me"))
    cycles = [c for _, c in st["cycles"]]
    recent_sym = _recent_symptoms(log, today)
    counted = []
    for k, label in SYMPTOMS.items():
        worst = worst_severity(log, today, k)
        counted.append((label + (f" (worst: {SEVERITY_OPTIONS[worst].lower()})" if worst else ""),
                        sum(1 for s in recent_sym if s.get(k))))
    weights = [(s["date"], s["weight_kg"]) for s in log["symptoms"] if s.get("weight_kg")]
    keys = [a["key"] for a in flagged]
    questions = [GP_QUESTIONS[k] for k in GP_QUESTIONS if k in keys]
    questions.append("What can I do now to manage my symptoms?")
    return {
        "prepared": today.strftime("%d %B %Y"),
        "about": about,
        "concern": (extras.get("concern") or "").strip(),
        "impact": impact,
        "flagged": flagged,
        "tests": tests,
        "medicines": (extras.get("medicines") or "").strip(),
        "tried": list(extras.get("tried") or []),
        "tried_notes": (extras.get("tried_notes") or "").strip(),
        "answers": [(q, a.strip()) for q, a in (extras.get("answers") or []) if a and a.strip()],
        "periods": periods,
        "cycles": cycles,
        "typical_cycle": st["typical_cycle"],
        "days_since_last": st["days_since_last"],
        "symptoms": [(label, n) for label, n in counted if n],
        "weight_change": ((weights[0][1], _d(weights[0][0])), (weights[-1][1], _d(weights[-1][0])))
        if len(weights) >= 2 else None,
        "result": result,
        "questions": questions,
    }


def gp_summary(log: dict, today: date, alert_list: list[dict], result: dict | None,
               extras: dict | None = None) -> str:
    """Markdown version of the GP summary (shown as an on-screen preview)."""
    d = summary_data(log, today, alert_list, result, extras)
    lines = ["# My health summary for my GP",
             f"Prepared on {d['prepared']} from information I logged myself in HerHealth "
             "(a concept prototype, not an NHS service). This is not a diagnosis.", ""]
    if d["about"]:
        lines += ["## About me", *[f"- {k}: {v}" for k, v in d["about"]], ""]
    if d["concern"]:
        lines += ["## What I'd like help with", d["concern"], ""]
    if d["impact"]:
        value, day = d["impact"]
        lines += ["## How my symptoms affect me",
                  f"- Worst day in the last month: {value}/10 ({impact_label(value)}), "
                  f"{day.strftime('%d %b')}", ""]
    lines += ["## Things the app flagged (based on NHS and NICE guidance)"]
    lines += [f"- {'URGENT: ' if a['level'] == 'urgent' else ''}{a['title']} "
              f"(source: {a['source']})" for a in d["flagged"]] or ["- Nothing flagged"]
    if d["tests"]:
        lines += ["", "## Tests I'd like to ask about", *[f"- {t}" for t in d["tests"]]]
    if d["medicines"] or d["tried"] or d["tried_notes"]:
        lines += ["", "## Medicines and treatments"]
        if d["medicines"]:
            lines.append(f"- Taking now: {d['medicines']}")
        if d["tried"]:
            lines.append(f"- Tried: {', '.join(d['tried'])}")
        if d["tried_notes"]:
            lines.append(f"- What helped or didn't: {d['tried_notes']}")
    if d["answers"]:
        lines += ["", "## More about my symptoms", *[f"- **{q}** {a}" for q, a in d["answers"]]]
    lines += ["", "## Periods logged", "| Start | Length (days) | Flow | Pain (0-10) |",
              "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {e} |" for a, b, c, e, _ in d["periods"]]
    lines += ["", "## Cycle lengths"]
    if d["cycles"]:
        lines += [f"- Typical: {d['typical_cycle']:.0f} days (range {min(d['cycles'])} to "
                  f"{max(d['cycles'])})",
                  f"- All cycles: {', '.join(str(c) for c in d['cycles'])} days"]
    else:
        lines.append("- Not enough periods logged yet")
    if d["days_since_last"] is not None:
        lines.append(f"- Days since last period started: {d['days_since_last']}")
    lines += ["", "## Symptoms logged in the last 6 months"]
    lines += [f"- {label}: {n} time(s)" for label, n in d["symptoms"]] or ["- None logged"]
    if d["weight_change"]:
        (w0, d0), (w1, d1) = d["weight_change"]
        lines.append(f"- Weight: {w0} kg ({d0.strftime('%b %Y')}) to {w1} kg ({d1.strftime('%b %Y')})")
    if result:
        lines += ["", "## PMOS pattern check (screening estimate, not a diagnosis)",
                  f"- Screening score: {round(result['risk'] * 100)}/100. {result['band_text']}"]
    lines += ["", "## Questions I'd like to ask", *[f"- {q}" for q in d["questions"]]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Made-up demo profiles for presentations (clearly labelled in the app)
# ---------------------------------------------------------------------------
def _demo_periods(log, today, cycles, days_since_last, lengths, flows, pains):
    starts = [today - timedelta(days=days_since_last)]
    for c in reversed(cycles):
        starts.insert(0, starts[0] - timedelta(days=c))
    for s, n, f, pn in zip(starts, lengths, flows, pains):
        add_period(log, s, s + timedelta(days=n - 1), f, pn)
    return starts


def demo_sophie(today: date) -> dict:
    """Sophie, 23: cycles gradually becoming irregular, with skin and hair symptoms."""
    log = empty_log()
    log["profile"].update({"age": 23, "height_cm": 158, "last_screening": SCREENING_OPTIONS[0]})
    starts = _demo_periods(log, today, [31, 34, 42, 47, 38, 55], 49,
                           [5, 6, 5, 8, 6, 5, 6],
                           ["Medium", "Medium", "Heavy", VERY_HEAVY, "Heavy", "Medium", "Heavy"],
                           [4, 5, 6, 7, 5, 4, 6])
    weights = [61.0, 61.4, 62.3, 63.0, 64.1, 64.9, 65.8]
    for i, s in enumerate(starts):
        add_symptoms(log, s + timedelta(days=10), {
            "acne": i != 1, "hair_growth": i >= 2, "weight_gain": i >= 4,
            "tiredness": i in (3, 5, 6), "low_mood": i == 5}, weights[i])
    return log


def demo_emma(today: date) -> dict:
    """Emma, 31: regular cycles but painful, heavy periods."""
    log = empty_log()
    log["profile"].update({"age": 31, "height_cm": 167, "weight_kg": 63.5,
                           "last_screening": SCREENING_OPTIONS[1]})
    starts = _demo_periods(log, today, [28, 27, 29, 28, 30, 28], 12,
                           [6, 7, 6, 7, 7, 6, 7],
                           ["Heavy", "Heavy", VERY_HEAVY, "Heavy", VERY_HEAVY, "Heavy", VERY_HEAVY],
                           [6, 7, 6, 7, 7, 6, 7])
    for i, s in enumerate(starts):
        add_symptoms(log, s + timedelta(days=1), {
            "pain_stops_activities": i >= 2, "pain_poo": i in (3, 5, 6), "clots": i >= 3,
            "flooding": i in (4, 6), "tiredness": i >= 4, "breathless": i == 6})
        add_symptoms(log, s + timedelta(days=14), {"pain_sex": i in (2, 5), "pelvic_pain": i >= 4})
    return log


def demo_claire(today: date) -> dict:
    """Claire, 47: cycles changing, with hot flushes, night sweats and poor sleep."""
    log = empty_log()
    log["profile"].update({"age": 47, "height_cm": 164, "weight_kg": 70.2,
                           "last_screening": SCREENING_OPTIONS[2]})
    starts = _demo_periods(log, today, [27, 25, 33, 41, 23, 46], 30,
                           [5, 5, 4, 6, 3, 5, 4],
                           ["Medium", "Medium", "Light", "Heavy", "Light", "Medium", "Light"],
                           [3, 3, 2, 4, 2, 3, 2])
    for i, s in enumerate(starts):
        add_symptoms(log, s + timedelta(days=8), {
            "hot_flushes": i >= 3, "night_sweats": i >= 4, "poor_sleep": i >= 2,
            "brain_fog": i >= 5, "low_mood": i == 6})
    return log


DEMOS = {
    "Sophie, 23: cycles becoming irregular": demo_sophie,
    "Emma, 31: painful, heavy periods": demo_emma,
    "Claire, 47: hot flushes and changing cycles": demo_claire,
}


def demo_log(today: date) -> dict:
    """Default demo profile (kept for backwards compatibility)."""
    return demo_sophie(today)
