"""Import Apple Health / Apple Watch data from the Health app's "Export All Health Data" file.

Why a file? Apple Health has no web API: Apple Watch data lives in the Health app on the
user's iPhone, and only a native iPhone app (HealthKit) or the user's own export can get
it out. So the user exports `export.zip` and uploads it here.

Data minimisation:
  - we stream through the file and read ONLY the record types listed below (periods,
    spotting, cycle symptoms, Apple's cycle-deviation alerts, weight, height, sleeping wrist
    temperature, resting heart rate) from the last 12 months;
  - everything else (workouts, routes, heart rhythm, medical records, ...) is skipped
    without being kept;
  - date of birth is only used to work out age, then thrown away;
  - nothing is stored on a server or sent to an AI.

Record names follow Apple's HealthKit identifiers (developer.apple.com/documentation/healthkit).
"""

from __future__ import annotations

import io
import random
import zipfile
from datetime import date, datetime, timedelta
from statistics import median

try:  # safer XML parsing for files uploaded by users
    from defusedxml.ElementTree import iterparse
except ImportError:  # pragma: no cover - fallback if defusedxml isn't installed
    from xml.etree.ElementTree import iterparse

from . import tracker

IMPORT_DAYS = 365
SOURCE = "Apple Health"

CAT = "HKCategoryTypeIdentifier"
QTY = "HKQuantityTypeIdentifier"

FLOW_TYPE = CAT + "MenstrualFlow"
FLOW_VALUES = {                       # Apple's flow levels stop at "heavy"
    "HKCategoryValueMenstrualFlowLight": "Light",
    "HKCategoryValueMenstrualFlowMedium": "Medium",
    "HKCategoryValueMenstrualFlowHeavy": "Heavy",
    "HKCategoryValueMenstrualFlowUnspecified": "Medium",
}
FLOW_ORDER = {"Light": 0, "Medium": 1, "Heavy": 2}

# Apple Health symptom -> HerHealth symptom
SYMPTOM_MAP = {
    CAT + "IntermenstrualBleeding": "bleeding_between",
    CAT + "BleedingAfterMenopause": "postmeno_bleeding",
    CAT + "Acne": "acne",
    CAT + "HairLoss": "hair_loss",
    CAT + "Fatigue": "tiredness",
    CAT + "ShortnessOfBreath": "breathless",
    CAT + "RapidPoundingOrFlutteringHeartbeat": "breathless",
    CAT + "MoodChanges": "low_mood",
    CAT + "HotFlashes": "hot_flushes",
    CAT + "NightSweats": "night_sweats",
    CAT + "SleepChanges": "poor_sleep",
    CAT + "MemoryLapse": "brain_fog",
    CAT + "Bloating": "bloating",
    CAT + "PelvicPain": "pelvic_pain",       # only used on days outside a period
}
# Read so we can report them, but not mapped (no exact HerHealth equivalent)
UNMAPPED = {
    CAT + "AbdominalCramps": "Abdominal cramps",
    CAT + "LowerBackPain": "Lower back pain",
    CAT + "Headache": "Headache",
    CAT + "BreastPain": "Breast pain",
    CAT + "AppetiteChanges": "Appetite changes",
    CAT + "VaginalDryness": "Vaginal dryness",
}
DEVIATIONS = {
    CAT + "IrregularMenstrualCycles": (
        "irregular", "Apple Watch: irregular cycles",
        "at least a 17-day difference between your shortest and longest cycles in the last "
        "6 months"),
    CAT + "InfrequentMenstrualCycles": (
        "infrequent", "Apple Watch: infrequent periods",
        "only one or two periods in the last 6 months"),
    CAT + "ProlongedMenstrualPeriods": (
        "prolonged", "Apple Watch: prolonged periods",
        "bleeding for 10 or more days, at least twice in the last 6 months"),
    CAT + "PersistentIntermenstrualBleeding": (
        "spotting", "Apple Watch: persistent spotting",
        "spotting in at least two of your cycles in the last 6 months"),
}
WEIGHT, HEIGHT = QTY + "BodyMass", QTY + "Height"
WRIST_TEMP, RESTING_HR = QTY + "AppleSleepingWristTemperature", QTY + "RestingHeartRate"
WANTED = ({FLOW_TYPE, WEIGHT, HEIGHT, WRIST_TEMP, RESTING_HR}
          | set(SYMPTOM_MAP) | set(UNMAPPED) | set(DEVIATIONS))
NOT_PRESENT = {"HKCategoryValueSeverityNotPresent", "HKCategoryValuePresenceNotPresent"}
SEVERITY_VALUES = {"HKCategoryValueSeverityMild": 1, "HKCategoryValueSeverityModerate": 2,
                   "HKCategoryValueSeveritySevere": 3}
TOP_LEVEL = {"Record", "Workout", "ActivitySummary", "ClinicalRecord", "Correlation",
             "Audiogram", "VisionPrescription", "Me", "ExportDate"}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _when(text: str) -> date:
    """Apple dates look like '2026-09-14 08:12:33 +0100'; keep the person's local date."""
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S %z").date()
    except (TypeError, ValueError):
        return date.fromisoformat(str(text)[:10])


def _kg(value: float, unit: str) -> float | None:
    factor = {"kg": 1.0, "lb": 0.45359237, "st": 6.35029318, "g": 0.001}.get(unit)
    return round(value * factor, 1) if factor else None


def _cm(value: float, unit: str) -> float | None:
    factor = {"cm": 1.0, "m": 100.0, "in": 2.54, "ft": 30.48, "mm": 0.1}.get(unit)
    return round(value * factor, 1) if factor else None


def _open_xml(data: bytes):
    """Accept the export.zip from the Health app, or the export.xml inside it."""
    if data[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(data))
        names = [n for n in z.namelist() if n.endswith("export.xml")]
        if not names:
            raise ValueError("This zip doesn't contain an export.xml file. Is it the file "
                             "from Health > Export All Health Data?")
        return z.open(sorted(names, key=len)[0])
    return io.BytesIO(data)


def parse_export(data: bytes, today: date) -> dict:
    cutoff = today - timedelta(days=IMPORT_DAYS)
    flow_days: dict[date, str] = {}
    cycle_starts: set[date] = set()
    symptoms: dict[date, set[str]] = {}
    severity: dict[date, dict[str, int]] = {}
    pelvic: set[date] = set()
    unmapped: dict[str, int] = {}
    deviations, weights, temps, rhr = [], {}, {}, {}
    heights: list[tuple[date, float]] = []
    dob = None
    used = ignored = 0

    fh = _open_xml(data)
    root = None
    try:
        for event, el in iterparse(fh, events=("start", "end")):
            if event == "start":
                if root is None:
                    root = el
                continue
            tag = el.tag
            if tag == "Me":
                dob = el.get("HKCharacteristicTypeIdentifierDateOfBirth") or None
            elif tag == "Record":
                rtype = el.get("type")
                if rtype not in WANTED:
                    ignored += 1
                else:
                    day = _when(el.get("startDate"))
                    if day < cutoff or day > today:
                        ignored += 1
                    else:
                        used += 1
                        value = el.get("value")
                        if rtype == FLOW_TYPE:
                            flow = FLOW_VALUES.get(value)
                            if flow:
                                if FLOW_ORDER[flow] >= FLOW_ORDER.get(flow_days.get(day), -1):
                                    flow_days[day] = flow
                                for m in el.iter("MetadataEntry"):
                                    if m.get("key") == "HKMenstrualCycleStart" and m.get("value") == "1":
                                        cycle_starts.add(day)
                        elif rtype in SYMPTOM_MAP:
                            if value not in NOT_PRESENT:
                                key = SYMPTOM_MAP[rtype]
                                level = SEVERITY_VALUES.get(value)
                                if level:
                                    old = severity.setdefault(day, {}).get(key, 0)
                                    severity[day][key] = max(old, level)
                                if key == "pelvic_pain":
                                    pelvic.add(day)
                                else:
                                    symptoms.setdefault(day, set()).add(key)
                        elif rtype in UNMAPPED:
                            if value not in NOT_PRESENT:
                                unmapped[UNMAPPED[rtype]] = unmapped.get(UNMAPPED[rtype], 0) + 1
                        elif rtype in DEVIATIONS:
                            deviations.append({"kind": DEVIATIONS[rtype][0], "date": day.isoformat()})
                        else:
                            try:
                                num = float(value)
                            except (TypeError, ValueError):
                                num = None
                            unit = el.get("unit", "")
                            if num is not None:
                                if rtype == WEIGHT and _kg(num, unit):
                                    weights[day] = _kg(num, unit)
                                elif rtype == HEIGHT and _cm(num, unit):
                                    heights.append((day, _cm(num, unit)))
                                elif rtype == WRIST_TEMP:
                                    temps[day] = num if unit != "degF" else (num - 32) * 5 / 9
                                elif rtype == RESTING_HR:
                                    rhr.setdefault(day, []).append(num)
            if tag in TOP_LEVEL and root is not None:
                root.clear()                      # drop everything already read: keeps memory low
    except Exception as exc:  # malformed / not an Apple Health file
        raise ValueError("That file couldn't be read as an Apple Health export.") from exc

    periods = _group_periods(flow_days, cycle_starts)
    in_period = {p["start"] + timedelta(days=i)
                 for p in periods for i in range((p["end"] - p["start"]).days + 1)}
    for day in pelvic - in_period:                 # our symptom is "outside periods"
        symptoms.setdefault(day, set()).add("pelvic_pain")
    for day in pelvic & in_period:                 # drop severity for in-period pelvic pain
        severity.get(day, {}).pop("pelvic_pain", None)

    age = None
    if dob:
        try:
            b = date.fromisoformat(dob[:10])
            age = today.year - b.year - ((today.month, today.day) < (b.month, b.day))
        except ValueError:
            age = None

    return {
        "periods": periods,
        "symptoms": symptoms,
        "severity": severity,
        "weights": weights,
        "height_cm": sorted(heights)[-1][1] if heights else None,
        "age": age,                                # date of birth itself is not kept
        "deviations": deviations,
        "wrist_temp": sorted((d.isoformat(), round(v, 2)) for d, v in temps.items()),
        "resting_hr": sorted((d.isoformat(), round(median(v), 1)) for d, v in rhr.items()),
        "unmapped": unmapped,
        "records_used": used,
        "records_ignored": ignored,
    }


def _group_periods(flow_days: dict[date, str], cycle_starts: set[date]) -> list[dict]:
    """Turn daily flow records into periods (a gap of more than 2 days starts a new one)."""
    periods: list[dict] = []
    for day in sorted(flow_days):
        last = periods[-1] if periods else None
        if last and ((day - last["end"]).days <= 2 and not
                     (day in cycle_starts and (day - last["end"]).days > 1)):
            last["end"] = day
            if FLOW_ORDER[flow_days[day]] > FLOW_ORDER[last["flow"]]:
                last["flow"] = flow_days[day]
        else:
            periods.append({"start": day, "end": day, "flow": flow_days[day]})
    return periods


# ---------------------------------------------------------------------------
# Merging into the HerHealth log (never overwrites what the user logged by hand)
# ---------------------------------------------------------------------------
def merge_into_log(log: dict, parsed: dict, today: date) -> dict:
    added_periods = 0
    existing_starts = [tracker._d(p["start"]) for p in log["periods"]]
    for p in parsed["periods"]:
        if (any(abs((p["start"] - s).days) <= 2 for s in existing_starts)
                or tracker.overlapping(log, p["start"], p["end"], today)):
            continue                               # the user already logged this period
        tracker.add_period(log, p["start"], p["end"], p["flow"], None, source=SOURCE)
        added_periods += 1

    by_day = {tracker._d(s["date"]): s for s in log["symptoms"]}
    days = set(parsed["symptoms"]) | set(parsed["weights"])
    for day in days:
        old = by_day.get(day, {})
        flags = {k: bool(old.get(k)) or k in parsed["symptoms"].get(day, set())
                 for k in tracker.SYMPTOMS}
        weight = old.get("weight_kg") or parsed["weights"].get(day)
        sev = dict(old.get("severity") or {})
        for k, v in parsed.get("severity", {}).get(day, {}).items():
            sev[k] = max(sev.get(k, 0), v)
        tracker.add_symptoms(log, day, flags, weight, old.get("impact"), sev)

    prof = log["profile"]
    filled = []
    if prof.get("age") is None and parsed["age"] is not None:
        prof["age"] = parsed["age"]
        filled.append("age")
    if prof.get("height_cm") is None and parsed["height_cm"]:
        prof["height_cm"] = parsed["height_cm"]
        filled.append("height")

    log["wearable"] = {
        "source": SOURCE,
        "imported_on": today.isoformat(),
        "deviations": parsed["deviations"],
        "wrist_temp": parsed["wrist_temp"],
        "resting_hr": parsed["resting_hr"],
        "unmapped": parsed["unmapped"],
    }
    return {
        "periods": added_periods,
        "periods_skipped": len(parsed["periods"]) - added_periods,
        "symptom_days": len(parsed["symptoms"]),
        "weights": len(parsed["weights"]),
        "filled": filled,
        "deviations": len(parsed["deviations"]),
        "records_used": parsed["records_used"],
        "records_ignored": parsed["records_ignored"],
        "unmapped": parsed["unmapped"],
    }


def deviation_cards(log: dict, today: date) -> list[dict]:
    """Apple's own cycle-deviation alerts, shown alongside (not instead of) the NHS rules."""
    out, seen = [], set()
    for dev in sorted(log.get("wearable", {}).get("deviations", []), key=lambda d: d["date"],
                      reverse=True):
        day = tracker._d(dev["date"])
        if dev["kind"] in seen or (today - day).days > tracker.RECENT_DAYS:
            continue
        seen.add(dev["kind"])
        _, title, rule = next(v for v in DEVIATIONS.values() if v[0] == dev["kind"])
        out.append({
            "key": "apple_" + dev["kind"], "level": "info", "title": title,
            "detail": (f"Apple's Cycle Tracking flagged this on {day.strftime('%d %b %Y')}, after "
                       f"you confirmed your cycle history. Apple's rule: {rule}. HerHealth also "
                       "checks your imported periods against NHS guidance in the cards above."),
            "link": "https://support.apple.com/120356",
            "source": "Apple HealthKit cycle deviation detection",
            "evidence": ("This is Apple's own rule, worked out on your iPhone. Its thresholds "
                         "differ from the NHS ones (for example, the NHS calls cycles irregular "
                         "if they're shorter than 21 or longer than 35 days). We haven't tested "
                         "Apple's rule ourselves."),
        })
    return out


# ---------------------------------------------------------------------------
# Made-up sample export for demos ("Olivia", 26 — not a real person)
# ---------------------------------------------------------------------------
_DEVICE = ("&lt;&lt;HKDevice: 0x0&gt;, name:Apple Watch, manufacturer:Apple Inc., "
           "model:Watch, hardware:Watch7,1, software:12.0&gt;")


def _rec(rtype, day, value, unit=None, hour=8, meta=None, source="Olivia's Apple Watch"):
    ts = f"{day.isoformat()} {hour:02d}:00:00 +0100"
    attrs = (f'type="{rtype}" sourceName="{source}" sourceVersion="12.0" device="{_DEVICE}" '
             + (f'unit="{unit}" ' if unit else "")
             + f'creationDate="{ts}" startDate="{ts}" endDate="{ts}" value="{value}"')
    if meta:
        inner = "".join(f'<MetadataEntry key="{k}" value="{v}"/>' for k, v in meta.items())
        return f"  <Record {attrs}>{inner}</Record>\n"
    return f"  <Record {attrs}/>\n"


def make_sample_export(today: date) -> bytes:
    """A small, clearly made-up export.zip in Apple's format, with dates relative to today."""
    rnd = random.Random(26)
    dob = date(today.year - 26, 3, 14) if (today.month, today.day) >= (3, 14) else \
        date(today.year - 27, 3, 14)
    rows = []
    cycles = [33, 41, 36, 52, 38, 47]
    starts = [today - timedelta(days=20)]
    for c in reversed(cycles):
        starts.insert(0, starts[0] - timedelta(days=c))
    pattern = ["Medium", "Heavy", "Heavy", "Medium", "Light", "Light"]
    for i, s in enumerate(starts):
        for d in range(5 + (i % 2)):
            flow = "HKCategoryValueMenstrualFlow" + pattern[min(d, len(pattern) - 1)]
            rows.append(_rec(FLOW_TYPE, s + timedelta(days=d), flow,
                             meta={"HKMenstrualCycleStart": 1 if d == 0 else 0,
                                   "HKWasUserEntered": 1}))
        rows.append(_rec(CAT + "AbdominalCramps", s, "HKCategoryValueSeverityModerate"))
        rows.append(_rec(CAT + "Acne", s + timedelta(days=9), "HKCategoryValueSeverityModerate"))
        if i >= 3:
            rows.append(_rec(CAT + "Fatigue", s + timedelta(days=12), "HKCategoryValueSeverityMild"))
            rows.append(_rec(CAT + "HairLoss", s + timedelta(days=15), "HKCategoryValueSeverityMild"))
        if i in (4, 5):
            rows.append(_rec(CAT + "IntermenstrualBleeding", s + timedelta(days=17),
                             "HKCategoryValueNotApplicable"))
        if i == 5:
            rows.append(_rec(CAT + "MoodChanges", s + timedelta(days=20),
                             "HKCategoryValuePresencePresent"))
    for k in range(4):
        rows.append(_rec(CAT + "Bloating", today - timedelta(days=3 + k * 6),
                         "HKCategoryValueSeverityMild"))
    for kind in ("IrregularMenstrualCycles", "PersistentIntermenstrualBleeding"):
        rows.append(_rec(CAT + kind, today - timedelta(days=24), "HKCategoryValueNotApplicable",
                         source="Health"))
    rows.append(_rec(HEIGHT, today - timedelta(days=300), "165", unit="cm", source="Health"))
    for m in range(7):
        rows.append(_rec(WEIGHT, today - timedelta(days=30 * (6 - m) + 2),
                         f"{68.0 + m * 0.55:.1f}", unit="kg", source="Health"))
    ovulations = [s + timedelta(days=max(c - 14, 12)) for s, c in zip(starts, cycles + [33])]
    for back in range(180, 0, -1):
        day = today - timedelta(days=back)
        luteal = any(o <= day < o + timedelta(days=13) for o in ovulations)
        temp = 35.55 + (0.3 if luteal else 0.0) + rnd.gauss(0, 0.08)
        rows.append(_rec(WRIST_TEMP, day, f"{temp:.2f}", unit="degC", hour=4))
        rows.append(_rec(RESTING_HR, day, f"{61 + rnd.gauss(0, 2.0):.0f}", unit="count/min"))
        rows.append(_rec(QTY + "StepCount", day, f"{rnd.randint(4000, 12000)}", unit="count"))
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!DOCTYPE HealthData [\n"
        "<!ELEMENT HealthData (ExportDate,Me,Record*)>\n"
        "<!ATTLIST HealthData locale CDATA #REQUIRED>\n"
        "<!ELEMENT ExportDate EMPTY>\n<!ATTLIST ExportDate value CDATA #REQUIRED>\n"
        "<!ELEMENT Me EMPTY>\n"
        "<!ELEMENT Record (MetadataEntry*)>\n"
        "<!ELEMENT MetadataEntry EMPTY>\n"
        "]>\n"
        "<!-- MADE-UP SAMPLE for the HerHealth demo. 'Olivia' is not a real person. -->\n"
        '<HealthData locale="en_GB">\n'
        f' <ExportDate value="{today.isoformat()} 09:00:00 +0100"/>\n'
        f' <Me HKCharacteristicTypeIdentifierDateOfBirth="{dob.isoformat()}" '
        'HKCharacteristicTypeIdentifierBiologicalSex="HKBiologicalSexFemale" '
        'HKCharacteristicTypeIdentifierBloodType="HKBloodTypeNotSet" '
        'HKCharacteristicTypeIdentifierFitzpatrickSkinType="HKFitzpatrickSkinTypeNotSet"/>\n'
        + "".join(rows) + "</HealthData>\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("apple_health_export/export.xml", xml)
    return buf.getvalue()
