# HerHealth: spotting women's health warning signs earlier

> A **concept prototype** for a women's health feature in the **NHS App** ("My Health"), built at the **Leeds AI Society AI in Healthcare Hackathon** (24–25 Oct 2026, Helix, University of Leeds). **Not affiliated with or endorsed by the NHS.**

**🔗 Live demo:** https://herhealth-ai-week.streamlit.app/ (on the *My health* tab, pick a demo profile and click **Load demo profile**)

**Women's conditions are diagnosed late.**
- **PMOS (polyendocrine metabolic ovarian syndrome, formerly PCOS):** up to 70% of people with it are estimated to be undiagnosed (WHO estimate, as reported by ACP Internist, 2024). NICE's first UK guideline (draft, July 2026) suggests checking women with irregular periods.
- **Endometriosis:** diagnosis takes **8 years 10 months** on average in the UK (Endometriosis UK, 2024).

The warning signs (irregular cycles, heavy or painful periods, hot flushes, bleeding that should be checked, bloating that won't go away) are often dismissed or treated one at a time, so the pattern gets missed.

**HerHealth** is designed to fit the NHS 10 Year Health Plan's shifts from *analogue to digital* and *sickness to prevention*. Women log their periods and symptoms over time. The app:
- applies **NHS and NICE guidance automatically** across women's health: cycles, heavy periods and anaemia, endometriosis symptoms, perimenopause, bleeding after sex or after the menopause, persistent bloating, and cervical screening;
- shows, on every alert, **where the rule comes from and how reliable that sign is**, quoting published studies;
- runs a **machine-learning pattern check for PMOS** from the log itself, and shows its **measured accuracy in plain numbers**;
- produces a **summary to take to a GP appointment**.

> ⚠️ This is a screening and education tool, **not a diagnosis**.

---

## What it does

| Tab | What it does |
|---|---|
| **My health** (main) | Log periods (dates, flow, pain) and symptoms (pain, bleeding, skin and hair, energy and mood, menopause-type symptoms, tummy). Shows a cycle-length chart, **NHS-style care cards** (blue = see a GP, red = urgent) with *How reliable is this sign?* and a link to the nhs.uk page, a **PMOS pattern check** with its accuracy, and a **downloadable GP summary** with tailored questions. **Download / delete my data**, plus three made-up **demo profiles** (Sophie, 23; Emma, 31; Claire, 47) |
| **Ask about my symptoms** | Chat in plain English. An LLM turns free text into structured answers and asks only the missing questions. Simple replies like yes, no or numbers are parsed instantly without the LLM |
| **Quick check** | The same PMOS model through a short form, with no AI at all |

**Safety:** red-flag phrases (severe pain, very heavy bleeding, fainting, bleeding in pregnancy, bleeding after the menopause, chest pain, self-harm) trigger urgent-care advice immediately.

<!-- TODO: add 2-3 screenshots here (My health with alerts, pattern check accuracy, chat) -->

### Why only PMOS gets a machine-learning model
We only train a model where there is **open data with self-reported symptoms and confirmed diagnoses**. For PMOS there is (Kaggle, 541 patients). For the other conditions there isn't, and published studies show symptoms alone can't detect them reliably (for example, endometriosis symptom questionnaires reach an AUC of only about 0.68). So for those we use **transparent NHS/NICE rules** and quote the published evidence, rather than inventing an accuracy figure.

### How reliable are these checks?
| Check | Based on | How it's backed | What the evidence says |
|---|---|---|---|
| PMOS pattern check | Our machine learning model, trained on a public dataset (541 people) | Tested by us (5-fold cross-validation) | Flags about 86 in 100 people with PMOS; about 24 in 100 without it are flagged too |
| Irregular, missed or long periods | NHS guidance | Clinical guidance | NHS 'see a GP if' thresholds (21 to 35 days; 3 missed periods; over 7 days) |
| Heavy periods and anaemia | NHS; NICE NG88 | Published studies | A product-counting chart picked up 58% to 98% of heavy bleeding across studies |
| Endometriosis symptoms | NICE NG73 | Published studies | Symptoms alone: AUC about 0.68, so we flag rather than score |
| Perimenopause | NICE NG23 / QS143 | Clinical guidance | Over-45s are diagnosed from symptoms alone; no reliable blood test |
| Bleeding after the menopause | NHS; NICE NG12 | Review of 129 studies | About 9 in 100 have womb cancer; about 9 in 10 womb cancers cause it |
| Persistent bloating | NHS; NICE NG12 | Published studies | A symptom checklist spotted 57% of early and 80% of advanced ovarian cancers |
| Bleeding between periods or after sex | NHS | Clinical guidance | Usually harmless, but the NHS says it should always be checked |
| Cervical screening reminder | NHS screening programme | National programme | Every 5 years from 25 to 64; more often if HPV is found |

### Alert rules
| Rule | Level | Source |
|---|---|---|
| A cycle shorter than 21 or longer than 35 days | See a GP | [NHS: irregular periods](https://www.nhs.uk/symptoms/irregular-periods/) |
| No period for 90+ days (about 3 missed periods) | See a GP | [NHS: missed or late periods](https://www.nhs.uk/symptoms/missed-or-late-periods/) |
| A period lasting longer than 7 days | See a GP | NHS: irregular periods |
| Changing a pad or tampon every 1–2 hours, clots bigger than a 10p, flooding, or two products at once (plus anaemia advice if tired or breathless) | See a GP | [NHS: heavy periods](https://www.nhs.uk/conditions/heavy-periods/), [NICE NG88](https://www.nice.org.uk/guidance/ng88) |
| Severe period pain (8/10 or more) | Urgent | [NHS: period pain](https://www.nhs.uk/symptoms/period-pain/) |
| Pain that stops normal activities, pelvic pain, pain during sex, pain pooing or peeing during a period, or 7/10+ pain in 2 periods | See a GP | [NICE NG73](https://www.nice.org.uk/guidance/ng73), [NHS: endometriosis](https://www.nhs.uk/conditions/endometriosis/) |
| Bleeding between periods or after sex | See a GP | [NHS: bleeding between periods or after sex](https://www.nhs.uk/symptoms/vaginal-bleeding-between-periods-or-after-sex/) |
| Any bleeding after the menopause | Urgent | [NHS: post-menopausal bleeding](https://www.nhs.uk/symptoms/post-menopausal-bleeding/), NICE NG12 |
| Bloating on 12 or more days in a month | See a GP | [NHS: bloating](https://www.nhs.uk/symptoms/bloating/), NICE NG12 |
| Hot flushes or night sweats, or 2+ menopause-type symptoms (with different advice under 45) | See a GP | [NHS: menopause symptoms](https://www.nhs.uk/conditions/menopause-and-perimenopause/symptoms/), NICE NG23 |
| Aged 25–64 and last cervical screening over 5 years ago | See a GP | [NHS: cervical screening](https://www.nhs.uk/tests-and-treatments/cervical-screening/) |
| Irregular periods together with hair growth, acne, weight gain, tiredness or dark skin patches | See a GP | [NHS: PMOS](https://www.nhs.uk/conditions/polyendocrine-metabolic-ovarian-syndrome-pmos/) |

### How we show the PMOS model's accuracy
- **No single "accuracy" figure.** It is misleading with imbalanced data: a model that says "no" to everyone would score 67% on our data.
- **Plain numbers with 95% ranges:** of 100 people with PMOS it flags about 86 (80% to 90%); of 100 without, about 24 are flagged too (20% to 29%). Shown as a 100-square picture.
- **What a flag means in real life:** 33% of the training data had PMOS, but the general population figure is up to 13% (WHO). At that rate only about **3 in 10** people flagged would actually have PMOS, so a flag means "worth checking with a GP".
- **Extra caveats** appear when the score is less reliable: ages outside 20–48, or irregular periods in someone over 40 with menopause-type symptoms (perimenopause can look like PMOS).

## How it works

```
User text ──► LLM extracts structured answers (JSON schema, strict) ──► follow-up questions
                                   │
                                   ▼
             scikit-learn model trained on a public dataset ──► screening score
                                   │      + per-person factors (what moved the score)
                                   ▼
             LLM explains in plain English + GP questions ──► guardrails ──► user
```

**Key design decision: the LLM never makes the prediction.** LLM outputs can't be validated and can be confidently wrong. A small, transparent ML model does the prediction and is evaluated with cross-validation. The LLM is only the *interface*: it understands free text and explains the result kindly.

### Two models
| Model | Inputs | Purpose |
|---|---|---|
| **Screening** (used in the app; *My health* fills its inputs from the log) | 11 self-reportable answers: age, BMI, cycle regularity, period length, weight gain, excess hair growth, skin darkening, hair loss, acne, fast food, exercise | Works for anyone, with no tests |
| **Clinical** (comparison) | Screening inputs + hormone blood tests + ultrasound follicle counts | Shows how much the lab tests add |

The self-reported inputs map onto two of the three internationally used (Rotterdam) criteria for PMOS (formerly PCOS): irregular cycles and signs of excess androgens.

### Evaluation
- 5-fold **stratified cross-validation** (small dataset, so no single lucky split)
- Logistic regression vs random forest, with the best chosen by **ROC-AUC**
- The decision threshold is tuned for **high sensitivity (≥ 85%)**, because a screening tool should rarely miss a case
- Reported: sensitivity, specificity, PPV, NPV, plus a **bias check by age group and BMI band**
- Full results: [`reports/model_card.md`](reports/model_card.md) and [`reports/figures/`](reports/figures)

### Results (541 patients, 177 diagnosed with PMOS/PCOS; 5-fold cross-validation)

| Model | Inputs | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| **Screening** (in the app) | 11 self-reported answers, **no tests** | **0.90** | **86%** | 76% |
| Clinical (comparison) | + hormone blood tests + ultrasound | 0.96 | 86% | 94% |

**Key finding: answers a woman already knows recover most of the signal** (AUC 0.90 vs 0.96 with blood tests and scans). The strongest self-reported signals were darkened skin patches, excess hair growth, unexplained weight gain and irregular periods.

![ROC curves](reports/figures/roc_curves.png)

## Responsible use of data

See **[`RESPONSIBLE_DATA.md`](RESPONSIBLE_DATA.md)**. In short:
- **Open public dataset only** (Kaggle, CC BY-NC-SA 4.0). We collected no patient data, and our pipeline drops the only identifier-like column (hospital file numbers).
- **Nothing users enter is stored.** Obvious identifiers are scrubbed before any text is sent to the LLM API, and the form mode uses no external API at all.
- **Bias and limitations are measured and stated:** the data comes from one region, it's a small sample, and the scores aren't calibrated probabilities.

## Run it locally

```bash
git clone <this repo> && cd herhealth-ai
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Get the data (see data/README.md), then train
python -m src.train --data data/raw/PCOS_data_without_infertility.xlsx

# 2. (Optional) enable the AI chat
cp .env.example .env        # then paste your LLM API key (e.g. Groq) into .env (never commit it)

# 3. Run the app
streamlit run app.py
```

No data yet? `python scripts/make_synthetic_data.py`, then train on `data/synthetic/pcos_synthetic.csv`. The app will clearly show **"Demo mode: synthetic data"**.

Tests: `pytest -q`

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub, **public**, *including* `models/pcos_screening.joblib`.
2. On share.streamlit.io, choose **New app**, pick the repo, and set the main file to `app.py`.
3. In **Advanced settings → Secrets**, add your LLM provider. Any OpenAI-compatible API works. For example, with Groq's free tier:
   ```toml
   OPENAI_API_KEY = "your-groq-key"
   OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
   OPENAI_MODEL = "openai/gpt-oss-120b"
   ```
   See `.env.example` for Gemini and OpenAI settings.

## Repo structure
```
app.py                  Streamlit app: My health tracker, chat, quick check, accuracy display (NHS-style UI)
src/tracker.py          health log, NHS/NICE alert rules with evidence, reliability table, pattern-check inputs, GP summary, demo profiles
src/config.py           feature definitions and questions
src/data.py             loading/cleaning the Kaggle file
src/train.py            cross-validated training, threshold, bias check, model card, figures
src/predict.py          scoring + per-person explanations
src/llm.py              LLM extraction (strict JSON schema, any OpenAI-compatible API) and explanations, with offline fallback
src/guardrails.py       red flags, PII scrubbing, no-diagnosis wording, disclaimer
scripts/make_synthetic_data.py   synthetic test data (same columns as the real file)
tests/test_pipeline.py  end-to-end tests
reports/                metrics.json, model_card.md, figures
```

## Tech
Python · pandas · scikit-learn · Streamlit · Groq via the OpenAI-compatible API (structured outputs) · Altair · matplotlib · pytest

## Limitations and next steps
- The training data is small and from a single region, so the next step is validating on more diverse data.
- Scores show similarity to diagnosed cases. They aren't calibrated probabilities of having PMOS.
- Next: co-design with patients and GPs, add other under-diagnosed conditions (e.g. endometriosis, where UK diagnosis takes **8 years 10 months** on average, per Endometriosis UK, 2024), and multilingual support.

## Team
<!-- TODO: list every teammate (name + GitHub) and add them as repo collaborators -->
- Salma Bushra Islam
- Hareem Ghouri
- Ching Kiu Chan
- Jinesh Dugar

## Disclosure of AI tools
Per the hackathon rules: parts of this code and documentation were drafted with the help of AI assistants (Anthropic's Claude), then reviewed, run and tested by the team. The app itself calls an LLM at runtime (Groq's `openai/gpt-oss-120b`, through the OpenAI-compatible API) for the chat features only.

## Data credit and licence
Prasoon Kottarathil, *Polycystic ovary syndrome (PCOS)*, Kaggle (2020):
https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos.
It was collected from 10 hospitals across Kerala, India.

The data is licensed **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**, and the original file is included unmodified in `data/raw/`. The trained models are derived from it, so they are for **non-commercial use only** and are shared under the same licence. See [`data/raw/ATTRIBUTION.md`](data/raw/ATTRIBUTION.md).
