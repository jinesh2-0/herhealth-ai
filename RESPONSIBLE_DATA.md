# Responsible use of data

This is written for the hackathon's **Most Responsible Use of Data** award, and to answer the data question on Devpost.

## 1. Is the data appropriate?
- **Source:** the public, anonymised "Polycystic ovary syndrome (PCOS)" dataset on Kaggle (541 patients; clinical, symptom, hormone and ultrasound variables). Per the dataset description, it was collected from hospitals in Kerala, India.
- **No real patient data was collected or used by us**, in line with the hackathon rules. For testing we also generate clearly labelled **synthetic** data (`scripts/make_synthetic_data.py`). Synthetic results are never reported as findings, and the app shows a "demo mode" banner whenever a synthetic model is loaded.
- **Why this dataset:** it is one of the few open PCOS datasets that includes both *self-reportable symptoms* and *clinical measurements*. That lets us test our main idea honestly: how well can screening work with **only** information a person already knows?
- **Licence:** **CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike). We credit the author, use the data only for this non-commercial hackathon, include the original file unmodified, and share the derived models under the same terms (`data/raw/ATTRIBUTION.md`). Any commercial use would need a different data source or the author's permission.
- **Provenance and consent gap:** the dataset page says the data came from 10 hospitals in Kerala, India, but it doesn't document how patients consented or how the data was anonymised. We flag this openly. Before any real-world use, we would only train on data with documented consent and governance.

## 2. Privacy
- **In the training data:** the file contains no names, but it does include hospital "Patient File No." values. Our pipeline drops these and every other unused column, so the models only ever see the 27 clinical and symptom variables.
- **Nothing a user enters is stored.** There are no accounts and no database, and answers live only in the browser session.
- **Data minimisation:** the screening model needs only 11 answers. We never ask for names, dates of birth, contact details or NHS numbers.
- **PII scrubbing:** before any chat text is sent to the OpenAI API, `src/guardrails.py` removes emails, phone numbers, NHS-style numbers and UK postcodes. Users are also told not to share identifying details.
- **Choice of LLM provider:** the app works with any OpenAI-compatible API (OpenAI, Groq, Gemini). Free tiers can have different data terms. For example, Google's terms for its unpaid Gemini API tier allow it to use submitted content to improve its products. That's another reason we scrub identifiers first, and why production use would need a paid tier with a data processing agreement.
- **A no-AI option:** the *Quick check* tab runs entirely inside the app, with no external API calls, for people who don't want their words sent to a third party.
- **Secrets:** API keys are kept in `.env` or Streamlit secrets and never committed (`.gitignore`).

## 3. Consent and transparency
- Clear on-screen statements that this is a screening estimate, not a diagnosis, and what data is used.
- The sidebar shows the model's measured sensitivity and specificity and how many people it was trained on.
- Each result shows which answers influenced it, so it is never a black box.

## 4. Bias and fairness
- **What we measured:** `src/train.py` reports AUC, sensitivity and specificity for each **age group** and **BMI band** (`reports/model_card.md`).
- **What we found** (screening model):
  - Performance is fairly consistent across age groups (AUC 0.85–0.97).
  - **Specificity drops for people with a BMI of 30 or more (0.58, versus about 0.80 for a
    healthy BMI).** The tool raises more false alarms for people with higher BMI, probably
    because weight-related answers overlap with PCOS signs. This could reinforce weight stigma,
    so explanations never present weight as "the cause". We name this openly as the first thing
    to fix with more data.
  - Specificity is also lower for under-25s (0.67).
  - The under-18.5 BMI group has only 10 PCOS cases, so its numbers are very uncertain.
  - The data only covers ages 20–48. The app warns users outside that range that their result
    is less reliable.
- **What we know is missing:**
  - The data comes from **one region of India**, so performance for other ethnicities and healthcare settings is unknown. PCOS can present differently across ethnic groups, for example in hair growth patterns.
  - We have no data on socio-economic status, so we can't assess equity across groups.
  - The sample is small, so subgroup estimates are uncertain. Groups under 15 people are not scored.
- **Design choices to reduce harm:**
  - The threshold is tuned for **high sensitivity**, so the tool errs towards suggesting a GP visit rather than falsely reassuring.
  - **Low scores never say "you don't have PCOS".** People are still encouraged to see a GP if symptoms bother them.
  - The score is presented as a **screening score**, not a probability, because it isn't calibrated for the general population: the dataset has a much higher PCOS rate than the public.

## 5. Safety
- Red-flag symptoms (severe pain, very heavy bleeding, fainting, pregnancy with bleeding, bleeding after the menopause, chest pain, thoughts of self-harm) trigger **immediate urgent-care advice**, whatever the model says.
- LLM outputs pass through a filter that rewrites any diagnostic phrasing ("you have PCOS") and always adds the disclaimer.
- The LLM never makes the prediction. It only converts free text into the model's inputs and explains the validated model's output.

## 6. What would be needed before real-world use
- Validation on diverse UK data under proper governance (ethics approval, a data protection impact assessment).
- Co-design with patients and clinicians, and clinical safety review (DCB0129 / DCB0160).
- Assessment as a medical device under UK regulations.

## 7. Tracking data (My health) and the NHS App vision
- **In this prototype nothing is stored.** The period and symptom log lives only in the user's browser session. The user can **download** it as a file to keep, **re-upload** it later, or **delete** it with one click.
- **The log is never sent to an AI.** Alerts are simple, transparent rules taken from nhs.uk "see a GP if" advice. The pattern check runs the local ML model, and its explanation uses a fixed template, not the LLM.
- **Alerts lean towards safety.** They point to a GP or NHS 111 and link the NHS page they come from. They never diagnose.
- **How it would work inside the NHS App** (our proposal, not current NHS policy):
  - It would sit behind NHS login, with data held in the NHS's secure infrastructure under UK GDPR and the NHS Data Security and Protection Toolkit.
  - It would need clinical safety cases (DCB0129/DCB0160) and a data protection impact assessment.
  - Users would get explicit opt-in consent, the ability to see, export and delete their data, and no sharing with third parties.
  - Any LLM provider would need an NHS-approved data processing agreement.
- **Terminology:** we follow the NHS's current name, **polyendocrine metabolic ovarian syndrome (PMOS)**, formerly PCOS. The training dataset itself still uses "PCOS".

## 8. Honest accuracy, and why most checks are rules rather than models
- **We only train a model where suitable open data exists.** PMOS is the only condition with a public dataset that has self-reported symptoms *and* confirmed diagnoses. For endometriosis, heavy periods, perimenopause and gynaecological cancer warning signs we found no such dataset, and published studies show symptoms alone can't detect these reliably (for example, endometriosis symptom questionnaires reach an AUC of about 0.68). Training weak models on unsuitable data would give false confidence.
- **So those checks are transparent rules from NHS and NICE guidance** (NG73 endometriosis, NG88 heavy menstrual bleeding, NG23 menopause, NG12 suspected cancer, and the NHS cervical screening programme). Every alert names its source and has a *How reliable is this sign?* section quoting the published evidence. We never claim our own accuracy for these rules, because we haven't tested them.
- **The PMOS model's accuracy is shown in plain numbers, not as one "accuracy" figure**, which is misleading with imbalanced data: of 100 people with PMOS it flags about 86 (95% range 80% to 90%), and of 100 without it about 24 are flagged too (20% to 29%).
- **We show what a flag means in real life.** The training data is 33% PMOS, but the general population figure is up to 13% (WHO). At that rate only about 3 in 10 people flagged would have PMOS, so every result says a flag means "worth checking with a GP", never "you have it", and a low score never rules it out.
- **Known blind spots are shown to the user:** ages outside 20–48, more false alarms for a BMI of 30 or more, and irregular periods in people over 40 with menopause-type symptoms (perimenopause can look like PMOS to the model). The check is switched off for people who say their periods have stopped.
- **Wording around cancer warning signs is deliberately calm and factual:** for example, "about 9 in 100 people with bleeding after the menopause have womb cancer, but about 9 in 10 womb cancers cause it, so it should always be checked".
