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
