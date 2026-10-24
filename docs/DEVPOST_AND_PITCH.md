# Devpost draft + 4-minute pitch plan

> This is a draft. Edit it in your own words, fill in the real numbers after training on the
> Kaggle data, and remember the rules: **declare pre-existing work and AI tools**.

## Devpost answers (draft)

**Inspiration:** Up to 70% of women with PCOS are estimated to be undiagnosed. Its symptoms
(irregular periods, acne, excess hair, weight gain) are often treated one by one, so nobody joins
the dots. Endometriosis takes 8 years 10 months on average to diagnose in the UK. Women's symptoms
get missed, and we wanted to help people walk into a GP appointment with the pattern in hand.

**What it does:** HerHealth AI is a two-minute, private screening check for PCOS that uses only
information you already know about yourself, with no blood tests. Describe your symptoms in a
chat, or fill in a quick form. You get a screening score, the answers that influenced it, a
plain-English explanation and questions to ask your GP. Red-flag symptoms trigger urgent-care advice.

**How we built it:** We trained scikit-learn models (logistic regression and random forest) on a
public, anonymised Kaggle PCOS dataset, using 5-fold stratified cross-validation. We tuned the
threshold for high sensitivity and checked for bias across age and BMI groups. We compared a
self-report "screening" model with a "clinical" model that adds lab and ultrasound values. The
OpenAI API is the interface only: strict-JSON extraction of answers from free text, and
plain-English explanations. It never makes the prediction. Guardrails scrub personal details
before API calls, block diagnostic wording and catch red flags. Built in Python and Streamlit.

**Challenges:** Making an LLM useful without letting it "diagnose"; a small, single-region
dataset; choosing a threshold that balances missed cases against false alarms.

**Accomplishments:** A working end-to-end demo, and a screening model that needs no tests,
reaching cross-validated AUC 0.90 with 86% sensitivity and 76% specificity, compared with AUC
0.96 for a model that also uses hormone blood tests and ultrasound. In other words, self-reported
answers alone recover most of the signal. We also report its limits honestly, including more
false alarms for people with a BMI of 30 or more.

**What we learned:** Accuracy is misleading with imbalanced medical data. Sensitivity and
specificity matter more. LLMs are best as translators, not decision-makers.

**What's next:** Validation on diverse UK data with proper governance, co-design with patients
and GPs, more under-diagnosed conditions (endometriosis), and multilingual support.

**Data question:** see `RESPONSIBLE_DATA.md`. Paste the key points: open anonymised data only,
nothing stored, PII scrubbing, a no-AI mode, a bias check by age and BMI, high-sensitivity
threshold, and limitations stated.

**Built with:** python, scikit-learn, pandas, streamlit, openai, matplotlib

## 4-minute pitch (then 2 minutes of Q&A)

| Time | Who | What |
|---|---|---|
| 0:00–0:40 | Speaker 1 | **The problem:** a short story of "Priya": 4 GP visits, acne treated separately, periods "just irregular". Up to 70% undiagnosed. |
| 0:40–1:10 | Speaker 1 | **The idea:** a two-minute check using only what you already know. It gives you the pattern and the right questions for your GP. |
| 1:10–2:40 | Speaker 2 | **Live demo:** chat in plain English, follow-up questions, result, what influenced it, GP questions. Then show a red-flag message. (**Have a backup screen recording.**) |
| 2:40–3:30 | Speaker 3 | **Under the hood:** LLM = translator, ML = test. Cross-validated, sensitivity-first threshold, screening vs clinical comparison, bias table. |
| 3:30–4:00 | Speaker 1 | **Responsible and next steps:** no data stored, PII scrubbing, limitations, validation plan. Close with impact. |

### Likely judge questions (prepare answers)
- *Why not just ask ChatGPT?* Its answers can't be validated, aren't consistent, and it can make
  things up. Our model's performance is measured, and the LLM only translates.
- *How do you handle bias?* Show the subgroup table, name the single-region limitation, and explain
  the high-sensitivity threshold.
- *Is this a medical device?* As a diagnostic tool it would be, so we frame it as screening and
  education, and list what's needed for real use (validation, DCB0129, regulatory assessment).
- *What does the score mean?* Similarity to diagnosed cases in the training data. It's not a
  calibrated probability.
- *How would the NHS use it?* As a pre-appointment tool that structures symptoms for the GP
  consultation.
