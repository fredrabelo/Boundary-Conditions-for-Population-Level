# When Synthetic Populations Fail — research artifact

Data, scripts and result tables for the paper *"When Synthetic Populations Fail:
Evidence for Boundary Conditions in Population-Level Behavioural Fidelity"*
(under double-blind review). The study compares LLM-generated respondent
populations (`claude-haiku-4-5`, `gemini-3.7-flash`) with the human outcomes of a
public Brazilian survey on adaptive residential behaviour and air-conditioning use
(Ramos et al., 2021; 3,259 residents, 281 cities), under strict holdout of every
target outcome.

## Design in brief

- **Population.** 248 respondents sampled proportionally across four climate groups
  from the 3,224 complete-case survey rows (extremely hot 62, very hot 28, hot 131,
  warm 27). 120 of them have air conditioning in the bedroom and form the Module 3
  population.
- **Grounding.** Each synthetic respondent receives 26 allowlisted contextual fields
  (27 in Module 3, which adds bedroom-AC eligibility). No target outcome is ever
  part of the grounding.
- **Modules.** (1) adoption: ventilation preference and household AC ownership;
  (2) habitual hot- and cold-weather actions (multi-select); (3) AC usage among
  bedroom-AC owners: frequency, hours per day, cooling setpoint.
- **Conditions.** *Baseline*: explanations required, generic "all applicable"
  checklist. *Instrument Calibration*: explanations removed and a habitual-observation
  policy for Module 2. *Narrative Grounding*: the same facts rendered as narrative
  paragraphs (Modules 1 and 3 only).
- **Runs.** 16 batch runs, 7,120 item responses, metered cost US$3.0289.
- **Evaluation.** Population level: prevalences, distributions, gradients, total
  variation distance, standard deviations; paired respondent bootstrap
  (10,000 resamples, seed 20260907, percentile 95% intervals).

## Repository layout

```
data/
  raw/                       public survey release (Ramos et al., 2020)
    DATA.CSV                     3,259 questionnaire responses
    NOTES.xlsx                   variable notes
    Questionnaire.pdf            original questionnaire
  processed/
    respondents.json             248 sampled respondents: grounding fields and held-out human outcomes
    module3_subset.json          external keys of the 120 bedroom-AC respondents
    grounding_spec.json          evidence keys visible to each module, and the outcomes held out
  instruments/
    instruments.json             question wording (Portuguese), options and ranges, per condition and module
  manifests/
    run_manifests.json           model, provider, grounding configuration and timestamps of each run
    run_costs.json               metered tokens and cost per run
  model_outputs/
    synthetic_responses.csv      7,120 rows: run, condition, module, provider, model,
                                 external_key, question_key, draw_index, value_json, explanation
results/                     tables written by scripts/02_analyze_results.py
scripts/
  01_build_sample.py            raw survey -> respondents.json, module3_subset.json, grounding_spec.json
  02_analyze_results.py         model outputs + human outcomes -> every statistic in the paper
  03_export_model_outputs_from_db.py   extraction of the exported files from the run database (audit only)
```

## Reproducing the results

```bash
pip install -r requirements.txt
cd scripts
python3 01_build_sample.py        # rebuilds data/processed/ from data/raw/DATA.CSV
python3 02_analyze_results.py     # prints all reported numbers; writes ../results/*.csv
```

`02_analyze_results.py` needs only `data/processed/`, `data/model_outputs/` and
`data/manifests/run_costs.json`. It prints, per module: prevalences and errors,
paired-bootstrap changes, income and climate gradients, action-prevalence vectors
(MAE, Pearson, Spearman, mean selected count), frequency distributions with total
variation distance, hours and setpoint means and standard deviations, and the
pre-specified recovery criteria, including the counts summarised in the recovery
figure. The complete output is stored in `results/analysis_output.txt`.

`03_export_model_outputs_from_db.py` reads a private database through the
environment variables `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`; it cannot be
run without that database and is not needed for reproduction.

## Data notes

- Instruments and model responses are in Portuguese, the language of the survey.
  `02_analyze_results.py` maps each option to the survey's English variable labels.
- `explanation` is filled only in the Baseline condition; the other conditions
  removed explanations from the response schema.
- Module 2 answers are stored as the comma-joined list of selected options.
- Setpoint is missing for 12 of the 120 human bedroom-AC respondents; human
  setpoint statistics use n = 108, model statistics use all 120 respondents.

## Scope

The repository contains the survey data, derived respondent files, instrument
wording, run configurations, all model outputs and the analysis code. It does not
contain the source code of the platform that generated the synthetic respondents
or the grounding-text compiler; the fields each module receives are listed in
`data/processed/grounding_spec.json` and the paragraph structure of the narrative
condition in `data/manifests/run_manifests.json`.

## Data source

Ramos, G., et al. (2021). Adaptive behaviour and air conditioning use in Brazilian
residential buildings. *Building Research & Information*, 49(5), 496–511.
https://doi.org/10.1080/09613218.2020.1804314

Ramos, G., et al. (2020). Dataset for adaptive behaviour and air conditioning use in
Brazilian residential buildings [Data set]. Mendeley Data.
https://doi.org/10.17632/zwjxzgkkn7.1

Files in `data/raw/` are redistributed unchanged from that release and remain subject
to its terms.
