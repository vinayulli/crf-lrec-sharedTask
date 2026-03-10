"""
DSPy pipeline for extracting CRF annotations from clinical notes.

Usage:
    from pipeline import CRFExtractor, configure_lm

    configure_lm("openai/gpt-4o-mini")
    extractor = CRFExtractor()
    result = extractor(clinical_note="77-year-old patient...")
    print(result.crf)  # CRFOutput instance
"""

import dspy

from schema import (
    ArterialBloodGas,
    CRFOutput,
    CurrentPresentation,
    AcuteDiagnoses,
    Devices,
    EpilepsyAssessment,
    ImagingAndDiagnostics,
    InfectionScreening,
    LabValues,
    MedicalHistory,
    Outcome,
    SocialContext,
    SyncopeAssessment,
    Treatments,
    VitalSigns,
)


# ── Configuration ────────────────────────────────────────────────────────────

def configure_lm(model: str = "openai/gpt-4o-mini", **kwargs):
    """Configure the DSPy language model.

    Args:
        model: LiteLLM-style model string, e.g. "openai/gpt-4o-mini",
               "openai/gpt-4o", "anthropic/claude-3-haiku", etc.
        **kwargs: Extra args passed to dspy.LM (temperature, max_tokens, ...).
    """
    lm = dspy.LM(model, **kwargs)
    dspy.configure(lm=lm)
    return lm


# ── Signatures (one per sub-model) ──────────────────────────────────────────
# Each signature extracts one logical group from the clinical note.
# The docstring is the LLM instruction — keep it clear and specific.

EXTRACT_INSTRUCTION = (
    "You are a clinical NLP system extracting structured information from "
    "emergency department clinical notes. "
    "Extract ONLY what is explicitly stated or clearly implied in the note. "
    "Leave fields as None/null when the information is not mentioned, "
    "uncertain, or cannot be determined from the text."
)


class ExtractMedicalHistory(dspy.Signature):
    """Extract the patient's medical history and chronic conditions from the clinical note.

    Look for: chronic diseases (pulmonary, cardiac, renal, metabolic, rheumatologic),
    dialysis, neoplasia, cardiovascular/vascular disease, neurological conditions
    (dementia, neurodegeneration, neuropsychiatric, neuropathy, epilepsy),
    substance abuse history, allergies, recent trauma, immunosuppression, pregnancy.

    CRITICAL RULES — read carefully:
    - None means the condition is NOT MENTIONED AT ALL in the note. This is the DEFAULT.
      A typical patient will have MOST fields as None.
    - False means the note EXPLICITLY STATES the condition is absent
      (e.g., "no allergies", "denies drug use", "no history of epilepsy").
    - Do NOT set False just because the condition is absent from a list of past medical
      history. Absence from a list means None (not mentioned), NOT False (ruled out).
    - For chronic disease status fields (ChronicStatus enum): use "certainly not chronic"
      or "possibly chronic" ONLY when the note explicitly discusses that specific condition.
      If the condition is not mentioned at all, return None.
    - For epilepsy fields: ONLY set True/False if epilepsy is explicitly discussed.
      Do NOT infer epilepsy status from seizure presentations alone.
    - For pregnancy: ONLY set True if explicitly stated pregnant. None if not mentioned.
    - For active neoplasia: ONLY extract if cancer/neoplasia is explicitly mentioned
      in the patient's history. Do NOT infer from unrelated findings.
    - When in doubt, prefer None over False. False requires explicit negation."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    medical_history: MedicalHistory = dspy.OutputField(desc="Extracted medical history and chronic conditions")


class ExtractVitalSigns(dspy.Signature):
    """Extract vital signs and functional status from the clinical note.

    Look for: blood pressure (normotensive/hypertensive/hypotensive),
    heart rate (normocardic/tachycardic/bradycardic),
    body temperature (normothermic/hyperthermic/hypothermic),
    respiratory rate (eupneic/tachypneic/bradypneic),
    SpO2 percentage, level of consciousness (AVPU: A/V/P/U),
    level of autonomy/mobility (walking independently/with aids/bedridden).

    CRITICAL RULES:
    - Return None for any vital sign NOT EXPLICITLY MENTIONED or clearly implied.
    - Do NOT infer vital signs from other findings. E.g., do not infer heart rate
      category from ECG findings, do not infer respiratory rate from dyspnea status.
    - Only extract values from the TRIAGE or PHYSICAL EXAM sections of the current visit.
    - For SpO2: extract only if an explicit percentage or saturation value is given.
    - For level of consciousness: only extract if AVPU, GCS, or explicit terms like
      "alert", "responsive to voice/pain", "unresponsive" are stated.
    - For level of autonomy: only extract if mobility is explicitly described.
      Do NOT infer mobility from age or diagnosis alone."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    vital_signs: VitalSigns = dspy.OutputField(desc="Extracted vital signs")


class ExtractLabValues(dspy.Signature):
    """Extract laboratory test results from the clinical note.

    Look for: hemoglobin, leukocytes/WBC, platelets, creatinine, blood glucose,
    potassium, sodium, calcium, CRP, D-dimer, troponin, BNP/NT-proBNP,
    transaminases (AST/ALT), creatinine kinase, INR, lactates,
    blood alcohol, blood drug levels, urine drug test.

    CRITICAL RULES for extracting values:
    - Return ONLY the numeric value as a short string (e.g., "149", "5.2", "0.69", "24530").
    - NEVER include label prefixes like "WBC", "CRP", "Hb", "Na", "PLT", "KCL", "HS trop T".
      Strip ALL labels — return ONLY the number or qualitative result.
      Examples: "WBC 24530" → return "24530". "CRP 0.69" → return "0.69".
      "Hb 12" → return "12". "Na 136" → return "136". "KCL 5.08" → return "5.08".
    - For qualitative results, return the result verbatim (e.g., "neg", "nn", "positive").
      Do NOT normalize or interpret abbreviations.
    - When a lab test is reported multiple times, use the FIRST mentioned value.
    - Only extract results from the CURRENT clinical encounter. Ignore values
      from previous visits, past records, or historical references.
    - Return None if the lab test is not mentioned in the current encounter.
    - Do NOT extract values from medication names or dosages (e.g., "KCL" in a
      medication list is NOT a potassium lab result)."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    lab_values: LabValues = dspy.OutputField(desc="Extracted laboratory values")


class ExtractABG(dspy.Signature):
    """Extract arterial blood gas (ABG) values from the clinical note.

    Look for: pH, PaO2, PaCO2, HCO3-/bicarbonate.

    Return the numeric value as it appears in the note.

    CRITICAL RULES:
    - Extract values from any blood gas analysis (ABG/EGA/BGA) that appears in the note.
    - If the note reports blood gas values WITHOUT specifying arterial vs venous,
      ASSUME they are arterial and extract them. Most ED blood gases are arterial.
    - Only skip extraction if the note EXPLICITLY labels them as VENOUS (VBG).
    - Look for pH, pO2/PaO2, pCO2/PaCO2, HCO3/bicarbonate in lab sections,
      blood gas panels, and any reported gas analysis results.
    - Return the numeric value only (e.g., "7.39", "43", "25").
    - Return None only if no blood gas values are mentioned at all."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    arterial_blood_gas: ArterialBloodGas = dspy.OutputField(desc="Extracted ABG values")


class ExtractImaging(dspy.Signature):
    """Extract imaging and diagnostic test results from the clinical note.

    Look for results of: chest X-ray, chest CT, abdomen CT, brain CT, brain MRI,
    cardiac ultrasound, thoracic ultrasound, compression ultrasound (CUS/DVT),
    ECG, ECG monitoring, EEG, pulmonary scintigraphy, gastroscopy.

    CRITICAL RULES:
    - True = the test was PERFORMED AND an abnormality was found.
    - False = the test was PERFORMED AND the result was normal / no abnormality.
    - None = the test was NOT PERFORMED or NOT MENTIONED in the note.
    - Do NOT set True/False for a test unless the note explicitly mentions that
      specific test being performed AND describes results.
    - "ECG abnormality" refers to a standard 12-lead ECG. "ECG monitoring" is
      separate (continuous cardiac monitoring). Do not confuse the two.
    - When in doubt whether a test was performed, prefer None."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    imaging_and_diagnostics: ImagingAndDiagnostics = dspy.OutputField(desc="Extracted imaging and diagnostic results")


class ExtractCurrentPresentation(dspy.Signature):
    """Extract current symptoms and clinical presentation from the clinical note.

    Look for: dyspnea, respiratory distress, chest pain, agitation,
    blood in stool, foreign body in airways, general condition deterioration,
    head/other trauma, hemorrhage, concussive head trauma.

    CRITICAL RULES for True / False / None:

    TRUE = the symptom/finding IS CURRENTLY present:
    - "dyspneic", "shortness of breath", "difficulty breathing" → dyspnea True
    - "chest pain", "thoracic pain" → chest_pain True
    - "agitated", "restless", "combative" → agitation True
    - Patient presents TO THE ED for dyspnea → presence_of_dyspnea True

    FALSE = EXPLICIT negation or specific clinical descriptors:
    - "no chest pain", "denies dyspnea" → False
    - "eupnoeic"/"eupnoic"/"eupneic" → dyspnea False
    - "clear airways"/"no foreign body" → foreign_body_in_airways False
    - "alert"/"cooperative"/"calm"/"oriented" → agitation False
    - "no bleeding"/"no hemorrhage" → hemorrhage False

    NONE = the symptom is NOT DISCUSSED AT ALL in the note.

    COMMON MISTAKES TO AVOID:
    - Do NOT set head_or_other_trauma=True from fall history alone. Only True
      if trauma/injury is explicitly noted on current presentation.
    - Do NOT set concussive_head_trauma=True unless a head concussion is
      explicitly diagnosed or described.
    - Do NOT set general_condition_deterioration=True from acute illness alone.
      Only True if the note explicitly states the patient's condition has deteriorated.
    - Do NOT confuse foreign_body_in_airways with aspiration. Foreign body means
      a physical object lodged in the airway.
    - presence_of_dyspnea: if the patient's chief complaint or reason for ED visit
      is dyspnea/breathlessness, set True. If "eupneic" at exam, set False.
    - When in doubt between False and None, choose None unless there is clear
      explicit negation or the specific clinical descriptors listed above."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    current_presentation: CurrentPresentation = dspy.OutputField(desc="Extracted current presentation")


class ExtractAcuteDiagnoses(dspy.Signature):
    """Extract acute diagnoses and conditions from the clinical note.

    Look for: pneumonia, aspiration pneumonia, pulmonary embolism, pneumothorax,
    acute coronary syndrome, heart failure, acute pulmonary edema,
    cardiac tamponade, aortic dissection, arrhythmia, severe anemia,
    intoxication, respiratory failure, asthma/COPD exacerbation, COVID-19.

    CRITICAL RULES:
    - True = the diagnosis is EXPLICITLY NAMED by the clinician as a diagnosis,
      impression, or confirmed finding. The WORD for the diagnosis must appear.
      (e.g., "diagnosed with pneumonia", "PE on CT", "atrial fibrillation").
    - False = the diagnosis is EXPLICITLY RULED OUT
      (e.g., "PE excluded", "no pneumothorax", "troponin negative rules out ACS").
    - None = the diagnosis is not mentioned or discussed at all. This is the DEFAULT.

    COMMON MISTAKES TO AVOID:
    - arrhythmia: Do NOT set True just because ECG shows abnormalities or because
      the patient has a pacemaker/cardiac history. Only True if an arrhythmia is
      EXPLICITLY DIAGNOSED (e.g., "atrial fibrillation", "SVT", "VT", "arrhythmia").
    - covid_19: Do NOT set True/False from swab tests alone. Only True if COVID-19
      is DIAGNOSED. Only False if explicitly ruled out. Swab results go in
      infection_screening, not here.
    - intoxication: Do NOT set True from alcohol/drug levels alone. Only True if
      the clinician explicitly diagnoses intoxication.
    - pneumonia: Do NOT infer from chest X-ray findings alone. The word "pneumonia"
      or clear equivalent must be used by the clinician.
    - Do NOT infer ANY diagnosis from symptoms, lab results, imaging findings,
      or treatments alone. The diagnosis must be explicitly stated."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    acute_diagnoses: AcuteDiagnoses = dspy.OutputField(desc="Extracted acute diagnoses")


class ExtractTreatments(dspy.Signature):
    """Extract treatments, interventions, and medication info from the clinical note.

    Look for: oxygen/ventilation, bronchodilators, diuretics, fluids, steroids,
    blood transfusions, CPR, thoracentesis, palliative care,
    antihypertensive therapy, anticoagulant/antiplatelet therapy,
    antiepileptic therapy, poly-pharmacological therapy (5+ medications),
    compliance with antiepileptic therapy.

    CRITICAL RULES:
    - True = the treatment is EXPLICITLY stated as administered, prescribed,
      or currently in place during this encounter.
    - False = the treatment is EXPLICITLY stated as NOT given or not in place.
    - None = the treatment is NOT MENTIONED at all. This is the DEFAULT.
    - Do NOT infer treatments from diagnoses or symptoms alone.
    - For "antiepileptic therapy already in place": True only if the patient
      is documented as already being ON antiepileptic medication before this visit.
    - For "poly-pharmacological therapy": True only if the note explicitly lists
      5 or more chronic/home medications, or explicitly states polypharmacy.
    - For "anticoagulants or antiplatelet": True if the patient is ON anticoagulant
      or antiplatelet therapy (e.g., warfarin, aspirin, clopidogrel, DOACs).
    - Do NOT set True for treatments that are only mentioned as future plans,
      recommendations, or possibilities — only for actually administered/in-place.
    - When in doubt, prefer None over True or False."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    treatments: Treatments = dspy.OutputField(desc="Extracted treatments and medications")


class ExtractEpilepsyAssessment(dspy.Signature):
    """Extract epilepsy and seizure-related assessment from the clinical note.

    Look for: tonic-clonic seizures, further seizures in ED, first epilepsy episode,
    stiffness during episode, eye deviation, pale skin, drooling, tongue bite,
    postcritical drowsiness/confusion, duration of unconsciousness,
    duration of consciousness recovery.

    CRITICAL RULES:
    - This section is ONLY relevant if the patient had a seizure or epileptic episode.
    - If the patient did NOT present with seizures/epilepsy, ALL fields should be None.
    - Do NOT extract epilepsy assessment data for patients presenting with syncope,
      altered consciousness, or other non-seizure conditions.
    - Even for seizure patients, only extract what is EXPLICITLY described.
      Do NOT infer symptoms of the seizure episode.
    - Return None for anything not mentioned."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    epilepsy_assessment: EpilepsyAssessment = dspy.OutputField(desc="Extracted epilepsy assessment")


class ExtractSyncopeAssessment(dspy.Signature):
    """Extract syncope-related assessment from the clinical note.

    Look for: situational syncope, transient loss of consciousness (TLOC) during
    effort or while supine, supine-to-standing BP test, carotid sinus massage,
    prodromal symptoms, situation description (coughing, straining, pain, phlebotomy).

    CRITICAL RULES:
    - This section is relevant if the patient had syncope, near-syncope, fainting,
      collapse, or transient loss of consciousness (TLOC).
    - Look carefully for syncope-related details even in patients whose primary
      complaint is dyspnea — syncope may be a secondary finding.
    - situational_syncope: True if syncope occurred in a specific situation
      (coughing, micturition, defecation, swallowing, etc.).
    - presence_of_prodromal_symptoms: True if the patient had warning symptoms
      before the event (dizziness, nausea, visual changes, sweating, etc.).
    - situation_description: extract the triggering situation as a brief string
      if described (e.g., "coughing", "straining", "standing up").
    - If the patient did NOT have any syncope-related event, ALL fields should be None.
    - Return None for anything not mentioned."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    syncope_assessment: SyncopeAssessment = dspy.OutputField(desc="Extracted syncope assessment")


class ExtractInfectionScreening(dspy.Signature):
    """Extract infection screening and consultation info from the clinical note.

    Look for: influenza or other infections identified, SARS-CoV-2 swab test result,
    neurologist consultation requested or performed.

    CRITICAL RULES:
    - None = not mentioned in the note at all. This is the DEFAULT for most patients.
    - True = EXPLICITLY identified, confirmed, or performed.
    - False = EXPLICITLY stated as negative or not performed.

    SPECIFIC FIELD RULES:
    - sars_cov2_swab_test: True ONLY if a COVID swab/test is explicitly mentioned
      AND the result is positive. False ONLY if swab is mentioned AND result is negative.
      None if no swab test is mentioned at all.
    - influenza_and_infections: True ONLY if an infection (influenza, URI, UTI, etc.)
      is explicitly identified/confirmed. False ONLY if infection is explicitly ruled out.
      None if not discussed.
    - neurologist_consultation: True ONLY if a neurologist consult is explicitly
      requested or performed. None if not mentioned.

    COMMON MISTAKES TO AVOID:
    - Do NOT set sars_cov2_swab_test to True/False just because COVID is mentioned
      in a historical or screening context.
    - Do NOT infer infections from symptoms (fever, cough) alone.
    - Absence of mention = None, NOT False."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    infection_screening: InfectionScreening = dspy.OutputField(desc="Extracted infection screening")


class ExtractSocialContext(dspy.Signature):
    """Extract social and living context from the clinical note.

    Look for: whether patient lives alone, homelessness, need for but absence
    of a caregiver, problematic family context.

    CRITICAL RULES:
    - None = not mentioned at all. This is the DEFAULT for most patients.
    - Only extract social context that is EXPLICITLY stated in the note.
    - Do NOT infer living situation from age, diagnosis, or other clinical data.
    - living_alone: True only if explicitly stated the patient lives alone.
    - homelessness: True only if explicitly stated.
    - need_but_absence_of_caregiver: True only if explicitly noted.
    - problematic_family_context: True only if explicitly described.
    - Return None for anything not mentioned."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    social_context: SocialContext = dspy.OutputField(desc="Extracted social context")


class ExtractDevices(dspy.Signature):
    """Extract information about implanted cardiac devices from the clinical note.

    Look for: presence of a pacemaker, presence of an implanted defibrillator (ICD).

    CRITICAL RULES:
    - True = the note EXPLICITLY states the patient has a pacemaker or defibrillator.
    - False = the note EXPLICITLY states the patient does NOT have these devices.
    - None = not mentioned at all. This is the DEFAULT.
    - Do NOT infer device presence from cardiac history or arrhythmia alone.
    - Return None if not mentioned."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    devices: Devices = dspy.OutputField(desc="Extracted device information")


class ExtractOutcome(dspy.Signature):
    """Extract patient outcome information from the clinical note.

    Look for: whether dyspnea improved during the ED stay,
    whether the patient's overall condition improved.

    CRITICAL RULES:
    - None = improvement is NOT EXPLICITLY discussed. This is the DEFAULT.
    - True = the note EXPLICITLY states that dyspnea or condition IMPROVED
      (e.g., "dyspnea resolved", "condition improved", "patient better").
    - False = the note EXPLICITLY states NO improvement or worsening.
    - Do NOT infer improvement from treatment administration alone.
      Giving oxygen does NOT mean dyspnea improved.
    - Do NOT infer improvement from discharge. Being discharged does NOT
      mean the condition improved.
    - Do NOT set True/False based on clinical reasoning or assumptions.
      Only set if the note EXPLICITLY discusses improvement or lack thereof.
    - Most notes do NOT explicitly discuss outcome — expect None for most cases."""

    clinical_note: str = dspy.InputField(desc="Emergency department clinical note")
    outcome: Outcome = dspy.OutputField(desc="Extracted outcome information")


# ── Mapping of group name → signature class ─────────────────────────────────

GROUP_SIGNATURES = {
    "medical_history": ExtractMedicalHistory,
    "vital_signs": ExtractVitalSigns,
    "lab_values": ExtractLabValues,
    "arterial_blood_gas": ExtractABG,
    "imaging_and_diagnostics": ExtractImaging,
    "current_presentation": ExtractCurrentPresentation,
    "acute_diagnoses": ExtractAcuteDiagnoses,
    "treatments": ExtractTreatments,
    "epilepsy_assessment": ExtractEpilepsyAssessment,
    "syncope_assessment": ExtractSyncopeAssessment,
    "infection_screening": ExtractInfectionScreening,
    "social_context": ExtractSocialContext,
    "devices": ExtractDevices,
    "outcome": ExtractOutcome,
}

# The output field name in each signature (matches the group name)
GROUP_OUTPUT_FIELD = {
    "medical_history": "medical_history",
    "vital_signs": "vital_signs",
    "lab_values": "lab_values",
    "arterial_blood_gas": "arterial_blood_gas",
    "imaging_and_diagnostics": "imaging_and_diagnostics",
    "current_presentation": "current_presentation",
    "acute_diagnoses": "acute_diagnoses",
    "treatments": "treatments",
    "epilepsy_assessment": "epilepsy_assessment",
    "syncope_assessment": "syncope_assessment",
    "infection_screening": "infection_screening",
    "social_context": "social_context",
    "devices": "devices",
    "outcome": "outcome",
}


# ── Main extractor module ───────────────────────────────────────────────────

class CRFExtractor(dspy.Module):
    """Extracts all 134 CRF fields from a clinical note.

    Runs 14 typed predictors (one per category) and combines results
    into a single CRFOutput.
    """

    def __init__(self, use_cot: bool = True):
        super().__init__()
        predictor_cls = dspy.ChainOfThought if use_cot else dspy.Predict

        # Create one predictor per group
        # NOTE: Do NOT use self.predictors — it shadows dspy.Module.predictors() method
        self._extractor_names = []
        for group_name, sig_cls in GROUP_SIGNATURES.items():
            predictor = predictor_cls(sig_cls)
            # Register as a named sub-module so DSPy can track it
            setattr(self, f"extract_{group_name}", predictor)
            self._extractor_names.append(group_name)

    def forward(self, clinical_note: str) -> dspy.Prediction:
        # Run all 14 extractors
        results = {}
        for group_name in self._extractor_names:
            output_field = GROUP_OUTPUT_FIELD[group_name]
            predictor = getattr(self, f"extract_{group_name}")
            prediction = predictor(clinical_note=clinical_note)
            results[group_name] = getattr(prediction, output_field)

        crf = CRFOutput(**results)
        return dspy.Prediction(crf=crf)


# ── Single extractor module (for per-extractor optimization) ─────────────

class SingleExtractor(dspy.Module):
    """Wraps a single predictor for per-extractor MIPROv2 optimization.

    Uses the same `extract_{group_name}` naming convention as CRFExtractor
    so saved JSON keys are compatible.
    """

    def __init__(self, group_name: str, use_cot: bool = True):
        super().__init__()
        self.group_name = group_name
        self.output_field = GROUP_OUTPUT_FIELD[group_name]
        sig_cls = GROUP_SIGNATURES[group_name]
        predictor_cls = dspy.ChainOfThought if use_cot else dspy.Predict
        setattr(self, f"extract_{group_name}", predictor_cls(sig_cls))

    def forward(self, clinical_note: str) -> dspy.Prediction:
        predictor = getattr(self, f"extract_{self.group_name}")
        pred = predictor(clinical_note=clinical_note)
        sub_model = getattr(pred, self.output_field)
        return dspy.Prediction(**{self.output_field: sub_model})


def compose_optimized_extractors(
    group_json_dir: str,
    use_cot: bool = True,
    fallback_groups: str = "default",
) -> CRFExtractor:
    """Build a CRFExtractor from individually optimized per-group JSON files.

    Args:
        group_json_dir: Directory containing {group_name}.json files
        use_cot: Whether to use ChainOfThought
        fallback_groups: What to do for groups without a JSON file:
            "default" = use default prompts, "skip" = raise error
    """
    from pathlib import Path

    full_extractor = CRFExtractor(use_cot=use_cot)
    json_dir = Path(group_json_dir)

    for group_name in GROUP_SIGNATURES:
        json_path = json_dir / f"{group_name}.json"
        if not json_path.exists():
            if fallback_groups == "default":
                continue  # Keep default prompts
            raise FileNotFoundError(f"Missing optimized file: {json_path}")

        # Load the single-extractor program
        single = SingleExtractor(group_name, use_cot=use_cot)
        single.load(str(json_path))

        # Transfer optimized state into the full extractor
        # ChainOfThought wraps a Predict in .predict; demos/signature live there
        src = getattr(single, f"extract_{group_name}")
        dst = getattr(full_extractor, f"extract_{group_name}")
        src_inner = getattr(src, "predict", src)
        dst_inner = getattr(dst, "predict", dst)
        dst_inner.demos = src_inner.demos
        dst_inner.signature = src_inner.signature

    return full_extractor


# ── Metric for DSPy optimization ────────────────────────────────────────────

def crf_field_accuracy(example, prediction, trace=None) -> float:
    """Compute field-level accuracy between ground truth and prediction.

    Returns a score between 0.0 and 1.0 (fraction of 134 fields correct).
    Used as the DSPy optimization metric.
    """
    from schema import crf_to_annotations

    gt_crf = example.crf  # Ground truth CRFOutput
    pred_crf = prediction.crf  # Predicted CRFOutput

    gt_anns = {a["item"]: a["ground_truth"] for a in crf_to_annotations(gt_crf)}
    pred_anns = {a["item"]: a["ground_truth"] for a in crf_to_annotations(pred_crf)}

    correct = sum(1 for k in gt_anns if gt_anns[k] == pred_anns.get(k))
    return correct / len(gt_anns)


def crf_known_field_accuracy(example, prediction, trace=None) -> float:
    """Accuracy only on fields where the ground truth is NOT 'unknown'.

    This is a harder, more meaningful metric — measures how well the model
    extracts information that IS present in the note.
    """
    from schema import crf_to_annotations

    gt_crf = example.crf
    pred_crf = prediction.crf

    gt_anns = {a["item"]: a["ground_truth"] for a in crf_to_annotations(gt_crf)}
    pred_anns = {a["item"]: a["ground_truth"] for a in crf_to_annotations(pred_crf)}

    known_keys = [k for k in gt_anns if gt_anns[k] != "unknown"]
    if not known_keys:
        return 1.0  # No known fields to evaluate

    correct = sum(1 for k in known_keys if gt_anns[k] == pred_anns.get(k))
    return correct / len(known_keys)
