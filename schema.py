"""
Output schema for NLP-FBK/dyspnea-crf-development dataset.

134 annotation keys extracted from clinical notes, grouped into logical categories.
Each field is Optional — None means "unknown" (not mentioned / not determinable).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class ChronicStatus(str, Enum):
    CERTAINLY_CHRONIC = "certainly chronic"
    POSSIBLY_CHRONIC = "possibly chronic"
    CERTAINLY_NOT_CHRONIC = "certainly not chronic"


class NeoplasmStatus(str, Enum):
    CERTAINLY_ACTIVE = "certainly active"
    POSSIBLY_ACTIVE = "possibly active"
    CERTAINLY_NOT_ACTIVE = "certainly not active"


class BloodPressureStatus(str, Enum):
    NORMOTENSIVE = "normotensive"
    HYPERTENSIVE = "hypertensive"
    HYPOTENSIVE = "hypotensive"


class HeartRateStatus(str, Enum):
    NORMOCARDIC = "normocardic"
    TACHYCARDIC = "tachycardic"
    BRADYCARDIC = "bradycardic"


class TemperatureStatus(str, Enum):
    NORMOTHERMIC = "normothermic"
    HYPERTHERMIC = "hyperthermic"
    HYPOTHERMIC = "hypothermic"


class RespiratoryRateStatus(str, Enum):
    EUPNEIC = "eupneic"
    TACHYPNEIC = "tachypneic"
    BRADYPNEIC = "bradypneic"


class MobilityLevel(str, Enum):
    WALKING_INDEPENDENTLY = "walking independently"
    WALKING_WITH_AUXILIARY_AIDS = "walking with auxiliary aids"
    WALKING_WITH_PHYSICAL_ASSISTANCE = "walking with physical assistance"
    BEDRIDDEN = "bedridden"


class ConsciousnessLevel(str, Enum):
    """AVPU scale."""
    ALERT = "A"
    VOICE = "V"
    PAIN = "P"


# ── Sub-models ───────────────────────────────────────────────────────────────

class MedicalHistory(BaseModel):
    """Chronic conditions and past medical history (22 fields)."""

    chronic_pulmonary_disease: Optional[ChronicStatus] = Field(None, description="Chronic pulmonary disease status")
    chronic_respiratory_failure: Optional[ChronicStatus] = Field(None, description="Chronic respiratory failure status")
    chronic_cardiac_failure: Optional[ChronicStatus] = Field(None, description="Chronic cardiac failure status")
    chronic_renal_failure: Optional[ChronicStatus] = Field(None, description="Chronic renal failure status")
    chronic_metabolic_failure: Optional[ChronicStatus] = Field(None, description="Chronic metabolic failure status")
    chronic_rheumatologic_disease: Optional[ChronicStatus] = Field(None, description="Chronic rheumatologic disease status")
    chronic_dialysis: Optional[bool] = Field(None, description="Patient on chronic dialysis")
    active_neoplasia: Optional[NeoplasmStatus] = Field(None, description="Active neoplasia / cancer status")
    cardiovascular_diseases: Optional[bool] = Field(None, description="History of cardiovascular diseases")
    diffuse_vascular_disease: Optional[bool] = Field(None, description="Diffuse vascular disease present")
    dementia: Optional[bool] = Field(None, description="Dementia diagnosed")
    neurodegenerative_diseases: Optional[bool] = Field(None, description="Neurodegenerative diseases present")
    neuropsychiatric_disorders: Optional[bool] = Field(None, description="Neuropsychiatric disorders present")
    peripheral_neuropathy: Optional[bool] = Field(None, description="Peripheral neuropathy present")
    epilepsy: Optional[bool] = Field(None, description="Epilepsy or epileptic seizure history")
    known_history_of_epilepsy: Optional[bool] = Field(None, description="Previously known epilepsy diagnosis")
    history_of_alcohol_abuse: Optional[bool] = Field(None, description="History of alcohol abuse")
    history_of_drug_abuse: Optional[bool] = Field(None, description="History of drug abuse")
    history_of_allergy: Optional[bool] = Field(None, description="Known allergies")
    history_of_recent_trauma: Optional[bool] = Field(None, description="Recent trauma history")
    immunosuppression: Optional[bool] = Field(None, description="Immunosuppressed state")
    pregnancy: Optional[bool] = Field(None, description="Currently pregnant")


class VitalSigns(BaseModel):
    """Vital signs and functional status (7 fields)."""

    blood_pressure: Optional[BloodPressureStatus] = Field(None, description="Blood pressure category")
    heart_rate: Optional[HeartRateStatus] = Field(None, description="Heart rate category")
    body_temperature: Optional[TemperatureStatus] = Field(None, description="Body temperature category")
    respiratory_rate: Optional[RespiratoryRateStatus] = Field(None, description="Respiratory rate category")
    spo2: Optional[str] = Field(None, description="SpO2 percentage, e.g. '97%'")
    level_of_consciousness: Optional[ConsciousnessLevel] = Field(None, description="AVPU consciousness level")
    level_of_autonomy: Optional[MobilityLevel] = Field(None, description="Patient mobility level")


class LabValues(BaseModel):
    """Laboratory test results — all free-text numeric values (19 fields)."""

    hemoglobin: Optional[str] = Field(None, description="Hemoglobin value (g/dL)")
    leukocytes: Optional[str] = Field(None, description="White blood cell count")
    platelets: Optional[str] = Field(None, description="Platelet count")
    creatinine: Optional[str] = Field(None, description="Serum creatinine value")
    blood_glucose: Optional[str] = Field(None, description="Blood glucose level")
    blood_potassium: Optional[str] = Field(None, description="Serum potassium level")
    blood_sodium: Optional[str] = Field(None, description="Serum sodium level")
    blood_calcium: Optional[str] = Field(None, description="Serum calcium level")
    c_reactive_protein: Optional[str] = Field(None, description="CRP value")
    d_dimer: Optional[str] = Field(None, description="D-dimer value")
    troponin: Optional[str] = Field(None, description="Troponin value")
    bnp_or_nt_pro_bnp: Optional[str] = Field(None, description="BNP or NT-proBNP value")
    transaminases: Optional[str] = Field(None, description="AST/ALT transaminase values")
    serum_creatinine_kinase: Optional[str] = Field(None, description="Serum CK value")
    inr: Optional[str] = Field(None, description="INR value")
    lactates: Optional[str] = Field(None, description="Lactate level")
    blood_alcohol: Optional[str] = Field(None, description="Blood alcohol level")
    blood_drug_dosage: Optional[str] = Field(None, description="Blood drug dosage/level")
    urine_drug_test: Optional[str] = Field(None, description="Urine drug test result")


class ArterialBloodGas(BaseModel):
    """Arterial blood gas analysis (4 fields)."""

    ph: Optional[str] = Field(None, description="Arterial pH value")
    pao2: Optional[str] = Field(None, description="Partial pressure of oxygen (PaO2)")
    paco2: Optional[str] = Field(None, description="Partial pressure of CO2 (PaCO2)")
    hco3: Optional[str] = Field(None, description="Bicarbonate (HCO3-) level")


class ImagingAndDiagnostics(BaseModel):
    """Imaging and diagnostic test results — True=abnormality found (13 fields)."""

    chest_rx_abnormality: Optional[bool] = Field(None, description="Chest X-ray abnormality found")
    chest_ct_abnormality: Optional[bool] = Field(None, description="Chest CT abnormality found")
    abdomen_ct_abnormality: Optional[bool] = Field(None, description="Abdomen CT abnormality found")
    brain_ct_abnormality: Optional[bool] = Field(None, description="Brain CT abnormality found")
    brain_mri_abnormality: Optional[bool] = Field(None, description="Brain MRI abnormality found")
    cardiac_ultrasound_abnormality: Optional[bool] = Field(None, description="Cardiac ultrasound abnormality found")
    thoracic_ultrasound_abnormality: Optional[bool] = Field(None, description="Thoracic ultrasound abnormality found")
    compression_ultrasound_abnormality: Optional[bool] = Field(None, description="CUS abnormality found (DVT screening)")
    ecg_abnormality: Optional[bool] = Field(None, description="ECG abnormality found")
    ecg_monitoring_abnormality: Optional[bool] = Field(None, description="ECG monitoring abnormality found")
    eeg_abnormality: Optional[bool] = Field(None, description="EEG abnormality found")
    pulmonary_scintigraphy_abnormality: Optional[bool] = Field(None, description="Pulmonary scintigraphy abnormality found")
    gastroscopy_abnormality: Optional[bool] = Field(None, description="Gastroscopy abnormality found")


class CurrentPresentation(BaseModel):
    """Current symptoms and clinical presentation (10 fields)."""

    presence_of_dyspnea: Optional[bool] = Field(None, description="Dyspnea present on arrival")
    presence_of_respiratory_distress: Optional[bool] = Field(None, description="Respiratory distress currently present")
    chest_pain: Optional[bool] = Field(None, description="Chest pain reported")
    agitation: Optional[bool] = Field(None, description="Patient is agitated")
    blood_in_the_stool: Optional[bool] = Field(None, description="Blood found in stool")
    foreign_body_in_airways: Optional[bool] = Field(None, description="Foreign body in airways suspected/found")
    general_condition_deterioration: Optional[bool] = Field(None, description="General condition has deteriorated")
    head_or_other_trauma: Optional[bool] = Field(None, description="Head or other district trauma present")
    hemorrhage: Optional[bool] = Field(None, description="Active hemorrhage")
    concussive_head_trauma: Optional[bool] = Field(None, description="Concussive head trauma present")


class AcuteDiagnoses(BaseModel):
    """Acute diagnoses and conditions identified (16 fields)."""

    pneumonia: Optional[bool] = Field(None, description="Pneumonia diagnosed")
    ab_ingestis_pneumonia: Optional[bool] = Field(None, description="Aspiration pneumonia diagnosed")
    pulmonary_embolism: Optional[bool] = Field(None, description="Pulmonary embolism diagnosed")
    pneumothorax: Optional[bool] = Field(None, description="Pneumothorax identified")
    acute_coronary_syndrome: Optional[bool] = Field(None, description="Acute coronary syndrome diagnosed")
    heart_failure: Optional[bool] = Field(None, description="Heart failure (acute) diagnosed")
    acute_pulmonary_edema: Optional[bool] = Field(None, description="Acute pulmonary edema present")
    cardiac_tamponade: Optional[bool] = Field(None, description="Cardiac tamponade identified")
    aortic_dissection: Optional[bool] = Field(None, description="Aortic dissection diagnosed")
    arrhythmia: Optional[bool] = Field(None, description="Arrhythmia identified")
    severe_anemia: Optional[bool] = Field(None, description="Severe anemia present")
    intoxication: Optional[bool] = Field(None, description="Intoxication identified")
    respiratory_failure: Optional[bool] = Field(None, description="Respiratory failure diagnosed")
    asthma_exacerbation: Optional[bool] = Field(None, description="Asthma exacerbation")
    copd_exacerbation: Optional[bool] = Field(None, description="COPD exacerbation")
    covid_19: Optional[bool] = Field(None, description="COVID-19 diagnosed")


class Treatments(BaseModel):
    """Treatments, interventions, and medications administered (14 fields)."""

    administration_of_oxygen_ventilation: Optional[bool] = Field(None, description="Oxygen or ventilation administered")
    administration_of_bronchodilators: Optional[bool] = Field(None, description="Bronchodilators administered")
    administration_of_diuretics: Optional[bool] = Field(None, description="Diuretics administered")
    administration_of_fluids: Optional[bool] = Field(None, description="IV fluids administered")
    administration_of_steroids: Optional[bool] = Field(None, description="Steroids administered")
    blood_transfusions: Optional[bool] = Field(None, description="Blood transfusion performed")
    cardio_pulmonary_resuscitation: Optional[bool] = Field(None, description="CPR performed")
    performance_of_thoracentesis: Optional[bool] = Field(None, description="Thoracentesis performed")
    palliative_care: Optional[bool] = Field(None, description="Palliative care initiated")
    antihypertensive_therapy: Optional[bool] = Field(None, description="On antihypertensive therapy")
    anticoagulants_or_antiplatelet: Optional[bool] = Field(None, description="On anticoagulant or antiplatelet therapy")
    antiepileptic_therapy: Optional[bool] = Field(None, description="Antiepileptic therapy already in place")
    poly_pharmacological_therapy: Optional[bool] = Field(None, description="Poly-pharmacological therapy (5+ medications)")
    compliance_with_antiepileptic_therapy: Optional[bool] = Field(None, description="Compliant with antiepileptic therapy")


class EpilepsyAssessment(BaseModel):
    """Epilepsy / seizure-specific assessment (11 fields)."""

    tonic_clonic_seizures: Optional[bool] = Field(None, description="Tonic-clonic seizures observed")
    further_seizures_in_ed: Optional[bool] = Field(None, description="Further seizures occurred in the ED")
    first_episode_of_epilepsy: Optional[bool] = Field(None, description="First ever epileptic episode")
    stiffness_during_episode: Optional[bool] = Field(None, description="Stiffness during the episode")
    eye_deviation_during_episode: Optional[bool] = Field(None, description="Eye deviation during the episode")
    pale_skin_during_episode: Optional[bool] = Field(None, description="Pale skin during the episode")
    drooling_during_episode: Optional[bool] = Field(None, description="Drooling during the episode")
    tongue_bite: Optional[bool] = Field(None, description="Tongue bite present")
    drowsiness_confusion_postcritical: Optional[bool] = Field(None, description="Drowsiness/confusion/disorientation as postcritical state")
    duration_of_unconsciousness: Optional[str] = Field(None, description="Duration of patient's unconsciousness")
    duration_of_consciousness_recovery: Optional[str] = Field(None, description="Duration of consciousness recovery")


class SyncopeAssessment(BaseModel):
    """Syncope-specific assessment (7 fields)."""

    situational_syncope: Optional[bool] = Field(None, description="Situational syncope identified")
    tloc_during_effort: Optional[bool] = Field(None, description="Transient loss of consciousness during effort")
    tloc_while_supine: Optional[bool] = Field(None, description="Transient loss of consciousness while supine")
    supine_to_standing_bp_test: Optional[bool] = Field(None, description="Supine-to-standing systolic BP test performed")
    carotid_sinus_massage: Optional[bool] = Field(None, description="Carotid sinus massage performed")
    presence_of_prodromal_symptoms: Optional[bool] = Field(None, description="Prodromal symptoms present before event")
    situation_description: Optional[str] = Field(None, description="Situation description (coughing, straining, sudden abdominal pain, phlebotomy, etc.)")


class InfectionScreening(BaseModel):
    """Infection and screening results (3 fields)."""

    influenza_and_infections: Optional[bool] = Field(None, description="Influenza or various infections identified")
    sars_cov2_swab_test: Optional[bool] = Field(None, description="SARS-CoV-2 swab test result (True=positive)")
    neurologist_consultation: Optional[bool] = Field(None, description="Neurologist consultation requested/performed")


class SocialContext(BaseModel):
    """Social and living context (4 fields)."""

    living_alone: Optional[bool] = Field(None, description="Patient lives alone")
    homelessness: Optional[bool] = Field(None, description="Patient is homeless")
    need_but_absence_of_caregiver: Optional[bool] = Field(None, description="Needs a caregiver but has none")
    problematic_family_context: Optional[bool] = Field(None, description="Problematic family context identified")


class Devices(BaseModel):
    """Implanted cardiac devices (2 fields)."""

    presence_of_pacemaker: Optional[bool] = Field(None, description="Patient has a pacemaker")
    presence_of_defibrillator: Optional[bool] = Field(None, description="Patient has an implanted defibrillator")


class Outcome(BaseModel):
    """Patient outcome during ED stay (2 fields)."""

    improvement_of_dyspnea: Optional[bool] = Field(None, description="Dyspnea improved during stay")
    improvement_of_patient_conditions: Optional[bool] = Field(None, description="Overall patient condition improved")


# ── Main output model ────────────────────────────────────────────────────────

class CRFOutput(BaseModel):
    """
    Complete CRF (Case Report Form) output for a single clinical note.

    Extracted from emergency department clinical notes for dyspnea patients.
    134 total fields across 14 categories. None = unknown / not mentioned.
    """

    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    vital_signs: VitalSigns = Field(default_factory=VitalSigns)
    lab_values: LabValues = Field(default_factory=LabValues)
    arterial_blood_gas: ArterialBloodGas = Field(default_factory=ArterialBloodGas)
    imaging_and_diagnostics: ImagingAndDiagnostics = Field(default_factory=ImagingAndDiagnostics)
    current_presentation: CurrentPresentation = Field(default_factory=CurrentPresentation)
    acute_diagnoses: AcuteDiagnoses = Field(default_factory=AcuteDiagnoses)
    treatments: Treatments = Field(default_factory=Treatments)
    epilepsy_assessment: EpilepsyAssessment = Field(default_factory=EpilepsyAssessment)
    syncope_assessment: SyncopeAssessment = Field(default_factory=SyncopeAssessment)
    infection_screening: InfectionScreening = Field(default_factory=InfectionScreening)
    social_context: SocialContext = Field(default_factory=SocialContext)
    devices: Devices = Field(default_factory=Devices)
    outcome: Outcome = Field(default_factory=Outcome)


# ── Mapping: dataset key → schema field path ────────────────────────────────

ANNOTATION_KEY_TO_FIELD: dict[str, tuple[str, str]] = {
    # medical_history
    "chronic pulmonary disease": ("medical_history", "chronic_pulmonary_disease"),
    "chronic respiratory failure": ("medical_history", "chronic_respiratory_failure"),
    "chronic cardiac failure": ("medical_history", "chronic_cardiac_failure"),
    "chronic renal failure": ("medical_history", "chronic_renal_failure"),
    "chronic metabolic failure": ("medical_history", "chronic_metabolic_failure"),
    "chronic rheumatologic disease": ("medical_history", "chronic_rheumatologic_disease"),
    "chronic dialysis": ("medical_history", "chronic_dialysis"),
    "active neoplasia": ("medical_history", "active_neoplasia"),
    "cardiovascular diseases": ("medical_history", "cardiovascular_diseases"),
    "diffuse vascular disease": ("medical_history", "diffuse_vascular_disease"),
    "dementia": ("medical_history", "dementia"),
    "neurodegenerative diseases": ("medical_history", "neurodegenerative_diseases"),
    "neuropsychiatric disorders": ("medical_history", "neuropsychiatric_disorders"),
    "peripheral neuropathy": ("medical_history", "peripheral_neuropathy"),
    "epilepsy / epileptic seizure": ("medical_history", "epilepsy"),
    "known history of epilepsy": ("medical_history", "known_history_of_epilepsy"),
    "history of alcohol abuse": ("medical_history", "history_of_alcohol_abuse"),
    "history of drug abuse": ("medical_history", "history_of_drug_abuse"),
    "history of allergy": ("medical_history", "history_of_allergy"),
    "history of recent trauma": ("medical_history", "history_of_recent_trauma"),
    "immunosuppression": ("medical_history", "immunosuppression"),
    "pregnancy": ("medical_history", "pregnancy"),
    # vital_signs
    "blood pressure": ("vital_signs", "blood_pressure"),
    "heart rate": ("vital_signs", "heart_rate"),
    "body temperature": ("vital_signs", "body_temperature"),
    "respiratory rate": ("vital_signs", "respiratory_rate"),
    "spo2": ("vital_signs", "spo2"),
    "level of consciousness": ("vital_signs", "level_of_consciousness"),
    "level of autonomy (mobility)": ("vital_signs", "level_of_autonomy"),
    # lab_values
    "hemoglobin": ("lab_values", "hemoglobin"),
    "leukocytes": ("lab_values", "leukocytes"),
    "platelets": ("lab_values", "platelets"),
    "creatinine": ("lab_values", "creatinine"),
    "blood glucose": ("lab_values", "blood_glucose"),
    "blood potassium": ("lab_values", "blood_potassium"),
    "blood sodium": ("lab_values", "blood_sodium"),
    "blood calcium": ("lab_values", "blood_calcium"),
    "c-reactive protein": ("lab_values", "c_reactive_protein"),
    "d-dimer": ("lab_values", "d_dimer"),
    "troponin": ("lab_values", "troponin"),
    "bnp or nt-pro-bnp": ("lab_values", "bnp_or_nt_pro_bnp"),
    "transaminases": ("lab_values", "transaminases"),
    "serum creatinine kinase": ("lab_values", "serum_creatinine_kinase"),
    "inr": ("lab_values", "inr"),
    "lactates": ("lab_values", "lactates"),
    "blood alcohol": ("lab_values", "blood_alcohol"),
    "blood drug dosage": ("lab_values", "blood_drug_dosage"),
    "urine drug test": ("lab_values", "urine_drug_test"),
    # arterial_blood_gas
    "ph": ("arterial_blood_gas", "ph"),
    "pa02": ("arterial_blood_gas", "pao2"),
    "pac02": ("arterial_blood_gas", "paco2"),
    "hc03-": ("arterial_blood_gas", "hco3"),
    # imaging_and_diagnostics
    "chest rx, any abnormalities": ("imaging_and_diagnostics", "chest_rx_abnormality"),
    "chest ct scan, any abnormality": ("imaging_and_diagnostics", "chest_ct_abnormality"),
    "abdomen ct scan, any abnormality": ("imaging_and_diagnostics", "abdomen_ct_abnormality"),
    "brain ct scan, any abnormality": ("imaging_and_diagnostics", "brain_ct_abnormality"),
    "brain mri, any abnormality": ("imaging_and_diagnostics", "brain_mri_abnormality"),
    "cardiac ultrasound, any abnormality": ("imaging_and_diagnostics", "cardiac_ultrasound_abnormality"),
    "thoracic ultrasound, any abnormalities": ("imaging_and_diagnostics", "thoracic_ultrasound_abnormality"),
    "compression ultrasound (cus), any abnormality": ("imaging_and_diagnostics", "compression_ultrasound_abnormality"),
    "ecg, any abnormality": ("imaging_and_diagnostics", "ecg_abnormality"),
    "ecg monitoring, any abnormality": ("imaging_and_diagnostics", "ecg_monitoring_abnormality"),
    "eeg, any abnormality": ("imaging_and_diagnostics", "eeg_abnormality"),
    "pulmonary scintigraphy, any abnormality": ("imaging_and_diagnostics", "pulmonary_scintigraphy_abnormality"),
    "gastroscopy , any abnormalities": ("imaging_and_diagnostics", "gastroscopy_abnormality"),
    # current_presentation
    "presence of dyspnea": ("current_presentation", "presence_of_dyspnea"),
    "presence of respiratory distress": ("current_presentation", "presence_of_respiratory_distress"),
    "chest pain": ("current_presentation", "chest_pain"),
    "agitation": ("current_presentation", "agitation"),
    "blood in the stool": ("current_presentation", "blood_in_the_stool"),
    "foreign body in the airways": ("current_presentation", "foreign_body_in_airways"),
    "general condition deterioration": ("current_presentation", "general_condition_deterioration"),
    "head or other districts trauma": ("current_presentation", "head_or_other_trauma"),
    "hemorrhage": ("current_presentation", "hemorrhage"),
    "concussive head trauma": ("current_presentation", "concussive_head_trauma"),
    # acute_diagnoses
    "pneumonia": ("acute_diagnoses", "pneumonia"),
    "ab ingestis pneumonia": ("acute_diagnoses", "ab_ingestis_pneumonia"),
    "pulmonary embolism": ("acute_diagnoses", "pulmonary_embolism"),
    "pneumothorax": ("acute_diagnoses", "pneumothorax"),
    "acute coronary syndrome": ("acute_diagnoses", "acute_coronary_syndrome"),
    "heart failure": ("acute_diagnoses", "heart_failure"),
    "acute pulmonary edema": ("acute_diagnoses", "acute_pulmonary_edema"),
    "cardiac tamponade": ("acute_diagnoses", "cardiac_tamponade"),
    "aortic dissection": ("acute_diagnoses", "aortic_dissection"),
    "arrhythmia": ("acute_diagnoses", "arrhythmia"),
    "severe anemia": ("acute_diagnoses", "severe_anemia"),
    "intoxication": ("acute_diagnoses", "intoxication"),
    "respiratory failure": ("acute_diagnoses", "respiratory_failure"),
    "asthma exacerbation": ("acute_diagnoses", "asthma_exacerbation"),
    "copd exacerbation": ("acute_diagnoses", "copd_exacerbation"),
    "covid 19": ("acute_diagnoses", "covid_19"),
    # treatments
    "administration of oxygen/ventilation": ("treatments", "administration_of_oxygen_ventilation"),
    "administration of bronchodilators": ("treatments", "administration_of_bronchodilators"),
    "administration of diuretics": ("treatments", "administration_of_diuretics"),
    "administration of fluids": ("treatments", "administration_of_fluids"),
    "administration of steroids": ("treatments", "administration_of_steroids"),
    "blood transfusions": ("treatments", "blood_transfusions"),
    "cardio-pulmonary resuscitation": ("treatments", "cardio_pulmonary_resuscitation"),
    "performance of thoracentesis": ("treatments", "performance_of_thoracentesis"),
    "palliative care": ("treatments", "palliative_care"),
    "antihypertensive therapy": ("treatments", "antihypertensive_therapy"),
    "anticoagulants or antiplatelet drug therapy": ("treatments", "anticoagulants_or_antiplatelet"),
    "antiepileptic therapy already in place": ("treatments", "antiepileptic_therapy"),
    "poly-pharmacological therapy": ("treatments", "poly_pharmacological_therapy"),
    "compliance with antiepileptic therapy": ("treatments", "compliance_with_antiepileptic_therapy"),
    # epilepsy_assessment
    "tonic-clonic seizures": ("epilepsy_assessment", "tonic_clonic_seizures"),
    "further seizures in the ed": ("epilepsy_assessment", "further_seizures_in_ed"),
    "first episod of epilepsy": ("epilepsy_assessment", "first_episode_of_epilepsy"),
    "stiffness during the episode": ("epilepsy_assessment", "stiffness_during_episode"),
    "eye deviation during the episode": ("epilepsy_assessment", "eye_deviation_during_episode"),
    "pale skin during the episode": ("epilepsy_assessment", "pale_skin_during_episode"),
    "drooling during the episode": ("epilepsy_assessment", "drooling_during_episode"),
    "tongue bite": ("epilepsy_assessment", "tongue_bite"),
    "drowsiness, confusion, disorientation as postcritical state": ("epilepsy_assessment", "drowsiness_confusion_postcritical"),
    "duration of the patient's unconsciousness": ("epilepsy_assessment", "duration_of_unconsciousness"),
    "duration of the patient's consciousness recovery": ("epilepsy_assessment", "duration_of_consciousness_recovery"),
    # syncope_assessment
    "situational syncope": ("syncope_assessment", "situational_syncope"),
    "tloc during effort": ("syncope_assessment", "tloc_during_effort"),
    "tloc while supine": ("syncope_assessment", "tloc_while_supine"),
    "supine-to-standing systolic blood pressure test": ("syncope_assessment", "supine_to_standing_bp_test"),
    "carotid sinus massage": ("syncope_assessment", "carotid_sinus_massage"),
    "presence of prodromal symptoms": ("syncope_assessment", "presence_of_prodromal_symptoms"),
    "situation description, like coughing, prolonged periods of straining, sudden abdominal pain, phlebotomy": ("syncope_assessment", "situation_description"),
    # infection_screening
    "influenza and various infections": ("infection_screening", "influenza_and_infections"),
    "sars-cov-2 swab test": ("infection_screening", "sars_cov2_swab_test"),
    "neurologist consultation": ("infection_screening", "neurologist_consultation"),
    # social_context
    "living alone": ("social_context", "living_alone"),
    "homelessness": ("social_context", "homelessness"),
    "need but absence of a caregiver": ("social_context", "need_but_absence_of_caregiver"),
    "problematic family context": ("social_context", "problematic_family_context"),
    # devices
    "presence of pacemaker": ("devices", "presence_of_pacemaker"),
    "presence of defibrillator": ("devices", "presence_of_defibrillator"),
    # outcome
    "improvement of dyspnea": ("outcome", "improvement_of_dyspnea"),
    "improvement of patient\u2019s conditions": ("outcome", "improvement_of_patient_conditions"),
}


def _get_field_type(group: str, field: str) -> type:
    """Get the inner type of a schema field (unwrapping Optional)."""
    group_cls = globals()[_group_class_name(group)]
    annotation = group_cls.model_fields[field].annotation
    # Optional[X] is Union[X, None] — grab X
    args = getattr(annotation, "__args__", None)
    if args:
        return args[0]
    return annotation


def ground_truth_to_field_value(annotation_key: str, ground_truth: str):
    """Convert a raw dataset ground_truth string to the typed schema value."""
    if ground_truth == "unknown":
        return None

    mapping = ANNOTATION_KEY_TO_FIELD.get(annotation_key)
    if mapping is None:
        return ground_truth

    group, field = mapping
    field_type = _get_field_type(group, field)

    # For string fields, return the raw value as-is (lab values, durations, etc.)
    if field_type is str:
        return ground_truth

    # Binary y/n → bool
    if ground_truth == "y":
        return True
    if ground_truth == "n":
        return False
    # "current" for respiratory distress → True
    if ground_truth == "current":
        return True

    # Enum values — return the string directly, Pydantic will coerce
    return ground_truth


def dataset_row_to_crf(annotations: list[dict]) -> CRFOutput:
    """Convert a dataset row's annotations list into a CRFOutput instance."""
    groups: dict[str, dict] = {}

    for ann in annotations:
        key = ann["item"]
        gt = ann["ground_truth"]
        mapping = ANNOTATION_KEY_TO_FIELD.get(key)
        if mapping is None:
            continue
        group, field = mapping
        value = ground_truth_to_field_value(key, gt)
        if group not in groups:
            groups[group] = {}
        groups[group][field] = value

    return CRFOutput(**{g: globals()[_group_class_name(g)](**fields) for g, fields in groups.items()})


def _group_class_name(group_key: str) -> str:
    """Convert group key like 'medical_history' to class name 'MedicalHistory'."""
    return "".join(w.capitalize() for w in group_key.split("_"))


# Fields where True maps to a non-"y" ground_truth value
_BOOL_TRUE_OVERRIDES: dict[tuple[str, str], str] = {
    ("current_presentation", "presence_of_respiratory_distress"): "current",
}

def crf_to_annotations(crf: CRFOutput) -> list[dict]:
    """Convert a CRFOutput back to the dataset annotation format."""
    # Build reverse mapping
    field_to_key = {v: k for k, v in ANNOTATION_KEY_TO_FIELD.items()}
    annotations = []

    for group_name, (_cls_ignore) in [
        ("medical_history", MedicalHistory),
        ("vital_signs", VitalSigns),
        ("lab_values", LabValues),
        ("arterial_blood_gas", ArterialBloodGas),
        ("imaging_and_diagnostics", ImagingAndDiagnostics),
        ("current_presentation", CurrentPresentation),
        ("acute_diagnoses", AcuteDiagnoses),
        ("treatments", Treatments),
        ("epilepsy_assessment", EpilepsyAssessment),
        ("syncope_assessment", SyncopeAssessment),
        ("infection_screening", InfectionScreening),
        ("social_context", SocialContext),
        ("devices", Devices),
        ("outcome", Outcome),
    ]:
        sub_model = getattr(crf, group_name)
        for field_name, value in sub_model:
            dataset_key = field_to_key.get((group_name, field_name))
            if dataset_key is None:
                continue
            field_key = (group_name, field_name)
            if value is None:
                gt = "unknown"
            elif isinstance(value, bool) and value and field_key in _BOOL_TRUE_OVERRIDES:
                gt = _BOOL_TRUE_OVERRIDES[field_key]
            elif isinstance(value, bool) and not value and field_key in _BOOL_TRUE_OVERRIDES:
                gt = "n"  # explicit no for fields with special True overrides
            elif isinstance(value, bool):
                gt = "y" if value else "n"
            elif hasattr(value, "value"):  # Enum
                gt = value.value
            else:
                gt = str(value)
            annotations.append({"item": dataset_key, "ground_truth": gt})

    return annotations
