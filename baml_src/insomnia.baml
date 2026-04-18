// ============================================================
// insomnia.baml — v2.1: Two-pass architecture + Definition 2 precision fix
//
// Pass 1 (this file): LLM extracts structured evidence only — no labels.
// Pass 2 (inference.py): derive_labels() computes all labels deterministically.
//
// v2.1 changelog:
//   - Added CRITICAL DEFINITION 2 EXTRACTION RULES section to fix
//     three over-extraction failure modes identified in validation:
//     1. Sleep complaints double-counted as Definition 2
//     2. Pre-existing psychiatric diagnoses counted as mood_disturbance
//     3. ICU/acute illness symptoms counted as daytime impairment
//
// Deprecated v1 single-pass functions are preserved in:
//   baml_src/insomnia_v1.baml.bak
// ============================================================

// ──────────────────────────────────────────────────────────────
// Schema for Pass 1 output
// ──────────────────────────────────────────────────────────────

class SleepDifficulty {
  criterion "trouble_initiating" | "trouble_maintaining" | "early_awakening" | "explicit_insomnia" @description("Which Definition 1 criterion this evidence supports")
  citation string @description("Verbatim minimal substring copied character-for-character from the note")
  match_regex string @description("A regex string that uniquely locates the citation in the note")
}

class DaytimeImpairment {
  criterion "fatigue_malaise" | "impaired_attention" | "impaired_performance" | "mood_disturbance" | "daytime_sleepiness" | "behavioral_problems" | "decreased_motivation" | "error_proneness" | "sleep_dissatisfaction" | "explicit_insomnia" @description("Which Definition 2 criterion this evidence supports")
  citation string @description("Verbatim minimal substring copied character-for-character from the note")
  match_regex string @description("A regex string that uniquely locates the citation in the note")
}

class MedicationMention {
  name string @description("Medication name exactly as it appears in the note (may include misspellings or brand variants)")
  normalized_name string @description("Canonical medication name, e.g. 'Zolpidem' even if note says 'zolpdem'")
  med_type "primary" | "secondary" @description("'primary' = Rule B medication list; 'secondary' = Rule C medication list")
  citation string @description("Verbatim substring from the note showing the medication mention in context")
}

class ClinicalNoteExtraction {
  sleep_difficulty SleepDifficulty[] @description("All Definition 1 (difficulty sleeping) criteria found; empty array if none")
  daytime_impairment DaytimeImpairment[] @description("All Definition 2 (daytime impairment) criteria found; empty array if none")
  medications MedicationMention[] @description("All primary or secondary insomnia medications mentioned in the note; empty array if none")
  reasoning string @description("One or two sentences describing any exclusions applied, or 'No exclusions applied'")
}

// ──────────────────────────────────────────────────────────────
// Pass 1 function: evidence extraction only (no label decisions)
// ──────────────────────────────────────────────────────────────

function ExtractClinicalEvidence(note_text: string, few_shot_context: string) -> ClinicalNoteExtraction {
  client CustomGemini
  prompt #"
    {{ _.role("system") }}
    You extract structured clinical evidence from discharge summaries for an insomnia detection task.
    Your job is EVIDENCE EXTRACTION ONLY — do not output insomnia labels or verdicts.
    Labels are derived deterministically in a separate post-processing step from your output.

    ═══════════════════════════════════════════════════════════════
    DEFINITION 1: DIFFICULTY SLEEPING  →  sleep_difficulty array
    ═══════════════════════════════════════════════════════════════

    Extract one entry per criterion instance found. Use these exact criterion keys:

      trouble_initiating   — trouble initiating sleep (can't fall asleep, difficulty falling asleep)
      trouble_maintaining  — trouble maintaining sleep (frequent awakenings, can't stay asleep)
      early_awakening      — waking earlier than desired
      explicit_insomnia    — explicit mention of "insomnia" as a sleep symptom

    ═══════════════════════════════════════════════════════════════
    DEFINITION 2: DAYTIME IMPAIRMENT  →  daytime_impairment array
    ═══════════════════════════════════════════════════════════════

    Extract one entry per criterion instance found. Use these exact criterion keys:

      fatigue_malaise        — fatigue or malaise
      impaired_attention     — impaired attention, concentration, or memory
      impaired_performance   — impaired social, family, occupational, or academic performance
      mood_disturbance       — mood disturbance or irritability
      daytime_sleepiness     — daytime sleepiness
      behavioral_problems    — behavioral problems (hyperactivity, impulsivity, aggression)
      decreased_motivation   — decreased motivation, energy, or initiative
      error_proneness        — proneness to errors or accidents
      sleep_dissatisfaction  — concerns or dissatisfaction with sleep
      explicit_insomnia      — explicit mention of "insomnia" as a daytime consequence

    ═══════════════════════════════════════════════════════════════
    MEDICATIONS  →  medications array
    ═══════════════════════════════════════════════════════════════

    Extract one entry per mention of a medication from either list below.
    Set med_type = "primary" for List B medications, "secondary" for List C medications.
    Omit all other medications entirely.

    PRIMARY insomnia medications (med_type = "primary") — Rule B:
      Estazolam, Eszopiclone (Lunesta), Flurazepam (Dalmane), Lemborexant (Dayvigo),
      Quazepam (Doral), Ramelteon (Rozerem), Suvorexant (Belsomra), Temazepam (Restoril),
      Triazolam (Halcion), Zaleplon (Sonata), Zolpidem (Ambien)

    SECONDARY insomnia medications (med_type = "secondary") — Rule C:
      Acamprosate (Campral), Alprazolam (Xanax), Clonazepam (Klonopin), Clonidine (Catapres),
      Diazepam (Valium), Diphenhydramine (Benadryl), Doxepin (Silenor), Gabapentin (Neurontin),
      Hydroxyzine (Vistaril/Atarax), Lorazepam (Ativan), Melatonin, Mirtazapine (Remeron),
      Olanzapine (Zyprexa), Quetiapine (Seroquel), Trazodone (Desyrel)

    Recognize brand names, generic names, and common misspellings:
    e.g., "zolpdem" → Zolpidem, "clozepam" → Clonazepam, "Zyprexia" → Olanzapine, "ambien" → Zolpidem.

    ═══════════════════════════════════════════════════════════════
    CRITICAL: DEFINITION 2 EXTRACTION RULES
    ═══════════════════════════════════════════════════════════════

    Definition 2 (daytime impairment) is the most commonly over-extracted category.
    Apply these three rules strictly — they take priority over any general extraction instinct:

    1. NO DOUBLE-COUNTING SLEEP COMPLAINTS
       Symptoms about difficulty sleeping (unable to sleep, trouble sleeping, can't
       fall asleep, has not slept, sleeplessness, complaining about sleep) are ONLY
       Definition 1 evidence. Do NOT also extract them as Definition 2 under
       sleep_dissatisfaction or any other criterion.
       Only extract a Definition 2 entry if the note describes a SEPARATE DAYTIME
       CONSEQUENCE of poor sleep — e.g., "couldn't sleep and was exhausted the next
       day" → Definition 1: "couldn't sleep", Definition 2: "exhausted the next day".
       The sleep complaint itself is never Definition 2 evidence.

    2. PRE-EXISTING PSYCHIATRIC CONDITIONS ≠ MOOD DISTURBANCE
       Depression, anxiety, bipolar disorder, PTSD, and similar diagnoses listed in
       past medical history, problem lists, admission diagnoses, or medication
       indication sections are comorbidities — NOT Definition 2 mood_disturbance.
       Only extract mood_disturbance if the note explicitly describes mood changes
       AS A RESULT OF or IN CONNECTION WITH sleep problems (e.g., "irritable due to
       poor sleep last night").

    3. ICU / ACUTE ILLNESS SYMPTOMS ≠ DAYTIME IMPAIRMENT
       In ICU and acute care notes, general symptoms of illness — lethargy,
       sleepiness, fatigue, low energy, napping, sedation — are expected features
       of being acutely ill, medicated, or post-operative.
       Do NOT extract these as Definition 2 evidence UNLESS the note explicitly
       attributes them to poor sleep or insomnia.
       EXCLUDE: "patient lethargic", "sleepy", "napping throughout the day",
                "falls asleep during conversation" (when no sleep problem is mentioned)
       INCLUDE: "patient lethargic, states he has not slept in 3 days"
                (lethargy explicitly linked to sleep deprivation)

    ═══════════════════════════════════════════════════════════════
    EXCLUSIONS — do NOT extract the following as evidence
    ═══════════════════════════════════════════════════════════════

    Definition 1 exclusions:
    • Sleep difficulty explicitly caused by pain, discomfort, or physical symptoms does NOT count.
      Excluded: "trouble sleeping due to pain", "unable to sleep because uncomfortable",
                "difficulty sleeping secondary to back pain"

    Definition 2 exclusions (overnight nursing shift notes):
    • If the note is from an overnight nursing shift, daytime impairment symptoms generally
      do NOT count — they may reflect normal nighttime behaviour rather than insomnia impairment.
    • EXCEPTIONS that still count even in overnight notes:
        – sleep_dissatisfaction (patient concerns or dissatisfaction with sleep quality)
        – explicit_insomnia (explicit mention of "insomnia")

    Medication ↔ symptom overlap:
    • Medications from the primary/secondary lists must NOT also be extracted as
      Definition 1 or 2 evidence.
    • EXCEPTION: A medication NOT on either list that is explicitly linked to sleep
      (e.g., "Serax for sleep") CAN be extracted as Definition 1 or 2 evidence.

    ═══════════════════════════════════════════════════════════════
    CITATION GUIDELINES
    ═══════════════════════════════════════════════════════════════

    • Citations must be verbatim substrings — copy exactly, character for character.
    • Select the SMALLEST span with complete meaning.
    • Include important modifiers: negations ("denies insomnia"), temporal cues ("last night"),
      quantities ("slept only 2 hours").
    • For patient-reported speech, extract content WITHOUT surrounding quotation marks.
    • match_regex: write a simple regex that uniquely locates this span in the note.
      For unambiguous tokens, re.escape(citation) is sufficient. Add \b word boundaries
      or short surrounding context only when the same token appears multiple times in
      different contexts.

    Return empty arrays for any category where no qualifying evidence is found.
    In the reasoning field, briefly note any exclusions you applied, or write "No exclusions applied."

    {% if few_shot_context %}
    ═══════════════════════════════════════════════════════════════
    EXAMPLES — similar clinical notes with correct extractions
    ═══════════════════════════════════════════════════════════════

    The following examples show correct evidence extraction on similar
    clinical notes. Use them to calibrate your extraction decisions —
    especially for what DOES and DOES NOT count as Definition 2 evidence.

    {{ few_shot_context }}
    {% endif %}

    {{ _.role("user") }}
    Clinical note:
    {{ note_text }}

    {{ ctx.output_format }}
  "#
}