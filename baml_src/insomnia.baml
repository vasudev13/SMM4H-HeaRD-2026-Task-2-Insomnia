// Insomnia shared-task: classification + evidence extraction (quotes, not raw spans).

class InsomniaBinary {
  insomnia "yes" | "no" @description("Whether the patient has insomnia per Rule A, Rule B, or Rule C in the task instructions")
}

class RuleEvidence {
  label "yes" | "no"
  evidence_quotes string[] @description("Verbatim substrings copied from the note that support the label; use empty list when label is no")
}

class InsomniaEvidenceExtraction {
  definition_1 RuleEvidence @description("Definition 1: Difficulty Sleeping")
  definition_2 RuleEvidence @description("Definition 2: Daytime Impairment")
  rule_b RuleEvidence @description("Rule B: primary insomnia medications")
  rule_c RuleEvidence @description("Rule C: secondary insomnia medications with Definition 1 or 2 symptoms")
}

function ClassifyInsomnia(note_text: string) -> InsomniaBinary {
  client CustomGemini
  prompt #"
    {{ _.role("system") }}
    You classify clinical notes for the insomnia detection shared task.

    Use these definitions and rules exactly:

    **Definition 1: Difficulty Sleeping**
    The patient is considered to have difficulty sleeping if they report any of the following:
    1. Trouble initiating sleep.
    2. Trouble maintaining sleep.
    3. Waking up earlier than desired.
    4. An explicit mention of insomnia.

    **Definition 2: Daytime Impairment**
    The patient is considered to have daytime impairment if they report any of the following:
    1. Fatigue or malaise.
    2. Impaired attention, concentration, or memory.
    3. Impaired social, family, occupational, or academic performance.
    4. Mood disturbance or irritability.
    5. Daytime sleepiness.
    6. Behavioral problems such as hyperactivity, impulsivity, or aggression.
    7. Decreased motivation, energy, or initiative.
    8. Proneness to errors or accidents.
    9. Concerns or dissatisfaction with sleep.
    10. An explicit mention of insomnia.

    **Rule A:** The patient is considered to have insomnia if they meet both Definition 1 (difficulty sleeping) and Definition 2 (daytime impairment).

    **Rule B**: The patient has insomnia if prescribed any of the following primary insomnia medications: Estazolam, Eszopiclone, Flurazepam, Lemborexant, Quazepam, Ramelteon, Suvorexant, Temazepam, Triazolam, Zaleplon, Zolpidem.

    **Rule C**: The patient has insomnia if prescribed any of the following secondary insomnia medications and reports any symptoms from Definition 1 (difficulty sleeping) or Definition 2 (daytime impairment): Acamprosate, Alprazolam, Clonazepam, Clonidine, Diazepam, Diphenhydramine, Doxepin, Gabapentin, Hydroxyzine, Lorazepam, Melatonin, Mirtazapine, Olanzapine, Quetiapine, Trazodone.

    **Insomnia status**: The patient is considered to have insomnia if they meet the criteria of Rule A, Rule B, or Rule C ("yes"). If none of these rules are met, label "no".

    {{ _.role("user") }}
    Clinical note:
    {{ note_text }}

    {{ ctx.output_format }}
  "#
}

function ExtractInsomniaEvidence(note_text: string) -> InsomniaEvidenceExtraction {
  client CustomGemini
  prompt #"
    {{ _.role("system") }}
    You extract evidence for insomnia sub-definitions from a clinical note.

    For each of the four blocks below, set label to "yes" or "no" according to the same rules as the task:

    **Definition 1: Difficulty Sleeping** — trouble initiating/maintaining sleep, early waking, or explicit insomnia.
    **Definition 2: Daytime Impairment** — fatigue, cognitive/social/mood issues, sleepiness, etc., as listed in the task, or explicit insomnia.
    **Rule B** — patient on a listed primary insomnia medication (Estazolam, Eszopiclone, Flurazepam, Lemborexant, Quazepam, Ramelteon, Suvorexant, Temazepam, Triazolam, Zaleplon, Zolpidem).
    **Rule C** — patient on a listed secondary medication AND symptoms from Definition 1 or 2.

    For each block, fill evidence_quotes with **verbatim** substrings copied from the note that justify the label. If label is "no", use an empty list. Do not invent text; quotes must appear exactly in the note (including whitespace where relevant).

    {{ _.role("user") }}
    Clinical note:
    {{ note_text }}

    {{ ctx.output_format }}
  "#
}
