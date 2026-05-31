"""Render typed DiagnosisOutput into the markdown shown in the chat UI.

Every plant_diagnosis response carries a permanent disclaimer footer reminding
the user this is a triage tool, not a treatment prescription.
"""

from __future__ import annotations

from cdiapp.schema import DiagnosisOutput, NotAPlant, PlantDiagnosis

DISCLAIMER = (
    "_This is a triage tool, not a treatment prescription. "
    "Confirm with your local extension officer (KVK) or a qualified agronomist "
    "before applying any chemical treatment._"
)


def render(output: DiagnosisOutput) -> str:
    if isinstance(output, NotAPlant):
        return f"I can only diagnose plant leaves. {output.reason}"
    if isinstance(output, PlantDiagnosis):
        return _render_diagnosis(output)
    raise TypeError(f"Unsupported output kind: {type(output).__name__}")


def _render_diagnosis(d: PlantDiagnosis) -> str:
    symptoms = "\n".join(f"- {s}" for s in d.symptoms)
    next_steps = "\n".join(f"{i}. {step}" for i, step in enumerate(d.next_steps, 1))

    return (
        f"### Diagnosis\n\n"
        f"**Likely:** {d.disease}\n"
        f"**Confidence:** {d.confidence}\n\n"
        f"### Symptoms\n\n"
        f"{symptoms}\n\n"
        f"### Severity\n\n"
        f"{d.severity} — {d.severity_explanation}\n\n"
        f"### Next steps\n\n"
        f"{next_steps}\n\n"
        f"---\n\n"
        f"{DISCLAIMER}"
    )
