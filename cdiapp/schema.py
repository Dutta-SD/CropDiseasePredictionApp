"""Output schema for the LLM.

The LLM returns one of two shapes (discriminated by `kind`):
  - plant_diagnosis: a triage report
  - not_a_plant:     refusal with a short reason

Validation lives at the wire boundary; everything downstream operates on typed
objects. A SchemaViolationError carries the raw response so the retry layer can
feed it back to the model for self-correction.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, ValidationError, conlist


class PlantDiagnosis(BaseModel):
    kind: Literal["plant_diagnosis"]
    disease: str = Field(min_length=1, max_length=120)
    confidence: Literal["High", "Medium", "Low"]
    symptoms: conlist(str, min_length=1, max_length=5)
    severity: Literal["Mild", "Moderate", "Severe", "Unknown"]
    severity_explanation: str = Field(min_length=1, max_length=400)
    next_steps: conlist(str, min_length=1, max_length=5)


class NotAPlant(BaseModel):
    kind: Literal["not_a_plant"]
    reason: str = Field(min_length=1, max_length=400)


DiagnosisOutput = Annotated[
    PlantDiagnosis | NotAPlant,
    Field(discriminator="kind"),
]


class SchemaViolationError(ValueError):
    """Raised when an LLM response cannot be parsed into DiagnosisOutput."""

    def __init__(self, raw: str, validation_error: Exception):
        self.raw = raw
        self.validation_error = validation_error
        super().__init__(f"LLM response did not match schema: {validation_error}")


__all__ = [
    "DiagnosisOutput",
    "NotAPlant",
    "PlantDiagnosis",
    "SchemaViolationError",
    "ValidationError",
]
