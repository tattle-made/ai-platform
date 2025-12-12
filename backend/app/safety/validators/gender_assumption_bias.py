from enum import Enum
from guardrails import OnFailAction
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
    Validator
)
from pathlib import Path
from typing import Callable, ClassVar, List, Literal, Optional, Union, Annotated

import pandas
import re
from app.safety.validators.constants import GENDER_BIAS_LIST_FILENAME
from app.safety.validators.base_validator_config import BaseValidatorConfig

class BiasCategories(Enum):
    Generic = "generic"
    Healthcare = "healthcare"
    Education = "education"
    All = "all"

@register_validator(name="gender-assumption-bias", data_type="string")
class GenderAssumptionBias(Validator):
    """
    Validate text for the presence of gender assumption in LLM generated outputs.
    """

    def __init__(
        self, 
        categories: Optional[List[BiasCategories]] = [BiasCategories.All],
        on_fail: Optional[Callable] = OnFailAction.FIX
    ):
        self.gender_bias_list = self.load_gender_bias_list(categories)
        self.text = None
        super().__init__(on_fail=on_fail)

    def _validate(self, value: str, metadata: dict = None) -> ValidationResult:
        self.text = value
        detected_biased_words = []
        bias_check = False

        for entry in self.gender_bias_list:
            word = entry["word"]
            neutral_term = entry["neutral-term"]

            pattern = rf"\b{re.escape(word)}\b"

            if re.search(pattern, self.text, flags=re.IGNORECASE):
                detected_biased_words.append(word)

                self.text = re.sub(pattern, neutral_term, self.text, flags=re.IGNORECASE)
                bias_check = True

        if bias_check:
            return FailResult(
                error_message=f"Detected gender assumption bias: {detected_biased_words}",
                fix_value=self.text
            )

        return PassResult(value=self.text)

    def load_gender_bias_list(self, categories):
        BASE_DIR = Path(__file__).resolve().parent.parent  # goes up from validators/ to src/
        file_path = f"{BASE_DIR}/validators/gender_assumption_bias/{GENDER_BIAS_LIST_FILENAME}"
        neutral_term_col = 'neutral-term'
        gender_bias_list = []

        df = pandas.read_csv(file_path)
        df['word'] = df['word'].str.lower()
        df[neutral_term_col] = df[neutral_term_col].str.lower()

        for category in categories:
            if category == BiasCategories.All:
                temp = df
            else:
                temp = df[df['type'] == category]

            rows = temp.to_dict(orient="records")
            for row in rows:
                gender_bias_list.append({
                    "word": row["word"],
                    neutral_term_col: row[neutral_term_col]
                })
        return gender_bias_list

    
class GenderAssumptionBiasSafetyValidatorConfig(BaseValidatorConfig):
    type: Literal["gender_assumption_bias"]
    categories: Optional[List[BiasCategories]] = [BiasCategories.All]
    validator_cls: ClassVar = GenderAssumptionBias