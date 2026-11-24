from get_llm import get_llm
from query_data import query_rag

import pytest
from langchain_google_genai.chat_models import GoogleRateLimitError

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
Does the actual response match the expected response without contradicting it?
Respond ONLY with the single word 'true' or 'false'. Do not include any explanation.
"""


def query_and_validate(question: str, expected_response: str) -> bool:
    try:
        response_text, _sources = query_rag(question)

        prompt = EVAL_PROMPT.format(
            expected_response=expected_response, actual_response=response_text
        )

        judge_model = get_llm(is_judge=True)
        response = judge_model.invoke(prompt)
        response_text = response.text.lower()
    except GoogleRateLimitError:
        pytest.skip("Ran out of API tokens/quota on both primary and fallback models.")

    print(prompt)

    if "true" in response_text:
        print("\033[92m" + f"Response: {response_text}" + "\033[0m")
        return True
    elif "false" in response_text:
        print("\033[91m" + f"Response: {response_text}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false': {response_text}"
        )


# --- Example Positive & Negative Unit Tests ---
def test_surgery_fasting_rules():
    assert query_and_validate(
        question="What kind of anaesthetic is used for varicose vein surgery?",
        expected_response="General anaesthetic",
    )


def test_discomfort_timeline():
    assert query_and_validate(
        question="When is leg discomfort usually at its worst after varicose vein surgery, and when does it resolve?",
        expected_response="Symptoms are usually at their worst on the 8th to 10th day after the operation and usually resolve 12 to 14 days after the operation.",
    )


def test_dvt_and_exercise():
    assert query_and_validate(
        question="Should I just rest in bed after the surgery, or should I exercise?",
        expected_response="Regular daily exercise like walking or using an exercise bike is recommended because resting up raises the risk of deep vein thrombosis (DVT).",
    )


def test_alternative_options():
    assert query_and_validate(
        question="What should I do if I don't want a surgery for varicose veins?",
        expected_response="You don't have to get a treatment if you don't mind the appearance and they don't cause significant symptoms. You can wear stockings, get injections or laser",
    )


def test_stitches_removal_negative():
    # The PDF states the stitches lie beneath the skin and do NOT need to be removed.
    assert not query_and_validate(
        question="Do I need to come back to have my stitches removed after varicose vein surgery?",
        expected_response="Yes, you will need to return to the clinic to have your stitches removed.",
    )
