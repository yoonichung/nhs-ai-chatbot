from langchain_google_genai import ChatGoogleGenerativeAI
import config


def get_llm(is_judge: bool = False):
    kwargs = {"thinking_level": "low", "max_tokens": 500} if is_judge else {}

    # max_retries=1 -> 1 attempt total, 0 retries, so it falls back immediately on 429
    primary = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        max_retries=1,
        **kwargs,
    )
    fallback = ChatGoogleGenerativeAI(
        model=config.GEMINI_FALLBACK_MODEL,
        max_retries=2,  # Allow 1 retry on the fallback in case of a brief per-minute blip
        **kwargs,
    )

    return primary.with_fallbacks([fallback])
