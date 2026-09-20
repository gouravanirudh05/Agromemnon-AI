"""Loads the model every agent shares.

Bedrock is the primary and Gemini is the fallback, wired together with Strands'
`ModelRouter`. The router asks its strategy for a candidate before the first call
and again after any failed one, so a Bedrock throttle, outage or model error moves
the turn onto Gemini mid-conversation instead of losing it. `FallbackStrategy` is
ordered failover until a candidate starts failing repeatedly, after which the
healthier one is preferred and a success re-arms both.

Bedrock leads because the rest of the system already lives in this account: the
call is IAM-authorized by the runtime's execution role rather than a shared API
key, the farmer's question never leaves AWS, and it is billed with everything
else. Gemini stays because a single-provider agent is down when that provider is.

**Why Nova and not Claude.** Claude on Bedrock needs the Anthropic use-case
details form submitted for the account, and account 222758971755 has not
submitted it — every `us.anthropic.*` id returns ResourceNotFoundException
("Model use case details have not been submitted"). Nova needs no form and is
invocable today. Once the form is approved, switching is a `TEXT_MODEL_ID`
change on the runtime plus the matching resource in
policies/bedrock-text-model.json — no code change.

The same pairing carries the photo path. `load_vision_model()` builds the same
Bedrock-then-Gemini router on a vision-capable model, so plant_doctor survives a
Bedrock outage the way the text agents do. Strands normalises an image content
block onto both providers, so one message shape serves both.

The Gemini key is resolved by `secret_store`, from `GEMINI_API_KEY` locally or the
Secrets Manager secret named by `GEMINI_API_KEY_SECRET` when deployed.

A missing or unreadable Gemini key is no longer fatal. It once was: this module
read `os.environ["GEMINI_API_KEY"]` at import time while the deploy config never
set it, so a deployed container raised KeyError on its first import and the
runtime failed before any farmer question reached it. Now that Gemini is the
fallback rather than the only model, losing it costs redundancy, not the answer —
so the key is resolved defensively and its absence is logged and carried on from,
matching how every other AWS-backed capability here degrades.
"""

import logging
import os

from strands.models.bedrock import BedrockModel
from strands.models.gemini import GeminiModel
from strands.models.routing import ModelRouter, RoutingCandidate

import secret_store

logger = logging.getLogger(__name__)

# Nova Pro rather than Nova Lite: the orchestrator's whole job is choosing which
# specialists a question belongs to and calling several of them in one turn, and
# the lighter model picks a single specialist for questions that span two.
TEXT_MODEL_ID = os.environ.get("TEXT_MODEL_ID", "openai.gpt-5.5")
TEXT_MODEL_REGION = (
    os.environ.get("TEXT_MODEL_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
)
GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-3.1-flash-lite")


def _api_key() -> str | None:
    """The Gemini key, or None when this deployment has not been given one."""
    key = secret_store.resolve("GEMINI_API_KEY")
    if key is None:
        logger.warning(
            "No Gemini credentials; running on %s with no fallback. Set GEMINI_API_KEY "
            "for local runs, or GEMINI_API_KEY_SECRET to a Secrets Manager secret name "
            "for a deployed runtime.",
            TEXT_MODEL_ID,
        )
    return key


def _gemini(model_id: str, params: dict | None = None) -> GeminiModel | None:
    """A Gemini model on this deployment's key, or None when there is no key."""
    api_key = _api_key()
    if api_key is None:
        return None
    config = {"model_id": model_id, "client_args": {"api_key": api_key}}
    if params:
        config["params"] = params
    return GeminiModel(**config)


def _with_fallback(bedrock: BedrockModel, gemini: GeminiModel | None):
    """Bedrock in front, Gemini behind it, or the bare Bedrock model when no key.

    A router of one candidate would only add a failover that cannot fire.
    """
    if gemini is None:
        return bedrock
    # Named so the routing logs say which provider a turn switched to, rather than
    # printing a model id the on-call reader has to recognise.
    return ModelRouter(
        [
            RoutingCandidate(model=bedrock, name="bedrock"),
            RoutingCandidate(model=gemini, name="gemini"),
        ]
    )


def load_model():
    """The text model for the orchestrator and every specialist.

    Returns a `ModelRouter` running Bedrock with Gemini behind it, or the bare
    Bedrock model when no Gemini key is available.

    Each call builds fresh model objects. The router tracks per-candidate health
    and refuses to route to one model instance twice, so conversations must not
    share instances; a router is, however, safe to share across the agents built
    for one conversation, which is what the orchestrator does.
    """
    return _with_fallback(
        BedrockModel(model_id=TEXT_MODEL_ID, region_name=TEXT_MODEL_REGION),
        _gemini(GEMINI_MODEL_ID),
    )


# Nova Pro reads images and is invocable by this account today, which is the whole
# reason it is the default: every `us.anthropic.*` id returns
# ResourceNotFoundException until the use-case form is submitted, and a vision
# agent pointed at a model it cannot call is a plant_doctor that never answers.
# Verified against Bedrock on 2026-09-19 — Nova Pro, Nova Lite, Llama 4 Maverick,
# Pixtral Large and Qwen3-VL all invoke; the Anthropic family does not.
VISION_MODEL_ID = os.environ.get("VISION_MODEL_ID", "openai.gpt-5.5")
VISION_MODEL_REGION = (
    os.environ.get("VISION_MODEL_REGION") or os.environ.get("AWS_REGION") or "us-east-1"
)
# Defaults to the text model's Gemini id so there is one key and one knob; override
# only to give the photo path a different Gemini model from the text path.
GEMINI_VISION_MODEL_ID = os.environ.get("GEMINI_VISION_MODEL_ID", GEMINI_MODEL_ID)

# The diagnosis plus treatment runs long in Hindi or Kannada, where the script
# costs more tokens per word than English. Set on both candidates so a failover
# does not silently truncate the half-written answer it is retrying.
VISION_MAX_TOKENS = 2048


def load_vision_model():
    """The model for the agents that have to look at a photograph.

    Separate from load_model() because the two jobs have different requirements.
    Reading a leaf photograph and deciding what the damage is, when a wrong call
    becomes a spray a farmer pays for, is the one judgement in this system where a
    weaker model is expensive.

    Gemini sits behind Bedrock here as it does for text. The photo is the farmer's
    own field, so keeping it inside AWS is worth something — but a plant_doctor
    that returns nothing because Bedrock is throttling is worth less, and Strands
    normalises the same image content block onto both providers, so the failover
    costs no extra code.
    """
    return _with_fallback(
        BedrockModel(
            model_id=VISION_MODEL_ID,
            region_name=VISION_MODEL_REGION,
            max_tokens=VISION_MAX_TOKENS,
        ),
        _gemini(GEMINI_VISION_MODEL_ID, {"max_output_tokens": VISION_MAX_TOKENS}),
    )
