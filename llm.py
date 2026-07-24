from typing import Generator, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from config import SYSTEM_PROMPT, get_api_key, get_env_model, APP_TITLE, APP_REFERER

# Centralized Prompt Templates Registry
PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "ad_copy": PromptTemplate.from_template(
        "Act as an expert copywriter. Generate high-converting advertisement copy for {platform}.\n"
        "Brand/Product: {brand_name}\n"
        "Description: {description}\n"
        "Target Audience: {target_audience}\n"
        "Key Selling Points: {key_points}\n"
        "Tone: {tone}\n\n"
        "Provide 3 distinct ad variants. Format each variant inside a ``` markdown code block so it is easily copyable.\n"
        "Each variant should contain:\n"
        "- Headline\n"
        "- Primary Text\n"
        "- Call-to-Action (CTA)\n\n"
        "Make sure you don't add things like Headline, Primary text and CTA\n"
        "Separate the variants using --- horizontal dividers."
    ),
    "social_caption": PromptTemplate.from_template(
        "Generate 3 engaging social media captions for {platform}.\n"
        "Product/Topic: {brand_name} - {description}\n"
        "Key Details to Include: {key_points}\n"
        "Target Audience: {target_audience}\n"
        "Tone of Voice: {tone}\n\n"
        "Provide 3 distinct caption options. Wrap each option inside its own ``` markdown code block for easy one-click copying. Include emojis, readable line breaks, and hashtags.\n"
        "Separate the options using --- horizontal dividers."
    ),
    "email_campaign": PromptTemplate.from_template(
        "Write a high-converting email marketing campaign.\n"
        "Campaign Goal: {campaign_goal}\n"
        "Product/Brand: {brand_name}\n"
        "Product Details: {description}\n"
        "Target Audience: {target_audience}\n"
        "Key Offer/CTA: {key_points}\n"
        "Tone: {tone}\n\n"
        "Provide:\n"
        "- Subject Lines (wrap options in ``` code blocks)\n"
        "- Preview Text (wrap in ``` code blocks)\n"
        "--- \n"
        "- Complete Email Body with clear placeholders like [Name] and CTA button placements (wrap the body in a ``` code block)."
    ),
    "product_description": PromptTemplate.from_template(
        "Write a compelling, benefit-driven product description.\n"
        "Product Name: {brand_name}\n"
        "Product Features: {description}\n"
        "Target Customer: {target_audience}\n"
        "Unique Value Proposition: {key_points}\n"
        "Tone: {tone}\n\n"
        "Structure the output with:\n"
        "1. A catchy headline (in ``` block)\n"
        "2. A persuasive narrative paragraph (in ``` block)\n"
        "3. A bulleted list of core features & specs translated into user benefits.\n"
        "Separate sections using --- horizontal dividers."
    ),
    "slogan": PromptTemplate.from_template(
        "Generate 10 catchy, memorable promotional slogans or taglines.\n"
        "Brand/Product: {brand_name}\n"
        "Core Message: {description} (Value: {key_points})\n"
        "Target Audience: {target_audience}\n"
        "Tone: {tone}\n\n"
        "Provide 10 slogans. Wrap each slogan inside a markdown ``` code block so it can be copied individually with one click.\n"
        "Separate groups with --- horizontal dividers."
    ),
    "marketing_ideas": PromptTemplate.from_template(
        "Brainstorm 5 creative, out-of-the-box marketing ideas or campaign concepts.\n"
        "Product/Service: {brand_name}\n"
        "Product Details: {description}\n"
        "Target Audience: {target_audience}\n"
        "Marketing Goal: {campaign_goal} (Key Message: {key_points})\n"
        "Tone/Vibe: {tone}\n\n"
        "For each of the 5 ideas, provide:\n"
        "- Idea Title\n"
        "- Concept & Execution Plan (wrap actionable summary in a ``` code block)\n"
        "- Expected Impact\n"
        "Separate each idea with --- horizontal dividers."
    )
}

def get_llm_instance(provider: str = "openrouter") -> ChatOpenAI:
    """
    Factory function to instantiate LangChain LLM instances.
    Extensible to support other providers (e.g., 'openai', 'anthropic', 'google', 'ollama') in upcoming versions.
    """
    api_key = get_api_key()
    model = get_env_model()
    
    if not api_key:
        raise ValueError("**OPENROUTER_API_KEY** is not set in `.env`. Please add your key to `.env` or use the settings modal.")
    if not model:
        raise ValueError("**LLM_MODEL** is not set in `.env`. Please specify a model name in `.env` (e.g. `LLM_MODEL=google/gemma-4-31b-it:free`).")

    if provider == "openrouter":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            streaming=True,
            timeout=45.0,
            default_headers={
                "HTTP-Referer": APP_REFERER,
                "X-Title": APP_TITLE
            }
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def build_prompt(tool_type: str, inputs: Dict[str, Any]) -> Dict[str, str]:
    """
    Build and return system prompt and formatted user prompt using LangChain PromptTemplates.
    """
    if tool_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Invalid tool type '{tool_type}'")
    
    template = PROMPT_TEMPLATES[tool_type]
    formatted_user_prompt = template.format(
        brand_name=inputs.get("brand_name", ""),
        description=inputs.get("description", ""),
        target_audience=inputs.get("target_audience", ""),
        key_points=inputs.get("key_points", ""),
        tone=inputs.get("tone", "Professional"),
        platform=inputs.get("platform", "General"),
        campaign_goal=inputs.get("campaign_goal", "Brand Awareness")
    )
    
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": formatted_user_prompt
    }

def stream_llm_messages(messages: list, provider: str = "openrouter") -> Generator[str, None, None]:
    """
    Unified core generator to stream LLM responses using LangChain.
    """
    try:
        llm = get_llm_instance(provider=provider)
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
    except ValueError as ve:
        yield f"Error: {str(ve)}"
    except Exception as e:
        yield f"Connection Error: {str(e)}"

def stream_marketing_content(tool_type: str, inputs: Dict[str, Any], provider: str = "openrouter") -> Generator[str, None, None]:
    """
    Formulates prompt for specific tool_type and yields streaming chunks from LangChain.
    """
    try:
        prompt_data = build_prompt(tool_type, inputs)
    except ValueError as e:
        yield f"Error: {str(e)}"
        return

    messages = [
        SystemMessage(content=prompt_data["system_prompt"]),
        HumanMessage(content=prompt_data["user_prompt"])
    ]
    yield from stream_llm_messages(messages, provider=provider)

def stream_raw_prompt(raw_prompt: str, provider: str = "openrouter") -> Generator[str, None, None]:
    """
    Streams LangChain LLM response for a raw user-supplied prompt string.
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=raw_prompt)
    ]
    yield from stream_llm_messages(messages, provider=provider)
