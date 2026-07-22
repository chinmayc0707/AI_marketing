import os
import json
import requests
from dotenv import load_dotenv

def build_prompt(tool_type, inputs):
    """
    Build and return the system prompt and formatted user prompt
    for a given tool_type and inputs dict.
    Returns a dict: { "system_prompt": str, "user_prompt": str }
    or raises ValueError for invalid tool_type.
    """
    system_prompt = (
        "You are an elite, high-converting growth marketer and copywriter.\n"
        "IMPORTANT FORMATTING INSTRUCTIONS:\n"
        "1. Put every final copy inside a markdown code block (using ```) so the user can easily copy individual quotes with one click.\n"
        "2. Separate distinct options or sections with horizontal line dividers (using ---)."
    )
    
    prompts = {
        "ad_copy": (
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
            "- Call-to-Action (CTA)\n\n""Make sure you don't add things like Headline, Primary text and CTA"
            "Separate the variants using --- horizontal dividers."
        ),
        "social_caption": (
            "Generate 3 engaging social media captions for {platform}.\n"
            "Product/Topic: {brand_name} - {description}\n"
            "Key Details to Include: {key_points}\n"
            "Target Audience: {target_audience}\n"
            "Tone of Voice: {tone}\n\n"
            "Provide 3 distinct caption options. Wrap each option inside its own ``` markdown code block for easy one-click copying. Include emojis, readable line breaks, and hashtags.\n"
            "Separate the options using --- horizontal dividers."
        ),
        "email_campaign": (
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
        "product_description": (
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
        "slogan": (
            "Generate 10 catchy, memorable promotional slogans or taglines.\n"
            "Brand/Product: {brand_name}\n"
            "Core Message: {description} (Value: {key_points})\n"
            "Target Audience: {target_audience}\n"
            "Tone: {tone}\n\n"
            "Provide 10 slogans. Wrap each slogan inside a markdown ``` code block so it can be copied individually with one click.\n"
            "Separate groups with --- horizontal dividers."
        ),
        "marketing_ideas": (
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
    
    if tool_type not in prompts:
        raise ValueError(f"Invalid tool type '{tool_type}'")
        
    formatted_prompt = prompts[tool_type].format(
        brand_name=inputs.get("brand_name", ""),
        description=inputs.get("description", ""),
        target_audience=inputs.get("target_audience", ""),
        key_points=inputs.get("key_points", ""),
        tone=inputs.get("tone", "Professional"),
        platform=inputs.get("platform", "General"),
        campaign_goal=inputs.get("campaign_goal", "Brand Awareness")
    )

    return {
        "system_prompt": system_prompt,
        "user_prompt": formatted_prompt
    }


def stream_marketing_content(tool_type, inputs):
    """
    Formulates a prompt for the specific tool_type and returns a generator
    yielding response chunks from OpenRouter. Strictly reads LLM_MODEL from .env.
    """
    # Always reload env variables so any user changes to .env take effect immediately
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip("'\"").strip()
    model = os.getenv("LLM_MODEL", "").strip("'\"").strip()
    
    if not api_key:
        yield "Error: **OPENROUTER_API_KEY** is not set in `.env`. Please add your key to `.env` or use the settings modal."
        return

    if not model:
        yield "Error: **LLM_MODEL** is not set in `.env`. Please specify a model name in `.env` (e.g. `LLM_MODEL=google/gemma-4-31b-it:free`)."
        return

    try:
        prompt_data = build_prompt(tool_type, inputs)
    except ValueError as e:
        yield f"Error: {str(e)}"
        return

    system_prompt = prompt_data["system_prompt"]
    formatted_prompt = prompt_data["user_prompt"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Airbnb Marketing Generator",
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ],
        "stream": True
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            stream=True,
            timeout=45
        )
        
        if response.status_code != 200:
            yield f"Error from OpenRouter API (Status {response.status_code}): {response.text}"
            return

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_content = line_str[6:]
                    if data_content.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_content)
                        text = chunk['choices'][0]['delta'].get('content', '')
                        if text:
                            yield text
                    except Exception:
                        pass
    except Exception as e:
        yield f"Connection Error: {str(e)}"


def stream_raw_prompt(raw_prompt):
    """
    Stream LLM response using a raw user-supplied prompt string.
    Uses the same system prompt as form-based generation.
    """
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip("'\"").strip()
    model = os.getenv("LLM_MODEL", "").strip("'\"").strip()
    
    if not api_key:
        yield "Error: **OPENROUTER_API_KEY** is not set in `.env`."
        return

    if not model:
        yield "Error: **LLM_MODEL** is not set in `.env`."
        return

    system_prompt = (
        "You are an elite, high-converting growth marketer and copywriter.\n"
        "IMPORTANT FORMATTING INSTRUCTIONS:\n"
        "1. Put every final copy inside a markdown code block (using ```) so the user can easily copy individual quotes with one click.\n"
        "2. Separate distinct options or sections with horizontal line dividers (using ---)."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Airbnb Marketing Generator",
    }

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_prompt}
        ],
        "stream": True
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            stream=True,
            timeout=45
        )
        
        if response.status_code != 200:
            yield f"Error from OpenRouter API (Status {response.status_code}): {response.text}"
            return

        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith("data: "):
                    data_content = line_str[6:]
                    if data_content.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_content)
                        text = chunk['choices'][0]['delta'].get('content', '')
                        if text:
                            yield text
                    except Exception:
                        pass
    except Exception as e:
        yield f"Connection Error: {str(e)}"
