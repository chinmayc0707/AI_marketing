import os
import json
from flask import Flask, render_template, request, Response, jsonify
from dotenv import load_dotenv, set_key
from llm import stream_marketing_content, build_prompt

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path, override=True)

app = Flask(__name__)

def get_current_env_model():
    load_dotenv(dotenv_path, override=True)
    return os.getenv("LLM_MODEL", "").strip("'\"").strip()

def get_current_api_key():
    load_dotenv(dotenv_path, override=True)
    return os.getenv("OPENROUTER_API_KEY", "").strip("'\"").strip()

@app.route('/')
def index():
    model = get_current_env_model()
    key = get_current_api_key()
    has_api_key = bool(key)
    api_key_masked = ""
    if has_api_key:
        if len(key) > 8:
            api_key_masked = key[:4] + "..." + key[-4:]
        else:
            api_key_masked = "********"
            
    return render_template(
        'index.html',
        active_model=model,
        has_api_key=has_api_key,
        api_key_masked=api_key_masked
    )

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json() or {}
    tool_type = data.get('tool_type', 'ad_copy')
    
    inputs = {
        'brand_name': data.get('brand_name', ''),
        'description': data.get('description', ''),
        'target_audience': data.get('target_audience', ''),
        'key_points': data.get('key_points', ''),
        'tone': data.get('tone', 'Professional'),
        'platform': data.get('platform', 'General'),
        'campaign_goal': data.get('campaign_goal', 'Brand Awareness')
    }

    def generate_stream():
        for chunk in stream_marketing_content(tool_type, inputs):
            payload = json.dumps({"content": chunk})
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')

@app.route('/preview-prompt', methods=['POST'])
def preview_prompt():
    data = request.get_json() or {}
    tool_type = data.get('tool_type', 'ad_copy')
    
    inputs = {
        'brand_name': data.get('brand_name', ''),
        'description': data.get('description', ''),
        'target_audience': data.get('target_audience', ''),
        'key_points': data.get('key_points', ''),
        'tone': data.get('tone', 'Professional'),
        'platform': data.get('platform', 'General'),
        'campaign_goal': data.get('campaign_goal', 'Brand Awareness')
    }

    try:
        prompt_data = build_prompt(tool_type, inputs)
        return jsonify({
            "system_prompt": prompt_data["system_prompt"],
            "user_prompt": prompt_data["user_prompt"]
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/generate-raw', methods=['POST'])
def generate_raw():
    """Stream LLM response from a raw user-supplied prompt string."""
    from llm import stream_raw_prompt
    data = request.get_json() or {}
    raw_prompt = data.get('prompt', '')

    if not raw_prompt.strip():
        return Response("data: {\"content\": \"Error: Empty prompt.\"}\n\ndata: [DONE]\n\n", mimetype='text/event-stream')

    def generate_stream():
        for chunk in stream_raw_prompt(raw_prompt):
            payload = json.dumps({"content": chunk})
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')

@app.route('/settings', methods=['POST'])
def update_settings():
    data = request.get_json() or {}
    api_key = data.get('api_key')
    
    if api_key is not None and api_key.strip():
        clean_key = api_key.strip().strip("'\"")
        os.environ['OPENROUTER_API_KEY'] = clean_key
        set_key(dotenv_path, 'OPENROUTER_API_KEY', clean_key)
        
    current_model = get_current_env_model()
    has_api_key = bool(get_current_api_key())
    
    return jsonify({
        "status": "success",
        "active_model": current_model,
        "has_api_key": has_api_key
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
