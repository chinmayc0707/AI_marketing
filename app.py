import json
from flask import Flask, render_template, request, Response, jsonify
from config import get_env_model, get_api_key, get_masked_api_key, update_api_key
from llm import stream_marketing_content, stream_raw_prompt, build_prompt

app = Flask(__name__)

def _extract_request_payload(data: dict) -> tuple[str, dict]:
    """Helper to extract tool_type and formatted inputs from request JSON."""
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
    return tool_type, inputs

@app.route('/')
def index():
    model = get_env_model()
    has_api_key, api_key_masked = get_masked_api_key()
    return render_template(
        'index.html',
        active_model=model,
        has_api_key=has_api_key,
        api_key_masked=api_key_masked
    )

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json() or {}
    tool_type, inputs = _extract_request_payload(data)

    def generate_stream():
        for chunk in stream_marketing_content(tool_type, inputs):
            payload = json.dumps({"content": chunk})
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')

@app.route('/preview-prompt', methods=['POST'])
def preview_prompt():
    data = request.get_json() or {}
    tool_type, inputs = _extract_request_payload(data)

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
    
    if api_key:
        update_api_key(api_key)
        
    current_model = get_env_model()
    has_api_key = bool(get_api_key())
    
    return jsonify({
        "status": "success",
        "active_model": current_model,
        "has_api_key": has_api_key
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
