import os
from flask import Flask, jsonify, request
from google import genai
from google.genai import types
from google.genai.errors import APIError

app = Flask(__name__)

# 1. Grab your Gemini API key from environment variables
# Note: Set this in your PythonAnywhere environment or .env file
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_FALLBACK_KEY_IF_NOT_SET")

# 2. Configure the HTTP proxy options for PythonAnywhere's whitelist proxy
# The SDK uses httpx under the hood, so we pass the proxy into client_args
http_options = types.HttpOptions(
    client_args={'proxy': 'http://proxy.server:3128'},
    async_client_args={'proxy': 'http://proxy.server:3128'}
)

# 3. Initialize the GenAI client with proxy configurations
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=http_options
)

@app.route('/')
def home():
    return jsonify({"status": "healthy", "message": "Flask server running with Gemini SDK"}), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Expects a JSON payload: {"message": "Your prompt here"}
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' parameter in request body"}), 400
    
    user_prompt = data['message']
    
    try:
        # Generate text using the recommended default model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_prompt
        )
        
        return jsonify({
            "success": True,
            "response": response.text
        }), 200

    except APIError as e:
        # Catches Google API specific exceptions (bad keys, quota limits, etc.)
        app.logger.error(f"Gemini API Error: {str(e)}")
        return jsonify({"error": "Gemini service error", "details": str(e)}), 502
    except Exception as e:
        # Generic fallback error handler
        app.logger.error(f"Server Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred processing your request"}), 500

if __name__ == '__main__':
    # Used only for local development testing; PythonAnywhere uses WSGI
    app.run(debug=True)