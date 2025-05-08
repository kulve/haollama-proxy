# SPDX-FileCopyrightText: 2025 Tuomas Kulve
#
# SPDX-License-Identifier: MIT
"""
A Flask-based proxy server for Ollama.

This proxy server sits between a client application (e.g., Home Assistant)
and an Ollama instance. It allows for:
- Centralized configuration of the Ollama URL.
- Modification of requests and responses, such as removing empty <think> tags.
- Ensuring specific parameters like 'stream: false' for certain endpoints.
"""
import toml
import requests
import re
from flask import Flask, request, jsonify, Response
import json
import os

# Load configuration from TOML file
APP_NAME = "haollama-proxy"
CONFIG_FILE_NAME = "haollama-proxy.toml"

xdg_config_home = os.environ.get('XDG_CONFIG_HOME')
if xdg_config_home:
    CONFIG_DIR = os.path.join(xdg_config_home, APP_NAME)
else:
    CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".config", APP_NAME)

CONFIG_FILE = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)

# Ensure the configuration directory exists
os.makedirs(CONFIG_DIR, exist_ok=True)

# Try to load the config, or create a default one if it doesn't exist
try:
    config = toml.load(CONFIG_FILE)
except FileNotFoundError:
    print(f"Configuration file not found at {CONFIG_FILE}. Creating a default one.")
    # Define your default configuration here
    default_config = {
        "ollama": {"url": "http://localhost:11434"},
        "proxy": {"host": "0.0.0.0", "port": 12434}
        # Add other default settings as needed
    }
    with open(CONFIG_FILE, "w") as f:
        toml.dump(default_config, f)
    config = default_config
except Exception as e:
    print(f"Error loading configuration file {CONFIG_FILE}: {e}")
    # Fallback to some very basic defaults or exit
    config = {
        "ollama": {"url": "http://localhost:11434"},
        "proxy": {"host": "0.0.0.0", "port": 12434}
    }

OLLAMA_URL = config["ollama"]["url"]

app = Flask(__name__)

def query_ollama(prompt):
    """
    Sends a prompt to the Ollama /api/generate endpoint and returns the response.

    This function is not currently used by the main proxy routes but can be
    a utility for direct generation if needed.

    Args:
        prompt (str): The prompt to send to Ollama.

    Returns:
        str: The 'response' field from the Ollama JSON output, or an empty string.
    """
    data = {"prompt": prompt}
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=data)
    response.raise_for_status()
    return response.json().get("response", "")

def remove_empty_think_tags(text):
    """
    Removes <think>...</think> tags that only contain whitespace or newlines.

    Args:
        text (str): The input string.

    Returns:
        str: The string with empty <think> tags removed.
    """
    # Remove <think>...</think> tags that contain only whitespace/newlines
    return re.sub(r'<think>\s*</think>', '', text, flags=re.IGNORECASE | re.DOTALL)

@app.route('/api/chat', methods=["POST"])
def proxy_chat():
    """
    Proxies requests to the Ollama /api/chat endpoint, handling streaming.

    It forwards the request to the configured Ollama URL. If the Ollama
    response is streamed, this function processes each line (expected to be
    a JSON object), removes empty <think> tags from the message content,
    and streams back the modified NDJSON.

    Returns:
        flask.Response: A streaming response of NDJSON objects.
    """
    # Forward the request as-is (with stream true)
    headers = {key: value for key, value in request.headers if key.lower() != 'host'}
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        headers=headers,
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
        params=request.args,
        stream=True
    )
    def ndjson_stream():
        """Generates NDJSON lines from the Ollama stream."""
        for line_bytes in resp.iter_lines(): # iter_lines() yields bytes
            if not line_bytes:
                continue  # skip empty lines
            try:
                # line_bytes is a complete JSON object string from Ollama, as bytes.
                # json.loads can handle bytes (UTF-8, UTF-16, or UTF-32 encoded).
                data = json.loads(line_bytes)

                # Clean up <think> tags if present
                if "message" in data and "content" in data["message"]:
                    data["message"]["content"] = remove_empty_think_tags(data["message"]["content"])

                # Yield the modified JSON object as a UTF-8 encoded string, followed by a newline.
                yield (json.dumps(data) + "\n").encode('utf-8')
            except json.JSONDecodeError as e:
                line_str_for_log = line_bytes.decode('utf-8', errors='replace')
                print(f"Error parsing JSON line from Ollama: {e}. Line: '{line_str_for_log}'")
                # Do not yield anything on error, to avoid sending malformed data.
            except Exception as e:
                line_str_for_log = line_bytes.decode('utf-8', errors='replace')
                print(f"Generic error processing line from Ollama: {e}. Line: '{line_str_for_log}'")
                # Do not yield anything on error.

    return Response(ndjson_stream(), mimetype="application/x-ndjson; charset=utf-8")

@app.route('/<path:path>', methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
def proxy_all(path):
    """
    A catch-all proxy for other Ollama API endpoints.

    This forwards requests to any path not specifically handled by
    /api/chat or /api/generate. It adds an Authorization header if
    OLLAMA_API_KEY is configured.

    Args:
        path (str): The path of the request.

    Returns:
        flask.Response: The direct response from the Ollama server.
    """
    # Avoid double-handling /api/generate and /api/chat
    if path == "api/generate" and request.method == "POST":
        return proxy_generate()
    if path == "api/chat" and request.method == "POST":
        return proxy_chat()

    # Forward all other requests to the real Ollama server
    url = f"{OLLAMA_URL}/{path}"
    headers = {key: value for key, value in request.headers if key.lower() != 'host'}
    resp = requests.request(
        method=request.method,
        url=url,
        headers=headers,
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
        params=request.args
    )
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    response_headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
    return Response(resp.content, resp.status_code, response_headers)


if __name__ == "__main__":
    app.run(host=config["proxy"]["host"], port=config["proxy"]["port"])
