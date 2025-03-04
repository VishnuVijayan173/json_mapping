from flask import Flask, request, Response, jsonify, stream_with_context
import json
import requests
import re
from flask_cors import CORS
from typing import Dict, Any, List, Union
import logging
import re
import time
import tiktoken
import traceback

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://gen-ai-demo.openai.azure.com/openai/deployments/open-ai-gpt35/chat/completions?api-version=2023-07-01-preview"
AZURE_OPENAI_API_KEY = 'f39237474a6546c8ad3f14d3931ff7d7'

def call_azure_openai(messages, max_retries=3, retry_delay=5, max_tokens=500):
    for attempt in range(max_retries):
        try:
            url = f"{AZURE_OPENAI_ENDPOINT}"
            headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
            data = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3
            }
            
            logging.debug(f"Sending request to Azure OpenAI API: URL={url}")
            logging.debug(f"Request data: {json.dumps(data)}")
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            content = result['choices'][0]['message']['content']
            logging.debug(f"Received response from Azure OpenAI API: {content[:100]}...")
            
            return content
        except requests.exceptions.RequestException as e:
            logging.error(f"Error in Azure OpenAI API call (attempt {attempt + 1}): {str(e)}")
            if hasattr(e, 'response'):
                logging.error(f"Response status code: {e.response.status_code}")
                logging.error(f"Response content: {e.response.text}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise  # Re-raise the exception if all retries are exhausted
    return None

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def flatten_json(json_obj, prefix=''):
    flattened = {}
    
    # Handle the case where the input is a list of dictionaries
    if isinstance(json_obj, list) and all(isinstance(item, dict) for item in json_obj):
        for i, item in enumerate(json_obj):
            flattened.update(flatten_json(item, f"[{i}]."))
        return flattened
    
    # Original functionality
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            new_key = f"{prefix}{key}"
            if isinstance(value, (dict, list)):
                flattened.update(flatten_json(value, f"{new_key}."))
            else:
                flattened[new_key] = value
    elif isinstance(json_obj, list):
        for i, item in enumerate(json_obj):
            new_key = f"{prefix}[{i}]"
            if isinstance(item, (dict, list)):
                flattened.update(flatten_json(item, f"{new_key}."))
            else:
                flattened[new_key] = item
    else:
        flattened[prefix.rstrip('.')] = json_obj
    return flattened

# def stream_json(json_data, chunk_size=100):
#     items = list(json_data.items())
#     for i in range(0, len(items), chunk_size):
#         yield dict(items[i:i+chunk_size])

def is_json_empty(json_data):
    if json_data is None:
        return True
    if isinstance(json_data, (dict, list)):
        return len(json_data) == 0
    return False

def chunk_nested_json(data, max_tokens=3000, max_depth=100):
    """Chunk the nested JSON data into smaller parts to fit within the token limit."""
    chunks = []
    current_chunk = {}
    current_tokens = 0

    def process_item(key, value):
        nonlocal current_chunk, current_tokens
        item_json = json.dumps({key: value})
        item_tokens = num_tokens_from_string(item_json)
        
        if current_tokens + item_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = {}
            current_tokens = 0
        
        current_chunk[key] = value
        current_tokens += item_tokens

    def traverse(obj, prefix='', depth=0):
        if depth > max_depth:
            logging.warning(f"Maximum depth of {max_depth} reached. Truncating nested structure.")
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    traverse(v, new_key)
                else:
                    process_item(new_key, v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{prefix}[]"
                if isinstance(v, (dict, list)):
                    traverse(v, new_key)
                else:
                    process_item(new_key, v)

    traverse(data)
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def truncate_large_arrays(json_data, max_array_size=1000):
    def truncate(obj):
        if isinstance(obj, list):
            if len(obj) > max_array_size:
                logging.warning(f"Array truncated from {len(obj)} to {max_array_size} elements")
                return obj[:max_array_size]
            return [truncate(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: truncate(v) for k, v in obj.items()}
        else:
            return obj

    return truncate(json_data)

def clean_mapped_data(data):
    if isinstance(data, dict):
        return {k: clean_mapped_data(v) for k, v in data.items() if v is not None and v != {}}
    elif isinstance(data, list):
        cleaned = [clean_mapped_data(item) for item in data if item is not None and item != {}]
        return cleaned if cleaned else None
    else:
        return data


def get_matches(source: Union[Dict[Any, Any], List[Any]], target: Union[Dict[Any, Any], List[Any]]):
    source_chunks = chunk_nested_json(source, max_tokens=3000)
    target_chunks = chunk_nested_json(target, max_tokens=3000)
    
    all_mapped = {}

    for source_chunk in source_chunks:
        for target_chunk in target_chunks:
            prompt = f"""
                Map the source JSON chunk to the target JSON structure and provide matches.
                Source chunk: {json.dumps(source_chunk)}
                Target chunk: {json.dumps(target_chunk)}
                
                Consider the following guidelines:
                1. Maintain the hierarchical structure of the target JSON.
                2. Map conceptually similar fields (e.g., 'skills' should map to fields related to abilities or competencies).
                3. Avoid mapping unrelated concepts (e.g., don't map personal skills to project activities).
                4. Preserve array structures where appropriate.
                5. If a direct mapping is not clear, prefer to omit the field rather than forcing an incorrect mapping.

                For each mapping, return a JSON array of objects, each with 'source_key', 'target_key', and 'confidence' score (0.00-1.00).
                Respond ONLY with the JSON array, no additional text.
                """

            messages = [
                {"role": "system", "content": "You are a JSON mapping expert."},
                {"role": "user", "content": prompt}
            ]

            try:
                response = call_azure_openai(messages)
                
                if response:
                    logging.debug(f"Raw API response: {response}")
                    
                    # Try to extract JSON from the text response
                    json_start = response.find('[')
                    json_end = response.rfind(']') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = response[json_start:json_end]
                        try:
                            result = json.loads(json_str)
                        except json.JSONDecodeError:
                            # If JSON parsing fails, try to extract individual mappings
                            mappings = re.findall(r'(\d+)\.\s+Mapping:\s+{(.+?)}\s+Confidence:\s+(\d+)', response, re.DOTALL)
                            result = [
                                {
                                    "source_key": mapping[1].split('->')[0].strip().strip('"'),
                                    "target_key": mapping[1].split('->')[1].strip().strip('"'),
                                    "confidence": int(mapping[2]) / 10  # Convert to 0-1 scale
                                }
                                for mapping in mappings
                            ]
                    else:
                        raise ValueError("Unable to extract valid JSON or mappings from the response")

                    logging.debug(f"Parsed response: {json.dumps(result, indent=2)}")
                    
                    if isinstance(result, list):
                        for match in result:
                            if all(key in match for key in ['source_key', 'target_key', 'confidence']):
                                yield match
                                if match['confidence'] > 0.7:  # You can adjust this threshold
                                    all_mapped[match['source_key']] = match['target_key']
                    else:
                        logging.error(f"Unexpected response format from API: {result}")
                else:
                    logging.error("No response from Azure OpenAI API")
            except Exception as e:
                logging.error(f"Error during API call or processing: {str(e)}")
                logging.error(f"Full error: {traceback.format_exc()}")

    yield {"status": "Mapping complete", "mapped": all_mapped}

@app.route('/get_matches', methods=['POST'])
def get_matches_api():
    if 'source_file' not in request.files or 'target_file' not in request.files:
        return jsonify({"error": "Missing source_file or target_file"}), 400

    source_file = request.files['source_file']
    target_file = request.files['target_file']

    try:
        source_json = json.load(source_file)
        target_json = json.load(target_file)

        if is_json_empty(source_json) or is_json_empty(target_json):
            return jsonify({"error": "Source or target JSON is empty"}), 400
        
        source_json = truncate_large_arrays(source_json)
        target_json = truncate_large_arrays(target_json)

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON file"}), 400

    def generate():
        yield json.dumps({"status": "Processing..."}) + "\n"
        try:
            for result in get_matches(source_json, target_json):
                yield json.dumps(result) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return Response(stream_with_context(generate()), content_type='application/json')

def get_json_structure(json_data):
    if isinstance(json_data, list):
        return "list_of_dicts"
    elif isinstance(json_data, dict) and any(isinstance(v, list) for v in json_data.values()):
        return "dict_with_list"
    else:
        return "unknown"

def count_items(json_data):
    structure = get_json_structure(json_data)
    if structure == "list_of_dicts":
        return len(json_data)
    elif structure == "dict_with_list":
        return len(next(iter(json_data.values())))
    else:
        return 0

def get_main_list(json_data):
    structure = get_json_structure(json_data)
    if structure == "list_of_dicts":
        return json_data
    elif structure == "dict_with_list":
        return next(iter(json_data.values()))
    else:
        return []

def map_json(source: Union[Dict[Any, Any], List[Any]], target: Union[Dict[Any, Any], List[Any]], matches: List[Dict[str, Any]]):
    def flatten(data, prefix='', index=None):
        items = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    items.extend(flatten(v, new_key, index))
                else:
                    items.append((new_key, v, index))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                new_key = f"{prefix}[]"
                if isinstance(v, (dict, list)):
                    items.extend(flatten(v, new_key, i))
                else:
                    items.append((new_key, v, i))
        else:
            # Handle non-dict, non-list data (simple JSON)
            items.append((prefix, data, index))
        return items

    def unflatten(items):
        result = {}
        for key, value, index in items:
            if not key:  # Handle root-level simple value
                return value
            parts = key.split('.')
            d = result
            for part in parts[:-1]:
                if part.endswith('[]'):
                    part = part[:-2]
                    if part not in d:
                        d[part] = []
                    while len(d[part]) <= index:
                        d[part].append({})
                    d = d[part][index]
                else:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
            last_part = parts[-1]
            if last_part.endswith('[]'):
                last_part = last_part[:-2]
                if last_part not in d:
                    d[last_part] = []
                d[last_part].append(value)
            else:
                d[last_part] = value
        return result

    matches_dict = {match['source_key']: match['target_key'] for match in matches}
    
    flattened_source = flatten(source)
    mapped_items = []
    for source_key, value, index in flattened_source:
        if source_key in matches_dict:
            target_key = matches_dict[source_key]
            mapped_items.append((target_key, value, index))
    
    result = unflatten(mapped_items)
    
    # Handle case where result is a simple value
    if not isinstance(result, dict):
        result = {next(iter(matches_dict.values())): result}
    
    return {"status": "Processing complete", "mapped_data": result}

@app.route('/map_json', methods=['POST'])
def map_json_api():
    if 'source_file' not in request.files or 'target_file' not in request.files:
        return jsonify({"error": "Missing source_file or target_file"}), 400

    source_file = request.files['source_file']
    target_file = request.files['target_file']

    try:
        source_json = json.load(source_file)
        target_json = json.load(target_file)
        logging.info(f"Source JSON loaded. Structure: {type(source_json)}")

        if is_json_empty(source_json) or is_json_empty(target_json):
            return jsonify({"error": "Source or target JSON is empty"}), 400
        
        source_json = truncate_large_arrays(source_json)
        target_json = truncate_large_arrays(target_json)

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON file: {str(e)}"}), 400

        

    def generate():
        yield json.dumps({"status": "Processing..."}) + "\n"
        try:
            matches = []
            for result in get_matches(source_json, target_json):
                if 'source_key' in result and 'target_key' in result and 'confidence' in result:
                    matches.append(result)
                yield json.dumps({"status": "Mapping in progress", "match": result}) + "\n"
            
            final_result = clean_mapped_data(map_json(source_json, target_json, matches))
            
            yield json.dumps(final_result) + "\n"
        except Exception as e:
            logging.error(f"Error during mapping: {str(e)}")
            logging.error(traceback.format_exc())
            yield json.dumps({"error": str(e), "traceback": traceback.format_exc()}) + "\n"

    return Response(stream_with_context(generate()), content_type='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
	
