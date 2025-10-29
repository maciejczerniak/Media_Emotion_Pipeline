import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import language_tool_python as ltp
import pandas as pd
import requests
from tqdm import tqdm


def correct_with_language_tools(text: str, tool: ltp.LanguageTool) -> str:
    """Fast correction with LanguageTool"""
    matches = tool.check(text)
    corrected_text = ltp.utils.correct(text, matches)
    return corrected_text

def correct_with_ollama(text: str, ollama_api_url: str, model: str) -> str:
    """Correct text using Ollama API with JSON response"""
    headers = {'Content-Type': 'application/json'}

    data = {
        'model': model,
        'prompt': f'''Popraw błędy ortograficzne i gramatyczne w tekście. Odpowiedz TYLKO w formacie JSON bez dodatkowych wyjaśnień.

Tekst: "{text}"

{{"corrected": "poprawiony tekst tutaj"}}''',
        'stream': False,
        'temperature': 0.1,
        'format': 'json',
    }

    try:
        response = requests.post(ollama_api_url, headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')

            try:
                corrected_data = json.loads(response_text)
                return corrected_data.get('corrected', text)
            except json.JSONDecodeError:
                return text
        else:
            return text
    except Exception as e:
        print(f"Error with Ollama: {e}")
        return text

def orchestrate_corrections(text: str, tool: ltp.LanguageTool, ollama_api_url: str, model: str) -> str:
    """Two-stage correction pipeline"""
    text_after_tool = correct_with_language_tools(text, tool)
    final_corrected_text = correct_with_ollama(text_after_tool, ollama_api_url, model)
    return final_corrected_text

def auto_correct_batch(df: pd.DataFrame,
                       text_column: str,
                       tool: ltp.LanguageTool,
                       ollama_api_url: str,
                       model: str,
                       max_workers: int = 4) -> pd.DataFrame:
    """
    Parallel correction with progress bars

    Args:
        df: Input dataframe
        text_column: Column containing text to correct
        tool: LanguageTool instance
        ollama_api_url: Ollama API endpoint
        model: Model name
        max_workers: Number of parallel workers (default: 4)
    """
    df = df.copy()
    texts = df[text_column].tolist()

    # Stage 1: LanguageTool corrections (fast, can parallelize)
    # print("Stage 1: Running LanguageTool corrections...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        stage1_results = list(tqdm(
            executor.map(lambda x: correct_with_language_tools(x, tool), texts),
            total=len(texts),
            desc="LanguageTool"
        ))

    # Stage 2: Ollama corrections (slower, parallel with progress)
#     print("\nStage 2: Running Ollama corrections...")
    corrected_texts = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(correct_with_ollama, text, ollama_api_url, model): idx
            for idx, text in enumerate(stage1_results)
        }

        # Process completed tasks with progress bar
        results_dict = {}
        for future in tqdm(as_completed(future_to_idx), total=len(stage1_results), desc="Ollama"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results_dict[idx] = result
            except Exception as e:
                print(f"\nError processing text {idx}: {e}")
                results_dict[idx] = stage1_results[idx]  # Fallback to stage 1 result

        # Sort results by original index
        corrected_texts = [results_dict[i] for i in range(len(stage1_results))]

    df['corrected_text'] = corrected_texts
    return df