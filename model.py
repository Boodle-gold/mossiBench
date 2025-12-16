import time
# We import OpenAI inside the function so it doesn't crash if you don't have it installed
# for local-only runs.

def get_pred_api(client, model_name, prompt, args):
    """
    Sends the constructed prompt to an API.
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful translator. Please provide only what you believe the translation of the given sentance to be and NOTHING else"},
                {"role": "user", "content": prompt}
            ],
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            top_p=args.top_p,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        time.sleep(2) # Wait a bit before retrying if you want to add retry logic
        return ""