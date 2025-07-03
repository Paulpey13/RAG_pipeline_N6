import requests

MISTRAL_API_KEY = "6atvb0C3Zbpdr0TNvC6STCG6qDCsxJ02"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

def ask_mistral(question):
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "mistral-medium-2505",
        "messages": [
            {"role": "system", "content": "Tu es un assistant intelligent qui répond toujours en français."},
            {"role": "user", "content": question}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
    }

    response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        print(f"Erreur API Mistral: {response.status_code} {response.text}")
        return None

# Exemple d'utilisation
question = "Explique-moi la théorie de la relativité en termes simples."
print(ask_mistral(question))
