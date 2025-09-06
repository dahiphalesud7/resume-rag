import requests
import streamlit as st
import json

def query_perplexity_llm(query: str, resumes: list, api_key: str) -> str:
    context = "\n\n".join([f"Resume {i+1}:\n{doc}" for i, doc in enumerate(resumes)])
    context = context[:]  # Optional truncation

    prompt = f"""
You are an AI assistant helping to evaluate candidates.Based on the job description and resumes,take a look at the all retrived candidates analyse them all and tell me who are most deserving and why and alos specify the reasone you can draw the table to explain it?
Job Description: "{query}"

Here are the top retrieved resumes:
{context}


"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar-pro",   
        "messages": [{"role": "user", "content": prompt.strip()}],
        "temperature": 0.7
    }

    # 👇 Show full request for debugging
    st.markdown("###Sent Payload:")
    st.code(json.dumps(payload, indent=2), language="json")

    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload
    )

    # 👇 Show response error if failed
    if response.status_code != 200:
        st.error(f"❌ LLM API Error: {response.status_code}")
        st.markdown("### 🔍 Full API Error Response:")
        st.code(response.text, language="json")
        return "⚠️ LLM failed to respond properly."

    return response.json()["choices"][0]["message"]["content"]

