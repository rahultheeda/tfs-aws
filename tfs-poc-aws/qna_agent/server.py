import os
import json
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID")

if not AWS_REGION or not MODEL_ID:
    raise ValueError("AWS_REGION or BEDROCK_MODEL_ID missing in .env")

# ---------------------------------------------------
# Initialize Bedrock Runtime Client
# ---------------------------------------------------
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION
)

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(
    title="Toyota Financial Services RAG QnA Agent",
    version="2.0.0"
)

# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------
class QuestionRequest(BaseModel):
    question: str
    context: str  # Retrieved knowledge base content


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------
@app.get("/health")
def health():
    return {"status": "running"}


# ---------------------------------------------------
# QnA Endpoint (RAG-based)
# ---------------------------------------------------
@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        retrieved_context = request.context
        user_question = request.question

        # ---------------------------------------------------
        # TFS Enterprise RAG Prompt
        # ---------------------------------------------------
        prompt = f"""
# 📚 RAG Query Agent - Toyota Financial Services Assistant

You are the official Toyota Financial Services (TFS) virtual assistant.

You must answer user questions using ONLY the retrieved knowledge base content provided below.

────────────────────
YOUR ROLE
────────────────────
1. Use ONLY the retrieved content.
2. Summarize clearly and concisely.
3. Do NOT add external knowledge.
4. Do NOT infer beyond the provided content.

────────────────────
SCOPE RULES
────────────────────
- You can answer ONLY questions related to Toyota Financial Services (TFS).
- If the question is unrelated or generic:
  Respond exactly:
  "I am a TFS assistant and can only answer questions related to Toyota Financial Services."

- If the question is a greeting:
  Respond:
  "Hi, I'm the Toyota Financial Services virtual assistant."

- If relevant information is NOT found in the retrieved context:
  Respond exactly:
  "I don't have information about this in the TFS knowledge base. This topic may be out of scope or not available."

────────────────────
ANSWER STYLE
────────────────────
- Clear and professional
- Concise but complete
- Use bullet points when listing items
- No repetition
- No speculation
- No mention of internal systems, documents, PDFs, or filenames

────────────────────
SOURCE RULES (VERY IMPORTANT)
────────────────────
Only for valid TFS answers:

1. Add a heading exactly: Sources
2. Include ONLY relevant public URLs (https://...) from the context.
3. Include ALL relevant URLs.
4. One URL per line.
5. Do NOT include generic homepage links.
6. Do NOT modify URLs.
7. If no relevant public URLs exist, omit the Sources section entirely.

────────────────────
RETRIEVED CONTEXT
────────────────────
{retrieved_context}

────────────────────
USER QUESTION
────────────────────
{user_question}

────────────────────
Now provide your answer following ALL rules strictly.
""".strip()

        # ---------------------------------------------------
        # Nova Pro Payload (Converse Format)
        # ---------------------------------------------------
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt}
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 1200,
                "temperature": 0.0,   # deterministic for RAG
                "topP": 0.9
            }
        }

        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response["body"].read())

        answer = response_body["output"]["message"]["content"][0]["text"]

        return {
            "question": user_question,
            "answer": answer.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
