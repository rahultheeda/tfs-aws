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
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")

if not AWS_REGION or not MODEL_ID or not KNOWLEDGE_BASE_ID:
    raise ValueError("AWS_REGION, BEDROCK_MODEL_ID or BEDROCK_KB_ID missing in .env")

# ---------------------------------------------------
# Initialize Bedrock Clients
# ---------------------------------------------------
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION
)

bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=AWS_REGION
)

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(
    title="Toyota Financial Services RAG QnA Agent",
    version="3.0.0"
)

# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------
class QuestionRequest(BaseModel):
    question: str


# ---------------------------------------------------
# Health Check
# ---------------------------------------------------
@app.get("/health")
def health():
    return {"status": "running"}


# ---------------------------------------------------
# QnA Endpoint (Bedrock Knowledge Base RAG)
# ---------------------------------------------------
@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:

        user_question = request.question

        # ---------------------------------------------------
        # 1️⃣ Retrieve from Bedrock Knowledge Base
        # ---------------------------------------------------
        retrieval_response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={
                "text": user_question
            },
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5
                }
            }
        )

        retrieval_results = retrieval_response.get("retrievalResults", [])

        retrieved_context = "\n\n".join(
            [item["content"]["text"] for item in retrieval_results]
        )

        print("\n----- RETRIEVED CONTEXT -----\n")
        print(retrieved_context)
        print("\n-----------------------------\n")

        # ---------------------------------------------------
        # 2️⃣ Build Prompt
        # ---------------------------------------------------
        prompt = f""" # 📚 RAG Query Agent - Toyota Financial Services Assistant You are the official Toyota Financial Services (TFS) virtual assistant. You must answer user questions using ONLY the retrieved knowledge base content provided below. ──────────────────── YOUR ROLE ──────────────────── 1. Use ONLY the retrieved content. 2. Summarize clearly and concisely. 3. Do NOT add external knowledge. 4. Do NOT infer beyond the provided content. ──────────────────── SCOPE RULES ──────────────────── - You can answer ONLY questions related to Toyota Financial Services (TFS). - If the question is unrelated or generic: Respond exactly: "I am a TFS assistant and can only answer questions related to Toyota Financial Services." - If the question is a greeting: Respond: "Hi, I'm the Toyota Financial Services virtual assistant." - If relevant information is NOT found in the retrieved context: Respond exactly: "I don't have information about this in the TFS knowledge base. This topic may be out of scope or not available." ──────────────────── ANSWER STYLE ──────────────────── - Clear and professional - Concise but complete - Use bullet points when listing items - No repetition - No speculation - No mention of internal systems, documents, PDFs, or filenames ──────────────────── SOURCE RULES (VERY IMPORTANT) ──────────────────── Only for valid TFS answers: 1. Add a heading exactly: Sources 2. Include ONLY relevant public URLs (https://...) from the context. 3. Include ALL relevant URLs. 4. One URL per line. 5. Do NOT include generic homepage links. 6. Do NOT modify URLs. 7. If no relevant public URLs exist, omit the Sources section entirely.
────────────────────
RETRIEVED CONTEXT
────────────────────
{retrieved_context}

────────────────────
USER QUESTION
────────────────────
{user_question}
""".strip()

        # ---------------------------------------------------
        # 3️⃣ Invoke Nova Pro
        # ---------------------------------------------------
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 1200,
                "temperature": 0.0,
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
