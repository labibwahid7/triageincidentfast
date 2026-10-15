# triageincidentfast
import os
import logging
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI, OpenAIError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TriageService")

app = FastAPI(title="AI Incident Triage & Resolution API")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

# ==========================================
# 1. INPUT & OUTPUT SCHEMAS (GUARDRAILS)
# ==========================================

class IncomingTicketWebhook(BaseModel):
    """Validates incoming webhook payloads from Power Automate / Ticketing System."""
    ticket_id: str
    subject: str
    description: Optional[str] = ""
    requester_email: Optional[str] = "company.com"
    affected_system: Optional[str] = "General IT"

    # Guardrail: Handle explicit 'null' values sent by webhook JSON
    @field_validator("description", "affected_system", mode="before")
    @classmethod
    def handle_null_strings(cls, v):
        if v is None:
            return "None provided"
        return str(v).strip()


class TriageResolutionOutput(BaseModel):
    """Strict schema enforced on the OpenAI API response."""
    category: Literal[
        "Identity & Access (Entra ID)",
        "Endpoint / Intune",
        "CRM / Dynamics 365",
        "Power Platform / Automation",
        "Network / VPN",
        "Other"
    ]
    severity: Literal["Low", "Medium", "High", "Critical"]
    suspected_root_cause: str = Field(
        description="Concise 1-sentence hypothesis of the technical root cause."
    )
    troubleshooting_steps: List[str] = Field(
        description="Step-by-step resolution actions for the Tier-1 engineer."
    )
    suggested_user_reply: str = Field(
        description="Polite, ready-to-send initial response to the user."
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )


# ==========================================
# 2. MOCK KNOWLEDGE BASE LOOKUP
# ==========================================

KNOWLEDGE_BASE = {
    "entra": "KB-101: For MFA lockouts or Conditional Access blocks, verify sign-in logs in Entra ID -> Check CA policy 'Require MFA for Admins' -> Reset MFA session in Azure Portal.",
    "intune": "KB-204: For device compliance failures, run 'Sync' in Company Portal -> Check BitLocker status -> Verify Primary User assignment in Intune.",
    "dynamics": "KB-309: For Dataverse / Dynamics 365 webhook mapping errors, inspect Power Automate run history -> Verify GUID lookup syntax `/contacts(guid)` -> Check null field handling.",
    "default": "KB-000: Collect user logs, verify network connectivity, and check service health dashboard."
}

def retrieve_kb_context(text: str) -> str:
    """Fetches relevant internal KB articles based on ticket keywords."""
    text_lower = text.lower()
    matched_kbs = []
    if any(k in text_lower for k in ["login", "mfa", "password", "entra", "sso", "access"]):
        matched_kbs.append(KNOWLEDGE_BASE["entra"])
    if any(k in text_lower for k in ["laptop", "compliant", "intune", "bitlocker", "device"]):
        matched_kbs.append(KNOWLEDGE_BASE["intune"])
    if any(k in text_lower for k in ["dynamics", "crm", "dataverse", "flow", "webhook"]):
        matched_kbs.append(KNOWLEDGE_BASE["dynamics"])
    
    return "\n".join(matched_kbs) if matched_kbs else KNOWLEDGE_BASE["default"]


# ==========================================
# 3. CORE AI TRIAGE LOGIC (WITH FALLBACKS)
# ==========================================

def analyze_ticket_with_ai(ticket: IncomingTicketWebhook, kb_context: str) -> TriageResolutionOutput:
    """Calls OpenAI Structured Outputs with prompt guardrails and fallback handling."""
    system_prompt = (
        "You are an expert Tier-2 Application Support & Cloud Engineer. "
        "Analyze the incoming support ticket and the provided Internal Knowledge Base context. "
        "Classify the issue accurately, identify the likely root cause, and provide actionable "
        "troubleshooting steps for the support queue. Do not invent tools outside our stack."
    )

    user_prompt = f"""
    Ticket ID: {ticket.ticket_id}
    Affected System: {ticket.affected_system}
    Subject: {ticket.subject}
    Description: {ticket.description}

    Relevant Internal Knowledge Base Context:
    {kb_context}
    """

    try:
        # Using OpenAI's beta.chat.completions.parse to guarantee schema compliance
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=TriageResolutionOutput,
            temperature=0.1, # Low temperature for deterministic triage
        )
        return completion.choices[0].message.parsed

    except OpenAIError as e:
        logger.error(f"OpenAI API failed for ticket {ticket.ticket_id}: {str(e)}")
        # Deterministic Fallback so the Power Automate pipeline never breaks
        return TriageResolutionOutput(
            category="Other",
            severity="Medium",
            suspected_root_cause="AI Triage unavailable — manual review required.",
            troubleshooting_steps=[
                "Review ticket description manually.",
                f"Reference standard KB: {kb_context}"
            ],
            suggested_user_reply="We have received your ticket and an engineer is reviewing your issue.",
            confidence_score=0.0
        )


# ==========================================
# 4. WEBHOOK ENDPOINT FOR POWER AUTOMATE
# ==========================================

@app.post("/api/v1/triage", response_model=TriageResolutionOutput)
async def triage_ticket(ticket: IncomingTicketWebhook):
    logger.info(f"Received Ticket #{ticket.ticket_id}: {ticket.subject}")
    
    # Step 1: Fetch relevant internal KB context
    combined_text = f"{ticket.subject} {ticket.description} {ticket.affected_system}"
    kb_context = retrieve_kb_context(combined_text)
    
    # Step 2: Run AI classification & drafting
    result = analyze_ticket_with_ai(ticket, kb_context)
    
    logger.info(f"Triage complete for #{ticket.ticket_id} -> Category: {result.category} | Severity: {result.severity}")
    return result
