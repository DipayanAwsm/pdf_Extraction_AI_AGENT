import os
import glob
import smtplib
from typing import TypedDict, Annotated, List, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Import Storage - Uses cloud storage (Azure/OneDrive) if configured, falls back to mock
try:
    import cloud_storage as storage
    print("☁️ Using cloud storage module")
except ImportError:
    import mock_storage as storage
    print("📁 Using mock storage module")

# ============================================================================
# Configuration
# ============================================================================

# Mock Email Database
LOB_EMAILS = {
    "AUTO": "madhumadhu112120052@gmail.com",
    "PROPERTY": "madhumadhu112120052@gmail.com",
    "GL": "madhumadhu112120052@gmail.com",
    "WC": "madhumadhu112120052@gmail.com",
    "UNKNOWN": "madhumadhu112120052@gmail.com"
}

# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    user_query: str
    account_name: Optional[str]
    policy_number: Optional[str]
    lob: Optional[str]
    date: Optional[str]
    found_files: List[Dict[str, str]]
    email_sent: bool
    error: Optional[str]
    logs: List[str]

# ============================================================================
# Models
# ============================================================================

class ExtractionResult(BaseModel):
    account_name: str = Field(description="The name of the Account or Company (e.g., Chubbs, Amex).")
    policy_number: str = Field(description="The policy number or claim number.")
    lob: str = Field(description="The Line of Business (AUTO, PROPERTY, GL, WC). Map 'work' to WC, 'vehicle' to AUTO.")
    date: str = Field(description="The effective date or report date mentioned (format: DD-MM-YYYY).")

# ============================================================================
# Nodes
# ============================================================================

# ============================================================================
# Valid Accounts and LOBs (derived from mock_documents structure)
# ============================================================================

VALID_ACCOUNTS = ["Amex", "Chubbs", "GlobalInsure", "TechCorp", "Travelers"]
VALID_LOBS = ["AUTO", "PROPERTY", "GL", "WC"]

def extract_info_node(state: AgentState):
    """
    Extracts Account, Policy, LoB, and Date from the user query using Gemini.
    """
    print("--- Extracting Info ---")
    query = state['user_query']
    
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "error": "GOOGLE_API_KEY not found in environment variables.",
            "logs": state['logs'] + ["Error: Missing GOOGLE_API_KEY"]
        }

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    parser = JsonOutputParser(pydantic_object=ExtractionResult)
    
    # Build dynamic prompt with all valid accounts and LOBs
    accounts_list = ", ".join(VALID_ACCOUNTS)
    lobs_list = ", ".join(VALID_LOBS)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert insurance document retrieval assistant. Your task is to extract structured information from the user's natural language request.

**Valid Account Names:** {accounts_list}
**Valid Lines of Business (LoB):** {lobs_list}

**Extraction Rules:**
1. **Account Name**: Match the user's input to one of the valid accounts. Handle variations:
   - "amex", "AMEX", "American Express" → "Amex"
   - "chubbs", "chubb", "CHUBBS" → "Chubbs"
   - "globalinsure", "global insure", "global" → "GlobalInsure"
   - "techcorp", "tech corp", "tech" → "TechCorp"
   - "travelers", "traveler", "TRAVELERS" → "Travelers"

2. **Line of Business (LoB)**: Map user terms to valid LOBs:
   - "auto", "vehicle", "car", "automobile", "motor", "accident" → "AUTO"
   - "property", "home", "house", "fire", "building", "real estate" → "PROPERTY"
   - "gl", "general liability", "liability", "general" → "GL"
   - "wc", "work", "workers comp", "workers compensation", "worker", "workplace", "injury" → "WC"

3. **Policy Number**: Extract any numeric identifier (policy number, claim number, reference number).

4. **Date**: Extract any date mentioned. Convert to format DD-MM-YYYY (e.g., "21-09-2024").
   - Handle formats like "21/09/2024", "September 21, 2024", "21 Sep 2024", etc.

**Important:** 
- Be flexible with spelling variations and typos.
- If unsure about account name, use the closest match from the valid list.
- Return ONLY valid JSON matching the schema.

{{format_instructions}}"""),
        ("user", "{query}")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})
        print(f"Extracted: {result}")
        return {
            "account_name": result.get("account_name"),
            "policy_number": result.get("policy_number"),
            "lob": result.get("lob").upper(),
            "date": result.get("date"),
            "logs": state['logs'] + [f"Extracted: {result}"]
        }
    except Exception as e:
        return {
            "error": f"Extraction failed: {str(e)}",
            "logs": state['logs'] + [f"Extraction Error: {str(e)}"]
        }

def locate_document_node(state: AgentState):
    """
    Searches for the PDF file using cloud storage (Azure/OneDrive) or mock storage.
    """
    print("--- Locating Document ---")
    if state.get("error"):
        return state

    account = state['account_name']
    lob = state['lob']
    policy = state['policy_number']
    date = state['date']
    
    found_files = storage.search_files(account, lob, policy, date)
    
    if found_files:
        print(f"Found {len(found_files)} files.")
        return {
            "found_files": found_files,
            "logs": state['logs'] + [f"Found {len(found_files)} files in storage."]
        }
    else:
        return {
            "found_files": [],
            "logs": state['logs'] + ["No files found matching criteria."]
        }

# Note: send_email_node is now intended to be called explicitly by the UI, 
# but we keep it in the graph if we want an automated flow. 
# For this requirement, the user wants to click a button.
# We will expose a separate function for sending email.

def send_email_action(file_info: Dict[str, str], lob: str, policy_number: str):
    """
    Standalone function to send email, called by UI.
    Handles both local files and cloud storage files.
    
    Args:
        file_info: Dictionary with file information (from search results)
        lob: Line of Business
        policy_number: Policy number
    """
    recipient = LOB_EMAILS.get(lob, LOB_EMAILS["UNKNOWN"])
    
    # Get local file path - download from cloud if needed
    try:
        if file_info.get("source") == "Mock Storage":
            file_path = file_info.get("path", file_info.get("full_path", ""))
        else:
            # Download from cloud storage to temp location
            file_path = storage.download_file(file_info)
    except Exception as e:
        return False, f"Failed to get file: {str(e)}"
    subject = f"Claims Document for Policy {policy_number} ({lob})"
    body = f"Please find attached the claims document for Policy {policy_number}."
    
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not smtp_username or not smtp_password:
        return False, "Missing SMTP credentials"

    msg = MIMEMultipart()
    msg['From'] = smtp_username
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        with open(file_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
        msg.attach(part)
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            
        return True, f"Email sent to {recipient}"
    except Exception as e:
        return False, str(e)

# ============================================================================
# Graph Construction
# ============================================================================

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("extract_info", extract_info_node)
    workflow.add_node("locate_document", locate_document_node)
    
    workflow.set_entry_point("extract_info")
    
    workflow.add_edge("extract_info", "locate_document")
    workflow.add_edge("locate_document", END)
    
    return workflow.compile()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Initialize Mock Data (only if using mock storage)
    if hasattr(storage, 'create_mock_data'):
        storage.create_mock_data()

    # Initialize Graph
    app = build_graph()
    
    # Example User Query
    user_input = "chubbs work 2456 date 21-09-2024"
    print(f"\n🤖 User Query: '{user_input}'\n")
    
    initial_state = {
        "user_query": user_input,
        "account_name": None,
        "policy_number": None,
        "lob": None,
        "date": None,
        "found_files": [],
        "email_sent": False,
        "error": None,
        "logs": []
    }
    
    # Run the graph
    result = app.invoke(initial_state)
    
    print("\n--- Final State ---")
    if result.get("error"):
        print(f"❌ Error: {result['error']}")
    else:
        print("✅ Workflow Completed Successfully")
        print(f"Extracted: Account={result['account_name']}, LoB={result['lob']}, Policy={result['policy_number']}, Date={result['date']}")
        print(f"Files Found: {len(result['found_files'])}")
        for f in result['found_files']:
            print(f" - {f['filename']} ({f['source']})")
