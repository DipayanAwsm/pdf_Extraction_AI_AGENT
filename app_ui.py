"""
Claims Document Management UI
=============================
A Streamlit app for searching, previewing, emailing, and parsing insurance documents.

Features:
1. Natural language search with AI-powered extraction
2. Editable dropdowns for Account, LOB, Policy, Date Range
3. Document preview
4. Email sending
5. PDF parsing using Google AI
"""

import os
import sys
import streamlit as st
from datetime import datetime, date
from typing import Optional, List, Dict
import tempfile
import json
import dotenv

dotenv.load_dotenv()

# Import modules
try:
    import cloud_storage as storage
except ImportError:
    import mock_storage as storage

from email_agent import build_graph, send_email_action, VALID_ACCOUNTS, VALID_LOBS

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Claims Document Manager",
    page_icon="📄",
    layout="wide"
)

# ============================================================================
# Session State Initialization
# ============================================================================

if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {
        'account_name': None,
        'lob': None,
        'policy_number': None,
        'start_date': None,
        'end_date': None
    }

if 'found_files' not in st.session_state:
    st.session_state.found_files = []

if 'selected_files' not in st.session_state:
    st.session_state.selected_files = []

if 'parse_results' not in st.session_state:
    st.session_state.parse_results = None

if 'email_status' not in st.session_state:
    st.session_state.email_status = None

if 'extraction_success' not in st.session_state:
    st.session_state.extraction_success = False

# ============================================================================
# AI Extraction Function
# ============================================================================

def extract_from_query(query: str) -> Dict:
    """Use the LangGraph agent to extract information from natural language query."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from pydantic import BaseModel, Field
        
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("❌ GOOGLE_API_KEY not found in environment variables.")
            return {}
        
        # Pydantic model for structured output
        class ExtractedInfo(BaseModel):
            """Extracted search criteria from user query."""
            account_name: Optional[str] = Field(
                default=None, 
                description="Account/Company name. Must be one of: Amex, Chubbs, GlobalInsure, TechCorp, Travelers"
            )
            lob: Optional[str] = Field(
                default=None, 
                description="Line of Business. Must be one of: AUTO, PROPERTY, GL, WC"
            )
            policy_number: Optional[str] = Field(
                default=None, 
                description="Policy or claim number (numeric identifier)"
            )
            start_date: Optional[str] = Field(
                default=None, 
                description="Start date in DD-MM-YYYY format"
            )
            end_date: Optional[str] = Field(
                default=None, 
                description="End date in DD-MM-YYYY format"
            )
        
        # Initialize LLM with structured output
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        structured_llm = llm.with_structured_output(ExtractedInfo)
        
        accounts_list = ", ".join(VALID_ACCOUNTS)
        lobs_list = ", ".join(VALID_LOBS)
        
        system_prompt = f"""You are an expert at extracting insurance document search criteria from natural language.

**Valid Account Names:** {accounts_list}
**Valid Lines of Business (LoB):** {lobs_list}

**Extraction Rules:**
1. **Account Name**: Match to valid accounts. Handle variations:
   - "amex", "American Express" → "Amex"
   - "chubbs", "chubb" → "Chubbs"
   - "globalinsure", "global" → "GlobalInsure"
   - "techcorp", "tech" → "TechCorp"
   - "travelers", "traveler" → "Travelers"

2. **Line of Business (LoB)**: Map to valid LOBs:
   - "auto", "vehicle", "car" → "AUTO"
   - "property", "home", "house", "fire" → "PROPERTY"
   - "gl", "general liability", "liability" → "GL"
   - "wc", "work", "workers comp" → "WC"

3. **Policy Number**: Extract numeric identifiers.

4. **Date Range**: Extract start_date and end_date in DD-MM-YYYY format.
   - "2024-2025" → start_date: "01-01-2024", end_date: "31-12-2025"
   - "less than 2024" → start_date: None, end_date: "31-12-2024"
   - "after 2023" → start_date: "01-01-2024", end_date: None
   - "between Sep 2024 and Dec 2024" → start_date: "01-09-2024", end_date: "31-12-2024"

Return None for any field that cannot be determined from the query."""

        # Invoke structured output
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        result: ExtractedInfo = structured_llm.invoke(messages)
        
        # Convert Pydantic model to dict
        extracted_dict = result.model_dump()
        print("Extraction result:", extracted_dict)
        return extracted_dict
        
    except Exception as e:
        st.error(f"Extraction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def parse_date_string(date_str: str) -> Optional[date]:
    """Parse date string to date object."""
    if not date_str:
        return None
    
    formats = ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d%m%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None

# ============================================================================
# PDF Parsing Function
# ============================================================================

def parse_pdf_with_google_ai(file_path: str) -> Dict:
    """Parse PDF using google_ai_parser methods."""
    try:
        # First, extract text from PDF
        import pdfplumber
        
        text_content = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() or ""
        
        if not text_content.strip():
            return {"error": "Could not extract text from PDF"}
        
        # Now use google_ai_parser methods
        from google_ai_parser import (
            load_config, 
            setup_openai_client, 
            classify_lobs_multi_openai,
            extract_fields_openai_chunked
        )
        
        # Load config
        cfg = load_config()
        if not cfg:
            return {"error": "Could not load config.py for OpenAI"}
        
        # Setup client
        client = setup_openai_client(cfg)
        if not client:
            return {"error": "Could not setup OpenAI client"}
        
        model = cfg['azure_deployment'] if cfg['use_azure'] else cfg['openai_model']
        
        # Classify LOBs
        lobs = classify_lobs_multi_openai(client, model, text_content)
        
        # Extract fields for each LOB
        results = []
        for lob in lobs:
            fields = extract_fields_openai_chunked(client, model, text_content, lob)
            results.append({
                "lob": lob,
                "carrier": fields.get("carrier", ""),
                "evaluation_date": fields.get("evaluation_date", ""),
                "claims": fields.get("claims", [])
            })
        
        return {
            "success": True,
            "source_file": file_path,
            "detected_lobs": lobs,
            "results": results,
            "text_preview": text_content[:500] + "..." if len(text_content) > 500 else text_content
        }
        
    except ImportError as e:
        return {"error": f"Missing dependency: {str(e)}"}
    except Exception as e:
        return {"error": f"Parse error: {str(e)}"}

# ============================================================================
# Main UI
# ============================================================================

st.title("📄 Claims Document Manager")
st.markdown("---")

# ============================================================================
# Section 1: Natural Language Search
# ============================================================================

st.subheader("🔍 Search Documents")

user_query = st.text_input(
    "Enter your search query:",
    placeholder="e.g., chubbs auto policy 12345 date 2024-2025",
    help="Type naturally - AI will extract Account, LOB, Policy, and Date Range"
)

col_extract, col_clear = st.columns([1, 5])

with col_extract:
    if st.button("🤖 Extract", type="primary"):
        if user_query:
            with st.spinner("Extracting information..."):
                extracted = extract_from_query(user_query)
                
                if extracted:
                    st.session_state.extracted_data = {
                        'account_name': extracted.get('account_name'),
                        'lob': extracted.get('lob'),
                        'policy_number': extracted.get('policy_number'),
                        'start_date': extracted.get('start_date'),
                        'end_date': extracted.get('end_date')
                    }
                    st.session_state.extraction_success = True
                    st.rerun()  # Rerun to update dropdowns with extracted values
                else:
                    st.warning("Could not extract information. Please fill manually.")

with col_clear:
    if st.button("🗑️ Clear"):
        st.session_state.extracted_data = {
            'account_name': None,
            'lob': None,
            'policy_number': None,
            'start_date': None,
            'end_date': None
        }
        st.session_state.found_files = []
        st.session_state.selected_files = []
        st.rerun()

st.markdown("---")

# ============================================================================
# Section 2: Editable Filters (Pre-populated from AI)
# ============================================================================

st.subheader("📋 Search Filters (Editable)")

# Update widget keys directly from extracted data (before widgets render)
if st.session_state.extraction_success:
    extracted = st.session_state.extracted_data
    
    # Update Account dropdown key
    if extracted.get('account_name') and extracted['account_name'] in VALID_ACCOUNTS:
        st.session_state['account_select'] = extracted['account_name']
    
    # Update LOB dropdown key
    if extracted.get('lob'):
        lob_upper = extracted['lob'].upper()
        if lob_upper in VALID_LOBS:
            st.session_state['lob_select'] = lob_upper
    
    # Update Policy input key
    if extracted.get('policy_number'):
        st.session_state['policy_input'] = extracted['policy_number']
    
    # Update Date input keys
    start_date_val = parse_date_string(extracted.get('start_date') or "")
    if start_date_val:
        st.session_state['start_date_input'] = start_date_val
    
    end_date_val = parse_date_string(extracted.get('end_date') or "")
    if end_date_val:
        st.session_state['end_date_input'] = end_date_val
    
    st.session_state.extraction_success = False  # Reset flag
    st.success("✅ Extracted successfully! Filters have been auto-filled.")

col1, col2, col3 = st.columns(3)

with col1:
    # Account dropdown
    account_options = ["-- Select --"] + VALID_ACCOUNTS
    selected_account = st.selectbox(
        "Account Name",
        options=account_options,
        key="account_select"
    )

with col2:
    # LOB dropdown
    lob_options = ["-- Select --"] + VALID_LOBS
    selected_lob = st.selectbox(
        "Line of Business",
        options=lob_options,
        key="lob_select"
    )

with col3:
    # Policy number
    policy_number = st.text_input(
        "Policy Number",
        key="policy_input"
    )

# Date range
col_date1, col_date2 = st.columns(2)

with col_date1:
    start_date = st.date_input(
        "Start Date",
        value=None,
        key="start_date_input"
    )

with col_date2:
    end_date = st.date_input(
        "End Date",
        value=None,
        key="end_date_input"
    )

st.markdown("---")

# ============================================================================
# Section 3: Fetch Button
# ============================================================================

if st.button("🔎 Fetch Documents", type="primary"):
    # Prepare search parameters
    account = selected_account if selected_account != "-- Select --" else None
    lob = selected_lob if selected_lob != "-- Select --" else None
    policy = policy_number if policy_number else None
    
    # For date, we'll use the start date for now (storage uses single date)
    date_str = None
    if start_date:
        date_str = start_date.strftime("%d-%m-%Y")
    
    with st.spinner("Searching documents..."):
        # Initialize mock data if needed
        if hasattr(storage, 'create_mock_data'):
            storage.create_mock_data()
        
        # Search
        results = storage.search_files(account, lob, policy, date_str)
        
        # Filter by date range if both dates provided
        if start_date and end_date and results:
            filtered_results = []
            for file in results:
                file_date_str = file.get('effective_date', '')
                file_date = parse_date_string(file_date_str)
                if file_date:
                    if start_date <= file_date <= end_date:
                        filtered_results.append(file)
                else:
                    filtered_results.append(file)  # Include if no date
            results = filtered_results
        
        st.session_state.found_files = results
        st.session_state.selected_files = []
    
    if results:
        st.success(f"✅ Found {len(results)} document(s)")
    else:
        st.warning("⚠️ No documents found matching criteria")

st.markdown("---")

# ============================================================================
# Section 4: Document Preview
# ============================================================================

if st.session_state.found_files:
    st.subheader("📁 Found Documents")
    
    for idx, file in enumerate(st.session_state.found_files):
        col_check, col_info = st.columns([0.5, 9.5])
        
        with col_check:
            is_selected = st.checkbox(
                "",
                key=f"file_select_{idx}",
                value=file in st.session_state.selected_files
            )
            
            if is_selected and file not in st.session_state.selected_files:
                st.session_state.selected_files.append(file)
            elif not is_selected and file in st.session_state.selected_files:
                st.session_state.selected_files.remove(file)
        
        with col_info:
            with st.expander(f"📄 {file.get('filename', 'Unknown')} - {file.get('source', 'Unknown')}", expanded=False):
                st.markdown(f"""
                | Field | Value |
                |-------|-------|
                | **Account** | {file.get('account', 'N/A')} |
                | **LOB** | {file.get('lob', 'N/A')} |
                | **Policy Number** | {file.get('policy_number', 'N/A')} |
                | **Effective Date** | {file.get('effective_date', 'N/A')} |
                | **Path** | `{file.get('folder_path', 'N/A')}` |
                | **Source** | {file.get('source', 'N/A')} |
                """)
                
                # Preview button for individual file
                if st.button(f"👁️ Preview", key=f"preview_{idx}"):
                    file_path = file.get('path', file.get('full_path', ''))
                    if os.path.exists(file_path):
                        try:
                            import pdfplumber
                            with pdfplumber.open(file_path) as pdf:
                                text = ""
                                for page in pdf.pages[:2]:  # First 2 pages
                                    text += page.extract_text() or ""
                            st.text_area("PDF Preview (first 2 pages)", text[:2000], height=300)
                        except Exception as e:
                            st.error(f"Could not preview: {e}")
                    else:
                        st.warning("File not found locally. May need to download from cloud.")
    
    st.markdown("---")
    
    # ============================================================================
    # Section 5: Action Buttons
    # ============================================================================
    
    st.subheader("⚡ Actions")
    
    selected_count = len(st.session_state.selected_files)
    st.info(f"📌 {selected_count} document(s) selected")
    
    col_email, col_parse, col_spacer = st.columns([1, 1, 4])
    
    with col_email:
        if st.button("📧 Send Email", disabled=selected_count == 0):
            if selected_count > 0:
                with st.spinner("Sending email(s)..."):
                    success_count = 0
                    for file in st.session_state.selected_files:
                        lob = file.get('lob', 'UNKNOWN')
                        policy = file.get('policy_number', 'Unknown')
                        
                        success, message = send_email_action(file, lob, policy)
                        
                        if success:
                            success_count += 1
                            st.success(f"✅ {file.get('filename')}: {message}")
                        else:
                            st.error(f"❌ {file.get('filename')}: {message}")
                    
                    st.session_state.email_status = f"Sent {success_count}/{selected_count} emails"
    
    with col_parse:
        if st.button("🔬 Parse PDF", disabled=selected_count == 0):
            if selected_count > 0:
                parse_results = []
                
                for file in st.session_state.selected_files:
                    file_path = file.get('path', file.get('full_path', ''))
                    
                    # Download from cloud if needed
                    if file.get('source') != 'Mock Storage' and not os.path.exists(file_path):
                        try:
                            file_path = storage.download_file(file)
                        except Exception as e:
                            parse_results.append({
                                "filename": file.get('filename'),
                                "error": f"Could not download: {e}"
                            })
                            continue
                    
                    with st.spinner(f"Parsing {file.get('filename')}..."):
                        result = parse_pdf_with_google_ai(file_path)
                        result['filename'] = file.get('filename')
                        parse_results.append(result)
                
                st.session_state.parse_results = parse_results

# ============================================================================
# Section 6: Parse Results Display
# ============================================================================

if st.session_state.parse_results:
    st.markdown("---")
    st.subheader("📊 Parse Results")
    
    for result in st.session_state.parse_results:
        filename = result.get('filename', 'Unknown')
        
        if result.get('error'):
            st.error(f"❌ {filename}: {result['error']}")
        else:
            with st.expander(f"✅ {filename} - {', '.join(result.get('detected_lobs', []))}", expanded=True):
                st.markdown(f"**Detected LOBs:** {', '.join(result.get('detected_lobs', []))}")
                
                for lob_result in result.get('results', []):
                    st.markdown(f"### {lob_result.get('lob', 'Unknown')} Claims")
                    st.markdown(f"**Carrier:** {lob_result.get('carrier', 'N/A')}")
                    st.markdown(f"**Evaluation Date:** {lob_result.get('evaluation_date', 'N/A')}")
                    
                    claims = lob_result.get('claims', [])
                    if claims:
                        import pandas as pd
                        df = pd.DataFrame(claims)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No claims extracted")
                
                # Show text preview
                if st.checkbox(f"Show text preview", key=f"text_preview_{filename}"):
                    st.text_area("Extracted Text", result.get('text_preview', ''), height=200)
    
    # Export button
    if st.button("📥 Export Results to Excel"):
        try:
            import pandas as pd
            
            all_claims = []
            for result in st.session_state.parse_results:
                if not result.get('error'):
                    for lob_result in result.get('results', []):
                        for claim in lob_result.get('claims', []):
                            claim['lob'] = lob_result.get('lob')
                            claim['carrier'] = lob_result.get('carrier')
                            claim['source_file'] = result.get('filename')
                            all_claims.append(claim)
            
            if all_claims:
                df = pd.DataFrame(all_claims)
                
                # Save to temp file
                temp_path = os.path.join(tempfile.gettempdir(), "parsed_claims.xlsx")
                df.to_excel(temp_path, index=False)
                
                with open(temp_path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Excel",
                        data=f.read(),
                        file_name="parsed_claims.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("No claims data to export")
                
        except Exception as e:
            st.error(f"Export error: {e}")

# ============================================================================
# Sidebar - Configuration Status
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### API Keys")
    st.markdown(f"- Google AI: {'✅' if os.getenv('GOOGLE_API_KEY') else '❌'}")
    st.markdown(f"- OpenAI: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
    
    st.markdown("### Storage")
    storage_mode = os.getenv('STORAGE_MODE', 'mock')
    st.markdown(f"- Mode: `{storage_mode}`")
    st.markdown(f"- Azure: {'✅' if os.getenv('AZURE_STORAGE_CONNECTION_STRING') else '❌'}")
    st.markdown(f"- OneDrive: {'✅' if os.getenv('ONEDRIVE_CLIENT_ID') else '❌'}")
    
    st.markdown("### Email")
    st.markdown(f"- SMTP: {'✅' if os.getenv('SMTP_USERNAME') else '❌'}")
    
    st.markdown("---")
    st.markdown("### Quick Help")
    st.markdown("""
    1. Enter a search query
    2. Click **Extract** to populate filters
    3. Adjust filters if needed
    4. Click **Fetch** to find documents
    5. Select documents
    6. **Send Email** or **Parse PDF**
    """)
