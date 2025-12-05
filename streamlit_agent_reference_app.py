#!/usr/bin/env python3
"""
Streamlit Agent Reference App
=============================
A reference implementation that:
1. Shows all PDFs from local file structure (mock_documents)
2. Uses config.py for Azure OpenAI configuration
3. Has AI agent that collects data from local files
4. Has mailing agent that sends extracted data
5. Allows selecting a single PDF and processing it like streamlit_e2e_openai_app.py
"""

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import tempfile
import shutil
import io

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    st.error("Please install openai: pip install openai>=1.30.0")
    st.stop()

try:
    import pdfplumber
except ImportError:
    st.error("Please install pdfplumber: pip install pdfplumber")
    st.stop()

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from PIL import Image

# Import email agent functionality
try:
    from email_agent import send_email_action, LOB_EMAILS
except ImportError:
    st.warning("email_agent module not found. Email functionality will be disabled.")
    send_email_action = None
    LOB_EMAILS = {}

# ============================================================================
# Configuration Functions
# ============================================================================

def load_config(config_file: str = "config.py") -> Dict[str, str]:
    """Load configuration from config.py file"""
    config_path = Path(config_file)
    if not config_path.exists():
        return None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        cfg = {}
        
        # OpenAI / Azure OpenAI settings
        cfg['use_azure'] = getattr(config_module, 'USE_AZURE_OPENAI', False)
        cfg['openai_api_key'] = getattr(config_module, 'OPENAI_API_KEY', None)
        cfg['openai_model'] = getattr(config_module, 'OPENAI_MODEL', 'gpt-4o-2024-08-06')
        cfg['azure_endpoint'] = getattr(config_module, 'AZURE_OPENAI_ENDPOINT', None)
        cfg['azure_api_key'] = getattr(config_module, 'AZURE_OPENAI_API_KEY', None)
        cfg['azure_deployment'] = getattr(config_module, 'AZURE_OPENAI_DEPLOYMENT_NAME', None)
        
        # Processing settings
        cfg['max_chunk_size'] = getattr(config_module, 'MAX_CHUNK_SIZE', 15000)
        cfg['api_delay'] = getattr(config_module, 'API_DELAY', 0.3)
        
        return cfg
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return None


def setup_openai_client(cfg: Dict[str, str]):
    """Setup OpenAI client based on configuration"""
    if not cfg:
        return None
    
    try:
        if cfg.get('use_azure'):
            if not all([cfg.get('azure_api_key'), cfg.get('azure_endpoint'), cfg.get('azure_deployment')]):
                return None
            
            client = OpenAI(
                api_key=cfg['azure_api_key'],
                base_url=f"{cfg['azure_endpoint'].rstrip('/')}/openai/deployments/{cfg['azure_deployment']}",
            )
            return client
        else:
            if not cfg.get('openai_api_key'):
                return None
            
            client = OpenAI(api_key=cfg['openai_api_key'])
            return client
    except Exception as e:
        st.error(f"Error setting up OpenAI client: {e}")
        return None


# ============================================================================
# Local File Structure Functions
# ============================================================================

def scan_local_pdfs(base_dir: str = "mock_documents") -> List[Dict[str, str]]:
    """
    Scan local file structure for PDFs
    Expected structure: AccountName/LOB/PolicyNo-Date/filename.pdf
    """
    pdf_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return pdf_files
    
    for account_dir in base_path.iterdir():
        if not account_dir.is_dir():
            continue
        
        account_name = account_dir.name
        
        for lob_dir in account_dir.iterdir():
            if not lob_dir.is_dir():
                continue
            
            lob = lob_dir.name
            
            for policy_dir in lob_dir.iterdir():
                if not policy_dir.is_dir():
                    continue
                
                # Parse policy number and date from folder name (e.g., "7890-10112024")
                folder_name = policy_dir.name
                policy_parts = folder_name.split('-', 1)
                policy_number = policy_parts[0] if policy_parts else ""
                effective_date = policy_parts[1] if len(policy_parts) > 1 else ""
                
                # Find PDFs in this folder
                for pdf_file in policy_dir.glob("*.pdf"):
                    pdf_files.append({
                        'filename': pdf_file.name,
                        'account': account_name,
                        'lob': lob,
                        'policy_number': policy_number,
                        'effective_date': effective_date,
                        'path': str(pdf_file),
                        'full_path': str(pdf_file),
                        'source': 'Local Files',
                        'folder_path': str(policy_dir)
                    })
    
    return sorted(pdf_files, key=lambda x: (x['account'], x['lob'], x['policy_number']))


# ============================================================================
# LLM-Powered Search Functions
# ============================================================================

def extract_search_criteria_with_llm(client, model: str, query: str, available_pdfs: List[Dict]) -> Dict:
    """
    Use LLM to extract search criteria from natural language query
    """
    # Get unique values from available PDFs for context
    unique_accounts = sorted(set(pdf['account'] for pdf in available_pdfs))
    unique_lobs = sorted(set(pdf['lob'] for pdf in available_pdfs))
    
    prompt = f"""Extract search criteria from the user's natural language query.

Available Accounts: {', '.join(unique_accounts)}
Available LOBs: {', '.join(unique_lobs)}

User Query: "{query}"

Extract the following information:
1. LOB (Line of Business): AUTO, PROPERTY, GL, WC, or empty if not specified
2. Account name: Match to one of the available accounts, or empty if not specified
3. Policy number: Any numeric identifier mentioned, or empty if not specified
4. Keywords: Important keywords from the query (e.g., "auto", "vehicle", "claim", etc.)

Return STRICT JSON ONLY with no markdown, no code blocks, no explanation:
{{
    "lob": "AUTO|PROPERTY|GL|WC|",
    "account": "account name or empty string",
    "policy_number": "policy number or empty string",
    "keywords": ["keyword1", "keyword2"]
}}

Examples:
- Query: "auto" → {{"lob": "AUTO", "account": "", "policy_number": "", "keywords": ["auto"]}}
- Query: "find WC documents" → {{"lob": "WC", "account": "", "policy_number": "", "keywords": ["WC", "documents"]}}
- Query: "Chubbs property" → {{"lob": "PROPERTY", "account": "Chubbs", "policy_number": "", "keywords": ["Chubbs", "property"]}}
- Query: "policy 7890" → {{"lob": "", "account": "", "policy_number": "7890", "keywords": ["policy", "7890"]}}
"""
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        
        content = resp.choices[0].message.content
        criteria = json.loads(content)
        
        # Normalize LOB
        if criteria.get('lob'):
            lob = criteria['lob'].upper().strip()
            if lob in ['AUTO', 'PROPERTY', 'GL', 'GENERAL LIABILITY', 'WC', 'WORKERS COMP']:
                if lob == 'GENERAL LIABILITY':
                    criteria['lob'] = 'GL'
                elif lob == 'WORKERS COMP':
                    criteria['lob'] = 'WC'
                else:
                    criteria['lob'] = lob
            else:
                criteria['lob'] = ''
        
        # Normalize account (fuzzy match)
        if criteria.get('account'):
            account_query = criteria['account'].lower().strip()
            for account in unique_accounts:
                if account_query in account.lower() or account.lower() in account_query:
                    criteria['account'] = account
                    break
        
        return criteria
    
    except Exception as e:
        st.warning(f"LLM search extraction failed: {e}. Using keyword matching instead.")
        # Fallback: simple keyword matching
        query_lower = query.lower()
        criteria = {
            'lob': '',
            'account': '',
            'policy_number': '',
            'keywords': query_lower.split()
        }
        
        # Simple LOB detection
        if any(kw in query_lower for kw in ['auto', 'vehicle', 'car', 'automobile']):
            criteria['lob'] = 'AUTO'
        elif any(kw in query_lower for kw in ['property', 'dwelling', 'building', 'fire']):
            criteria['lob'] = 'PROPERTY'
        elif any(kw in query_lower for kw in ['gl', 'general liability', 'liability']):
            criteria['lob'] = 'GL'
        elif any(kw in query_lower for kw in ['wc', 'workers comp', 'workers compensation', 'work']):
            criteria['lob'] = 'WC'
        
        return criteria


def filter_pdfs_by_criteria(pdf_files: List[Dict], criteria: Dict) -> List[Dict]:
    """
    Filter PDF files based on search criteria
    """
    filtered = []
    
    for pdf in pdf_files:
        match = True
        
        # Filter by LOB
        if criteria.get('lob'):
            if pdf['lob'].upper() != criteria['lob'].upper():
                match = False
        
        # Filter by Account
        if match and criteria.get('account'):
            if pdf['account'].lower() != criteria['account'].lower():
                match = False
        
        # Filter by Policy Number
        if match and criteria.get('policy_number'):
            policy_query = criteria['policy_number'].strip()
            if policy_query not in pdf['policy_number']:
                match = False
        
        # Filter by Keywords (in filename or other fields)
        if match and criteria.get('keywords'):
            pdf_text = f"{pdf['filename']} {pdf['account']} {pdf['lob']} {pdf['policy_number']}".lower()
            keyword_match = any(
                keyword.lower() in pdf_text 
                for keyword in criteria['keywords']
                if keyword.strip()
            )
            if not keyword_match:
                match = False
        
        if match:
            filtered.append(pdf)
    
    return filtered


# ============================================================================
# PDF Processing Functions (from streamlit_e2e_openai_app.py)
# ============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber"""
    text_content = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                
                # Also extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        for row in table:
                            if row:
                                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                                text_content.append(row_text)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""
    
    return "\n".join(text_content)


def _chunk_text(text: str, max_chars: int = 15000, overlap_chars: int = 800) -> List[str]:
    """Split text into chunks for processing"""
    chunks: List[str] = []
    if not text:
        return chunks
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            nl = text.rfind("\n", start, end)
            if nl != -1 and nl > start + 1000:
                end = nl
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap_chars)
    return chunks


def classify_lobs_multi_openai(client, model: str, text: str) -> List[str]:
    """Classify Lines of Business from text content"""
    prompt = f"""
You are an insurance domain expert. Determine ALL Lines of Business (LoBs) present in the content.
Choose any that apply from exactly these values: AUTO, GENERAL LIABILITY, WC, PROPERTY.
Return STRICT JSON ONLY with no commentary and no markdown. Use double quotes and valid JSON.
Schema: {{"lobs": ["AUTO"|"GENERAL LIABILITY"|"WC"|"PROPERTY", ...]}}
Content:\n{text[:10000]}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        obj = json.loads(content)
        lobs = obj.get('lobs') or []
        out = []
        for v in lobs:
            s = str(v).strip().upper()
            if s in {"AUTO", "GENERAL LIABILITY", "WC", "PROPERTY"} and s not in out:
                out.append(s)
        if out:
            return out
    except Exception:
        pass
    
    # Fallback heuristic
    t = text.upper()
    found = []
    if any(k in t for k in [" AUTO ", " AUTOMOBILE", " VEHICLE", " VIN ", " COLLISION", " COMPREHENSIVE", " LICENSE PLATE"]):
        found.append("AUTO")
    if any(k in t for k in [" GENERAL LIABILITY", " GL ", " PREMISES", " PRODUCTS LIABILITY", " CGL "]):
        found.append("GENERAL LIABILITY")
    if any(k in t for k in [" WORKERS' COMP", " WORKERS COMP", " WC ", " TTD", " TPD", " INDEMNITY"]):
        found.append("WC")
    if any(k in t for k in [" PROPERTY ", " DWELLING", " BUILDING", " CONTENTS", " FIRE", " THEFT"]):
        found.append("PROPERTY")
    return found or ["AUTO"]


def extract_fields_openai(client, model: str, text: str, lob: str) -> Dict:
    """Extract structured fields from text for a specific LoB"""
    lob = lob.upper()
    if lob == 'AUTO':
        schema = {
            "evaluation_date": "string",
            "carrier": "string",
            "claims": [{
                "claim_number": "string",
                "loss_date": "string",
                "paid_loss": "string",
                "reserve": "string",
                "alae": "string"
            }]
        }
    elif lob == 'PROPERTY':
        schema = {
            "evaluation_date": "string",
            "carrier": "string",
            "claims": [{
                "claim_number": "string",
                "loss_date": "string",
                "paid_loss": "string",
                "reserve": "string",
                "alae": "string"
            }]
        }
    elif lob in ('GENERAL LIABILITY', 'GL'):
        schema = {
            "evaluation_date": "string",
            "carrier": "string",
            "claims": [{
                "claim_number": "string",
                "loss_date": "string",
                "bi_paid_loss": "string",
                "pd_paid_loss": "string",
                "bi_reserve": "string",
                "pd_reserve": "string",
                "alae": "string"
            }]
        }
    else:  # WC
        schema = {
            "evaluation_date": "string",
            "carrier": "string",
            "claims": [{
                "claim_number": "string",
                "loss_date": "string",
                "Indemnity_paid_loss": "string",
                "Medical_paid_loss": "string",
                "Indemnity_reserve": "string",
                "Medical_reserve": "string",
                "ALAE": "string"
            }]
        }
        lob = 'WC'

    prompt = f"""
Extract structured fields from the content for LoB={lob}.
Return STRICT JSON ONLY matching this schema with no commentary and no markdown fences:
{schema}
Rules: ISO dates if possible; keep amounts/strings as-is; empty string if missing; preserve row order.
IMPORTANT: Extract the carrier/company name from the content. This is critical.

Content:\n{text}
"""
    max_attempts = 3
    delay_seconds = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=16000,
                response_format={"type": "json_object"}
            )
            content = resp.choices[0].message.content
            obj = json.loads(content)
            if isinstance(obj, dict) and 'claims' in obj and isinstance(obj['claims'], list):
                obj.setdefault('evaluation_date', '')
                obj.setdefault('carrier', '')
                return obj
        except Exception:
            if attempt == max_attempts:
                break
            time.sleep(delay_seconds)
            delay_seconds *= 2
            continue
    return {"evaluation_date": "", "carrier": "", "claims": []}


def extract_fields_openai_chunked(client, model: str, text: str, lob: str, progress_callback=None, max_chunk_size: int = 15000, api_delay: float = 0.3) -> Dict:
    """Extract fields with chunking for long documents"""
    chunks = _chunk_text(text, max_chars=max_chunk_size)
    if not chunks:
        chunks = [text]
    
    merged = {"evaluation_date": "", "carrier": "", "claims": []}
    for idx, part in enumerate(chunks):
        if progress_callback:
            progress_callback(idx + 1, len(chunks))
        
        result = extract_fields_openai(client, model, part, lob)
        if result.get('evaluation_date') and not merged['evaluation_date']:
            merged['evaluation_date'] = result.get('evaluation_date', '')
        if result.get('carrier') and not merged['carrier']:
            merged['carrier'] = result.get('carrier', '')
        if isinstance(result.get('claims'), list):
            merged['claims'].extend(result['claims'])
        time.sleep(api_delay)
    return merged


def process_pdf_with_openai(pdf_path: str, client, model: str, max_chunk_size: int = 15000, api_delay: float = 0.3) -> Dict:
    """
    Process PDF using OpenAI/Azure OpenAI
    Similar to text_lob_openai_extractor.py processing
    """
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return {"error": "No text extracted from PDF"}
    
    # Classify LOBs
    lobs = classify_lobs_multi_openai(client, model, text)
    
    # Extract fields for each LOB
    results = []
    for lob in lobs:
        fields = extract_fields_openai_chunked(client, model, text, lob, None, max_chunk_size, api_delay)
        results.append({
            'lob': lob,
            'carrier': fields.get('carrier', ''),
            'evaluation_date': fields.get('evaluation_date', ''),
            'claims': fields.get('claims', [])
        })
        time.sleep(api_delay)
    
    return {
        'success': True,
        'detected_lobs': lobs,
        'results': results,
        'source_file': Path(pdf_path).name
    }


# ============================================================================
# Email Agent Functions
# ============================================================================

def send_extracted_data_email(pdf_info: Dict, extracted_data: Dict, recipient_email: Optional[str] = None):
    """
    Send extracted data via email using email_agent
    """
    if not send_email_action:
        return False, "Email agent not available"
    
    # Determine recipient
    lob = pdf_info.get('lob', 'UNKNOWN')
    policy_number = pdf_info.get('policy_number', 'Unknown')
    
    if recipient_email:
        # Override default recipient
        original_recipient = LOB_EMAILS.get(lob)
        LOB_EMAILS[lob] = recipient_email
    
    try:
        # Create a summary of extracted data
        summary_text = f"""
Extracted Claims Data Summary:
- Account: {pdf_info.get('account', 'N/A')}
- LOB: {lob}
- Policy Number: {policy_number}
- Evaluation Date: {extracted_data.get('results', [{}])[0].get('evaluation_date', 'N/A') if extracted_data.get('results') else 'N/A'}

Detected LOBs: {', '.join(extracted_data.get('detected_lobs', []))}

Total Claims Extracted: {sum(len(r.get('claims', [])) for r in extracted_data.get('results', []))}
"""
        
        # Send email with PDF attachment
        success, message = send_email_action(pdf_info, lob, policy_number)
        
        if success:
            # Optionally send extracted data as JSON attachment
            # (This would require modifying send_email_action to accept additional data)
            return True, f"Email sent successfully: {message}"
        else:
            return False, f"Email failed: {message}"
    
    except Exception as e:
        return False, f"Error sending email: {str(e)}"
    finally:
        # Restore original recipient if overridden
        if recipient_email and lob in LOB_EMAILS:
            LOB_EMAILS[lob] = original_recipient


# ============================================================================
# Streamlit App
# ============================================================================

def main():
    st.set_page_config(
        page_title="AI Agent Reference App",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .pdf-card {
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 0.5rem;
            background-color: #f9f9f9;
        }
        .pdf-card:hover {
            background-color: #f0f0f0;
            border-color: #1f77b4;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🤖 AI Agent Reference App</h1>', unsafe_allow_html=True)
    
    # Load configuration
    cfg = load_config()
    
    if not cfg:
        st.error("❌ Could not load config.py. Please ensure config.py exists and is properly configured.")
        st.stop()
    
    # Sidebar - Configuration Status
    st.sidebar.header("⚙️ Configuration")
    
    if cfg.get('use_azure'):
        st.sidebar.success("✅ Azure OpenAI Configured")
        st.sidebar.write(f"**Endpoint:** {cfg.get('azure_endpoint', 'N/A')[:50]}...")
        st.sidebar.write(f"**Deployment:** {cfg.get('azure_deployment', 'N/A')}")
    elif cfg.get('openai_api_key'):
        st.sidebar.success("✅ OpenAI Configured")
        st.sidebar.write(f"**Model:** {cfg.get('openai_model', 'N/A')}")
    else:
        st.sidebar.error("❌ No API configuration found")
    
    st.sidebar.write(f"**Max Chunk Size:** {cfg.get('max_chunk_size', 15000)}")
    st.sidebar.write(f"**API Delay:** {cfg.get('api_delay', 0.3)}s")
    
    # Setup OpenAI client
    client = setup_openai_client(cfg)
    if not client:
        st.error("❌ Failed to setup OpenAI client. Check your configuration.")
        st.stop()
    
    model = cfg.get('azure_deployment') if cfg.get('use_azure') else cfg.get('openai_model', 'gpt-4o-2024-08-06')
    
    # ========================================================================
    # Section 1: Search Documents using LLM
    # ========================================================================
    st.header("🔍 Search Documents")
    
    # Automatically scan for PDFs on page load
    # Store in session state to avoid re-scanning on every rerun
    if 'pdf_files' not in st.session_state:
        with st.spinner("🔍 Scanning for PDF files..."):
            st.session_state['pdf_files'] = scan_local_pdfs("mock_documents")
    
    all_pdf_files = st.session_state['pdf_files']
    
    # Search interface
    col_search, col_clear = st.columns([4, 1])
    
    with col_search:
        search_query = st.text_input(
            "Search documents (e.g., 'auto', 'find WC documents', 'Chubbs property', 'policy 1234')",
            value=st.session_state.get('search_query', ''),
            placeholder="Type your search query here...",
            help="Use natural language to search. Examples: 'auto', 'WC', 'Chubbs', 'property claims', 'policy 7890'"
        )
    
    with col_clear:
        st.write("")  # Spacing
        if st.button("🗑️ Clear"):
            st.session_state['search_query'] = ''
            st.session_state['filtered_pdfs'] = None
            st.rerun()
    
    # Process search query with LLM
    filtered_pdfs = st.session_state.get('filtered_pdfs')
    
    if search_query:
        if st.button("🔎 Search", type="primary") or st.session_state.get('search_query') != search_query:
            st.session_state['search_query'] = search_query
            
            with st.spinner("🤖 Analyzing search query with AI..."):
                # Use LLM to extract search criteria
                search_criteria = extract_search_criteria_with_llm(client, model, search_query, all_pdf_files)
                
                # Filter PDFs based on criteria
                filtered_pdfs = filter_pdfs_by_criteria(all_pdf_files, search_criteria)
                st.session_state['filtered_pdfs'] = filtered_pdfs
                st.session_state['search_criteria'] = search_criteria
    
    # Display search results or all PDFs
    if filtered_pdfs is not None:
        pdf_files = filtered_pdfs
        search_criteria = st.session_state.get('search_criteria', {})
        
        if search_criteria:
            st.success(f"✅ Found {len(pdf_files)} document(s) matching your search")
            with st.expander("🔍 Search Criteria (click to see)", expanded=False):
                criteria_text = []
                if search_criteria.get('lob'):
                    criteria_text.append(f"**LOB:** {search_criteria['lob']}")
                if search_criteria.get('account'):
                    criteria_text.append(f"**Account:** {search_criteria['account']}")
                if search_criteria.get('policy_number'):
                    criteria_text.append(f"**Policy:** {search_criteria['policy_number']}")
                if search_criteria.get('keywords'):
                    criteria_text.append(f"**Keywords:** {', '.join(search_criteria['keywords'])}")
                st.markdown(" | ".join(criteria_text) if criteria_text else "No specific criteria")
    else:
        pdf_files = all_pdf_files
    
    # Refresh button
    col_refresh, col_count = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refresh PDF List"):
            with st.spinner("🔍 Scanning for PDF files..."):
                st.session_state['pdf_files'] = scan_local_pdfs("mock_documents")
                all_pdf_files = st.session_state['pdf_files']
                # Re-apply search if active
                if search_query:
                    search_criteria = extract_search_criteria_with_llm(client, model, search_query, all_pdf_files)
                    filtered_pdfs = filter_pdfs_by_criteria(all_pdf_files, search_criteria)
                    st.session_state['filtered_pdfs'] = filtered_pdfs
                else:
                    st.session_state['filtered_pdfs'] = None
            st.rerun()
    with col_count:
        if pdf_files:
            if filtered_pdfs is not None:
                st.info(f"📊 Showing {len(pdf_files)} of {len(all_pdf_files)} PDF file(s)")
            else:
                st.info(f"📊 Found {len(pdf_files)} PDF file(s) in local structure")
    
    if not pdf_files:
        st.warning("⚠️ No PDFs found in mock_documents directory. Please ensure the directory structure is correct.")
        st.info("Expected structure: mock_documents/AccountName/LOB/PolicyNo-Date/filename.pdf")
    else:
        st.success(f"✅ Found {len(pdf_files)} PDF file(s)")
        
        # Group by account
        accounts = {}
        for pdf in pdf_files:
            account = pdf['account']
            if account not in accounts:
                accounts[account] = []
            accounts[account].append(pdf)
        
        # Display PDFs grouped by account
        for account, pdfs in accounts.items():
            with st.expander(f"📂 {account} ({len(pdfs)} files)", expanded=False):
                for pdf in pdfs:
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{pdf['filename']}**")
                        st.caption(f"LOB: {pdf['lob']} | Policy: {pdf['policy_number']} | Date: {pdf['effective_date']}")
                    
                    with col2:
                        st.text(f"Path: {pdf['folder_path']}")
                    
                    with col3:
                        if st.button("Select", key=f"select_{pdf['path']}"):
                            st.session_state['selected_pdf'] = pdf
                            st.rerun()
        
        # ====================================================================
        # Section 2: Selected PDF Processing
        # ====================================================================
        st.markdown("---")
        st.header("🔬 Selected PDF Processing")
        
        selected_pdf = st.session_state.get('selected_pdf')
        
        if selected_pdf:
            st.info(f"📄 Selected: **{selected_pdf['filename']}**")
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Account", selected_pdf['account'])
            with col_info2:
                st.metric("LOB", selected_pdf['lob'])
            with col_info3:
                st.metric("Policy", selected_pdf['policy_number'])
            
            # PDF Preview
            if fitz:
                try:
                    doc = fitz.open(selected_pdf['path'])
                    total_pages = len(doc)
                    
                    st.subheader("📑 PDF Preview")
                    page_num = st.number_input("Page to preview", min_value=1, max_value=total_pages, value=1, step=1)
                    
                    page = doc[page_num - 1]
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    st.image(img, use_container_width=True, caption=f"Page {page_num} of {total_pages}")
                    
                    doc.close()
                except Exception as e:
                    st.warning(f"Could not preview PDF: {e}")
            
            # Processing Section
            st.subheader("🚀 Process PDF")
            
            if st.button("▶️ Start Processing", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("📖 Extracting text from PDF...")
                    progress_bar.progress(0.2)
                    
                    # Extract and process
                    status_text.text("🤖 Processing with AI...")
                    progress_bar.progress(0.5)
                    
                    extracted_data = process_pdf_with_openai(
                        selected_pdf['path'],
                        client,
                        model,
                        cfg.get('max_chunk_size', 15000),
                        cfg.get('api_delay', 0.3)
                    )
                    
                    progress_bar.progress(0.9)
                    
                    if extracted_data.get('success'):
                        st.session_state['extracted_data'] = extracted_data
                        st.session_state['processed_pdf'] = selected_pdf
                        
                        status_text.text("✅ Processing complete!")
                        progress_bar.progress(1.0)
                        st.success("🎉 PDF processed successfully!")
                    else:
                        st.error(f"❌ Processing failed: {extracted_data.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                finally:
                    progress_bar.empty()
            
            # ================================================================
            # Section 3: Display Extracted Results
            # ================================================================
            extracted_data = st.session_state.get('extracted_data')
            processed_pdf = st.session_state.get('processed_pdf')
            
            if extracted_data and processed_pdf and processed_pdf['path'] == selected_pdf['path']:
                st.markdown("---")
                st.header("📊 Extracted Data Results")
                
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Detected LOBs", len(extracted_data.get('detected_lobs', [])))
                with col2:
                    total_claims = sum(len(r.get('claims', [])) for r in extracted_data.get('results', []))
                    st.metric("Total Claims", total_claims)
                with col3:
                    carrier = extracted_data.get('results', [{}])[0].get('carrier', 'N/A') if extracted_data.get('results') else 'N/A'
                    st.metric("Carrier", carrier[:20] if carrier else 'N/A')
                
                # Display results by LOB
                for lob_result in extracted_data.get('results', []):
                    lob = lob_result.get('lob', 'Unknown')
                    claims = lob_result.get('claims', [])
                    
                    with st.expander(f"📋 {lob} Claims ({len(claims)} found)", expanded=True):
                        st.write(f"**Carrier:** {lob_result.get('carrier', 'N/A')}")
                        st.write(f"**Evaluation Date:** {lob_result.get('evaluation_date', 'N/A')}")
                        
                        if claims:
                            df = pd.DataFrame(claims)
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button for this LOB
                            output = pd.ExcelWriter(io.BytesIO(), engine='openpyxl')
                            df.to_excel(output, sheet_name=f"{lob}_claims", index=False)
                            output.close()
                            
                            excel_bytes = output.buffer.getvalue()
                            st.download_button(
                                label=f"📥 Download {lob} Claims (Excel)",
                                data=excel_bytes,
                                file_name=f"{selected_pdf['filename']}_{lob}_claims.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key=f"download_{lob}"
                            )
                        else:
                            st.info("No claims found for this LOB")
                
                # ============================================================
                # Section 4: Email Agent
                # ============================================================
                st.markdown("---")
                st.header("📧 Send Extracted Data via Email")
                
                col_email1, col_email2 = st.columns([2, 1])
                
                with col_email1:
                    recipient_email = st.text_input(
                        "Recipient Email (optional - uses default LOB email if empty)",
                        value="",
                        help="Leave empty to use default email for this LOB"
                    )
                
                with col_email2:
                    st.write("")  # Spacing
                    send_email_btn = st.button("📤 Send Email", type="primary")
                
                if send_email_btn:
                    if send_email_action:
                        with st.spinner("Sending email..."):
                            success, message = send_extracted_data_email(
                                selected_pdf,
                                extracted_data,
                                recipient_email if recipient_email else None
                            )
                            
                            if success:
                                st.success(f"✅ {message}")
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.error("Email agent not available. Please ensure email_agent.py is properly configured.")
                
                # Download all results
                st.markdown("---")
                st.subheader("📥 Download All Results")
                
                if st.button("💾 Download Complete Results (Excel)"):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for lob_result in extracted_data.get('results', []):
                            lob = lob_result.get('lob', 'Unknown')
                            claims = lob_result.get('claims', [])
                            if claims:
                                df = pd.DataFrame(claims)
                                sheet_name = f"{lob}_claims"[:31]  # Excel sheet name limit
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    output.seek(0)
                    st.download_button(
                        label="⬇️ Download All Claims (Excel)",
                        data=output.getvalue(),
                        file_name=f"{selected_pdf['filename']}_all_claims.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.info("👆 Select a PDF from the list above to begin processing")
    
    # Sidebar - Additional Info
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.write("""
    This app:
    1. Scans local file structure for PDFs
    2. Uses config.py for Azure OpenAI settings
    3. Processes selected PDFs with AI
    4. Sends extracted data via email
    """)
    
    if st.sidebar.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            if key not in ['config']:
                del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()

