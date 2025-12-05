#!/usr/bin/env python3
"""
Streamlit App for Claims PDF Parsing
Upload PDF files, extract claims data using AI, and download as Excel
"""

import io
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
        
        # AWS Bedrock / Claude settings
        cfg['aws_access_key'] = getattr(config_module, 'AWS_ACCESS_KEY', None)
        cfg['aws_secret_key'] = getattr(config_module, 'AWS_SECRET_KEY', None)
        cfg['aws_session_token'] = getattr(config_module, 'AWS_SESSION_TOKEN', None)
        cfg['aws_region'] = getattr(config_module, 'AWS_REGION', 'us-east-1')
        cfg['model_id'] = getattr(config_module, 'MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')
        
        # Processing settings
        cfg['extractor_engine'] = getattr(config_module, 'EXTRACTOR_ENGINE', 'claude')
        cfg['max_chunk_size'] = getattr(config_module, 'MAX_CHUNK_SIZE', 15000)
        cfg['api_delay'] = getattr(config_module, 'API_DELAY', 1)
        
        # Camelot settings
        cfg['use_camelot'] = getattr(config_module, 'USE_CAMELOT', True)
        cfg['camelot_flavors'] = getattr(config_module, 'CAMELOT_FLAVORS', ['lattice', 'stream'])
        cfg['camelot_pages'] = getattr(config_module, 'CAMELOT_PAGES', 'all')
        
        # Validation - check if at least one API is configured
        if cfg['use_azure']:
            missing = [k for k in ['azure_endpoint', 'azure_api_key', 'azure_deployment'] if not cfg[k]]
            if missing:
                return None
        elif not cfg['openai_api_key'] and not cfg['aws_access_key']:
            # Allow config without API keys if user will provide them via UI
            pass
        
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
            # Validate Azure OpenAI settings
            if not all([cfg.get('azure_api_key'), cfg.get('azure_endpoint'), cfg.get('azure_deployment')]):
                return None
            
            client = OpenAI(
                api_key=cfg['azure_api_key'],
                base_url=f"{cfg['azure_endpoint'].rstrip('/')}/openai/deployments/{cfg['azure_deployment']}",
            )
            return client
        else:
            # Validate OpenAI settings
            if not cfg.get('openai_api_key'):
                return None
            
            client = OpenAI(api_key=cfg['openai_api_key'])
            return client
    except Exception as e:
        st.error(f"Error setting up OpenAI client: {e}")
        return None


# ============================================================================
# PDF Text Extraction
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from uploaded PDF file"""
    text_content = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                
                # Also try to extract tables
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


# ============================================================================
# AI Processing Functions
# ============================================================================

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


def _extract_carrier_from_text(text: str) -> str:
    """Extract carrier name from text using regex patterns"""
    import re
    patterns = [
        r"\b(?:Carrier|company|insurer|provider)\s*[:\-]\s*([A-Za-z0-9 &'.\-/]+)",
        r"\b([A-Z][A-Za-z0-9 &'.\-/]+(?=Insurance|Ins|Corp|Corporation|Company|Co|LLC|Inc))\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) > 2:
                return candidate
    return ""


def _extract_carrier_from_filename(filename: str) -> str:
    """Extract carrier name from filename"""
    import re
    stem = Path(filename).stem.replace('_', ' ').replace('-', ' ').replace('.', ' ')
    m = re.search(r"\b([A-Z][A-Za-z0-9 &'.\-/]+(?=Insurance|Ins|Corp|Corporation|Company|Co|LLC|Inc))\b", stem, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


# ============================================================================
# Data Processing
# ============================================================================

def process_pdf_content(text: str, filename: str, client, model: str, progress_placeholder, max_chunk_size: int = 15000, api_delay: float = 0.3) -> List[Dict]:
    """Process extracted PDF text and return results"""
    results: List[Dict] = []
    
    progress_placeholder.text("🔍 Detecting Lines of Business...")
    lobs = classify_lobs_multi_openai(client, model, text)
    progress_placeholder.text(f"📋 Detected LoBs: {', '.join(lobs)}")
    
    for i, lob in enumerate(lobs):
        progress_placeholder.text(f"📊 Extracting {lob} claims ({i+1}/{len(lobs)})...")
        
        def update_progress(current, total):
            progress_placeholder.text(f"📊 Extracting {lob} claims - chunk {current}/{total}...")
        
        fields = extract_fields_openai_chunked(client, model, text, lob, update_progress, max_chunk_size, api_delay)
        carrier = fields.get('carrier') or _extract_carrier_from_text(text) or _extract_carrier_from_filename(filename)
        
        results.append({
            'lob': lob,
            'carrier': carrier,
            'fields': fields,
            'source_file': filename
        })
    
    return results


def results_to_dataframes(all_results: List[Dict]) -> Dict[str, pd.DataFrame]:
    """Convert processing results to DataFrames organized by LoB"""
    auto_rows: List[Dict] = []
    property_rows: List[Dict] = []
    gl_rows: List[Dict] = []
    wc_rows: List[Dict] = []
    
    for result in all_results:
        lob = result['lob']
        carrier = result['carrier']
        fields = result['fields']
        source_file = result['source_file']
        
        if lob == 'AUTO':
            for c in fields.get('claims', []):
                auto_rows.append({
                    'evaluation_date': fields.get('evaluation_date', ''),
                    'carrier': c.get('carrier', '') or carrier or fields.get('carrier', ''),
                    'claim_number': c.get('claim_number', ''),
                    'loss_date': c.get('loss_date', ''),
                    'paid_loss': c.get('paid_loss', ''),
                    'reserve': c.get('reserve', ''),
                    'alae': c.get('alae', ''),
                    'source_file': source_file
                })
        elif lob == 'PROPERTY':
            for c in fields.get('claims', []):
                property_rows.append({
                    'evaluation_date': fields.get('evaluation_date', ''),
                    'carrier': c.get('carrier', '') or carrier or fields.get('carrier', ''),
                    'claim_number': c.get('claim_number', ''),
                    'loss_date': c.get('loss_date', ''),
                    'paid_loss': c.get('paid_loss', ''),
                    'reserve': c.get('reserve', ''),
                    'alae': c.get('alae', ''),
                    'source_file': source_file
                })
        elif lob in ('GENERAL LIABILITY', 'GL'):
            for c in fields.get('claims', []):
                gl_rows.append({
                    'evaluation_date': fields.get('evaluation_date', ''),
                    'carrier': c.get('carrier', '') or carrier or fields.get('carrier', ''),
                    'claim_number': c.get('claim_number', ''),
                    'loss_date': c.get('loss_date', ''),
                    'bi_paid_loss': c.get('bi_paid_loss', ''),
                    'pd_paid_loss': c.get('pd_paid_loss', ''),
                    'bi_reserve': c.get('bi_reserve', ''),
                    'pd_reserve': c.get('pd_reserve', ''),
                    'alae': c.get('alae', ''),
                    'source_file': source_file
                })
        elif lob == 'WC':
            for c in fields.get('claims', []):
                wc_rows.append({
                    'evaluation_date': fields.get('evaluation_date', ''),
                    'carrier': c.get('carrier', '') or carrier or fields.get('carrier', ''),
                    'claim_number': c.get('claim_number', ''),
                    'loss_date': c.get('loss_date', ''),
                    'Indemnity_paid_loss': c.get('Indemnity_paid_loss', ''),
                    'Medical_paid_loss': c.get('Medical_paid_loss', ''),
                    'Indemnity_reserve': c.get('Indemnity_reserve', ''),
                    'Medical_reserve': c.get('Medical_reserve', ''),
                    'ALAE': c.get('ALAE', ''),
                    'source_file': source_file
                })
    
    per_lob = {}
    if auto_rows:
        per_lob['AUTO'] = pd.DataFrame(auto_rows, columns=['evaluation_date', 'carrier', 'claim_number', 'loss_date', 'paid_loss', 'reserve', 'alae', 'source_file'])
    if property_rows:
        per_lob['PROPERTY'] = pd.DataFrame(property_rows, columns=['evaluation_date', 'carrier', 'claim_number', 'loss_date', 'paid_loss', 'reserve', 'alae', 'source_file'])
    if gl_rows:
        per_lob['GL'] = pd.DataFrame(gl_rows, columns=['evaluation_date', 'carrier', 'claim_number', 'loss_date', 'bi_paid_loss', 'pd_paid_loss', 'bi_reserve', 'pd_reserve', 'alae', 'source_file'])
    if wc_rows:
        per_lob['WC'] = pd.DataFrame(wc_rows, columns=['evaluation_date', 'carrier', 'claim_number', 'loss_date', 'Indemnity_paid_loss', 'Medical_paid_loss', 'Indemnity_reserve', 'Medical_reserve', 'ALAE', 'source_file'])
    
    return per_lob


def create_excel_download(per_lob: Dict[str, pd.DataFrame]) -> bytes:
    """Create Excel file with all LoB sheets"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for lob, df in per_lob.items():
            if not df.empty:
                sheet_name = f"{lob.lower()}_claims"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.getvalue()


# ============================================================================
# Streamlit App
# ============================================================================

# EXL Custom CSS
EXL_CUSTOM_CSS = """
<style>
    /* Force light theme */
    :root {
        --background-color: #d6d3d1 !important;
        --secondary-background-color: #F1F5F9 !important;
        --text-color: #1E293B !important;
    }
    
    /* Main app background */
    .stApp {
        background-color: #d6d3d1 !important;
    }
    
    /* Sidebar styling - force light */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #1E293B !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #1E293B !important;
    }
    
    /* Hide Streamlit branding and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stDecoration"] {display: none;}
    
    /* Hide top-right hamburger menu */
    button[kind="header"] {display: none;}
    
    /* Custom header bar */
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 1rem 2rem;
        border-radius: 0 0 10px 10px;
        margin: -1rem -1rem 1rem -1rem;
        color: white;
    }
    
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 1.8rem;
    }
    
    .main-header .subtitle {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }
    
    /* EXL Logo styling */
    .exl-logo {
        font-weight: bold;
        font-size: 1.5rem;
        color: #1E3A8A;
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.4);
    }
    
    /* All buttons light theme */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s ease;
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
    }
    
    /* Cards/Expanders styling */
    .stExpander {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    
    /* Date inputs */
    .stDateInput > div > div > input {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #1E3A8A !important;
        font-weight: 700;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF !important;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        background-color: #F1F5F9 !important;
        color: #1E293B !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #ECFDF5 !important;
        border-left: 4px solid #10B981;
        border-radius: 0 8px 8px 0;
    }
    
    .stError {
        background-color: #FEF2F2 !important;
        border-left: 4px solid #EF4444;
        border-radius: 0 8px 8px 0;
    }
    
    /* Info messages */
    .stInfo {
        background-color: #EFF6FF !important;
        border-left: 4px solid #3B82F6;
        border-radius: 0 8px 8px 0;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #FFFBEB !important;
        border-left: 4px solid #F59E0B;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #1E3A8A;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #1E3A8A !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #1E3A8A;
    }
    
    /* Custom divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #CBD5E1, transparent);
        margin: 1.5rem 0;
    }
</style>
"""

def main():
    st.set_page_config(
        page_title="EXL Claims Document Manager",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None  # Hide the menu
    )
    
    # Inject custom CSS
    st.markdown(EXL_CUSTOM_CSS, unsafe_allow_html=True)
    
    # Custom header with EXL branding
    st.markdown("""
        <div class="main-header">
            <h1>📄 EXL Claims Document Manager</h1>
            <div class="subtitle">AI-Powered Insurance Document Processing</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load config at the start - available to both tabs
    if 'app_config' not in st.session_state:
        st.session_state['app_config'] = load_config()
    
    cfg = st.session_state.get('app_config')
    
    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["📤 PDF Parser", "🤖 AI Agent"])
    
    # ========================================================================
    # TAB 1: PDF Parser (Existing Functionality)
    # ========================================================================
    with tab1:
        st.markdown("Upload insurance claims PDFs to extract structured data and download as Excel.")
        
        # Sidebar for configuration
        st.sidebar.header("⚙️ Configuration")
        
        if cfg:
            st.sidebar.success("✅ Config loaded from config.py")
            use_config_file = st.sidebar.checkbox("Use config.py settings", value=True)
            
            # Display config info
            with st.sidebar.expander("📋 Config Details", expanded=False):
                st.write(f"**Extractor Engine:** {cfg.get('extractor_engine', 'claude')}")
                st.write(f"**Max Chunk Size:** {cfg.get('max_chunk_size', 15000)}")
                st.write(f"**API Delay:** {cfg.get('api_delay', 1)}s")
                
                # Show Azure OpenAI info if configured
                if cfg.get('use_azure'):
                    st.write(f"**Provider:** Azure OpenAI")
                    st.write(f"**Azure Endpoint:** {cfg.get('azure_endpoint', 'N/A')[:50]}...")
                    st.write(f"**Deployment:** {cfg.get('azure_deployment', 'N/A')}")
                elif cfg.get('openai_api_key'):
                    st.write(f"**Provider:** OpenAI")
                    st.write(f"**OpenAI Model:** {cfg.get('openai_model', 'N/A')}")
                
                # Show AWS/Claude info if configured
                if cfg.get('aws_access_key'):
                    st.write(f"**AWS Region:** {cfg.get('aws_region', 'us-east-1')}")
                    st.write(f"**Claude Model:** {cfg.get('model_id', 'N/A')}")
        else:
            use_config_file = False
            st.sidebar.info("No config.py found. Enter API keys below.")
        
        # Allow override only if user explicitly unchecks the config file option
        if not use_config_file or not cfg:
            st.sidebar.subheader("API Settings")
            use_azure = st.sidebar.checkbox("Use Azure OpenAI", value=cfg.get('use_azure', False) if cfg else False)
            
            if use_azure:
                azure_endpoint = st.sidebar.text_input(
                    "Azure Endpoint", 
                    value=cfg.get('azure_endpoint', '') if cfg else '',
                    type="default"
                )
                azure_api_key = st.sidebar.text_input(
                    "Azure API Key", 
                    value=cfg.get('azure_api_key', '') if cfg else '',
                    type="password"
                )
                azure_deployment = st.sidebar.text_input(
                    "Azure Deployment Name",
                    value=cfg.get('azure_deployment', '') if cfg else ''
                )
                cfg = {
                    'use_azure': True,
                    'azure_endpoint': azure_endpoint,
                    'azure_api_key': azure_api_key,
                    'azure_deployment': azure_deployment,
                    'max_chunk_size': cfg.get('max_chunk_size', 15000) if cfg else 15000,
                    'api_delay': cfg.get('api_delay', 0.3) if cfg else 0.3,
                }
            else:
                openai_api_key = st.sidebar.text_input(
                    "OpenAI API Key", 
                    value=cfg.get('openai_api_key', '') if cfg else '',
                    type="password"
                )
                openai_model = st.sidebar.selectbox(
                    "Model",
                    ["gpt-4o-2024-08-06", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                    index=0
                )
                cfg = {
                    'use_azure': False,
                    'openai_api_key': openai_api_key,
                    'openai_model': openai_model,
                    'max_chunk_size': cfg.get('max_chunk_size', 15000) if cfg else 15000,
                    'api_delay': cfg.get('api_delay', 0.3) if cfg else 0.3,
                }
            
            # Update session state with new config
            st.session_state['app_config'] = cfg
        else:
            # Ensure session state is updated with loaded config
            st.session_state['app_config'] = cfg
        
        # File uploader
        st.header("📤 Upload PDF Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more insurance claims PDF files"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) uploaded")
            
            # Display uploaded files
            with st.expander("View uploaded files"):
                for f in uploaded_files:
                    st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")
        
        # Process button
        if st.button("🚀 Process PDFs", type="primary", disabled=not uploaded_files):
            # Validate configuration
            if cfg.get('use_azure'):
                if not all([cfg.get('azure_endpoint'), cfg.get('azure_api_key'), cfg.get('azure_deployment')]):
                    st.error("Please provide all Azure OpenAI settings")
                    return
            else:
                if not cfg.get('openai_api_key'):
                    st.error("Please provide OpenAI API key")
                    return
            
            # Setup client
            client = setup_openai_client(cfg)
            if not client:
                st.error("Failed to setup OpenAI client. Check your API settings.")
                return
            
            model = cfg.get('azure_deployment') if cfg.get('use_azure') else cfg.get('openai_model', 'gpt-4o-2024-08-06')
            
            all_results = []
            
            # Process each file
            progress_bar = st.progress(0)
            status_container = st.container()
            
            for i, pdf_file in enumerate(uploaded_files):
                with status_container:
                    st.subheader(f"Processing: {pdf_file.name}")
                    progress_placeholder = st.empty()
                    
                    # Extract text from PDF
                    progress_placeholder.text("📖 Extracting text from PDF...")
                    text_content = extract_text_from_pdf(pdf_file)
                    
                    if not text_content:
                        st.warning(f"⚠️ Could not extract text from {pdf_file.name}")
                        continue
                    
                    progress_placeholder.text(f"📖 Extracted {len(text_content)} characters")
                    
                    # Process the content
                    max_chunk_size = cfg.get('max_chunk_size', 15000) if cfg else 15000
                    api_delay = cfg.get('api_delay', 0.3) if cfg else 0.3
                    results = process_pdf_content(
                        text_content,
                        pdf_file.name,
                        client,
                        model,
                        progress_placeholder,
                        max_chunk_size=max_chunk_size,
                        api_delay=api_delay
                    )
                    
                    all_results.extend(results)
                    progress_placeholder.text(f"✅ Completed processing {pdf_file.name}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Convert to DataFrames
            if all_results:
                st.success("✅ Processing complete!")
                
                per_lob = results_to_dataframes(all_results)
                
                # Store in session state for display
                st.session_state['per_lob'] = per_lob
                st.session_state['processed'] = True
            else:
                st.warning("No claims data found in the uploaded files.")
        
        # Display results
        if st.session_state.get('processed') and st.session_state.get('per_lob'):
            per_lob = st.session_state['per_lob']
            
            st.header("📊 Results")
            
            # Summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                auto_count = len(per_lob.get('AUTO', []))
                st.metric("AUTO Claims", auto_count)
            with col2:
                property_count = len(per_lob.get('PROPERTY', []))
                st.metric("PROPERTY Claims", property_count)
            with col3:
                gl_count = len(per_lob.get('GL', []))
                st.metric("GL Claims", gl_count)
            with col4:
                wc_count = len(per_lob.get('WC', []))
                st.metric("WC Claims", wc_count)
            
            # Display data in tabs
            tabs = st.tabs(["AUTO", "PROPERTY", "GL", "WC"])
            
            with tabs[0]:
                if 'AUTO' in per_lob and not per_lob['AUTO'].empty:
                    st.dataframe(per_lob['AUTO'], use_container_width=True)
                else:
                    st.info("No AUTO claims found")
            
            with tabs[1]:
                if 'PROPERTY' in per_lob and not per_lob['PROPERTY'].empty:
                    st.dataframe(per_lob['PROPERTY'], use_container_width=True)
                else:
                    st.info("No PROPERTY claims found")
            
            with tabs[2]:
                if 'GL' in per_lob and not per_lob['GL'].empty:
                    st.dataframe(per_lob['GL'], use_container_width=True)
                else:
                    st.info("No GL claims found")
            
            with tabs[3]:
                if 'WC' in per_lob and not per_lob['WC'].empty:
                    st.dataframe(per_lob['WC'], use_container_width=True)
                else:
                    st.info("No WC claims found")
            
            # Download button
            st.header("📥 Download Results")
            
            if any(lob in per_lob and not per_lob[lob].empty for lob in ['AUTO', 'PROPERTY', 'GL', 'WC']):
                excel_data = create_excel_download(per_lob)
                
                st.download_button(
                    label="⬇️ Download Excel File",
                    data=excel_data,
                    file_name="claims_extracted.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
            else:
                st.info("No data available to download")

    # ========================================================================
    # TAB 2: AI Agent (New Functionality)
    # ========================================================================
    with tab2:
        st.header("🤖 AI Claims Document Agent")
        st.markdown("Search for loss run reports using natural language, preview documents, send emails, and parse PDFs.")
        
        # Show config status
        app_cfg = st.session_state.get('app_config')
        if app_cfg:
            if app_cfg.get('use_azure'):
                st.info(f"✅ Using Azure OpenAI - Deployment: {app_cfg.get('azure_deployment', 'N/A')}")
            elif app_cfg.get('openai_api_key'):
                st.info(f"✅ Using OpenAI - Model: {app_cfg.get('openai_model', 'N/A')}")
            else:
                st.warning("⚠️ No OpenAI configuration found. Configure in Tab 1 or create config.py")
        else:
            st.warning("⚠️ No configuration found. Please configure in Tab 1 or create config.py")
        
        # Import the agent module dynamically
        try:
            import email_agent
            try:
                import cloud_storage as doc_storage
            except ImportError:
                import mock_storage as doc_storage
            
            # Initialize mock data if needed
            if 'mock_data_created' not in st.session_state:
                if hasattr(doc_storage, 'create_mock_data'):
                    doc_storage.create_mock_data()
                st.session_state['mock_data_created'] = True
                
        except ImportError as e:
            st.error(f"Agent dependencies missing: {e}")
            st.stop()

        # ====================================================================
        # Section 1: Natural Language Search Input
        # ====================================================================
        st.subheader("🔍 Search Documents")
        
        user_query = st.text_input(
            "Enter your search query:", 
            placeholder="e.g., chubbs auto policy 12345 date 2024-2025",
            help="Type naturally - AI will extract Account, LOB, Policy, and Date Range"
        )
        
        col_extract, col_clear = st.columns([1, 5])
        
        with col_extract:
            extract_clicked = st.button("🤖 Extract", type="primary")
        
        with col_clear:
            if st.button("🗑️ Clear All"):
                for key in ['extracted_info', 'agent_files', 'selected_agent_files', 'parse_results']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Extract information from query
        if extract_clicked and user_query:
            with st.spinner("Extracting information..."):
                try:
                    # Get config from session state
                    app_cfg = st.session_state.get('app_config')
                    
                    if not app_cfg:
                        st.error("❌ No configuration found. Please configure in Tab 1 or create config.py")
                    else:
                        # Setup OpenAI client
                        client = setup_openai_client(app_cfg)
                        if not client:
                            st.error("❌ Failed to setup OpenAI client. Check your configuration in Tab 1.")
                        else:
                            # Get model name
                            model = app_cfg.get('azure_deployment') if app_cfg.get('use_azure') else app_cfg.get('openai_model', 'gpt-4o-2024-08-06')
                            
                            accounts = ", ".join(email_agent.VALID_ACCOUNTS)
                            lobs = ", ".join(email_agent.VALID_LOBS)
                            
                            # Define JSON schema for extraction
                            schema = {
                                "account_name": "string (Account name from valid list)",
                                "lob": "string (Line of Business: AUTO, PROPERTY, GL, or WC)",
                                "policy_number": "string (Policy number or claim number)",
                                "start_date": "string (Start date in DD-MM-YYYY format)",
                                "end_date": "string (End date in DD-MM-YYYY format)"
                            }
                            
                            prompt = f"""Extract search criteria from the user query and return as JSON.

Valid Accounts: {accounts}
Valid LOBs: {lobs}

Extraction Rules:
- Map account variations: "chubbs/chubb" → "Chubbs", "amex" → "Amex", "globalinsure" → "GlobalInsure", "techcorp" → "TechCorp", "travelers" → "Travelers"
- Map LOB variations: "auto/vehicle/car" → "AUTO", "work/wc/workers comp" → "WC", "property" → "PROPERTY", "gl/general liability" → "GL"
- Date ranges: "2024-2025" → start_date: "01-01-2024", end_date: "31-12-2025"
- "less than 2024" or "before 2024" → end_date: "31-12-2024"
- "after 2024" → start_date: "01-01-2024"
- Return empty string "" for missing fields
- Always return valid JSON matching this schema: {schema}

User Query: {user_query}

Return ONLY valid JSON with no markdown, no code blocks, no explanation:"""
                            
                            try:
                                resp = client.chat.completions.create(
                                    model=model,
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.0,
                                    max_tokens=500,
                                    response_format={"type": "json_object"}
                                )
                                
                                content = resp.choices[0].message.content
                                extracted = json.loads(content)
                                
                                # Validate and normalize extracted data
                                if not isinstance(extracted, dict):
                                    extracted = {}
                                
                                # Normalize account name
                                account = extracted.get('account_name', '').strip()
                                if account:
                                    account_lower = account.lower()
                                    for valid_account in email_agent.VALID_ACCOUNTS:
                                        if valid_account.lower() in account_lower or account_lower in valid_account.lower():
                                            extracted['account_name'] = valid_account
                                            break
                                
                                # Normalize LOB
                                lob = extracted.get('lob', '').strip().upper()
                                if lob:
                                    if 'AUTO' in lob or 'VEHICLE' in lob or 'CAR' in lob:
                                        extracted['lob'] = 'AUTO'
                                    elif 'WC' in lob or 'WORK' in lob or 'WORKERS' in lob:
                                        extracted['lob'] = 'WC'
                                    elif 'PROPERTY' in lob:
                                        extracted['lob'] = 'PROPERTY'
                                    elif 'GL' in lob or 'GENERAL' in lob or 'LIABILITY' in lob:
                                        extracted['lob'] = 'GL'
                                    else:
                                        extracted['lob'] = lob
                                
                                # Ensure all required fields exist
                                extracted.setdefault('account_name', '')
                                extracted.setdefault('lob', '')
                                extracted.setdefault('policy_number', '')
                                extracted.setdefault('start_date', '')
                                extracted.setdefault('end_date', '')
                                
                                st.session_state['extracted_info'] = extracted
                                st.success("✅ Extracted!")
                                
                            except json.JSONDecodeError as e:
                                st.error(f"❌ Failed to parse JSON response: {e}")
                            except Exception as e:
                                st.error(f"❌ Extraction error: {e}")
                        
                except Exception as e:
                    st.error(f"Extraction error: {e}")
        
        st.markdown("---")
        
        # ====================================================================
        # Section 2: Editable Filters (Pre-populated from AI)
        # ====================================================================
        st.subheader("📋 Search Filters (Editable)")
        
        # Get extracted info or defaults
        extracted = st.session_state.get('extracted_info', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            account_options = ["-- All --"] + email_agent.VALID_ACCOUNTS
            default_idx = 0
            if extracted.get('account_name'):
                try:
                    default_idx = account_options.index(extracted['account_name'])
                except ValueError:
                    pass
            selected_account = st.selectbox("Account Name", account_options, index=default_idx)
        
        with col2:
            lob_options = ["-- All --"] + email_agent.VALID_LOBS
            default_idx = 0
            if extracted.get('lob'):
                lob_val = extracted['lob'].upper()
                try:
                    default_idx = lob_options.index(lob_val)
                except ValueError:
                    pass
            selected_lob = st.selectbox("Line of Business", lob_options, index=default_idx)
        
        with col3:
            policy_number = st.text_input("Policy Number", value=extracted.get('policy_number', ''))
        
        # Date Range
        col_d1, col_d2 = st.columns(2)
        
        def parse_date(date_str):
            if not date_str:
                return None
            for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except:
                    pass
            return None
        
        with col_d1:
            start_default = parse_date(extracted.get('start_date', ''))
            start_date = st.date_input("Start Date", value=start_default)
        
        with col_d2:
            end_default = parse_date(extracted.get('end_date', ''))
            end_date = st.date_input("End Date", value=end_default)
        
        st.markdown("---")
        
        # ====================================================================
        # Section 3: Fetch Button
        # ====================================================================
        if st.button("🔎 Fetch Documents", type="primary"):
            fetch_placeholder = st.empty()
            fetch_placeholder.info("🔍 Searching documents...")
            
            # Prepare search params
            acct = selected_account if selected_account != "-- All --" else None
            lob = selected_lob if selected_lob != "-- All --" else None
            policy = policy_number if policy_number else None
            date_str = start_date.strftime("%d-%m-%Y") if start_date else None
            
            # Search
            results = doc_storage.search_files(acct, lob, policy, date_str)
            
            # Filter by date range
            if start_date and end_date and results:
                fetch_placeholder.info("📅 Filtering by date range...")
                filtered = []
                for f in results:
                    f_date = parse_date(f.get('effective_date', ''))
                    if f_date and start_date <= f_date <= end_date:
                        filtered.append(f)
                    elif not f_date:
                        filtered.append(f)
                results = filtered
                results = filtered
            
            st.session_state['agent_files'] = results
            st.session_state['selected_agent_files'] = []
            
            if results:
                fetch_placeholder.success(f"✅ Found {len(results)} document(s)")
            else:
                fetch_placeholder.warning("⚠️ No documents found matching criteria")
        
        st.markdown("---")
        
        # ====================================================================
        # Section 4: Document Preview with Selection
        # ====================================================================
        if 'agent_files' in st.session_state and st.session_state['agent_files']:
            st.subheader("📁 Found Documents")
            
            files = st.session_state['agent_files']
            selected = st.session_state.get('selected_agent_files', [])
            
            for idx, f in enumerate(files):
                col_chk, col_info = st.columns([0.5, 9.5])
                
                with col_chk:
                    is_sel = st.checkbox("", key=f"sel_{idx}", value=f in selected)
                    if is_sel and f not in selected:
                        selected.append(f)
                    elif not is_sel and f in selected:
                        selected.remove(f)
                
                with col_info:
                    with st.expander(f"📄 {f.get('filename', 'Unknown')} | {f.get('account', '')} | {f.get('lob', '')}", expanded=False):
                        st.markdown(f"""
| Field | Value |
|-------|-------|
| **Account** | {f.get('account', 'N/A')} |
| **LOB** | {f.get('lob', 'N/A')} |
| **Policy** | {f.get('policy_number', 'N/A')} |
| **Date** | {f.get('effective_date', 'N/A')} |
| **Source** | {f.get('source', 'N/A')} |
| **Path** | `{f.get('folder_path', 'N/A')}` |
""")
                        # Preview button
                        if st.button(f"👁️ Preview PDF", key=f"preview_{idx}"):
                            file_path = f.get('path', f.get('full_path', ''))
                            if os.path.exists(file_path):
                                try:
                                    text = ""
                                    with pdfplumber.open(file_path) as pdf:
                                        for page in pdf.pages[:2]:
                                            text += (page.extract_text() or "") + "\n"
                                    st.text_area("PDF Text Preview", text[:3000], height=250)
                                except Exception as e:
                                    st.error(f"Preview error: {e}")
                            else:
                                st.warning("File not found locally")
            
            st.session_state['selected_agent_files'] = selected
            
            st.markdown("---")
            
            # ================================================================
            # Section 5: Action Buttons
            # ================================================================
            st.subheader("⚡ Actions")
            st.info(f"📌 {len(selected)} document(s) selected")
            
            col_email, col_parse, _ = st.columns([1, 1, 4])
            
            with col_email:
                if st.button("📧 Send Email", disabled=len(selected) == 0, type="primary"):
                    email_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    total = len(selected)
                    success_count = 0
                    
                    for idx, f in enumerate(selected):
                        email_placeholder.info(f"📤 Sending email for {f.get('filename')}...")
                        
                        lob = f.get('lob', 'UNKNOWN')
                        policy = f.get('policy_number', 'Unknown')
                        success, msg = email_agent.send_email_action(f, lob, policy)
                        
                        progress_bar.progress((idx + 1) / total)
                        
                        if success:
                            success_count += 1
                    
                    progress_bar.empty()
                    
                    if success_count == total:
                        email_placeholder.success(f"✅ Successfully sent {success_count}/{total} email(s)!")
                    elif success_count > 0:
                        email_placeholder.warning(f"⚠️ Sent {success_count}/{total} email(s). Some failed.")
                    else:
                        email_placeholder.error(f"❌ Failed to send emails. Check SMTP credentials.")
            
            with col_parse:
                if st.button("🔬 Parse PDF", disabled=len(selected) == 0, type="primary"):
                    parse_results = []
                    parse_placeholder = st.empty()
                    parse_progress = st.progress(0)
                    total_files = len(selected)
                    
                    for idx, f in enumerate(selected):
                        file_path = f.get('path', f.get('full_path', ''))
                        parse_placeholder.info(f"📄 Processing {f.get('filename')}... ({idx+1}/{total_files})")
                        
                        # Download if cloud
                        if f.get('source') != 'Mock Storage' and not os.path.exists(file_path):
                            try:
                                file_path = doc_storage.download_file(f)
                            except Exception as e:
                                parse_results.append({"filename": f.get('filename'), "error": str(e)})
                                parse_progress.progress((idx + 1) / total_files)
                                continue
                        
                        try:
                            # Extract text
                            parse_placeholder.info(f"📖 Extracting text from {f.get('filename')}...")
                            text = ""
                            with pdfplumber.open(file_path) as pdf:
                                for page in pdf.pages:
                                    text += (page.extract_text() or "") + "\n"
                            
                            if not text.strip():
                                parse_results.append({"filename": f.get('filename'), "error": "No text extracted"})
                                parse_progress.progress((idx + 1) / total_files)
                                continue
                            
                            # Use config from session state (shared with Tab 1)
                            app_cfg = st.session_state.get('app_config')
                            if app_cfg:
                                client = setup_openai_client(app_cfg)
                                if client:
                                    parse_placeholder.info(f"🤖 AI analyzing {f.get('filename')}...")
                                    model = app_cfg.get('azure_deployment') if app_cfg.get('use_azure') else app_cfg.get('openai_model', 'gpt-4o-2024-08-06')
                                    
                                    lobs = classify_lobs_multi_openai(client, model, text)
                                    results = []
                                    max_chunk_size = app_cfg.get('max_chunk_size', 15000)
                                    api_delay = app_cfg.get('api_delay', 0.3)
                                    for lob in lobs:
                                        parse_placeholder.info(f"📊 Extracting {lob} claims from {f.get('filename')}...")
                                        fields = extract_fields_openai_chunked(client, model, text, lob, None, max_chunk_size, api_delay)
                                        results.append({
                                            "lob": lob,
                                            "carrier": fields.get("carrier", ""),
                                            "evaluation_date": fields.get("evaluation_date", ""),
                                            "claims": fields.get("claims", [])
                                        })
                                    
                                    parse_results.append({
                                        "filename": f.get('filename'),
                                        "success": True,
                                        "detected_lobs": lobs,
                                        "results": results
                                    })
                                else:
                                    parse_results.append({"filename": f.get('filename'), "error": "Failed to setup OpenAI client. Check config.py settings."})
                            else:
                                parse_results.append({"filename": f.get('filename'), "error": "No config found. Please configure in Tab 1 or create config.py"})
                                
                        except Exception as e:
                            parse_results.append({"filename": f.get('filename'), "error": str(e)})
                        
                        parse_progress.progress((idx + 1) / total_files)
                    
                    parse_progress.empty()
                    success_parses = sum(1 for r in parse_results if r.get('success'))
                    if success_parses == total_files:
                        parse_placeholder.success(f"✅ Successfully parsed {success_parses}/{total_files} file(s)!")
                    elif success_parses > 0:
                        parse_placeholder.warning(f"⚠️ Parsed {success_parses}/{total_files} file(s). Some failed.")
                    else:
                        parse_placeholder.error(f"❌ Failed to parse files.")
                    
                    st.session_state['parse_results'] = parse_results
            
            # ================================================================
            # Section 6: Parse Results
            # ================================================================
            if 'parse_results' in st.session_state and st.session_state['parse_results']:
                st.markdown("---")
                st.subheader("📊 Parse Results")
                
                for res in st.session_state['parse_results']:
                    fname = res.get('filename', 'Unknown')
                    
                    if res.get('error'):
                        st.error(f"❌ {fname}: {res['error']}")
                    else:
                        with st.expander(f"✅ {fname} - {', '.join(res.get('detected_lobs', []))}", expanded=True):
                            for lob_res in res.get('results', []):
                                st.markdown(f"**{lob_res.get('lob')}** | Carrier: {lob_res.get('carrier')} | Eval Date: {lob_res.get('evaluation_date')}")
                                
                                claims = lob_res.get('claims', [])
                                if claims:
                                    st.dataframe(pd.DataFrame(claims), use_container_width=True)
                                else:
                                    st.info("No claims found")
                
                # Export
                if st.button("📥 Export to Excel"):
                    all_claims = []
                    for res in st.session_state['parse_results']:
                        if res.get('success'):
                            for lob_res in res.get('results', []):
                                for claim in lob_res.get('claims', []):
                                    claim['lob'] = lob_res.get('lob')
                                    claim['carrier'] = lob_res.get('carrier')
                                    claim['source'] = res.get('filename')
                                    all_claims.append(claim)
                    
                    if all_claims:
                        df = pd.DataFrame(all_claims)
                        output = io.BytesIO()
                        df.to_excel(output, index=False)
                        output.seek(0)
                        
                        st.download_button(
                            "⬇️ Download Excel",
                            data=output.getvalue(),
                            file_name="parsed_claims.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == "__main__":
    main()
