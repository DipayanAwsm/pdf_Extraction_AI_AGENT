#!/usr/bin/env python3
"""
Streamlit Agent Reference App
=============================
A reference implementation that:
1. Shows all PDFs from local file structure (mock_documents)
2. Uses config.py for processing configuration
3. Has search agent that finds documents based on LOB/keywords
4. Has mailing agent that sends extracted data based on LOB
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

# Load configuration from streamlit_config if available
try:
    from streamlit_config import (
        PYTHON_CMD, CONFIG_FILE, OCR_TIMEOUT, LOB_TIMEOUT, SHOW_DEBUG_INFO, SHOW_COMMAND_OUTPUT
    )
except ImportError:
    PYTHON_CMD = "python"
    CONFIG_FILE = "config.py"
    OCR_TIMEOUT = 1200
    LOB_TIMEOUT = 1800
    SHOW_DEBUG_INFO = True
    SHOW_COMMAND_OUTPUT = True

# OpenAI removed - using external scripts for processing

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
        return {}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        cfg = {}
        
        # Processing settings
        cfg['max_chunk_size'] = getattr(config_module, 'MAX_CHUNK_SIZE', 15000)
        cfg['api_delay'] = getattr(config_module, 'API_DELAY', 0.3)
        
        return cfg
    except Exception as e:
        st.warning(f"Could not load config: {e}")
        return {}


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

def extract_search_criteria(query: str, available_pdfs: List[Dict]) -> Dict:
    """
    Extract search criteria from natural language query using keyword matching
    """
    # Get unique values from available PDFs for context
    unique_accounts = sorted(set(pdf['account'] for pdf in available_pdfs))
    unique_lobs = sorted(set(pdf['lob'] for pdf in available_pdfs))
    
    query_lower = query.lower().strip()
    criteria = {
        'lob': '',
        'account': '',
        'policy_number': '',
        'keywords': query_lower.split()
    }
    
    # LOB detection
    if any(kw in query_lower for kw in ['auto', 'vehicle', 'car', 'automobile', 'motor']):
        criteria['lob'] = 'AUTO'
    elif any(kw in query_lower for kw in ['property', 'dwelling', 'building', 'fire', 'theft']):
        criteria['lob'] = 'PROPERTY'
    elif any(kw in query_lower for kw in ['gl', 'general liability', 'liability', 'cgl']):
        criteria['lob'] = 'GL'
    elif any(kw in query_lower for kw in ['wc', 'workers comp', 'workers compensation', 'work', 'workers']):
        criteria['lob'] = 'WC'
    
    # Account detection (fuzzy match)
    for account in unique_accounts:
        account_lower = account.lower()
        if account_lower in query_lower or query_lower in account_lower:
            criteria['account'] = account
            break
        # Check for partial matches
        account_words = account_lower.split()
        query_words = query_lower.split()
        if any(word in query_words for word in account_words if len(word) > 3):
            criteria['account'] = account
            break
    
    # Policy number detection (look for numbers)
    import re
    policy_numbers = re.findall(r'\d{4,}', query)
    if policy_numbers:
        criteria['policy_number'] = policy_numbers[0]
    
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
# PDF Processing Functions (EXACT COPY from streamlit_e2e_openai_app.py)
# ============================================================================

def create_directories():
    """Create necessary directories"""
    backup_dir = Path("./backup")
    tmp_dir = Path("./tmp")
    results_dir = Path("./results")
    
    backup_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    return backup_dir, tmp_dir, results_dir


def convert_pdf_to_text(pdf_path, tmp_dir, debug_log_container=None):
    """Convert PDF to text using fitzTest3.py"""
    try:
        cmd = [PYTHON_CMD, "fitzTest3.py", str(pdf_path), "--output", str(tmp_dir)]
        
        if st.session_state.get('debug_mode') and debug_log_container:
            # Real-time output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            log_buffer = []
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    output_lines.append(line)
                    log_line = f"[PDF->Text] {line}"
                    log_buffer.append(log_line)
                    if 'debug_logs' in st.session_state:
                        st.session_state.debug_logs.append(log_line)
            
            # Update debug log container with all lines
            if debug_log_container and log_buffer:
                debug_log_container.code('\n'.join(log_buffer), language='text')
            
            process.wait()
            result_code = process.returncode
            stdout_text = '\n'.join(output_lines)
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=OCR_TIMEOUT
            )
            result_code = result.returncode
            stdout_text = result.stdout
        
        if result_code == 0:
            output_lines = stdout_text.strip().split('\n')
            for line in output_lines:
                if line.startswith("SUCCESS:"):
                    return line.replace("SUCCESS:", "").strip(), None
            time.sleep(0.5)
            txts = list(Path(tmp_dir).glob("*_extracted.txt"))
            if txts:
                return str(txts[0]), None
            return None, "Text file not found after conversion"
        
        return None, stdout_text or "Conversion failed"
        
    except subprocess.TimeoutExpired:
        return None, "PDF conversion timed out"
    except Exception as e:
        return None, str(e)


def process_text_with_openai(text_file_path, results_dir, original_pdf_name, debug_log_container=None):
    """Process text file using text_lob_openai_extractor.py"""
    try:
        # Create timestamped output directory with original filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name_no_ext = Path(original_pdf_name).stem
        output_dir_name = f"{original_name_no_ext}_{timestamp}"
        output_dir = results_dir / output_dir_name
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            PYTHON_CMD, "text_lob_openai_extractor.py",
            str(text_file_path),
            "--config", CONFIG_FILE,
            "--out", str(output_dir)
        ]
        
        if st.session_state.get('debug_mode') and debug_log_container:
            # Real-time output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            log_buffer = []
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    output_lines.append(line)
                    log_line = f"[OpenAI] {line}"
                    log_buffer.append(log_line)
                    if 'debug_logs' in st.session_state:
                        st.session_state.debug_logs.append(log_line)
            
            # Update debug log container with all lines
            if debug_log_container and log_buffer:
                debug_log_container.code('\n'.join(log_buffer), language='text')
            
            process.wait()
            result_code = process.returncode
            stderr_text = '\n'.join(output_lines)
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=LOB_TIMEOUT
            )
            result_code = result.returncode
            stderr_text = result.stderr
        
        if result_code == 0:
            excel_files = list(output_dir.glob("result.xlsx"))
            if excel_files:
                # Rename result.xlsx to include original filename and timestamp
                original_result = excel_files[0]
                new_filename = f"{original_name_no_ext}_{timestamp}_result.xlsx"
                final_result_path = output_dir / new_filename
                
                # Copy the file with new name
                shutil.copy2(original_result, final_result_path)
                
                return str(final_result_path), None
            return None, "Extraction completed but result.xlsx not found"
        
        return None, stderr_text or "Extraction failed"
        
    except subprocess.TimeoutExpired:
        return None, "OpenAI extraction timed out"
    except Exception as e:
        return None, str(e)


# ============================================================================
# Email Agent Functions with CSV-based LOB Mapping
# ============================================================================

def load_lob_email_mapping(csv_path: str = "lob_email_mapping.csv") -> Dict[str, List[Dict]]:
    """
    Load LOB to email mapping from CSV file
    Returns: {LOB: [{'email': '...', 'description': '...'}, ...]}
    """
    mapping = {}
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        st.warning(f"⚠️ Email mapping CSV not found at {csv_path}. Creating default file.")
        # Create default CSV
        default_data = {
            'LOB': ['AUTO', 'AUTO', 'PROPERTY', 'PROPERTY', 'GL', 'GL', 'WC', 'WC'],
            'Email': [
                'auto-claims@example.com',
                'auto-processing@example.com',
                'property-claims@example.com',
                'property-processing@example.com',
                'gl-claims@example.com',
                'gl-processing@example.com',
                'wc-claims@example.com',
                'wc-processing@example.com'
            ],
            'Description': [
                'Auto Claims Department',
                'Auto Processing Team',
                'Property Claims Department',
                'Property Processing Team',
                'General Liability Claims Department',
                'General Liability Processing Team',
                'Workers Compensation Claims Department',
                'Workers Compensation Processing Team'
            ]
        }
        pd.DataFrame(default_data).to_csv(csv_file, index=False)
        st.info(f"✅ Created default email mapping file: {csv_path}")
    
    try:
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            lob = str(row.get('LOB', '')).strip().upper()
            email = str(row.get('Email', '')).strip()
            description = str(row.get('Description', '')).strip()
            
            if lob and email:
                if lob not in mapping:
                    mapping[lob] = []
                mapping[lob].append({
                    'email': email,
                    'description': description
                })
    except Exception as e:
        st.error(f"Error loading email mapping CSV: {e}")
    
    return mapping


def format_time(seconds):
    """Format seconds into readable time string"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} minutes {secs:.1f} seconds"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} hours {minutes} minutes {secs:.1f} seconds"


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
        # Send email with PDF attachment
        success, message = send_email_action(pdf_info, lob, policy_number)
        
        if success:
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
    
    # Sidebar - Configuration Status
    st.sidebar.header("⚙️ Configuration")
    
    if cfg:
        st.sidebar.success("✅ Config loaded")
        st.sidebar.write(f"**Max Chunk Size:** {cfg.get('max_chunk_size', 15000)}")
        st.sidebar.write(f"**API Delay:** {cfg.get('api_delay', 0.3)}s")
    else:
        st.sidebar.info("ℹ️ Using default settings")
    
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
            
            with st.spinner("🔍 Analyzing search query..."):
                # Extract search criteria using keyword matching
                search_criteria = extract_search_criteria(search_query, all_pdf_files)
                
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
                    search_criteria = extract_search_criteria(search_query, all_pdf_files)
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
            
            # Processing Section (EXACT COPY from streamlit_e2e_openai_app.py)
            st.subheader("🚀 Process PDF")
            
            # Initialize session state for processing
            if 'processing_complete' not in st.session_state:
                st.session_state.processing_complete = False
            if 'result_file' not in st.session_state:
                st.session_state.result_file = None
            if 'processing_status' not in st.session_state:
                st.session_state.processing_status = "Ready"
            if 'processing_times' not in st.session_state:
                st.session_state.processing_times = {}
            if 'debug_mode' not in st.session_state:
                st.session_state.debug_mode = False
            if 'debug_logs' not in st.session_state:
                st.session_state.debug_logs = []
            
            # Debug mode toggle
            debug_mode = st.checkbox("🐛 Debug Mode (Show Real-time Logs)", value=st.session_state.debug_mode)
            st.session_state.debug_mode = debug_mode
            
            if st.button("▶️ Start Processing", type="primary", disabled=st.session_state.processing_status == "Processing"):
                st.session_state.processing_status = "Processing"
                st.session_state.processing_times = {}
                
                # Create directories
                backup_dir, tmp_dir, results_dir = create_directories()
                
                # Copy PDF to backup
                backup_path = backup_dir / selected_pdf['filename']
                shutil.copy2(selected_pdf['path'], backup_path)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.empty()
                debug_log_container = st.empty() if st.session_state.debug_mode else None
                
                # Clear debug logs at start
                if st.session_state.debug_mode:
                    st.session_state.debug_logs = []
                
                try:
                    start_total = time.time()
                    
                    # Step 1: Convert PDF to text
                    status_text.text("Step 1/3: Converting PDF to text...")
                    progress_bar.progress(0.2)
                    
                    if st.session_state.debug_mode:
                        st.markdown("**📋 Real-time Debug Logs (PDF to Text):**")
                        debug_placeholder = st.empty()
                    else:
                        debug_placeholder = None
                    
                    start_ocr = time.time()
                    text_file_path, error = convert_pdf_to_text(backup_path, tmp_dir, debug_placeholder)
                    ocr_time = time.time() - start_ocr
                    st.session_state.processing_times['PDF to Text'] = ocr_time
                    
                    if not text_file_path:
                        st.session_state.processing_status = "Error"
                        st.error(f"Error: {error}")
                        return
                    
                    with log_container.expander("PDF Conversion Log", expanded=False):
                        st.text(f"[SUCCESS] Text file created: {text_file_path}")
                        st.text(f"Time taken: {format_time(ocr_time)}")
                    
                    # Step 2: Process with text_lob_openai_extractor.py (external script)
                    status_text.text("Step 2/3: Processing text with extraction script...")
                    progress_bar.progress(0.6)
                    
                    if st.session_state.debug_mode:
                        st.markdown("**📋 Real-time Debug Logs (Extraction):**")
                        debug_placeholder_extract = st.empty()
                    else:
                        debug_placeholder_extract = None
                    
                    start_extract = time.time()
                    result_file_path, error = process_text_with_openai(
                        text_file_path,
                        results_dir,
                        selected_pdf['filename'],
                        debug_placeholder_extract
                    )
                    extract_time = time.time() - start_extract
                    st.session_state.processing_times['Extraction'] = extract_time
                    
                    if not result_file_path:
                        st.session_state.processing_status = "Error"
                        st.error(f"Error: {error}")
                        return
                    
                    # Step 3: Complete
                    total_time = time.time() - start_total
                    st.session_state.processing_times['Total Time'] = total_time
                    
                    status_text.text("Step 3/3: Finalizing results...")
                    progress_bar.progress(1.0)
                    
                    st.session_state.processing_complete = True
                    st.session_state.result_file = result_file_path
                    st.session_state.processing_status = "Complete"
                    st.session_state['processed_pdf'] = selected_pdf
                    
                    st.success("🎉 Processing completed successfully!")
                    
                    # Load Excel to extract detected LOBs
                    try:
                        excel_data = pd.read_excel(result_file_path, sheet_name=None)
                        detected_lobs = []
                        for sheet_name in excel_data.keys():
                            sn = str(sheet_name).lower()
                            if 'auto' in sn:
                                detected_lobs.append('AUTO')
                            elif 'property' in sn:
                                detected_lobs.append('PROPERTY')
                            elif 'gl' in sn:
                                detected_lobs.append('GL')
                            elif 'wc' in sn:
                                detected_lobs.append('WC')
                        
                        st.session_state['detected_lobs'] = list(set(detected_lobs))
                    except:
                        st.session_state['detected_lobs'] = [selected_pdf.get('lob', 'AUTO')]
                
                except Exception as e:
                    st.session_state.processing_status = "Error"
                    st.error(f"Exception: {str(e)}")
                finally:
                    progress_bar.empty()
            
            # ================================================================
            # Section 3: Display Results (from streamlit_e2e_openai_app.py)
            # ================================================================
            if st.session_state.processing_complete and st.session_state.result_file:
                result_file = Path(st.session_state.result_file)
                processed_pdf = st.session_state.get('processed_pdf')
                
                if result_file.exists() and processed_pdf and processed_pdf['path'] == selected_pdf['path']:
                    st.markdown("---")
                    st.header("📊 Processing Results")
                    
                    # Display processing times
                    if st.session_state.processing_times:
                        st.markdown("**⏱️ Processing Times**")
                        times_df = pd.DataFrame([
                            {"Step": k, "Time": format_time(v)}
                            for k, v in st.session_state.processing_times.items()
                        ])
                        st.dataframe(times_df, use_container_width=True, hide_index=True)
                    
                    # Preview Excel content
                    try:
                        excel_data = pd.read_excel(result_file, sheet_name=None)
                        
                        # Display sheets
                        if len(excel_data) == 1:
                            sheet_name = list(excel_data.keys())[0]
                            df = excel_data[sheet_name]
                            st.write(f"**Sheet:** {sheet_name}")
                            st.dataframe(df, use_container_width=True)
                        else:
                            for sheet_name, df in excel_data.items():
                                with st.expander(f"📄 {sheet_name}", expanded=(sheet_name == list(excel_data.keys())[0])):
                                    st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not preview Excel content: {e}")
                    
                    # Download button
                    with open(result_file, "rb") as f:
                        st.download_button(
                            label="📥 Download Excel File",
                            data=f.read(),
                            file_name=result_file.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary"
                        )
                    
                    # ============================================================
                    # Section 4: Email Agent with CSV-based LOB Dropdown
                    # ============================================================
                    st.markdown("---")
                    st.header("📧 Send Extracted Data via Email")
                    
                    # Load email mapping from CSV
                    email_mapping = load_lob_email_mapping("lob_email_mapping.csv")
                    detected_lobs = st.session_state.get('detected_lobs', [selected_pdf.get('lob', 'AUTO')])
                    
                    if detected_lobs:
                        st.info(f"📋 Detected LOBs: {', '.join(detected_lobs)}")
                        
                        # Create dropdown for each detected LOB
                        selected_emails = {}
                        for lob in detected_lobs:
                            lob_emails = email_mapping.get(lob, [])
                            if lob_emails:
                                email_options = [f"{email['email']} - {email['description']}" for email in lob_emails]
                                selected_email_display = st.selectbox(
                                    f"Select Email for {lob}",
                                    options=email_options,
                                    key=f"email_select_{lob}",
                                    help=f"Choose recipient email for {lob} claims"
                                )
                                # Extract email from display string
                                selected_email = selected_email_display.split(' - ')[0] if ' - ' in selected_email_display else selected_email_display
                                selected_emails[lob] = selected_email
                            else:
                                st.warning(f"⚠️ No email mapping found for {lob} in CSV. Please add entries to lob_email_mapping.csv")
                                # Allow manual entry
                                manual_email = st.text_input(
                                    f"Enter Email for {lob}",
                                    key=f"manual_email_{lob}",
                                    help="Enter email address manually"
                                )
                                if manual_email:
                                    selected_emails[lob] = manual_email
                        
                        # Send email button
                        col_email1, col_email2 = st.columns([2, 1])
                        with col_email1:
                            st.write("")  # Spacing
                        with col_email2:
                            send_email_btn = st.button("📤 Send Email", type="primary")
                        
                        if send_email_btn:
                            if send_email_action:
                                with st.spinner("Sending email..."):
                                    # Send email for each LOB
                                    success_count = 0
                                    error_messages = []
                                    
                                    for lob, email in selected_emails.items():
                                        if email:
                                            success, message = send_extracted_data_email(
                                                selected_pdf,
                                                {'detected_lobs': [lob]},  # Simplified data structure
                                                email
                                            )
                                            
                                            if success:
                                                success_count += 1
                                                st.success(f"✅ Email sent to {email} for {lob}")
                                            else:
                                                error_messages.append(f"{lob}: {message}")
                                                st.error(f"❌ Failed to send email for {lob}: {message}")
                                    
                                    if success_count > 0:
                                        st.success(f"✅ Successfully sent {success_count} email(s)")
                                    if error_messages:
                                        st.error("❌ Some emails failed to send")
                            else:
                                st.error("Email agent not available. Please ensure email_agent.py is properly configured.")
                    else:
                        st.warning("⚠️ No LOBs detected. Cannot suggest email addresses.")
        else:
            st.info("👆 Select a PDF from the list above to begin processing")
    
    # Sidebar - Additional Info
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.write("""
    This app:
    1. Scans local file structure for PDFs
    2. Uses config.py for processing settings
    3. Processes selected PDFs using external scripts
    4. Sends extracted data via email with CSV-based LOB mapping
    """)
    
    if st.sidebar.button("🔄 Reset Session"):
        for key in list(st.session_state.keys()):
            if key not in ['config']:
                del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()

