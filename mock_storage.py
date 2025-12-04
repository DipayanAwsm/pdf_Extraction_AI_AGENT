"""
Mock Storage Module
===================
Simulates cloud storage (Azure Blob / OneDrive) for document search.

Folder Structure: AccountName/LOB/PolicyNo-EffectiveDate/
    
Examples:
    Chubbs/AUTO/2456-21092024/claim_report.pdf
    Amex/PROPERTY/7890-15102024/policy_document.pdf
    TechCorp/WC/1234-01082024/workers_comp_claim.pdf

User can search with ANY partial input - flexible matching enabled.
"""

import os
import glob
import re
from typing import List, Dict, Optional
from datetime import datetime

# Base directory for mock storage
MOCK_STORAGE_ROOT = os.path.join(os.path.dirname(__file__), "mock_documents")

# ============================================================================
# DOCUMENT REGISTRY - AccountName/LOB/PolicyNo-EffectiveDate
# ============================================================================

DOCUMENT_REGISTRY = [
    # Chubbs Account
    {"account": "Chubbs", "lob": "AUTO", "policy_number": "2456", "effective_date": "21-09-2024", "filename": "claim_report.pdf"},
    {"account": "Chubbs", "lob": "WC", "policy_number": "2456", "effective_date": "21-09-2024", "filename": "workers_comp_claim.pdf"},
    {"account": "Chubbs", "lob": "PROPERTY", "policy_number": "3344", "effective_date": "15-10-2024", "filename": "property_assessment.pdf"},
    {"account": "Chubbs", "lob": "GL", "policy_number": "5678", "effective_date": "01-11-2024", "filename": "general_liability.pdf"},
    # Amex Account
    {"account": "Amex", "lob": "AUTO", "policy_number": "7890", "effective_date": "10-11-2024", "filename": "vehicle_claim.pdf"},
    {"account": "Amex", "lob": "PROPERTY", "policy_number": "7890", "effective_date": "15-10-2024", "filename": "property_damage_report.pdf"},
    {"account": "Amex", "lob": "GL", "policy_number": "5555", "effective_date": "01-12-2024", "filename": "general_liability_claim.pdf"},
    {"account": "Amex", "lob": "WC", "policy_number": "1234", "effective_date": "01-01-2024", "filename": "workplace_injury.pdf"},
    # TechCorp Account
    {"account": "TechCorp", "lob": "WC", "policy_number": "1234", "effective_date": "01-08-2024", "filename": "workplace_injury.pdf"},
    {"account": "TechCorp", "lob": "GL", "policy_number": "1234", "effective_date": "01-08-2024", "filename": "liability_report.pdf"},
    {"account": "TechCorp", "lob": "AUTO", "policy_number": "4444", "effective_date": "15-09-2024", "filename": "fleet_accident.pdf"},
    # Travelers Account
    {"account": "Travelers", "lob": "GL", "policy_number": "9999", "effective_date": "15-05-2024", "filename": "loss_run_report.pdf"},
    {"account": "Travelers", "lob": "PROPERTY", "policy_number": "8888", "effective_date": "20-06-2024", "filename": "property_claim.pdf"},
    {"account": "Travelers", "lob": "AUTO", "policy_number": "7777", "effective_date": "10-07-2024", "filename": "auto_claim.pdf"},
    # GlobalInsure Account
    {"account": "GlobalInsure", "lob": "AUTO", "policy_number": "9999", "effective_date": "25-11-2024", "filename": "auto_accident_claim.pdf"},
    {"account": "GlobalInsure", "lob": "PROPERTY", "policy_number": "8888", "effective_date": "20-09-2024", "filename": "fire_damage_claim.pdf"},
    {"account": "GlobalInsure", "lob": "WC", "policy_number": "6666", "effective_date": "05-10-2024", "filename": "workers_comp.pdf"},
]

# LOB Aliases for flexible matching
LOB_ALIASES = {
    "work": "wc", "workers": "wc", "workers comp": "wc", "workerscomp": "wc", "worker": "wc",
    "vehicle": "auto", "car": "auto", "accident": "auto", "automobile": "auto",
    "house": "property", "home": "property", "fire": "property", "building": "property",
    "general": "gl", "liability": "gl", "general liability": "gl"
}


def _normalize_date(date_str: str) -> str:
    """Normalize date to folder format (no hyphens): DDMMYYYY"""
    if not date_str:
        return ""
    
    clean_date = re.sub(r'[-/.\s]', '', str(date_str))
    
    formats_to_try = [
        ("%d%m%Y", clean_date),
        ("%d-%m-%Y", date_str),
        ("%d/%m/%Y", date_str),
        ("%Y-%m-%d", date_str),
        ("%Y%m%d", clean_date),
    ]
    
    for fmt, val in formats_to_try:
        try:
            dt = datetime.strptime(val, fmt)
            return dt.strftime("%d%m%Y")
        except ValueError:
            continue
    
    return clean_date


def _normalize_string(s: str) -> str:
    """Normalize string for flexible matching."""
    if not s:
        return ""
    return str(s).lower().strip()


def _fuzzy_match(search_term: str, target: str) -> bool:
    """Flexible matching - partial match, case insensitive."""
    if not search_term:
        return True
    
    search_norm = _normalize_string(search_term)
    target_norm = _normalize_string(target)
    
    if search_norm in target_norm or target_norm in search_norm:
        return True
    
    search_clean = re.sub(r'[^a-z0-9]', '', search_norm)
    target_clean = re.sub(r'[^a-z0-9]', '', target_norm)
    
    return search_clean in target_clean or target_clean in search_clean


def get_folder_path(doc: Dict) -> str:
    """Generate folder path: AccountName/LOB/PolicyNo-EffectiveDate"""
    date_normalized = _normalize_date(doc["effective_date"])
    return f"{doc['account']}/{doc['lob']}/{doc['policy_number']}-{date_normalized}"


def get_full_path(doc: Dict) -> str:
    """Get full file path including filename."""
    folder = get_folder_path(doc)
    return os.path.join(MOCK_STORAGE_ROOT, folder, doc["filename"])


def search_files(
    account_name: Optional[str] = None,
    lob: Optional[str] = None,
    policy_number: Optional[str] = None,
    date: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Search for files with FLEXIBLE matching.
    
    User can enter ANY combination:
    - Account: "chubbs", "Chub", "CHUBBS", "chub" (partial OK)
    - LOB: "AUTO", "auto", "WC", "work", "workers comp", "vehicle"
    - Policy: "2456", "245" (partial OK)
    - Date: "21-09-2024", "21/09/2024", "21092024" (any format)
    
    Returns all matching files.
    """
    results = []
    
    search_account = _normalize_string(account_name) if account_name else None
    search_lob = _normalize_string(lob) if lob else None
    search_policy = _normalize_string(policy_number) if policy_number else None
    search_date = _normalize_date(date) if date else None
    
    # Resolve LOB aliases
    if search_lob and search_lob in LOB_ALIASES:
        search_lob = LOB_ALIASES[search_lob]
    
    print(f"🔍 Searching: Account={search_account or 'any'}, LOB={search_lob or 'any'}, Policy={search_policy or 'any'}, Date={search_date or 'any'}")
    
    for doc in DOCUMENT_REGISTRY:
        # Check Account (fuzzy match)
        if search_account and not _fuzzy_match(search_account, doc["account"]):
            continue
        
        # Check LOB (with alias resolution)
        if search_lob:
            doc_lob = _normalize_string(doc["lob"])
            if search_lob != doc_lob:
                continue
        
        # Check Policy Number (partial match)
        if search_policy and not _fuzzy_match(search_policy, doc["policy_number"]):
            continue
        
        # Check Date (normalized match)
        if search_date:
            doc_date = _normalize_date(doc["effective_date"])
            if search_date != doc_date:
                continue
        
        folder_path = get_folder_path(doc)
        full_path = get_full_path(doc)
        
        results.append({
            "filename": doc["filename"],
            "folder_path": folder_path,
            "full_path": full_path,
            "path": full_path,  # For backward compatibility
            "source": "Mock Storage",
            "account": doc["account"],
            "lob": doc["lob"],
            "policy_number": doc["policy_number"],
            "effective_date": doc["effective_date"]
        })
    
    print(f"✅ Found {len(results)} matching files")
    return results


def create_mock_data():
    """Create mock PDF files in: AccountName/LOB/PolicyNo-EffectiveDate/"""
    print("\n📁 Creating Mock Document Storage...")
    print(f"   Root: {MOCK_STORAGE_ROOT}")
    print(f"   Format: AccountName/LOB/PolicyNo-EffectiveDate/\n")
    
    os.makedirs(MOCK_STORAGE_ROOT, exist_ok=True)
    
    try:
        import fpdf
        use_fpdf = True
    except ImportError:
        use_fpdf = False
        print("   ℹ️ fpdf not installed, using simple PDF format")
    
    for doc in DOCUMENT_REGISTRY:
        folder_path = os.path.join(MOCK_STORAGE_ROOT, get_folder_path(doc))
        file_path = get_full_path(doc)
        
        os.makedirs(folder_path, exist_ok=True)
        
        if not os.path.exists(file_path):
            if use_fpdf:
                pdf = fpdf.FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Claim Document", ln=1, align="C")
                pdf.cell(200, 10, txt=f"Account: {doc['account']}", ln=1)
                pdf.cell(200, 10, txt=f"LOB: {doc['lob']}", ln=1)
                pdf.cell(200, 10, txt=f"Policy: {doc['policy_number']}", ln=1)
                pdf.cell(200, 10, txt=f"Date: {doc['effective_date']}", ln=1)
                pdf.output(file_path)
            else:
                with open(file_path, 'wb') as f:
                    content = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
trailer << /Size 4 /Root 1 0 R >>
%%EOF
Account: {doc['account']} | LOB: {doc['lob']} | Policy: {doc['policy_number']} | Date: {doc['effective_date']}
"""
                    f.write(content.encode())
            
            print(f"   ✅ {get_folder_path(doc)}/{doc['filename']}")
    
    print(f"\n✅ Mock storage ready with {len(DOCUMENT_REGISTRY)} documents\n")


def list_all_documents():
    """List all documents in the registry."""
    print("\n📋 Document Registry (Path: AccountName/LOB/PolicyNo-Date)")
    print("-" * 90)
    print(f"{'Account':<15} {'LOB':<10} {'Policy':<10} {'Date':<12} {'Folder Path':<40}")
    print("-" * 90)
    
    for doc in DOCUMENT_REGISTRY:
        folder = get_folder_path(doc)
        print(f"{doc['account']:<15} {doc['lob']:<10} {doc['policy_number']:<10} {doc['effective_date']:<12} {folder:<40}")
    
    print("-" * 90)
    print(f"Total: {len(DOCUMENT_REGISTRY)} documents\n")


if __name__ == "__main__":
    create_mock_data()
    list_all_documents()
    
    print("\n" + "="*60)
    print("🧪 FLEXIBLE SEARCH TESTS")
    print("="*60)
    
    test_cases = [
        {"account_name": "chubbs", "lob": "work", "policy_number": "2456", "date": "21-09-2024"},
        {"account_name": "Chub", "lob": "WC"},
        {"policy_number": "7890"},
        {"account_name": "amex", "lob": "auto"},
        {"date": "01-08-2024"},
        {"account_name": "techcorp"},
        {"lob": "gl"},
        {"account_name": "travel"},  # Partial match for Travelers
        {"lob": "vehicle"},  # Alias for AUTO
    ]
    
    for i, params in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {params} ---")
        results = search_files(**params)
        for r in results:
            print(f"  📄 {r['folder_path']}/{r['filename']}")
