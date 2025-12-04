"""
Cloud Storage Integration Module
================================
Real-world integration with Azure Blob Storage and OneDrive (Microsoft Graph API).

Folder Structure: AccountName/LOB/PolicyNo-EffectiveDate/

Required Environment Variables:
-------------------------------
# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_STORAGE_CONTAINER_NAME=your_container_name

# OneDrive (Microsoft Graph API)
ONEDRIVE_CLIENT_ID=your_app_client_id
ONEDRIVE_CLIENT_SECRET=your_app_client_secret
ONEDRIVE_TENANT_ID=your_tenant_id
ONEDRIVE_DRIVE_ID=your_drive_id (optional, uses default if not set)
ONEDRIVE_ROOT_FOLDER=Documents/Claims (optional, root folder path)

# Storage Mode
STORAGE_MODE=azure|onedrive|both|mock (default: mock)
"""

import os
import re
import io
import tempfile
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
import dotenv

dotenv.load_dotenv()

# ============================================================================
# LOB Aliases for flexible matching
# ============================================================================

LOB_ALIASES = {
    "work": "wc", "workers": "wc", "workers comp": "wc", "workerscomp": "wc", "worker": "wc",
    "vehicle": "auto", "car": "auto", "accident": "auto", "automobile": "auto",
    "house": "property", "home": "property", "fire": "property", "building": "property",
    "general": "gl", "liability": "gl", "general liability": "gl"
}

# ============================================================================
# Helper Functions
# ============================================================================

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


def _parse_path(path: str) -> Optional[Dict[str, str]]:
    """
    Parse a path like 'AccountName/LOB/PolicyNo-Date/filename.pdf' 
    into components.
    """
    parts = path.replace("\\", "/").strip("/").split("/")
    
    if len(parts) < 3:
        return None
    
    account = parts[-3] if len(parts) >= 3 else None
    lob = parts[-2] if len(parts) >= 2 else None
    folder_name = parts[-1] if len(parts) >= 1 else None
    filename = None
    
    # Check if last part is a file
    if folder_name and "." in folder_name:
        filename = folder_name
        folder_name = parts[-2] if len(parts) >= 2 else None
        lob = parts[-3] if len(parts) >= 3 else None
        account = parts[-4] if len(parts) >= 4 else None
    
    # Parse PolicyNo-Date from folder name
    policy_number = None
    effective_date = None
    
    if folder_name and "-" in folder_name:
        folder_parts = folder_name.split("-", 1)
        policy_number = folder_parts[0]
        if len(folder_parts) > 1:
            effective_date = folder_parts[1]
    
    return {
        "account": account,
        "lob": lob,
        "policy_number": policy_number,
        "effective_date": effective_date,
        "filename": filename
    }


# ============================================================================
# Abstract Storage Provider
# ============================================================================

class StorageProvider(ABC):
    """Abstract base class for storage providers."""
    
    @abstractmethod
    def list_files(self, prefix: str = "") -> List[Dict[str, str]]:
        """List all files in storage."""
        pass
    
    @abstractmethod
    def download_file(self, path: str) -> Tuple[bytes, str]:
        """Download a file and return (content, filename)."""
        pass
    
    @abstractmethod
    def get_file_url(self, path: str) -> str:
        """Get a URL or local path to the file."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the storage provider is configured and available."""
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this storage source."""
        pass


# ============================================================================
# Azure Blob Storage Provider
# ============================================================================

class AzureBlobStorageProvider(StorageProvider):
    """Azure Blob Storage integration."""
    
    def __init__(self):
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "claims-documents")
        self._client = None
        self._container_client = None
    
    def _get_client(self):
        """Lazy initialization of Azure Blob client."""
        if self._client is None:
            try:
                from azure.storage.blob import BlobServiceClient
                self._client = BlobServiceClient.from_connection_string(self.connection_string)
                self._container_client = self._client.get_container_client(self.container_name)
            except ImportError:
                raise ImportError("azure-storage-blob package not installed. Run: pip install azure-storage-blob")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Azure Blob Storage: {e}")
        return self._container_client
    
    def is_available(self) -> bool:
        """Check if Azure Blob Storage is configured."""
        return bool(self.connection_string)
    
    @property
    def source_name(self) -> str:
        return "Azure Blob Storage"
    
    def list_files(self, prefix: str = "") -> List[Dict[str, str]]:
        """List all files in the Azure Blob container."""
        if not self.is_available():
            return []
        
        try:
            container = self._get_client()
            files = []
            
            blobs = container.list_blobs(name_starts_with=prefix if prefix else None)
            
            for blob in blobs:
                if blob.name.lower().endswith('.pdf'):
                    parsed = _parse_path(blob.name)
                    if parsed:
                        files.append({
                            "filename": parsed.get("filename") or os.path.basename(blob.name),
                            "folder_path": os.path.dirname(blob.name),
                            "full_path": blob.name,
                            "path": blob.name,
                            "source": self.source_name,
                            "account": parsed.get("account", ""),
                            "lob": parsed.get("lob", ""),
                            "policy_number": parsed.get("policy_number", ""),
                            "effective_date": parsed.get("effective_date", ""),
                            "size": blob.size,
                            "last_modified": str(blob.last_modified) if blob.last_modified else ""
                        })
            
            return files
        except Exception as e:
            print(f"❌ Azure Blob error: {e}")
            return []
    
    def download_file(self, path: str) -> Tuple[bytes, str]:
        """Download a file from Azure Blob Storage."""
        container = self._get_client()
        blob_client = container.get_blob_client(path)
        content = blob_client.download_blob().readall()
        return content, os.path.basename(path)
    
    def download_to_temp(self, path: str) -> str:
        """Download file to a temporary location and return the path."""
        content, filename = self.download_file(path)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'wb') as f:
            f.write(content)
        return temp_path
    
    def get_file_url(self, path: str) -> str:
        """Get the blob URL (note: may require SAS token for access)."""
        container = self._get_client()
        blob_client = container.get_blob_client(path)
        return blob_client.url
    
    def upload_file(self, local_path: str, blob_path: str) -> bool:
        """Upload a file to Azure Blob Storage."""
        try:
            container = self._get_client()
            blob_client = container.get_blob_client(blob_path)
            
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)
            
            print(f"✅ Uploaded to Azure: {blob_path}")
            return True
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False


# ============================================================================
# OneDrive Storage Provider (Microsoft Graph API)
# ============================================================================

class OneDriveStorageProvider(StorageProvider):
    """OneDrive integration via Microsoft Graph API."""
    
    def __init__(self):
        self.client_id = os.getenv("ONEDRIVE_CLIENT_ID")
        self.client_secret = os.getenv("ONEDRIVE_CLIENT_SECRET")
        self.tenant_id = os.getenv("ONEDRIVE_TENANT_ID")
        self.drive_id = os.getenv("ONEDRIVE_DRIVE_ID")
        self.root_folder = os.getenv("ONEDRIVE_ROOT_FOLDER", "Documents/Claims")
        self._access_token = None
        self._token_expiry = None
    
    def is_available(self) -> bool:
        """Check if OneDrive is configured."""
        return all([self.client_id, self.client_secret, self.tenant_id])
    
    @property
    def source_name(self) -> str:
        return "OneDrive"
    
    def _get_access_token(self) -> str:
        """Get OAuth2 access token using client credentials flow."""
        import requests
        from datetime import datetime, timedelta
        
        # Check if we have a valid cached token
        if self._access_token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._access_token
        
        token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
        
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials"
        }
        
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        
        token_data = response.json()
        self._access_token = token_data["access_token"]
        self._token_expiry = datetime.now() + timedelta(seconds=token_data.get("expires_in", 3600) - 60)
        
        return self._access_token
    
    def _graph_request(self, endpoint: str, method: str = "GET", **kwargs) -> dict:
        """Make a request to Microsoft Graph API."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self._get_access_token()}",
            "Content-Type": "application/json"
        }
        
        base_url = "https://graph.microsoft.com/v1.0"
        url = f"{base_url}{endpoint}"
        
        response = requests.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        
        if response.content:
            return response.json()
        return {}
    
    def _get_drive_endpoint(self) -> str:
        """Get the drive endpoint based on configuration."""
        if self.drive_id:
            return f"/drives/{self.drive_id}"
        else:
            return "/me/drive"
    
    def list_files(self, prefix: str = "") -> List[Dict[str, str]]:
        """List all PDF files in OneDrive folder."""
        if not self.is_available():
            return []
        
        try:
            files = []
            folder_path = self.root_folder
            if prefix:
                folder_path = f"{folder_path}/{prefix}".strip("/")
            
            # Get folder contents recursively
            self._list_folder_recursive(folder_path, files)
            
            return files
        except Exception as e:
            print(f"❌ OneDrive error: {e}")
            return []
    
    def _list_folder_recursive(self, folder_path: str, files: List[Dict], current_path: str = ""):
        """Recursively list files in a folder."""
        try:
            drive = self._get_drive_endpoint()
            
            # URL encode the path
            encoded_path = folder_path.replace(" ", "%20")
            endpoint = f"{drive}/root:/{encoded_path}:/children"
            
            response = self._graph_request(endpoint)
            
            for item in response.get("value", []):
                item_path = f"{current_path}/{item['name']}".strip("/")
                
                if item.get("folder"):
                    # Recursively process subfolders
                    self._list_folder_recursive(f"{folder_path}/{item['name']}", files, item_path)
                elif item.get("file") and item["name"].lower().endswith(".pdf"):
                    # Parse the path to extract metadata
                    full_path = f"{folder_path}/{item['name']}"
                    parsed = _parse_path(full_path)
                    
                    files.append({
                        "filename": item["name"],
                        "folder_path": folder_path,
                        "full_path": full_path,
                        "path": full_path,
                        "source": self.source_name,
                        "account": parsed.get("account", "") if parsed else "",
                        "lob": parsed.get("lob", "") if parsed else "",
                        "policy_number": parsed.get("policy_number", "") if parsed else "",
                        "effective_date": parsed.get("effective_date", "") if parsed else "",
                        "size": item.get("size", 0),
                        "item_id": item["id"],
                        "web_url": item.get("webUrl", ""),
                        "last_modified": item.get("lastModifiedDateTime", "")
                    })
        except Exception as e:
            print(f"Warning: Could not list {folder_path}: {e}")
    
    def download_file(self, path: str) -> Tuple[bytes, str]:
        """Download a file from OneDrive."""
        import requests
        
        drive = self._get_drive_endpoint()
        encoded_path = path.replace(" ", "%20")
        
        # Get download URL
        endpoint = f"{drive}/root:/{encoded_path}:/content"
        
        headers = {"Authorization": f"Bearer {self._get_access_token()}"}
        url = f"https://graph.microsoft.com/v1.0{endpoint}"
        
        response = requests.get(url, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        return response.content, os.path.basename(path)
    
    def download_to_temp(self, path: str) -> str:
        """Download file to a temporary location and return the path."""
        content, filename = self.download_file(path)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        with open(temp_path, 'wb') as f:
            f.write(content)
        return temp_path
    
    def get_file_url(self, path: str) -> str:
        """Get the web URL for the file."""
        drive = self._get_drive_endpoint()
        encoded_path = path.replace(" ", "%20")
        
        endpoint = f"{drive}/root:/{encoded_path}"
        response = self._graph_request(endpoint)
        
        return response.get("webUrl", "")
    
    def upload_file(self, local_path: str, onedrive_path: str) -> bool:
        """Upload a file to OneDrive."""
        import requests
        
        try:
            drive = self._get_drive_endpoint()
            full_path = f"{self.root_folder}/{onedrive_path}".strip("/")
            encoded_path = full_path.replace(" ", "%20")
            
            # For files < 4MB, use simple upload
            file_size = os.path.getsize(local_path)
            
            if file_size < 4 * 1024 * 1024:  # 4MB
                endpoint = f"{drive}/root:/{encoded_path}:/content"
                url = f"https://graph.microsoft.com/v1.0{endpoint}"
                
                headers = {
                    "Authorization": f"Bearer {self._get_access_token()}",
                    "Content-Type": "application/octet-stream"
                }
                
                with open(local_path, "rb") as f:
                    response = requests.put(url, headers=headers, data=f)
                
                response.raise_for_status()
                print(f"✅ Uploaded to OneDrive: {onedrive_path}")
                return True
            else:
                # For larger files, use upload session (not implemented here)
                print("❌ File too large for simple upload. Implement upload session for files > 4MB")
                return False
                
        except Exception as e:
            print(f"❌ OneDrive upload failed: {e}")
            return False


# ============================================================================
# Unified Cloud Storage Manager
# ============================================================================

class CloudStorageManager:
    """
    Unified manager for multiple cloud storage providers.
    Supports Azure Blob Storage, OneDrive, and local mock storage.
    """
    
    def __init__(self):
        self.providers: List[StorageProvider] = []
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize storage providers based on configuration."""
        storage_mode = os.getenv("STORAGE_MODE", "mock").lower()
        
        print(f"📦 Initializing storage (mode: {storage_mode})")
        
        if storage_mode in ["azure", "both"]:
            azure = AzureBlobStorageProvider()
            if azure.is_available():
                self.providers.append(azure)
                print(f"   ✅ Azure Blob Storage configured")
            else:
                print(f"   ⚠️ Azure Blob Storage not configured (missing credentials)")
        
        if storage_mode in ["onedrive", "both"]:
            onedrive = OneDriveStorageProvider()
            if onedrive.is_available():
                self.providers.append(onedrive)
                print(f"   ✅ OneDrive configured")
            else:
                print(f"   ⚠️ OneDrive not configured (missing credentials)")
        
        if storage_mode == "mock" or not self.providers:
            # Fall back to mock storage
            print(f"   ℹ️ Using mock storage")
    
    def search_files(
        self,
        account_name: Optional[str] = None,
        lob: Optional[str] = None,
        policy_number: Optional[str] = None,
        date: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Search for files across all configured storage providers.
        Uses flexible matching similar to mock_storage.
        """
        all_results = []
        
        # Normalize search parameters
        search_account = _normalize_string(account_name) if account_name else None
        search_lob = _normalize_string(lob) if lob else None
        search_policy = _normalize_string(policy_number) if policy_number else None
        search_date = _normalize_date(date) if date else None
        
        # Resolve LOB aliases
        if search_lob and search_lob in LOB_ALIASES:
            search_lob = LOB_ALIASES[search_lob]
        
        print(f"🔍 Searching cloud storage: Account={search_account or 'any'}, LOB={search_lob or 'any'}, Policy={search_policy or 'any'}, Date={search_date or 'any'}")
        
        for provider in self.providers:
            try:
                files = provider.list_files()
                
                for file_info in files:
                    # Check Account (fuzzy match)
                    if search_account and not _fuzzy_match(search_account, file_info.get("account", "")):
                        continue
                    
                    # Check LOB
                    if search_lob:
                        file_lob = _normalize_string(file_info.get("lob", ""))
                        if search_lob != file_lob:
                            continue
                    
                    # Check Policy Number (partial match)
                    if search_policy and not _fuzzy_match(search_policy, file_info.get("policy_number", "")):
                        continue
                    
                    # Check Date
                    if search_date:
                        file_date = _normalize_date(file_info.get("effective_date", ""))
                        if search_date != file_date:
                            continue
                    
                    all_results.append(file_info)
                    
            except Exception as e:
                print(f"❌ Error searching {provider.source_name}: {e}")
        
        print(f"✅ Found {len(all_results)} matching files across cloud storage")
        return all_results
    
    def download_file(self, file_info: Dict[str, str]) -> str:
        """
        Download a file and return the local path.
        The file_info should contain 'source' and 'path' keys.
        """
        source = file_info.get("source", "")
        path = file_info.get("path", file_info.get("full_path", ""))
        
        for provider in self.providers:
            if provider.source_name == source:
                return provider.download_to_temp(path)
        
        raise ValueError(f"No provider found for source: {source}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available storage providers."""
        return [p.source_name for p in self.providers]


# ============================================================================
# Convenience Functions (drop-in replacement for mock_storage)
# ============================================================================

_manager: Optional[CloudStorageManager] = None

def _get_manager() -> CloudStorageManager:
    """Get or create the storage manager singleton."""
    global _manager
    if _manager is None:
        _manager = CloudStorageManager()
    return _manager


def search_files(
    account_name: Optional[str] = None,
    lob: Optional[str] = None,
    policy_number: Optional[str] = None,
    date: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Search for files across all configured cloud storage.
    This is a drop-in replacement for mock_storage.search_files().
    """
    manager = _get_manager()
    
    # If no cloud providers are configured, fall back to mock storage
    if not manager.providers:
        import mock_storage
        return mock_storage.search_files(account_name, lob, policy_number, date)
    
    return manager.search_files(account_name, lob, policy_number, date)


def download_file(file_info: Dict[str, str]) -> str:
    """Download a file and return the local path."""
    manager = _get_manager()
    
    # If source is mock storage, return the path directly
    if file_info.get("source") == "Mock Storage":
        return file_info.get("path", file_info.get("full_path", ""))
    
    return manager.download_file(file_info)


# ============================================================================
# Main - Test the integrations
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("☁️ CLOUD STORAGE INTEGRATION TEST")
    print("="*60)
    
    # Show current configuration
    print("\n📋 Configuration:")
    print(f"   STORAGE_MODE: {os.getenv('STORAGE_MODE', 'mock')}")
    print(f"   AZURE_STORAGE_CONNECTION_STRING: {'✅ Set' if os.getenv('AZURE_STORAGE_CONNECTION_STRING') else '❌ Not set'}")
    print(f"   AZURE_STORAGE_CONTAINER_NAME: {os.getenv('AZURE_STORAGE_CONTAINER_NAME', 'claims-documents')}")
    print(f"   ONEDRIVE_CLIENT_ID: {'✅ Set' if os.getenv('ONEDRIVE_CLIENT_ID') else '❌ Not set'}")
    print(f"   ONEDRIVE_CLIENT_SECRET: {'✅ Set' if os.getenv('ONEDRIVE_CLIENT_SECRET') else '❌ Not set'}")
    print(f"   ONEDRIVE_TENANT_ID: {'✅ Set' if os.getenv('ONEDRIVE_TENANT_ID') else '❌ Not set'}")
    
    # Test search
    print("\n🔍 Testing search...")
    results = search_files(account_name="chubbs", lob="auto")
    
    print(f"\nResults ({len(results)} files):")
    for r in results:
        print(f"   📄 {r.get('source')}: {r.get('folder_path')}/{r.get('filename')}")
