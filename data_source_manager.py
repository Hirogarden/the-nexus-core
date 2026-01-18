"""
The Nexus Core - External Data Source Manager
Handles USB drives and network sources with HIPAA compliance.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import shutil
import hashlib
from datetime import datetime


class DataSourceManager:
    """
    Manage external data sources (USB, network drives) with HIPAA compliance.
    """
    
    def __init__(self, base_path: str = "./nexus_data"):
        self.base_path = Path(base_path)
        self.sources_path = self.base_path / "external_sources"
        self.audit_log_path = self.base_path / "audit_logs"
        self.metadata_path = self.base_path / "source_metadata.json"
        
        # Create directories
        self.sources_path.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load source metadata."""
        if self.metadata_path.exists():
            try:
                return json.loads(self.metadata_path.read_text())
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
        
        return {"sources": {}, "last_updated": None}
    
    def _save_metadata(self):
        """Save source metadata."""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata_path.write_text(json.dumps(self.metadata, indent=2))
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def scan_external_source(
        self,
        source_path: str,
        source_type: str = "usb"
    ) -> Dict[str, Any]:
        """
        Scan external source for importable data.
        
        Args:
            source_path: Path to external source
            source_type: "usb", "network", "local"
        
        Returns:
            Scan results with file counts and metadata
        """
        source = Path(source_path)
        
        if not source.exists():
            return {
                "success": False,
                "error": "Source path does not exist",
                "path": source_path
            }
        
        scan_results = {
            "success": True,
            "path": str(source),
            "type": source_type,
            "files": {},
            "total_size": 0,
            "scan_time": datetime.now().isoformat()
        }
        
        # Supported file types
        file_types = {
            ".txt": "text",
            ".md": "markdown",
            ".pdf": "pdf",
            ".json": "json",
            ".csv": "csv",
            ".log": "log"
        }
        
        try:
            for file_path in source.rglob("*"):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    
                    if ext in file_types:
                        file_type = file_types[ext]
                        
                        if file_type not in scan_results["files"]:
                            scan_results["files"][file_type] = []
                        
                        file_info = {
                            "path": str(file_path),
                            "name": file_path.name,
                            "size": file_path.stat().st_size,
                            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        }
                        
                        scan_results["files"][file_type].append(file_info)
                        scan_results["total_size"] += file_info["size"]
        
        except Exception as e:
            scan_results["success"] = False
            scan_results["error"] = str(e)
        
        # Log scan operation
        self._audit_log("scan", {
            "source_path": source_path,
            "source_type": source_type,
            "files_found": sum(len(files) for files in scan_results.get("files", {}).values()),
            "success": scan_results["success"]
        })
        
        return scan_results
    
    def verify_source(
        self,
        source_path: str,
        check_integrity: bool = True
    ) -> Dict[str, Any]:
        """
        Verify external source is readable and data is not corrupted.
        
        Args:
            source_path: Path to verify
            check_integrity: Whether to check file integrity
        
        Returns:
            Verification results
        """
        source = Path(source_path)
        
        verification = {
            "path": str(source),
            "exists": source.exists(),
            "readable": False,
            "writable": False,
            "integrity_checked": check_integrity,
            "corrupted_files": [],
            "verify_time": datetime.now().isoformat()
        }
        
        if not source.exists():
            return verification
        
        # Check read/write permissions
        try:
            verification["readable"] = os.access(str(source), os.R_OK)
            verification["writable"] = os.access(str(source), os.W_OK)
        except:
            pass
        
        # Check file integrity if requested
        if check_integrity and source.is_dir():
            for file_path in source.rglob("*"):
                if file_path.is_file():
                    try:
                        # Try to read file
                        with open(file_path, 'rb') as f:
                            f.read(1024)  # Read first 1KB
                    except Exception as e:
                        verification["corrupted_files"].append({
                            "path": str(file_path),
                            "error": str(e)
                        })
        
        verification["is_valid"] = (
            verification["exists"] and
            verification["readable"] and
            len(verification["corrupted_files"]) == 0
        )
        
        self._audit_log("verify", {
            "source_path": source_path,
            "is_valid": verification["is_valid"],
            "corrupted_count": len(verification["corrupted_files"])
        })
        
        return verification
    
    def import_source(
        self,
        source_path: str,
        import_mode: str = "copy",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Import data from external source.
        
        Args:
            source_path: Path to import from
            import_mode: "copy" (full copy), "reference" (link only), "selective" (filtered)
            filters: File filters {"extensions": [".txt", ".md"], "max_size": 10485760}
        
        Returns:
            Import results
        """
        source = Path(source_path)
        
        if not source.exists():
            return {
                "success": False,
                "error": "Source does not exist",
                "imported_count": 0
            }
        
        # Generate unique import ID
        import_id = hashlib.md5(f"{source_path}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        import_dir = self.sources_path / import_id
        import_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "success": True,
            "import_id": import_id,
            "import_path": str(import_dir),
            "mode": import_mode,
            "imported_files": [],
            "skipped_files": [],
            "errors": [],
            "import_time": datetime.now().isoformat()
        }
        
        # Apply filters
        extensions = filters.get("extensions", []) if filters else []
        max_size = filters.get("max_size", float('inf')) if filters else float('inf')
        
        try:
            if import_mode == "copy":
                # Full copy of all files
                for file_path in source.rglob("*"):
                    if file_path.is_file():
                        # Check filters
                        if extensions and file_path.suffix not in extensions:
                            results["skipped_files"].append({
                                "path": str(file_path),
                                "reason": "extension_filter"
                            })
                            continue
                        
                        if file_path.stat().st_size > max_size:
                            results["skipped_files"].append({
                                "path": str(file_path),
                                "reason": "size_limit"
                            })
                            continue
                        
                        # Copy file
                        try:
                            rel_path = file_path.relative_to(source)
                            dest_path = import_dir / rel_path
                            dest_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            shutil.copy2(file_path, dest_path)
                            
                            results["imported_files"].append({
                                "source": str(file_path),
                                "destination": str(dest_path),
                                "size": file_path.stat().st_size
                            })
                        except Exception as e:
                            results["errors"].append({
                                "file": str(file_path),
                                "error": str(e)
                            })
            
            elif import_mode == "reference":
                # Store reference only (no copy)
                reference_file = import_dir / "source_reference.json"
                reference_data = {
                    "source_path": str(source),
                    "reference_time": datetime.now().isoformat(),
                    "note": "This is a reference link. Original files remain at source location."
                }
                reference_file.write_text(json.dumps(reference_data, indent=2))
                
                results["imported_files"].append({
                    "type": "reference",
                    "source": str(source),
                    "reference_file": str(reference_file)
                })
        
        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
        
        # Update metadata
        self.metadata["sources"][import_id] = {
            "source_path": str(source),
            "import_time": results["import_time"],
            "mode": import_mode,
            "file_count": len(results["imported_files"])
        }
        self._save_metadata()
        
        # Audit log
        self._audit_log("import", {
            "import_id": import_id,
            "source_path": str(source),
            "mode": import_mode,
            "imported_count": len(results["imported_files"]),
            "skipped_count": len(results["skipped_files"]),
            "success": results["success"]
        })
        
        return results
    
    def list_imported_sources(self) -> List[Dict[str, Any]]:
        """List all imported sources."""
        sources = []
        
        for import_id, source_info in self.metadata.get("sources", {}).items():
            sources.append({
                "import_id": import_id,
                **source_info
            })
        
        return sources
    
    def remove_imported_source(self, import_id: str) -> bool:
        """Remove an imported source."""
        import_dir = self.sources_path / import_id
        
        if not import_dir.exists():
            return False
        
        try:
            shutil.rmtree(import_dir)
            
            # Remove from metadata
            if import_id in self.metadata.get("sources", {}):
                del self.metadata["sources"][import_id]
                self._save_metadata()
            
            self._audit_log("remove", {
                "import_id": import_id,
                "success": True
            })
            
            return True
        
        except Exception as e:
            self._audit_log("remove", {
                "import_id": import_id,
                "success": False,
                "error": str(e)
            })
            return False
    
    def _audit_log(self, action: str, details: Dict[str, Any]):
        """
        HIPAA-compliant audit logging.
        
        Logs all data access and modifications for compliance.
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "details": details
        }
        
        # Daily log file
        log_file = self.audit_log_path / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not write audit log: {e}")
    
    def get_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs for compliance review.
        
        Args:
            start_date: Filter logs from this date
            end_date: Filter logs until this date
            action_filter: Filter by action type
        
        Returns:
            List of audit log entries
        """
        logs = []
        
        for log_file in self.audit_log_path.glob("audit_*.jsonl"):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        
                        # Apply filters
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        
                        if start_date and entry_time < start_date:
                            continue
                        if end_date and entry_time > end_date:
                            continue
                        if action_filter and entry["action"] != action_filter:
                            continue
                        
                        logs.append(entry)
            
            except Exception as e:
                print(f"Warning: Could not read log file {log_file}: {e}")
        
        # Sort by timestamp
        logs.sort(key=lambda x: x["timestamp"])
        
        return logs


# Windows compatibility
import os


if __name__ == "__main__":
    print("🌟 Data Source Manager Demo\n")
    
    manager = DataSourceManager("./demo_data_sources")
    
    # Demo: Scan a source (use current directory as example)
    print("1️⃣ Scanning source...")
    scan_results = manager.scan_external_source(".", source_type="local")
    
    if scan_results["success"]:
        print(f"✅ Found {sum(len(files) for files in scan_results['files'].values())} files")
        print(f"Total size: {scan_results['total_size']:,} bytes")
        for file_type, files in scan_results["files"].items():
            print(f"  - {file_type}: {len(files)} files")
    else:
        print(f"❌ Scan failed: {scan_results.get('error')}")
    
    print("\n2️⃣ Verifying source...")
    verification = manager.verify_source(".")
    print(f"Exists: {verification['exists']}")
    print(f"Readable: {verification['readable']}")
    print(f"Valid: {verification['is_valid']}")
    
    print("\n3️⃣ Viewing audit logs...")
    logs = manager.get_audit_logs()
    print(f"Found {len(logs)} audit entries")
    for log in logs[-3:]:  # Show last 3
        print(f"  - {log['timestamp']}: {log['action']}")
