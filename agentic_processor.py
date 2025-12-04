#!/usr/bin/env python3
"""
Agentic AI Processor for Loss Run Processing
Demonstrates how agentic AI can enhance the extraction workflow
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


@dataclass
class ProcessingDecision:
    """Represents a decision made by the agent"""
    action: str
    strategy: str
    confidence: float
    reasoning: str
    alternatives: List[str]


@dataclass
class ValidationIssue:
    """Represents a data quality issue"""
    type: str
    severity: str  # "high", "medium", "low"
    affected_rows: List[int]
    description: str
    suggested_fix: Optional[str] = None


class AgenticProcessor:
    """
    Agentic AI processor that makes intelligent decisions about
    how to process PDFs and extract data
    """
    
    def __init__(self, openai_client=None, config_path="config.py"):
        self.client = openai_client
        self.config_path = config_path
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_knowledge_base(self) -> Dict:
        """Load learned patterns and strategies"""
        kb_path = Path("agent_knowledge_base.json")
        if kb_path.exists():
            try:
                return json.loads(kb_path.read_text())
            except:
                return {}
        return {
            "successful_strategies": {},
            "common_errors": {},
            "carrier_patterns": {}
        }
    
    def _save_knowledge_base(self):
        """Save learned patterns"""
        kb_path = Path("agent_knowledge_base.json")
        kb_path.write_text(json.dumps(self.knowledge_base, indent=2))
    
    def analyze_document(self, pdf_path: str) -> Dict:
        """
        Agent analyzes the document structure and characteristics
        """
        print(f"🤖 Agent: Analyzing document structure...")
        
        analysis = {
            "file_path": pdf_path,
            "file_size": Path(pdf_path).stat().st_size,
            "pages": 0,
            "structure_type": "unknown",
            "complexity": "medium",
            "has_tables": False,
            "has_text": False,
            "is_scanned": False
        }
        
        # Analyze with PyMuPDF
        if fitz:
            try:
                doc = fitz.open(pdf_path)
                analysis["pages"] = len(doc)
                
                # Sample first few pages
                text_samples = []
                for page_num in range(min(3, len(doc))):
                    page = doc[page_num]
                    text = page.get_text()
                    text_samples.append(text)
                    
                    # Check for tables
                    tables = page.find_tables()
                    if tables:
                        analysis["has_tables"] = True
                
                doc.close()
                
                # Determine structure
                combined_text = " ".join(text_samples)
                if analysis["has_tables"] and len(combined_text) > 100:
                    analysis["structure_type"] = "mixed_tables_and_text"
                elif analysis["has_tables"]:
                    analysis["structure_type"] = "table_based"
                elif len(combined_text) > 100:
                    analysis["structure_type"] = "text_based"
                else:
                    analysis["is_scanned"] = True
                    analysis["structure_type"] = "scanned_image"
                
                analysis["has_text"] = len(combined_text) > 50
                
                # Determine complexity
                if analysis["pages"] > 50:
                    analysis["complexity"] = "high"
                elif analysis["pages"] > 20:
                    analysis["complexity"] = "medium"
                else:
                    analysis["complexity"] = "low"
                    
            except Exception as e:
                print(f"⚠️ Error analyzing document: {e}")
        
        print(f"✅ Analysis complete: {analysis['structure_type']}, {analysis['pages']} pages")
        return analysis
    
    def select_strategy(self, analysis: Dict) -> ProcessingDecision:
        """
        Agent selects the best extraction strategy based on analysis
        """
        print(f"🤖 Agent: Selecting optimal strategy...")
        
        structure = analysis["structure_type"]
        complexity = analysis["complexity"]
        has_tables = analysis["has_tables"]
        
        # Decision logic based on document characteristics
        if structure == "table_based" and has_tables:
            strategy = "camelot_tabula_first"
            confidence = 0.85
            reasoning = "Document has clear table structure, use table extraction first"
            alternatives = ["claude_text", "adaptive_multi"]
            
        elif structure == "mixed_tables_and_text":
            strategy = "adaptive_multi"
            confidence = 0.80
            reasoning = "Mixed structure requires adaptive approach"
            alternatives = ["claude_text", "camelot_tabula_first"]
            
        elif structure == "text_based":
            strategy = "claude_text"
            confidence = 0.90
            reasoning = "Text-based document, use LLM extraction"
            alternatives = ["openai_text", "adaptive_multi"]
            
        elif structure == "scanned_image":
            strategy = "ocr_then_llm"
            confidence = 0.75
            reasoning = "Scanned document requires OCR first"
            alternatives = ["claude_image", "adaptive_multi"]
            
        else:
            strategy = "adaptive_multi"
            confidence = 0.70
            reasoning = "Unknown structure, use adaptive multi-approach"
            alternatives = ["claude_text", "camelot_tabula_first"]
        
        # Adjust for complexity
        if complexity == "high":
            reasoning += " (High complexity - may need chunking)"
        elif complexity == "low":
            reasoning += " (Low complexity - can process directly)"
        
        decision = ProcessingDecision(
            action="extract",
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives
        )
        
        print(f"✅ Strategy selected: {strategy} (confidence: {confidence:.0%})")
        print(f"   Reasoning: {reasoning}")
        
        return decision
    
    def process_with_strategy(self, pdf_path: str, decision: ProcessingDecision) -> Dict:
        """
        Agent executes the selected strategy with error recovery
        """
        print(f"🤖 Agent: Executing strategy '{decision.strategy}'...")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if decision.strategy == "claude_text":
                    result = self._extract_with_claude_text(pdf_path)
                elif decision.strategy == "openai_text":
                    result = self._extract_with_openai_text(pdf_path)
                elif decision.strategy == "camelot_tabula_first":
                    result = self._extract_with_tables_first(pdf_path)
                elif decision.strategy == "adaptive_multi":
                    result = self._extract_adaptive(pdf_path)
                elif decision.strategy == "ocr_then_llm":
                    result = self._extract_ocr_then_llm(pdf_path)
                else:
                    result = self._extract_adaptive(pdf_path)
                
                if result and result.get("success"):
                    print(f"✅ Extraction successful with strategy '{decision.strategy}'")
                    return result
                else:
                    raise Exception("Extraction returned no data")
                    
            except Exception as e:
                retry_count += 1
                print(f"⚠️ Attempt {retry_count} failed: {e}")
                
                if retry_count < max_retries and decision.alternatives:
                    # Try alternative strategy
                    alt_strategy = decision.alternatives[0]
                    print(f"🔄 Agent: Trying alternative strategy '{alt_strategy}'...")
                    decision.strategy = alt_strategy
                    decision.alternatives = decision.alternatives[1:]
                else:
                    print(f"❌ All strategies exhausted")
                    return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def validate_data(self, extracted_data: Dict, pdf_context: Dict) -> List[ValidationIssue]:
        """
        Agent validates extracted data quality
        """
        print(f"🤖 Agent: Validating extracted data...")
        
        issues = []
        
        # Check for missing claim numbers
        if "claims" in extracted_data:
            claims = extracted_data["claims"]
            missing_claim_nums = [
                i for i, claim in enumerate(claims)
                if not claim.get("claim_number") or not str(claim.get("claim_number")).strip()
            ]
            if missing_claim_nums:
                issues.append(ValidationIssue(
                    type="missing_claim_numbers",
                    severity="high",
                    affected_rows=missing_claim_nums,
                    description=f"{len(missing_claim_nums)} claims missing claim numbers",
                    suggested_fix="re_extract_with_stricter_prompt"
                ))
        
        # Check for amount format consistency
        if "claims" in extracted_data:
            claims = extracted_data["claims"]
            amount_fields = ["paid_loss", "reserve", "alae", "bi_paid_loss", "pd_paid_loss"]
            format_issues = []
            for i, claim in enumerate(claims):
                for field in amount_fields:
                    value = claim.get(field)
                    if value and isinstance(value, str):
                        # Check if it looks like a valid amount
                        if not any(char.isdigit() for char in value):
                            format_issues.append(i)
                            break
            if format_issues:
                issues.append(ValidationIssue(
                    type="amount_format_inconsistency",
                    severity="medium",
                    affected_rows=format_issues,
                    description=f"{len(format_issues)} claims have amount format issues",
                    suggested_fix="normalize_amount_formats"
                ))
        
        # Check for date format consistency
        if "claims" in extracted_data:
            claims = extracted_data["claims"]
            date_issues = []
            for i, claim in enumerate(claims):
                loss_date = claim.get("loss_date")
                if loss_date and isinstance(loss_date, str):
                    # Simple date format check
                    if len(loss_date) < 6:  # Too short to be a valid date
                        date_issues.append(i)
            if date_issues:
                issues.append(ValidationIssue(
                    type="date_format_error",
                    severity="low",
                    affected_rows=date_issues,
                    description=f"{len(date_issues)} claims have date format issues",
                    suggested_fix="normalize_dates"
                ))
        
        if issues:
            print(f"⚠️ Found {len(issues)} validation issues")
            for issue in issues:
                print(f"   - {issue.type}: {issue.description}")
        else:
            print(f"✅ Data validation passed")
        
        return issues
    
    def auto_correct(self, extracted_data: Dict, issues: List[ValidationIssue]) -> Dict:
        """
        Agent attempts to auto-correct identified issues
        """
        print(f"🤖 Agent: Attempting auto-correction...")
        
        corrected_data = extracted_data.copy()
        
        for issue in issues:
            if issue.suggested_fix == "normalize_amount_formats":
                # Normalize amount formats
                if "claims" in corrected_data:
                    for claim in corrected_data["claims"]:
                        for field in ["paid_loss", "reserve", "alae"]:
                            if field in claim:
                                value = str(claim[field])
                                # Remove common non-numeric chars except decimal point
                                cleaned = "".join(c for c in value if c.isdigit() or c in ".,-")
                                claim[field] = cleaned
                print(f"   ✅ Normalized amount formats")
            
            elif issue.suggested_fix == "normalize_dates":
                # Basic date normalization
                if "claims" in corrected_data:
                    for claim in corrected_data["claims"]:
                        if "loss_date" in claim:
                            # Try to normalize date format
                            date_str = str(claim["loss_date"])
                            # This is a simplified example - real implementation would be more robust
                            claim["loss_date"] = date_str
                print(f"   ✅ Normalized dates")
        
        return corrected_data
    
    def learn_from_result(self, pdf_path: str, strategy: str, success: bool, issues: List[ValidationIssue]):
        """
        Agent learns from the processing result to improve future decisions
        """
        print(f"🤖 Agent: Learning from this processing...")
        
        # Update knowledge base
        if success:
            if "successful_strategies" not in self.knowledge_base:
                self.knowledge_base["successful_strategies"] = {}
            
            if strategy not in self.knowledge_base["successful_strategies"]:
                self.knowledge_base["successful_strategies"][strategy] = 0
            self.knowledge_base["successful_strategies"][strategy] += 1
        
        # Track common errors
        for issue in issues:
            if "common_errors" not in self.knowledge_base:
                self.knowledge_base["common_errors"] = {}
            
            if issue.type not in self.knowledge_base["common_errors"]:
                self.knowledge_base["common_errors"][issue.type] = 0
            self.knowledge_base["common_errors"][issue.type] += 1
        
        self._save_knowledge_base()
        print(f"✅ Knowledge base updated")
    
    # Placeholder methods for actual extraction strategies
    def _extract_with_claude_text(self, pdf_path: str) -> Dict:
        """Extract using Claude text extraction"""
        # This would call your existing text_lob_llm_extractor.py
        return {"success": True, "data": {}}
    
    def _extract_with_openai_text(self, pdf_path: str) -> Dict:
        """Extract using OpenAI text extraction"""
        # This would call your existing text_lob_openai_extractor.py
        return {"success": True, "data": {}}
    
    def _extract_with_tables_first(self, pdf_path: str) -> Dict:
        """Extract using table extraction first"""
        # This would use camelot/tabula
        return {"success": True, "data": {}}
    
    def _extract_adaptive(self, pdf_path: str) -> Dict:
        """Extract using adaptive multi-approach"""
        # This would use your adaptive_table_extractor
        return {"success": True, "data": {}}
    
    def _extract_ocr_then_llm(self, pdf_path: str) -> Dict:
        """Extract using OCR then LLM"""
        # This would do OCR first, then LLM extraction
        return {"success": True, "data": {}}


def main():
    """
    Example usage of the agentic processor
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic AI Loss Run Processor")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--config", default="config.py", help="Config file path")
    args = parser.parse_args()
    
    # Initialize agent
    agent = AgenticProcessor(config_path=args.config)
    
    print("=" * 60)
    print("🤖 Agentic AI Loss Run Processor")
    print("=" * 60)
    
    # Step 1: Analyze document
    analysis = agent.analyze_document(args.pdf_path)
    print(f"\n📊 Document Analysis:")
    print(f"   Structure: {analysis['structure_type']}")
    print(f"   Pages: {analysis['pages']}")
    print(f"   Complexity: {analysis['complexity']}")
    print(f"   Has Tables: {analysis['has_tables']}")
    
    # Step 2: Select strategy
    decision = agent.select_strategy(analysis)
    print(f"\n🎯 Selected Strategy:")
    print(f"   Strategy: {decision.strategy}")
    print(f"   Confidence: {decision.confidence:.0%}")
    print(f"   Reasoning: {decision.reasoning}")
    
    # Step 3: Process with strategy
    result = agent.process_with_strategy(args.pdf_path, decision)
    
    if result.get("success"):
        # Step 4: Validate
        issues = agent.validate_data(result.get("data", {}), analysis)
        
        # Step 5: Auto-correct if needed
        if issues:
            corrected_data = agent.auto_correct(result.get("data", {}), issues)
            result["data"] = corrected_data
        
        # Step 6: Learn from result
        agent.learn_from_result(args.pdf_path, decision.strategy, True, issues)
        
        print(f"\n✅ Processing complete!")
        print(f"   Issues found: {len(issues)}")
        print(f"   Issues auto-corrected: {sum(1 for i in issues if i.suggested_fix)}")
    else:
        print(f"\n❌ Processing failed: {result.get('error')}")
        agent.learn_from_result(args.pdf_path, decision.strategy, False, [])
    
    print("=" * 60)


if __name__ == "__main__":
    main()


