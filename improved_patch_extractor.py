"""
Improved Patch Extraction Module
Implements best practices for extracting patches from LLM responses.
"""

import re
import json
import hashlib
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from pydantic import BaseModel
from llm_provider import (
    responses_parse,
    MODEL_GPT4O_MINI,
)
import os


@dataclass
class ExtractedPatch:
    """Structured patch with metadata"""
    content: str
    format: str  # "unified_diff", "git_diff", "raw"
    confidence: float
    extraction_method: str
    validation_passed: bool
    checksum: str
    metadata: Dict[str, any] = None


class ImprovedPatchExtractor:
    """
    Best-practice patch extraction from LLM responses.
    
    Features:
    - Multiple extraction strategies
    - Format validation
    - Structured output parsing
    - Checksum verification
    - Metadata preservation
    """
    
    def __init__(self):
        self.extraction_strategies = [
            self.extract_from_structured_json,
            self.extract_from_code_blocks,
            self.extract_from_diff_markers,
            self.extract_with_llm_parsing,
            self.extract_fallback
        ]
    
    def extract_patch(self, llm_response: str) -> Optional[ExtractedPatch]:
        """
        Extract patch from LLM response using multiple strategies.
        Returns the first successfully extracted and validated patch.
        """
        
        for i, strategy in enumerate(self.extraction_strategies):
            try:
                patch = strategy(llm_response)
                if patch and patch.validation_passed:
                    print(f"✓ Extracted with strategy {i+1}: {strategy.__name__}")
                    return patch
            except Exception as e:
                print(f"  Strategy {i+1} failed: {e}")
                continue
        
        return None
    
    def extract_from_structured_json(self, response: str) -> Optional[ExtractedPatch]:
        """
        Extract from structured JSON response (preferred method).
        Expects: {"patch": "...", "explanation": "..."}
        """
        try:
            # Try to parse as JSON
            data = json.loads(response)
            
            if isinstance(data, dict):
                # Look for patch in common keys
                patch_keys = ['patch', 'patch_in_unified_diff_format', 'unified_diff', 'diff']
                for key in patch_keys:
                    if key in data and data[key]:
                        patch_content = data[key]
                        if self._validate_patch_format(patch_content):
                            return ExtractedPatch(
                                content=patch_content,
                                format="unified_diff",
                                confidence=0.95,
                                extraction_method="structured_json",
                                validation_passed=True,
                                checksum=self._compute_checksum(patch_content),
                                metadata={"explanation": data.get("explanation")}
                            )
        except json.JSONDecodeError:
            pass
        
        return None
    
    def extract_from_code_blocks(self, response: str) -> Optional[ExtractedPatch]:
        """
        Extract from markdown code blocks.
        Looks for ```diff or ``` blocks.
        """
        # Try ```diff blocks first
        diff_pattern = r'```diff\s*\n(.*?)\n```'
        matches = re.findall(diff_pattern, response, re.DOTALL)
        
        for match in matches:
            if self._validate_patch_format(match):
                return ExtractedPatch(
                    content=match.strip(),
                    format="unified_diff",
                    confidence=0.9,
                    extraction_method="code_block_diff",
                    validation_passed=True,
                    checksum=self._compute_checksum(match)
                )
        
        # Try generic ``` blocks
        generic_pattern = r'```\s*\n(.*?)\n```'
        matches = re.findall(generic_pattern, response, re.DOTALL)
        
        for match in matches:
            # Check if it looks like a patch
            if self._looks_like_patch(match) and self._validate_patch_format(match):
                return ExtractedPatch(
                    content=match.strip(),
                    format="unified_diff",
                    confidence=0.8,
                    extraction_method="code_block_generic",
                    validation_passed=True,
                    checksum=self._compute_checksum(match)
                )
        
        return None
    
    def extract_from_diff_markers(self, response: str) -> Optional[ExtractedPatch]:
        """
        Extract by identifying diff markers (diff --git, +++, ---, @@).
        """
        lines = response.splitlines()
        
        # Find start of patch (diff --git line)
        start_idx = None
        for i, line in enumerate(lines):
            if line.startswith('diff --git'):
                start_idx = i
                break
        
        if start_idx is None:
            return None
        
        # Extract until end or next non-patch content
        patch_lines = []
        for i in range(start_idx, len(lines)):
            line = lines[i]
            
            # Stop if we hit explanatory text after patch
            if patch_lines and not self._is_patch_line(line):
                # Allow a few non-patch lines (could be context)
                next_few = lines[i:i+3]
                if not any(self._is_patch_line(l) for l in next_few):
                    break
            
            patch_lines.append(line)
        
        if patch_lines:
            patch_content = '\n'.join(patch_lines)
            if self._validate_patch_format(patch_content):
                return ExtractedPatch(
                    content=patch_content,
                    format="unified_diff",
                    confidence=0.85,
                    extraction_method="diff_markers",
                    validation_passed=True,
                    checksum=self._compute_checksum(patch_content)
                )
        
        return None
    
    def extract_with_llm_parsing(self, response: str) -> Optional[ExtractedPatch]:
        """
        Use LLM to extract and clean the patch from response.
        """
        prompt = f"""Extract ONLY the unified diff patch from the following response.
Return ONLY the patch content, nothing else.

RESPONSE:
{response[:3000]}

PATCH:"""
        
        try:
            class PatchOnly(BaseModel):
                patch: str

            parsed = responses_parse(
                model=MODEL_GPT4O_MINI,
                input=[
                    {"role": "system", "content": "Extract only the unified diff. Output must match the expected schema."},
                    {"role": "user", "content": prompt}
                ],
                text_format=PatchOnly,
                temperature=0.1,
                max_output_tokens=2000
            )
            extracted = parsed.output_parsed.patch.strip()
            
            # Remove any remaining code block markers (defensive)
            extracted = re.sub(r'```(?:diff)?\s*\n?', '', extracted).strip()
            
            if self._validate_patch_format(extracted):
                return ExtractedPatch(
                    content=extracted,
                    format="unified_diff",
                    confidence=0.75,
                    extraction_method="llm_parsing",
                    validation_passed=True,
                    checksum=self._compute_checksum(extracted)
                )
        except Exception as e:
            print(f"LLM parsing failed: {e}")
        
        return None
    
    def extract_fallback(self, response: str) -> Optional[ExtractedPatch]:
        """
        Fallback: return entire response if it looks like a patch.
        """
        if self._looks_like_patch(response) and self._validate_patch_format(response):
            return ExtractedPatch(
                content=response.strip(),
                format="unified_diff",
                confidence=0.6,
                extraction_method="fallback_full_response",
                validation_passed=True,
                checksum=self._compute_checksum(response)
            )
        
        return None
    
    def _validate_patch_format(self, content: str) -> bool:
        """
        Validate that content is a proper unified diff.
        """
        if not content or len(content.strip()) == 0:
            return False
        
        lines = content.splitlines()
        
        # Must have minimum required elements
        has_diff_header = any(l.startswith('diff --git') for l in lines)
        has_file_markers = (any(l.startswith('---') for l in lines) and 
                           any(l.startswith('+++') for l in lines))
        has_hunk = any(l.startswith('@@') for l in lines)
        
        if not (has_diff_header and has_file_markers and has_hunk):
            return False
        
        # Validate hunk header format
        for line in lines:
            if line.startswith('@@'):
                if not re.match(r'@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@', line):
                    return False
        
        # Check that hunk lines are properly marked
        in_hunk = False
        for line in lines:
            if line.startswith('@@'):
                in_hunk = True
            elif in_hunk and line and not line.startswith(('diff', '---', '+++')):
                if line[0] not in (' ', '+', '-', '\\'):  # \\ for "\ No newline"
                    # Invalid hunk line
                    return False
        
        return True
    
    def _looks_like_patch(self, content: str) -> bool:
        """Quick check if content resembles a patch."""
        patch_indicators = ['diff --git', '@@', '---', '+++']
        return any(indicator in content for indicator in patch_indicators)
    
    def _is_patch_line(self, line: str) -> bool:
        """Check if line is part of a patch."""
        if not line:
            return True  # Empty lines are ok in patches
        
        return line.startswith((
            'diff ', 'index ', '---', '+++', '@@',
            '+', '-', ' ',  # Hunk content
            'new file', 'deleted file', 'rename ', 'copy ',
            'Binary files', '\\ No newline'
        ))
    
    def _compute_checksum(self, content: str) -> str:
        """Compute SHA256 checksum of patch."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def extract_multiple_patches(self, response: str) -> List[ExtractedPatch]:
        """
        Extract multiple patches from a single response.
        Useful when LLM returns patches for multiple files.
        """
        patches = []
        
        # Split by diff --git markers
        parts = re.split(r'(diff --git .*?$)', response, flags=re.MULTILINE)
        
        current_patch = []
        for part in parts:
            if part.startswith('diff --git'):
                if current_patch:
                    # Try to extract patch from accumulated content
                    patch_content = ''.join(current_patch)
                    extracted = self.extract_patch(patch_content)
                    if extracted:
                        patches.append(extracted)
                current_patch = [part]
            elif current_patch:
                current_patch.append(part)
        
        # Process last patch
        if current_patch:
            patch_content = ''.join(current_patch)
            extracted = self.extract_patch(patch_content)
            if extracted:
                patches.append(extracted)
        
        return patches


def integrate_with_llm_rewriter():
    """
    Example integration with LLMRewriter in core.py.
    This shows how to use ImprovedPatchExtractor.
    """
    extractor = ImprovedPatchExtractor()
    
    # Simulated LLM response
    llm_response = """
    Here's the improved patch:
    
    ```diff
    diff --git a/file.py b/file.py
    --- a/file.py
    +++ b/file.py
    @@ -1,3 +1,4 @@
     def example():
    -    return 1
    +    result: int = 1
    +    return result
    ```
    
    This adds proper type annotations.
    """
    
    # Extract patch
    patch = extractor.extract_patch(llm_response)
    
    if patch:
        print(f"✓ Extracted patch (confidence: {patch.confidence})")
        print(f"  Method: {patch.extraction_method}")
        print(f"  Checksum: {patch.checksum}")
        print(f"  Valid: {patch.validation_passed}")
        print(f"\nPatch content:\n{patch.content}")
    else:
        print("✗ Failed to extract patch")
    
    return patch


if __name__ == "__main__":
    integrate_with_llm_rewriter()
