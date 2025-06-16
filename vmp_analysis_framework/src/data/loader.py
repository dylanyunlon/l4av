"""
Data Loading and Preprocessing Module
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of VMP transformation data"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get('batch_size', 10000)

    def load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """Load JSONL file with progress tracking"""
        data = []
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading data from {filepath}")

        # Get total lines for progress bar
        total_lines = sum(1 for _ in open(filepath, 'r', encoding='utf-8'))

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Loading data"):
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue

        logger.info(f"Loaded {len(data)} entries")
        return data

    def stream_jsonl(self, filepath: str) -> Iterator[Dict[str, Any]]:
        """Stream JSONL file for memory-efficient processing"""
        filepath = Path(filepath)

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    yield json.loads(line.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue


class DataValidator:
    """Validates and cleans VMP transformation data"""

    REQUIRED_FIELDS = ['line', 'function', 'original_assembly', 'vmp_assembly', 'bytecode_size']

    def __init__(self):
        self.validation_stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'cleaned': 0,
            'errors': []
        }

    def validate_and_clean(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean the dataset"""
        clean_data = []

        for entry in tqdm(data, desc="Validating data"):
            self.validation_stats['total'] += 1

            if self._validate_entry(entry):
                cleaned_entry = self._clean_entry(entry)
                clean_data.append(cleaned_entry)
                self.validation_stats['valid'] += 1
            else:
                self.validation_stats['invalid'] += 1

        logger.info(f"Validation complete: {self.validation_stats}")
        return clean_data

    def _validate_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate a single entry"""
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in entry:
                self.validation_stats['errors'].append(f"Missing field: {field}")
                return False

        # Validate data types
        if not isinstance(entry['line'], int) or entry['line'] < 1:
            self.validation_stats['errors'].append("Invalid line number")
            return False

        if not isinstance(entry['bytecode_size'], int) or entry['bytecode_size'] < 0:
            self.validation_stats['errors'].append("Invalid bytecode size")
            return False

        # Validate assembly content
        if not entry['original_assembly'] or not entry['vmp_assembly']:
            self.validation_stats['errors'].append("Empty assembly content")
            return False

        return True

    def _clean_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and preprocess a single entry"""
        cleaned = entry.copy()

        # Extract function category from name
        cleaned['function_category'] = self._extract_function_category(entry['function'])

        # Parse assembly instructions count
        cleaned['original_instruction_count'] = self._count_instructions(entry['original_assembly'])
        cleaned['vmp_instruction_count'] = self._count_vmp_instructions(entry['vmp_assembly'])

        # Calculate basic metrics
        cleaned['code_expansion_ratio'] = (
                cleaned['bytecode_size'] / max(1, len(entry['original_assembly'].encode('utf-8')))
        )

        self.validation_stats['cleaned'] += 1
        return cleaned

    def _extract_function_category(self, function_name: str) -> str:
        """Extract category from function name"""
        categories = {
            'openssl': ['ssl', 'tls', 'crypto', 'aes', 'rsa', 'sha'],
            'crypt': ['crypt', 'encrypt', 'decrypt', 'hash'],
            'network': ['socket', 'net', 'tcp', 'udp', 'http'],
            'system': ['sys', 'kernel', 'os', 'file'],
            'audio': ['snd', 'audio', 'sound', 'wav', 'mp3'],
            'video': ['video', 'mpeg', 'h264', 'codec'],
            'math': ['math', 'calc', 'fft', 'matrix'],
            'compression': ['zip', 'gzip', 'compress', 'deflate']
        }

        function_lower = function_name.lower()
        for category, keywords in categories.items():
            if any(keyword in function_lower for keyword in keywords):
                return category

        return 'other'

    def _count_instructions(self, assembly: str) -> int:
        """Count x86-64 instructions in assembly text"""
        # Simple heuristic: count lines that look like instructions
        lines = assembly.strip().split('\n')
        instruction_pattern = re.compile(r'^\s*[a-zA-Z]+\s+')

        count = 0
        for line in lines:
            # Skip labels, comments, and directives
            if instruction_pattern.match(line) and not line.strip().endswith(':'):
                count += 1

        return count

    def _count_vmp_instructions(self, vmp_assembly: str) -> int:
        """Count VMP instructions including VM bytecode"""
        # Count both assembly instructions and bytecode entries
        instruction_count = self._count_instructions(vmp_assembly)

        # Look for bytecode arrays
        bytecode_matches = re.findall(r'db\s+(?:\d+(?:,\s*)?)+', vmp_assembly)
        bytecode_count = sum(len(match.split(',')) for match in bytecode_matches)

        return instruction_count + bytecode_count