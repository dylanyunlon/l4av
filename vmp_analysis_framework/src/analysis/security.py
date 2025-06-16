"""
Security Analysis Module for VMP Transformations
"""

import re
import logging
from typing import Dict, Any, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SecurityFeatures:
    """Container for security-related features"""
    anti_debug_score: float
    anti_tamper_score: float
    code_integrity_score: float
    dynamic_decryption_score: float
    anti_static_analysis_score: float
    vm_complexity_score: float
    obfuscation_layers: int
    security_mechanisms: List[str]
    vulnerabilities: List[str]


class SecurityAnalyzer:
    """Analyzes security aspects of VMP transformations"""

    # Security patterns to detect
    SECURITY_PATTERNS = {
        'anti_debug': [
            (r'rdtsc', 'Timing check (RDTSC)'),
            (r'rdtscp', 'Timing check (RDTSCP)'),
            (r'cpuid', 'CPUID detection'),
            (r'int\s+3', 'INT3 breakpoint detection'),
            (r'int\s+0x2d', 'INT 2D detection'),
            (r'pushf.*popf', 'Flag manipulation'),
            (r'sidt|sgdt|sldt', 'Descriptor table checks'),
            (r'str\s+', 'Task register check'),
            (r'IsDebuggerPresent', 'API-based detection'),
            (r'CheckRemoteDebuggerPresent', 'Remote debugger check'),
            (r'NtQueryInformationProcess', 'Process info query'),
            (r'GetTickCount', 'Timing analysis'),
        ],

        'anti_tamper': [
            (r'checksum|crc|hash', 'Checksum verification'),
            (r'vmp_checksum', 'VMP checksum'),
            (r'integrity', 'Integrity check'),
            (r'cmp.*\n.*j[ne][ez].*debugger_detected', 'Debugger detection branch'),
            (r'self.*modify', 'Self-modifying code'),
        ],

        'encryption': [
            (r'xor.*key|decrypt|encrypt', 'XOR encryption'),
            (r'aes|rc4|des', 'Standard crypto'),
            (r'pxor|pshufd|pshufb', 'SIMD crypto operations'),
            (r'dynamic.*decrypt', 'Dynamic decryption'),
            (r'vm_decrypt', 'VM-based decryption'),
        ],

        'obfuscation': [
            (r'jmp\s+\w+\s*\n\s*db\s+', 'Junk bytes after jump'),
            (r'(push.*pop|xor.*xor)', 'Dead code patterns'),
            (r'vm_handler|vm_dispatch', 'VM handlers'),
            (r'opaque.*predicate', 'Opaque predicates'),
            (r'control.*flow.*flatten', 'Control flow flattening'),
        ]
    }

    # Known vulnerability patterns
    VULNERABILITY_PATTERNS = [
        (r'strcpy|strcat(?!\w)', 'Unsafe string operations'),
        (r'gets(?!\w)', 'Unsafe input function'),
        (r'sprintf(?!\w)', 'Unsafe formatting'),
        (r'stack.*overflow', 'Potential stack overflow'),
        (r'buffer.*overflow', 'Potential buffer overflow'),
        (r'format.*string', 'Format string vulnerability'),
        (r'use.*after.*free', 'Use after free'),
        (r'double.*free', 'Double free'),
        (r'null.*deref', 'Null dereference'),
    ]

    def __init__(self):
        self.vm_pattern = re.compile(r'vm[_\s]*(handler|interpreter|dispatch|loop)', re.IGNORECASE)

    def analyze(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive security analysis"""
        logger.info("Starting security analysis")

        security_results = []

        for entry in data:
            sec_features = self._analyze_security_features(entry)
            security_results.append({
                'function': entry['function'],
                'category': entry.get('function_category', 'unknown'),
                'security': sec_features.__dict__,
                'bytecode_size': entry['bytecode_size']
            })

        # Aggregate security statistics
        aggregated_stats = self._aggregate_security_stats(security_results)

        # Identify security patterns and trends
        patterns = self._identify_security_patterns(security_results)

        return {
            'individual_results': security_results,
            'aggregated_stats': aggregated_stats,
            'security_patterns': patterns
        }

    def _analyze_security_features(self, entry: Dict[str, Any]) -> SecurityFeatures:
        """Analyze security features for a single transformation"""
        vmp_asm = entry['vmp_assembly']

        # Detect security mechanisms
        mechanisms = self._detect_security_mechanisms(vmp_asm)

        # Detect vulnerabilities
        vulnerabilities = self._detect_vulnerabilities(vmp_asm)

        # Calculate security scores
        features = SecurityFeatures(
            anti_debug_score=self._calculate_anti_debug_score(vmp_asm, mechanisms),
            anti_tamper_score=self._calculate_anti_tamper_score(vmp_asm, mechanisms),
            code_integrity_score=self._calculate_integrity_score(vmp_asm),
            dynamic_decryption_score=self._calculate_decryption_score(vmp_asm, mechanisms),
            anti_static_analysis_score=self._calculate_anti_static_score(vmp_asm, entry),
            vm_complexity_score=self._calculate_vm_complexity(vmp_asm),
            obfuscation_layers=self._count_obfuscation_layers(vmp_asm),
            security_mechanisms=mechanisms,
            vulnerabilities=vulnerabilities
        )

        return features

    def _detect_security_mechanisms(self, vmp_asm: str) -> List[str]:
        """Detect security mechanisms in VMP code"""
        mechanisms = []

        for category, patterns in self.SECURITY_PATTERNS.items():
            for pattern, description in patterns:
                if re.search(pattern, vmp_asm, re.IGNORECASE):
                    mechanisms.append(f"{category}: {description}")

        return list(set(mechanisms))  # Remove duplicates

    def _detect_vulnerabilities(self, vmp_asm: str) -> List[str]:
        """Detect potential vulnerabilities"""
        vulnerabilities = []

        for pattern, description in self.VULNERABILITY_PATTERNS:
            if re.search(pattern, vmp_asm, re.IGNORECASE):
                vulnerabilities.append(description)

        # Check for VMP-specific issues
        if 'vm_error' in vmp_asm and 'exception handler' not in vmp_asm.lower():
            vulnerabilities.append('Unhandled VM errors')

        if re.search(r'mov.*0xDEADBEEF', vmp_asm):
            vulnerabilities.append('Debug/test code present')

        return vulnerabilities

    def _calculate_anti_debug_score(self, vmp_asm: str, mechanisms: List[str]) -> float:
        """Calculate anti-debugging protection score"""
        score = 0.0

        # Count anti-debug mechanisms
        anti_debug_count = sum(1 for m in mechanisms if 'anti_debug:' in m)
        score += min(anti_debug_count * 0.15, 0.6)

        # Check for timing-based protection
        if any('timing' in m.lower() for m in mechanisms):
            score += 0.2

        # Check for exception-based protection
        if re.search(r'\.debugger_detected|\.vm_error', vmp_asm):
            score += 0.2

        return min(score, 1.0)

    def _calculate_anti_tamper_score(self, vmp_asm: str, mechanisms: List[str]) -> float:
        """Calculate anti-tampering protection score"""
        score = 0.0

        # Checksum/integrity checks
        if 'vmp_checksum' in vmp_asm:
            score += 0.3

        # Self-modifying code indicators
        if any('self-modify' in m.lower() for m in mechanisms):
            score += 0.2

        # Code verification patterns
        integrity_patterns = [
            r'cmp.*checksum',
            r'integrity.*check',
            r'verify.*code'
        ]

        for pattern in integrity_patterns:
            if re.search(pattern, vmp_asm, re.IGNORECASE):
                score += 0.15

        return min(score, 1.0)

    def _calculate_integrity_score(self, vmp_asm: str) -> float:
        """Calculate code integrity protection score"""
        score = 0.0

        # Look for integrity verification structures
        if 'vmp_checksum' in vmp_asm:
            score += 0.25

        # Hash/CRC patterns
        if re.search(r'(hash|crc|checksum)', vmp_asm, re.IGNORECASE):
            score += 0.25

        # VM integrity checks
        if re.search(r'vm.*integrity|verify.*vm', vmp_asm, re.IGNORECASE):
            score += 0.3

        # Jump table protection
        if 'vmp_interpreter_table' in vmp_asm:
            score += 0.2

        return min(score, 1.0)

    def _calculate_decryption_score(self, vmp_asm: str, mechanisms: List[str]) -> float:
        """Calculate dynamic decryption capability score"""
        score = 0.0

        # Check for encryption indicators
        encryption_count = sum(1 for m in mechanisms if 'encryption:' in m)
        score += min(encryption_count * 0.2, 0.6)

        # VM-based decryption
        if re.search(r'vm.*decrypt|decrypt.*handler', vmp_asm, re.IGNORECASE):
            score += 0.3

        # XOR chains
        xor_count = len(re.findall(r'xor\s+\w+,\s*\w+', vmp_asm))
        if xor_count > 5:
            score += 0.1

        return min(score, 1.0)

    def _calculate_anti_static_score(self, vmp_asm: str, entry: Dict[str, Any]) -> float:
        """Calculate anti-static analysis protection score"""
        score = 0.0

        # Code expansion makes static analysis harder
        expansion_rate = entry.get('code_expansion_ratio', 1.0)
        score += min((expansion_rate - 1) * 0.1, 0.3)

        # VM protection
        vm_count = len(self.vm_pattern.findall(vmp_asm))
        score += min(vm_count * 0.1, 0.4)

        # Obfuscation patterns
        obfuscation_patterns = [
            r'jmp.*\n.*db',  # Junk after jump
            r'(push.*pop){3,}',  # Repeated push/pop
            r'xor.*,.*\n.*xor.*,',  # XOR chains
        ]

        for pattern in obfuscation_patterns:
            if re.search(pattern, vmp_asm):
                score += 0.1

        return min(score, 1.0)

    def _calculate_vm_complexity(self, vmp_asm: str) -> float:
        """Calculate VM implementation complexity"""
        score = 0.0

        # Count VM components
        vm_components = {
            'handlers': len(re.findall(r'vm_handler_\d+', vmp_asm)),
            'dispatch': len(re.findall(r'vm_dispatch|vm_loop', vmp_asm)),
            'interpreter': len(re.findall(r'interpreter.*impl', vmp_asm)),
            'fetch': len(re.findall(r'vm_fetch|fetch.*decode', vmp_asm))
        }

        # Weighted complexity
        score += min(vm_components['handlers'] * 0.05, 0.3)
        score += min(vm_components['dispatch'] * 0.2, 0.3)
        score += min(vm_components['interpreter'] * 0.3, 0.3)
        score += min(vm_components['fetch'] * 0.1, 0.1)

        return score

    def _count_obfuscation_layers(self, vmp_asm: str) -> int:
        """Count distinct obfuscation layers"""
        layers = 0

        # VM layer
        if self.vm_pattern.search(vmp_asm):
            layers += 1

        # Control flow obfuscation
        if re.search(r'(opaque|flatten|dispatch)', vmp_asm, re.IGNORECASE):
            layers += 1

        # Data obfuscation
        if re.search(r'(encrypt|decrypt|xor.*key)', vmp_asm, re.IGNORECASE):
            layers += 1

        # Instruction obfuscation
        if re.search(r'(junk|dead.*code|nop\s+)', vmp_asm, re.IGNORECASE):
            layers += 1

        # Anti-analysis layer
        if re.search(r'(anti.*debug|anti.*tamper)', vmp_asm, re.IGNORECASE):
            layers += 1

        return layers

    def _aggregate_security_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate security statistics"""
        if not results:
            return {}

        # Group by category
        category_stats = defaultdict(lambda: defaultdict(list))

        for result in results:
            category = result['category']
            security = result['security']

            # Collect numeric scores
            for key, value in security.items():
                if isinstance(value, (int, float)):
                    category_stats[category][key].append(value)

        # Calculate statistics
        aggregated = {}

        for category, metrics in category_stats.items():
            category_agg = {}

            for metric, values in metrics.items():
                if values:
                    import numpy as np
                    category_agg[metric] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

            aggregated[category] = category_agg

        return aggregated

    def _identify_security_patterns(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify common security patterns and trends"""
        patterns = {
            'common_mechanisms': defaultdict(int),
            'common_vulnerabilities': defaultdict(int),
            'protection_combinations': defaultdict(int),
            'category_trends': defaultdict(dict)
        }

        for result in results:
            security = result['security']
            category = result['category']

            # Count mechanisms
            for mechanism in security['security_mechanisms']:
                patterns['common_mechanisms'][mechanism] += 1

            # Count vulnerabilities
            for vuln in security['vulnerabilities']:
                patterns['common_vulnerabilities'][vuln] += 1

            # Analyze protection combinations
            if len(security['security_mechanisms']) > 1:
                combo = tuple(sorted(security['security_mechanisms'][:2]))
                patterns['protection_combinations'][str(combo)] += 1

            # Category-specific trends
            if category not in patterns['category_trends']:
                patterns['category_trends'][category] = {
                    'avg_obfuscation_layers': [],
                    'avg_vm_complexity': [],
                    'has_vulnerabilities': 0,
                    'total': 0
                }

            patterns['category_trends'][category]['avg_obfuscation_layers'].append(
                security['obfuscation_layers']
            )
            patterns['category_trends'][category]['avg_vm_complexity'].append(
                security['vm_complexity_score']
            )
            if security['vulnerabilities']:
                patterns['category_trends'][category]['has_vulnerabilities'] += 1
            patterns['category_trends'][category]['total'] += 1

        # Calculate averages for trends
        for category, trend_data in patterns['category_trends'].items():
            if trend_data['avg_obfuscation_layers']:
                import numpy as np
                trend_data['avg_obfuscation_layers'] = np.mean(trend_data['avg_obfuscation_layers'])
                trend_data['avg_vm_complexity'] = np.mean(trend_data['avg_vm_complexity'])
                trend_data['vulnerability_rate'] = trend_data['has_vulnerabilities'] / trend_data['total']

        return patterns