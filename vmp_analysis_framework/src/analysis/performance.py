"""
Performance Analysis Module for VMP Transformations
"""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance-related metrics"""
    estimated_execution_overhead: float
    memory_overhead: float
    cache_impact_score: float
    branch_prediction_impact: float
    instruction_fetch_overhead: float
    register_pressure_score: float


class PerformanceAnalyzer:
    """Analyzes performance impact of VMP transformations"""

    # Instruction cycle estimates (simplified)
    INSTRUCTION_CYCLES = {
        # Basic arithmetic
        'add': 1, 'sub': 1, 'inc': 1, 'dec': 1,
        'mul': 3, 'imul': 3, 'div': 20, 'idiv': 20,

        # Logic operations
        'and': 1, 'or': 1, 'xor': 1, 'not': 1,
        'shl': 1, 'shr': 1, 'rol': 1, 'ror': 1,

        # Memory operations
        'mov': 1, 'movzx': 1, 'movsx': 1, 'lea': 1,
        'push': 2, 'pop': 2, 'call': 5, 'ret': 5,

        # Control flow
        'jmp': 1, 'je': 1, 'jne': 1, 'jz': 1, 'jnz': 1,
        'ja': 1, 'jb': 1, 'jg': 1, 'jl': 1,
        'loop': 2, 'cpuid': 100, 'rdtsc': 20,

        # VM operations (estimated)
        'vm_dispatch': 50, 'vm_handler': 30, 'vm_fetch': 10
    }

    def __init__(self):
        self.cache_line_size = 64  # bytes
        self.l1_cache_size = 32 * 1024  # 32KB
        self.l2_cache_size = 256 * 1024  # 256KB

    def analyze(self, data: List[Dict[str, Any]], metrics_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive performance analysis"""
        logger.info("Starting performance analysis")

        performance_results = []

        for entry, metrics in zip(data, metrics_results):
            if metrics is None:
                continue

            perf_metrics = self._analyze_performance(entry, metrics)
            performance_results.append({
                'function': entry['function'],
                'category': entry.get('function_category', 'unknown'),
                'performance': perf_metrics.__dict__,
                'bytecode_size': entry['bytecode_size']
            })

        # Aggregate statistics
        aggregated_stats = self._aggregate_performance_stats(performance_results)

        return {
            'individual_results': performance_results,
            'aggregated_stats': aggregated_stats
        }

    def _analyze_performance(self, entry: Dict[str, Any], metrics: Dict[str, Any]) -> PerformanceMetrics:
        """Analyze performance impact for a single transformation"""
        return PerformanceMetrics(
            estimated_execution_overhead=self._estimate_execution_overhead(entry, metrics),
            memory_overhead=self._calculate_memory_overhead(entry, metrics),
            cache_impact_score=self._calculate_cache_impact(entry, metrics),
            branch_prediction_impact=self._calculate_branch_prediction_impact(metrics),
            instruction_fetch_overhead=self._calculate_fetch_overhead(entry, metrics),
            register_pressure_score=self._calculate_register_pressure(metrics)
        )

    def _estimate_execution_overhead(self, entry: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """Estimate execution time overhead"""
        # Parse instructions from both versions
        orig_cycles = self._estimate_cycles(entry['original_assembly'])
        vmp_cycles = self._estimate_vmp_cycles(entry['vmp_assembly'], metrics)

        if orig_cycles == 0:
            return float('inf')

        return vmp_cycles / orig_cycles

    def _estimate_cycles(self, asm_text: str) -> int:
        """Estimate CPU cycles for assembly code"""
        cycles = 0

        # Simple pattern matching for instructions
        import re
        instruction_pattern = re.compile(r'^\s*([a-zA-Z]+)', re.MULTILINE)

        for match in instruction_pattern.finditer(asm_text):
            mnemonic = match.group(1).lower()
            cycles += self.INSTRUCTION_CYCLES.get(mnemonic, 2)  # Default 2 cycles

        return cycles

    def _estimate_vmp_cycles(self, vmp_asm: str, metrics: Dict[str, Any]) -> int:
        """Estimate cycles for VMP-protected code"""
        base_cycles = self._estimate_cycles(vmp_asm)

        # Add VM overhead
        vm_handlers = metrics['metrics'].get('vm_handler_count', 0)
        vm_overhead = vm_handlers * self.INSTRUCTION_CYCLES['vm_handler']

        # Add dispatch overhead based on bytecode size
        bytecode_size = metrics.get('bytecode_size', 0)
        dispatch_overhead = (bytecode_size // 10) * self.INSTRUCTION_CYCLES['vm_dispatch']

        # Add indirect jump overhead
        jump_density = metrics['metrics'].get('jump_density', 0)
        jump_overhead = int(base_cycles * jump_density * 0.2)  # 20% penalty for jumps

        return base_cycles + vm_overhead + dispatch_overhead + jump_overhead

    def _calculate_memory_overhead(self, entry: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """Calculate memory usage overhead"""
        original_size = len(entry['original_assembly'].encode('utf-8'))
        vmp_size = len(entry['vmp_assembly'].encode('utf-8')) + entry['bytecode_size']

        # Add VM state overhead (registers, stack, etc.)
        vm_state_overhead = 40 * 8  # 40 registers * 8 bytes
        vm_state_overhead += 1024  # Estimated VM stack

        total_vmp_size = vmp_size + vm_state_overhead

        if original_size == 0:
            return float('inf')

        return total_vmp_size / original_size

    def _calculate_cache_impact(self, entry: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """Calculate cache performance impact"""
        # Estimate working set size
        vmp_size = len(entry['vmp_assembly'].encode('utf-8')) + entry['bytecode_size']

        # Calculate cache lines needed
        cache_lines_needed = (vmp_size + self.cache_line_size - 1) // self.cache_line_size

        # Calculate cache pressure
        if vmp_size <= self.l1_cache_size:
            cache_pressure = 0.1  # Fits in L1
        elif vmp_size <= self.l2_cache_size:
            cache_pressure = 0.3  # Fits in L2
        else:
            cache_pressure = 0.7  # Spills to L3/memory

        # Factor in code locality (VM dispatch reduces locality)
        vm_handlers = metrics['metrics'].get('vm_handler_count', 0)
        locality_penalty = min(vm_handlers * 0.05, 0.3)

        return cache_pressure + locality_penalty

    def _calculate_branch_prediction_impact(self, metrics: Dict[str, Any]) -> float:
        """Calculate branch prediction performance impact"""
        # VMP typically uses indirect branches which are hard to predict
        jump_density = metrics['metrics'].get('jump_density', 0)
        vm_handlers = metrics['metrics'].get('vm_handler_count', 0)

        # Indirect branch penalty
        indirect_branch_penalty = min(vm_handlers * 0.1, 0.5)

        # High jump density reduces prediction accuracy
        jump_penalty = jump_density * 0.3

        return min(indirect_branch_penalty + jump_penalty, 1.0)

    def _calculate_fetch_overhead(self, entry: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """Calculate instruction fetch overhead"""
        # VMP increases code size and reduces fetch efficiency
        code_expansion = metrics['metrics'].get('code_expansion_rate', 1.0)

        # VM dispatch adds fetch overhead
        vm_overhead = metrics['metrics'].get('vm_handler_count', 0) * 0.05

        return (code_expansion - 1.0) * 0.2 + vm_overhead

    def _calculate_register_pressure(self, metrics: Dict[str, Any]) -> float:
        """Calculate register pressure score"""
        reg_pattern = metrics['metrics'].get('register_usage_pattern', {})

        # VMP typically uses more registers
        orig_regs = reg_pattern.get('original_unique_registers', 1)
        vmp_regs = reg_pattern.get('vmp_unique_registers', 1)

        if orig_regs == 0:
            return 1.0

        pressure_increase = (vmp_regs - orig_regs) / orig_regs

        # High register usage can cause spills
        spill_risk = min(vmp_regs / 16.0, 1.0)  # 16 general purpose registers

        return (pressure_increase + spill_risk) / 2

    def _aggregate_performance_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate performance statistics across all samples"""
        if not results:
            return {}

        # Group by category
        category_stats = defaultdict(list)

        for result in results:
            category = result['category']
            perf = result['performance']

            for metric, value in perf.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    category_stats[category].append({metric: value})

        # Calculate statistics per category
        aggregated = {}

        for category, measurements in category_stats.items():
            category_aggregated = {}

            # Combine all metrics
            all_metrics = defaultdict(list)
            for m in measurements:
                for metric, value in m.items():
                    all_metrics[metric].append(value)

            # Calculate stats for each metric
            for metric, values in all_metrics.items():
                if values:
                    category_aggregated[metric] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'p25': np.percentile(values, 25),
                        'p75': np.percentile(values, 75),
                        'p95': np.percentile(values, 95)
                    }

            aggregated[category] = category_aggregated

        # Overall statistics
        all_perf_values = defaultdict(list)

        for result in results:
            for metric, value in result['performance'].items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    all_perf_values[metric].append(value)

        overall_stats = {}
        for metric, values in all_perf_values.items():
            if values:
                overall_stats[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

        aggregated['overall'] = overall_stats

        return aggregated