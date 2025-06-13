# src/resources/instruction_set.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from ..core.opcodes import *

@dataclass
class VMPInstruction:
    """VMP instruction definition"""
    opcode: int
    name: str
    operand_count: int
    operand_types: List[str]
    handler: Optional[Callable] = None
    description: str = ""

class InstructionSet:
    """VMP instruction set resource"""
    
    def __init__(self):
        self.instructions = self._init_instructions()
        self.opcode_map = {inst.opcode: inst for inst in self.instructions}
        self.name_map = {inst.name: inst for inst in self.instructions}
    
    def _init_instructions(self) -> List[VMPInstruction]:
        """Initialize VMP instruction set"""
        return [
            VMPInstruction(
                opcode=ALLOCA_OP,
                name="ALLOCA",
                operand_count=2,
                operand_types=["variable", "offset"],
                description="Allocate space on stack"
            ),
            VMPInstruction(
                opcode=LOAD_OP,
                name="LOAD",
                operand_count=2,
                operand_types=["variable", "pointer"],
                description="Load value from memory"
            ),
            VMPInstruction(
                opcode=STORE_OP,
                name="STORE",
                operand_count=2,
                operand_types=["value", "pointer"],
                description="Store value to memory"
            ),
            VMPInstruction(
                opcode=BinaryOperator_OP,
                name="BINOP",
                operand_count=4,
                operand_types=["op_code", "result", "operand1", "operand2"],
                description="Binary arithmetic/logic operation"
            ),
            VMPInstruction(
                opcode=GEP_OP,
                name="GEP",
                operand_count=3,
                operand_types=["result", "pointer", "index"],
                description="Get element pointer"
            ),
            VMPInstruction(
                opcode=CMP_OP,
                name="CMP",
                operand_count=4,
                operand_types=["predicate", "result", "operand1", "operand2"],
                description="Compare values"
            ),
            VMPInstruction(
                opcode=CAST_OP,
                name="CAST",
                operand_count=2,
                operand_types=["result", "value"],
                description="Type cast"
            ),
            VMPInstruction(
                opcode=BR_OP,
                name="BR",
                operand_count=-1,  # Variable: 1 or 3
                operand_types=["target"] + ["condition", "true_target", "false_target"],
                description="Branch (conditional or unconditional)"
            ),
            VMPInstruction(
                opcode=Call_OP,
                name="CALL",
                operand_count=1,
                operand_types=["function_id"],
                description="Function call"
            ),
            VMPInstruction(
                opcode=Ret_OP,
                name="RET",
                operand_count=-1,  # Variable: 0 or 1
                operand_types=["value"],
                description="Return from function"
            ),
        ]
    
    def get_by_opcode(self, opcode: int) -> Optional[VMPInstruction]:
        """Get instruction by opcode"""
        return self.opcode_map.get(opcode)
    
    def get_by_name(self, name: str) -> Optional[VMPInstruction]:
        """Get instruction by name"""
        return self.name_map.get(name.upper())
    
    def get_binary_ops(self) -> Dict[str, int]:
        """Get binary operation mappings"""
        return {
            "ADD": BINOP_ADD,
            "SUB": BINOP_SUB,
            "MUL": BINOP_MUL,
            "UDIV": BINOP_UDIV,
            "SDIV": BINOP_SDIV,
            "UREM": BINOP_UREM,
            "SREM": BINOP_SREM,
            "SHL": BINOP_SHL,
            "LSHR": BINOP_LSHR,
            "ASHR": BINOP_ASHR,
            "AND": BINOP_AND,
            "OR": BINOP_OR,
            "XOR": BINOP_XOR
        }
    
    def get_comparison_ops(self) -> Dict[str, int]:
        """Get comparison operation mappings"""
        return {
            "EQ": ICMP_EQ,
            "NE": ICMP_NE,
            "UGT": ICMP_UGT,
            "UGE": ICMP_UGE,
            "ULT": ICMP_ULT,
            "ULE": ICMP_ULE,
            "SGT": ICMP_SGT,
            "SGE": ICMP_SGE,
            "SLT": ICMP_SLT,
            "SLE": ICMP_SLE
        }
    
    def validate_instruction(self, opcode: int, operands: List) -> bool:
        """Validate instruction format"""
        inst = self.get_by_opcode(opcode)
        if not inst:
            return False
        
        if inst.operand_count >= 0 and len(operands) != inst.operand_count:
            return False
        
        return True
    
    def get_instruction_size(self, opcode: int, operands: List) -> int:
        """Calculate instruction size in bytes"""
        # Base: opcode (1) + operand metadata
        size = 1
        
        for operand in operands:
            if isinstance(operand, dict):
                # Variable reference: size(1) + type(1) + offset(8)
                size += 10
            else:
                # Constant: size(1) + type(1) + value(varies)
                size += 2 + self._get_value_size(operand)
        
        return size
    
    def _get_value_size(self, value) -> int:
        """Get size needed to store value"""
        if isinstance(value, int):
            if -128 <= value <= 127:
                return 1
            elif -32768 <= value <= 32767:
                return 2
            elif -2147483648 <= value <= 2147483647:
                return 4
            else:
                return 8
        return 8  # Default to 8 bytes


# Singleton instance
instruction_set = InstructionSet()