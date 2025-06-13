# src/assembler/parser.py

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

@dataclass
class Instruction:
    """Represents a parsed assembly instruction"""
    mnemonic: str
    operands: List[str]
    line_number: int
    label: Optional[str] = None
    comment: Optional[str] = None

@dataclass
class ParsedAssembly:
    """Represents parsed assembly program"""
    instructions: List[Instruction]
    labels: Dict[str, int]
    data_section: Dict[str, Union[int, str, bytes]]
    
class AssemblyParser:
    """Parse assembly code into structured format"""
    
    def __init__(self):
        # Common x86/x64 instructions
        self.valid_mnemonics = {
            # Data transfer
            'mov', 'movzx', 'movsx', 'lea', 'push', 'pop',
            # Arithmetic
            'add', 'sub', 'mul', 'imul', 'div', 'idiv', 'inc', 'dec', 'neg',
            # Logic
            'and', 'or', 'xor', 'not', 'shl', 'shr', 'sal', 'sar', 'rol', 'ror',
            # Control flow
            'jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jl', 'jle', 'ja', 'jae', 'jb', 'jbe',
            'call', 'ret', 'loop',
            # Comparison
            'cmp', 'test',
            # Other
            'nop', 'int', 'syscall'
        }
        
        # Register names
        self.registers = {
            # 64-bit
            'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
            'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
            # 32-bit
            'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp',
            # 16-bit
            'ax', 'bx', 'cx', 'dx', 'si', 'di', 'bp', 'sp',
            # 8-bit
            'al', 'ah', 'bl', 'bh', 'cl', 'ch', 'dl', 'dh'
        }
    
    def parse_line(self, line: str, line_number: int) -> Optional[Instruction]:
        """Parse single line of assembly"""
        # Remove comments
        comment = None
        if ';' in line:
            line, comment = line.split(';', 1)
            comment = comment.strip()
        
        line = line.strip()
        if not line:
            return None
        
        # Check for label
        label = None
        if ':' in line:
            label, line = line.split(':', 1)
            label = label.strip()
            line = line.strip()
            
            if not line:
                # Label only line
                return Instruction(
                    mnemonic='label',
                    operands=[],
                    line_number=line_number,
                    label=label,
                    comment=comment
                )
        
        # Parse instruction
        tokens = re.split(r'[\s,]+', line)
        if not tokens:
            return None
        
        mnemonic = tokens[0].lower()
        operands = [op.strip() for op in tokens[1:] if op.strip()]
        
        return Instruction(
            mnemonic=mnemonic,
            operands=operands,
            line_number=line_number,
            label=label,
            comment=comment
        )
    
    def parse_operand(self, operand: str) -> dict:
        """Parse operand to determine its type"""
        operand = operand.strip()
        
        # Immediate value
        if operand.isdigit() or (operand.startswith('-') and operand[1:].isdigit()):
            return {'type': 'immediate', 'value': int(operand)}
        
        # Hex immediate
        if operand.startswith('0x') or operand.startswith('0X'):
            return {'type': 'immediate', 'value': int(operand, 16)}
        
        # Register
        if operand in self.registers:
            return {'type': 'register', 'name': operand}
        
        # Memory reference [...]
        if operand.startswith('[') and operand.endswith(']'):
            inner = operand[1:-1].strip()
            return {'type': 'memory', 'address': self.parse_address(inner)}
        
        # Label/symbol
        return {'type': 'label', 'name': operand}
    
    def parse_address(self, address_expr: str) -> dict:
        """Parse memory address expression"""
        # Simple cases
        if address_expr in self.registers:
            return {'base': address_expr}
        
        # More complex addressing modes would be parsed here
        # For now, return as string
        return {'expression': address_expr}
    
    def parse(self, assembly_code: str) -> ParsedAssembly:
        """Parse complete assembly program"""
        lines = assembly_code.strip().split('\n')
        instructions = []
        labels = {}
        data_section = {}
        
        current_section = 'text'
        instruction_index = 0
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Check for section directive
            if line.startswith('section') or line.startswith('.section'):
                if 'data' in line:
                    current_section = 'data'
                elif 'text' in line or 'code' in line:
                    current_section = 'text'
                continue
            
            if current_section == 'text':
                instruction = self.parse_line(line, line_num)
                if instruction:
                    if instruction.label:
                        labels[instruction.label] = instruction_index
                    if instruction.mnemonic != 'label':
                        instructions.append(instruction)
                        instruction_index += 1
            
            elif current_section == 'data':
                # Parse data declarations
                if line and not line.startswith(';'):
                    self.parse_data_declaration(line, data_section)
        
        return ParsedAssembly(
            instructions=instructions,
            labels=labels,
            data_section=data_section
        )
    
    def parse_data_declaration(self, line: str, data_section: dict):
        """Parse data section declarations"""
        # Simple parsing for common directives
        tokens = line.split()
        if len(tokens) >= 3:
            name = tokens[0]
            directive = tokens[1].lower()
            
            if directive in ['db', 'dw', 'dd', 'dq']:
                # Byte/word/dword/qword data
                values = ' '.join(tokens[2:]).split(',')
                data_section[name] = [self.parse_data_value(v.strip()) for v in values]
            elif directive == 'equ':
                # Constant
                data_section[name] = self.parse_data_value(tokens[2])
    
    def parse_data_value(self, value: str) -> Union[int, str]:
        """Parse data value"""
        value = value.strip()
        
        # String
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1]
        
        # Hex
        if value.startswith('0x') or value.startswith('0X'):
            return int(value, 16)
        
        # Decimal
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        
        # Character
        if value.startswith("'") and value.endswith("'"):
            return ord(value[1])
        
        return value