# src/assembler/converter.py

import random
from ..core.opcodes import *
from ..core.crypto import XorShift32

class AssemblyToVMPConverter:
    """Convert assembly code to VMP protected bytecode"""
    
    def __init__(self):
        self.bytecode = []
        self.data_segment = []
        self.labels = {}
        self.variables = {}
        self.current_offset = 0
        self.data_offset = 0
    
    def generate_seed(self):
        """Generate random seed for encryption"""
        return random.randint(1, 0xFFFFFFFF)
    
    def emit_seed(self, seed):
        """Emit 4-byte seed (unencrypted)"""
        for i in range(4):
            self.bytecode.append((seed >> (8 * i)) & 0xFF)
        self.current_offset += 4
    
    def emit_byte(self, byte):
        """Emit single byte"""
        self.bytecode.append(byte & 0xFF)
        self.current_offset += 1
    
    def emit_value(self, value, size):
        """Emit multi-byte value"""
        for i in range(size):
            self.bytecode.append((value >> (8 * i)) & 0xFF)
        self.current_offset += size
    
    def emit_variable_ref(self, var_name, size):
        """Emit variable reference"""
        if var_name not in self.variables:
            # Allocate new variable
            self.variables[var_name] = self.data_offset
            self.data_offset += size
        
        self.emit_byte(size)  # value_size
        self.emit_byte(0)     # value_type (0 = variable)
        self.emit_value(self.variables[var_name], POINTER_SIZE)
    
    def emit_constant(self, value, size):
        """Emit constant value"""
        self.emit_byte(size)  # value_size
        self.emit_byte(1)     # value_type (1 = constant)
        self.emit_value(value, size)
    
    def convert_assembly_line(self, line):
        """Convert single assembly instruction to VMP bytecode"""
        tokens = line.strip().split()
        if not tokens:
            return
        
        instruction = tokens[0].lower()
        
        # Handle labels
        if instruction.endswith(':'):
            label = instruction[:-1]
            self.labels[label] = self.current_offset
            return
        
        # Start new basic block
        opcode_seed = self.generate_seed()
        code_seed = self.generate_seed()
        self.emit_seed(opcode_seed)
        self.emit_seed(code_seed)
        
        # Encrypt opcode
        xorshift = XorShift32(opcode_seed)
        opcode_mapping = {}
        seen = set()
        
        for op in range(1, OP_TOTAL + 1):
            while True:
                encrypted = xorshift.next() & 0xFF
                if encrypted not in seen:
                    seen.add(encrypted)
                    opcode_mapping[op] = encrypted
                    break
        
        # Convert instruction
        if instruction == 'mov':
            # MOV destination, source -> LOAD/STORE
            dest = tokens[1].rstrip(',')
            src = tokens[2]
            
            if src.startswith('[') and src.endswith(']'):
                # LOAD from memory
                self.emit_byte(opcode_mapping[LOAD_OP])
                self.emit_variable_ref(dest, 8)  # destination
                ptr_name = src[1:-1]
                self.emit_variable_ref(ptr_name, 8)  # pointer
            else:
                # STORE immediate or register
                self.emit_byte(opcode_mapping[STORE_OP])
                if src.isdigit():
                    self.emit_constant(int(src), 8)
                else:
                    self.emit_variable_ref(src, 8)
                self.emit_variable_ref(dest, 8)  # destination
        
        elif instruction == 'add':
            # ADD dest, src1, src2
            self.emit_byte(opcode_mapping[BinaryOperator_OP])
            self.emit_byte(BINOP_ADD)
            
            dest = tokens[1].rstrip(',')
            self.emit_variable_ref(dest, 8)  # result
            
            # Operands
            if len(tokens) > 2:
                self.emit_variable_ref(tokens[2].rstrip(','), 8)
            if len(tokens) > 3:
                if tokens[3].isdigit():
                    self.emit_constant(int(tokens[3]), 8)
                else:
                    self.emit_variable_ref(tokens[3], 8)
        
        elif instruction == 'sub':
            # SUB dest, src1, src2
            self.emit_byte(opcode_mapping[BinaryOperator_OP])
            self.emit_byte(BINOP_SUB)
            
            dest = tokens[1].rstrip(',')
            self.emit_variable_ref(dest, 8)  # result
            
            # Operands
            if len(tokens) > 2:
                self.emit_variable_ref(tokens[2].rstrip(','), 8)
            if len(tokens) > 3:
                if tokens[3].isdigit():
                    self.emit_constant(int(tokens[3]), 8)
                else:
                    self.emit_variable_ref(tokens[3], 8)
        
        elif instruction == 'cmp':
            # CMP op1, op2
            self.emit_byte(opcode_mapping[CMP_OP])
            self.emit_byte(ICMP_EQ)  # Default to equality comparison
            
            # Result goes to flags register
            self.emit_variable_ref("_flags", 1)
            
            # Operands
            self.emit_variable_ref(tokens[1].rstrip(','), 8)
            if tokens[2].isdigit():
                self.emit_constant(int(tokens[2]), 8)
            else:
                self.emit_variable_ref(tokens[2], 8)
        
        elif instruction in ['jmp', 'je', 'jne', 'jg', 'jl']:
            # Branch instructions
            self.emit_byte(opcode_mapping[BR_OP])
            
            if instruction == 'jmp':
                # Unconditional branch
                self.emit_byte(0)
                label = tokens[1]
                # Placeholder for label address
                self.emit_value(0, POINTER_SIZE)  # Will be patched later
            else:
                # Conditional branch
                self.emit_byte(1)
                self.emit_variable_ref("_flags", 1)  # condition
                
                label = tokens[1]
                # True and false targets (placeholders)
                self.emit_value(0, POINTER_SIZE)  # true branch
                self.emit_value(0, POINTER_SIZE)  # false branch
        
        elif instruction == 'ret':
            # Return instruction
            self.emit_byte(opcode_mapping[Ret_OP])
            
            if len(tokens) > 1:
                # Return value
                if tokens[1].isdigit():
                    self.emit_byte(8)  # size
                    self.emit_byte(1)  # constant
                    self.emit_value(int(tokens[1]), 8)
                else:
                    self.emit_variable_ref(tokens[1], 8)
            else:
                # Void return
                self.emit_byte(0)
                self.emit_byte(0)
    
    def convert_assembly(self, assembly_code):
        """Convert complete assembly code to VMP bytecode"""
        self.bytecode = []
        self.data_segment = []
        self.labels = {}
        self.variables = {}
        self.current_offset = 0
        self.data_offset = 0
        
        # First pass: convert instructions and collect labels
        lines = assembly_code.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(';'):  # Skip comments
                self.convert_assembly_line(line)
        
        # Second pass: patch jump targets
        # (In a real implementation, we would need to track and patch label references)
        
        return bytes(self.bytecode), self.variables
    
    def generate_vmp_text(self, assembly_code):
        """Generate VMP protected assembly text representation"""
        bytecode, variables = self.convert_assembly(assembly_code)
        
        # Generate text representation
        vmp_text = "; VMP Protected Assembly\n"
        vmp_text += "; Generated by VMP Transformer\n\n"
        
        # Data section
        vmp_text += "section .data\n"
        vmp_text += f"    vmp_code_seg db {', '.join(str(b) for b in bytecode[:20])}"
        if len(bytecode) > 20:
            vmp_text += f", ... ; {len(bytecode)} bytes total\n"
        else:
            vmp_text += "\n"
        
        vmp_text += f"    vmp_data_seg times {self.data_offset} db 0\n\n"
        
        # Variable mapping
        vmp_text += "; Variable offsets:\n"
        for var, offset in variables.items():
            vmp_text += f";   {var}: offset {offset}\n"
        
        vmp_text += "\nsection .text\n"
        vmp_text += "    ; VMP interpreter entry point\n"
        vmp_text += "    call vmp_interpreter\n"
        vmp_text += "    ; Result in data_seg[0]\n"
        
        return vmp_text