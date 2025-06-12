# asm_transformer/parser.py
import re
from typing import List, Tuple, Dict, Any
from ..vmp_interpreter.opcodes import Opcode, BinaryOp, CompareOp

class InstructionParser:
    """汇编指令解析器"""
    
    def __init__(self):
        # 指令映射表
        self.instruction_map = {
            # 内存操作
            "mov": self._handle_mov,
            "movl": self._handle_mov,
            "movq": self._handle_mov,
            "lea": self._handle_lea,
            "push": self._handle_push,
            "pop": self._handle_pop,
            
            # 算术运算
            "add": self._handle_arithmetic,
            "sub": self._handle_arithmetic,
            "mul": self._handle_arithmetic,
            "imul": self._handle_arithmetic,
            "div": self._handle_arithmetic,
            "idiv": self._handle_arithmetic,
            
            # 逻辑运算
            "and": self._handle_logic,
            "or": self._handle_logic,
            "xor": self._handle_logic,
            "shl": self._handle_shift,
            "shr": self._handle_shift,
            "sar": self._handle_shift,
            
            # 比较和跳转
            "cmp": self._handle_cmp,
            "test": self._handle_test,
            "jmp": self._handle_jmp,
            "je": self._handle_conditional_jmp,
            "jne": self._handle_conditional_jmp,
            "jg": self._handle_conditional_jmp,
            "jge": self._handle_conditional_jmp,
            "jl": self._handle_conditional_jmp,
            "jle": self._handle_conditional_jmp,
            
            # 函数调用
            "call": self._handle_call,
            "ret": self._handle_ret,
        }
        
        # 寄存器映射
        self.register_map = {
            # 64位寄存器
            "rax": 0, "rbx": 1, "rcx": 2, "rdx": 3,
            "rsi": 4, "rdi": 5, "rbp": 6, "rsp": 7,
            "r8": 8, "r9": 9, "r10": 10, "r11": 11,
            "r12": 12, "r13": 13, "r14": 14, "r15": 15,
            
            # 32位寄存器
            "eax": 0, "ebx": 1, "ecx": 2, "edx": 3,
            "esi": 4, "edi": 5, "ebp": 6, "esp": 7,
            
            # 16位寄存器
            "ax": 0, "bx": 1, "cx": 2, "dx": 3,
            
            # 8位寄存器
            "al": 0, "bl": 1, "cl": 2, "dl": 3,
        }
    
    def parse_line(self, line: str) -> Tuple[str, List[str]]:
        """解析一行汇编代码"""
        # 移除注释
        line = line.split(';')[0].strip()
        if not line:
            return None, []
        
        # 分离指令和操作数
        parts = line.split(None, 1)
        if not parts:
            return None, []
        
        instruction = parts[0].lower()
        operands = []
        
        if len(parts) > 1:
            # 解析操作数
            operands_str = parts[1]
            operands = [op.strip() for op in operands_str.split(',')]
        
        return instruction, operands
    
    def parse_operand(self, operand: str) -> Dict[str, Any]:
        """解析操作数"""
        operand = operand.strip()
        
        # 立即数
        if operand.startswith('$'):
            value = operand[1:]
            if value.startswith('0x'):
                return {"type": "immediate", "value": int(value, 16)}
            else:
                return {"type": "immediate", "value": int(value)}
        
        # 寄存器
        if operand.startswith('%'):
            reg_name = operand[1:]
            if reg_name in self.register_map:
                return {"type": "register", "reg": self.register_map[reg_name]}
        
        # 内存地址
        if '(' in operand and ')' in operand:
            # 解析形如 offset(%base,%index,scale) 的地址
            match = re.match(r'(-?\d*)\(([^,)]+)(?:,([^,)]+)(?:,(\d+))?)?\)', operand)
            if match:
                offset = int(match.group(1) or 0)
                base = match.group(2).strip('%')
                index = match.group(3).strip('%') if match.group(3) else None
                scale = int(match.group(4) or 1) if match.group(4) else 1
                
                result = {
                    "type": "memory",
                    "offset": offset,
                    "base": self.register_map.get(base, 0)
                }
                
                if index:
                    result["index"] = self.register_map.get(index, 0)
                    result["scale"] = scale
                
                return result
        
        # 标签或符号
        return {"type": "label", "name": operand}
    
    def convert_to_vmp(self, instruction: str, operands: List[str]) -> bytes:
        """将指令转换为VMP字节码"""
        handler = self.instruction_map.get(instruction)
        if handler:
            return handler(instruction, operands)
        else:
            # 未知指令，返回NOP
            return bytes([0])
    
    def _handle_mov(self, inst: str, operands: List[str]) -> bytes:
        """处理MOV指令"""
        if len(operands) != 2:
            return bytes()
        
        src = self.parse_operand(operands[0])
        dst = self.parse_operand(operands[1])
        
        # 转换为VMP的LOAD/STORE操作
        bytecode = bytearray()
        
        if dst["type"] == "register":
            # 存储到寄存器
            bytecode.append(Opcode.LOAD_OP)
            # 添加操作数信息...
        
        return bytes(bytecode)
    
    def _handle_arithmetic(self, inst: str, operands: List[str]) -> bytes:
        """处理算术运算指令"""
        bytecode = bytearray()
        bytecode.append(Opcode.BINARY_OPERATOR_OP)
        
        # 映射指令到二元操作码
        op_map = {
            "add": BinaryOp.ADD,
            "sub": BinaryOp.SUB,
            "mul": BinaryOp.MUL,
            "imul": BinaryOp.MUL,
            "div": BinaryOp.UDIV,
            "idiv": BinaryOp.SDIV,
        }
        
        if inst in op_map:
            bytecode.append(op_map[inst])
        
        return bytes(bytecode)
    
    def _handle_cmp(self, inst: str, operands: List[str]) -> bytes:
        """处理CMP指令"""
        bytecode = bytearray()
        bytecode.append(Opcode.CMP_OP)
        # 添加比较操作的细节...
        return bytes(bytecode)
    
    def _handle_jmp(self, inst: str, operands: List[str]) -> bytes:
        """处理JMP指令"""
        bytecode = bytearray()
        bytecode.append(Opcode.BR_OP)
        bytecode.append(0)  # 无条件跳转
        return bytes(bytecode)
    
    def _handle_ret(self, inst: str, operands: List[str]) -> bytes:
        """处理RET指令"""
        return bytes([Opcode.RET_OP, 0, 0])
    
    # 其他处理函数的占位符
    def _handle_lea(self, inst: str, operands: List[str]) -> bytes:
        return bytes()
    
    def _handle_push(self, inst: str, operands: List[str]) -> bytes:
        return bytes()
    
    def _handle_pop(self, inst: str, operands: List[str]) -> bytes:
        return bytes()
    
    def _handle_logic(self, inst: str, operands: List[str]) -> bytes:
        return bytes()
    
    def _handle_shift(self, inst: str, operands: List[str]) -> bytes:
        return bytes()
    
    def _handle_test(self, inst: str, operands: List[str]) -> bytes:
        return bytes()
    
    def _handle_conditional_jmp(self, inst: str, operands: List[str]) -> bytes:
        return bytes()
    
    def _handle_call(self, inst: str, operands: List[str]) -> bytes:
        return bytes()

# asm_transformer/transformer.py
from typing import List, Dict, Any
import struct
from ..vmp_interpreter.utils import XorShift32

class ASMToVMPTransformer:
    """汇编到VMP转换器"""
    
    def __init__(self, protection_level: str = "basic", encryption_seed: int = 0x12345678):
        self.protection_level = protection_level
        self.encryption_seed = encryption_seed
        self.parser = InstructionParser()
        self.basic_block_id = 0
    
    def transform(self, asm_code: str) -> str:
        """将汇编代码转换为VMP保护的代码"""
        lines = asm_code.strip().split('\n')
        vmp_output = []
        
        # 添加VMP头部
        vmp_output.append("; VMP Protected Assembly")
        vmp_output.append(f"; Protection Level: {self.protection_level}")
        vmp_output.append(f"; Encryption Seed: {hex(self.encryption_seed)}")
        vmp_output.append("")
        
        # 生成数据段
        vmp_output.append("section .vmp_data")
        vmp_output.append("    vmp_code_seg db 5000 dup(0)")
        vmp_output.append("    vmp_data_seg db 5000 dup(0)")
        vmp_output.append("")
        
        # 生成代码段
        vmp_output.append("section .text")
        vmp_output.append("global _vmp_start")
        vmp_output.append("_vmp_start:")
        
        # 转换每条指令
        bytecode_buffer = bytearray()
        labels = {}  # 标签映射
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            # 处理标签
            if line.endswith(':'):
                label_name = line[:-1]
                labels[label_name] = len(bytecode_buffer)
                vmp_output.append(f"    ; Label: {label_name} at offset {len(bytecode_buffer)}")
                continue
            
            # 解析指令
            instruction, operands = self.parser.parse_line(line)
            if instruction:
                # 转换为VMP字节码
                vmp_bytecode = self._convert_instruction(instruction, operands)
                bytecode_buffer.extend(vmp_bytecode)
                
                # 添加注释
                vmp_output.append(f"    ; {line}")
                vmp_output.append(f"    ; VMP bytecode: {vmp_bytecode.hex()}")
        
        # 生成VMP加载器
        vmp_output.append("")
        vmp_output.append("    ; Initialize VMP interpreter")
        vmp_output.append("    mov rdi, vmp_code_seg")
        vmp_output.append("    mov rsi, vmp_data_seg")
        vmp_output.append(f"    mov rdx, {len(bytecode_buffer)}")
        vmp_output.append("    call _vmp_interpreter_init")
        vmp_output.append("")
        vmp_output.append("    ; Execute VMP code")
        vmp_output.append("    call _vmp_interpreter_execute")
        vmp_output.append("")
        vmp_output.append("    ; Cleanup and return")
        vmp_output.append("    xor eax, eax")
        vmp_output.append("    ret")
        
        # 添加VMP解释器存根
        vmp_output.extend(self._generate_interpreter_stub())
        
        # 嵌入字节码
        vmp_output.append("")
        vmp_output.append("; Embedded VMP bytecode")
        vmp_output.append("section .vmp_bytecode")
        vmp_output.append(f"    vmp_bytecode_data db {self._format_bytecode(bytecode_buffer)}")
        
        return '\n'.join(vmp_output)
    
    def _convert_instruction(self, instruction: str, operands: List[str]) -> bytes:
        """转换单条指令为VMP字节码"""
        # 基本块开始，添加加密种子
        if self._is_basic_block_start(instruction):
            bytecode = bytearray()
            
            # 添加操作码加密种子
            opcode_seed = self.encryption_seed ^ self.basic_block_id
            bytecode.extend(struct.pack('<I', opcode_seed))
            
            # 添加代码加密种子
            code_seed = (self.encryption_seed << 16) ^ self.basic_block_id
            bytecode.extend(struct.pack('<I', code_seed))
            
            self.basic_block_id += 1
            
            # 添加实际指令
            inst_bytecode = self.parser.convert_to_vmp(instruction, operands)
            
            # 根据保护级别加密
            if self.protection_level in ["advanced", "maximum"]:
                inst_bytecode = self._encrypt_bytecode(inst_bytecode, code_seed)
            
            bytecode.extend(inst_bytecode)
            return bytes(bytecode)
        else:
            # 普通指令
            return self.parser.convert_to_vmp(instruction, operands)
    
    def _is_basic_block_start(self, instruction: str) -> bool:
        """判断是否是基本块开始"""
        # 跳转目标、函数入口等是基本块开始
        return instruction in ["jmp", "je", "jne", "call", "ret"] or self.basic_block_id == 0
    
    def _encrypt_bytecode(self, bytecode: bytes, seed: int) -> bytes:
        """加密字节码"""
        xorshift = XorShift32(seed)
        encrypted = bytearray()
        
        for byte in bytecode:
            encrypted.append(byte ^ (xorshift.next() & 0xFF))
        
        return bytes(encrypted)
    
    def _format_bytecode(self, bytecode: bytearray) -> str:
        """格式化字节码为汇编数据"""
        hex_bytes = [f"0x{b:02x}" for b in bytecode]
        
        # 每行16个字节
        lines = []
        for i in range(0, len(hex_bytes), 16):
            line_bytes = hex_bytes[i:i+16]
            lines.append(", ".join(line_bytes))
        
        return ", \\\n        ".join(lines)
    
    def _generate_interpreter_stub(self) -> List[str]:
        """生成VMP解释器存根"""
        stub = [
            "",
            "; VMP Interpreter Stub",
            "_vmp_interpreter_init:",
            "    ; Save registers",
            "    push rbp",
            "    mov rbp, rsp",
            "    push rbx",
            "    push r12",
            "    push r13",
            "    push r14",
            "    push r15",
            "",
            "    ; Initialize interpreter state",
            "    ; rdi = code segment",
            "    ; rsi = data segment", 
            "    ; rdx = code size",
            "    mov [vmp_code_ptr], rdi",
            "    mov [vmp_data_ptr], rsi",
            "    mov [vmp_code_size], rdx",
            "    xor eax, eax",
            "    mov [vmp_ip], eax",
            "",
            "    ; Restore registers",
            "    pop r15",
            "    pop r14",
            "    pop r13",
            "    pop r12",
            "    pop rbx",
            "    pop rbp",
            "    ret",
            "",
            "_vmp_interpreter_execute:",
            "    ; Main interpreter loop",
            "    push rbp",
            "    mov rbp, rsp",
            "",
            "    ; TODO: Implement interpreter loop",
            "    ; This is a stub - actual implementation would be much larger",
            "",
            "    pop rbp",
            "    ret",
            "",
            "; VMP Interpreter Data",
            "section .bss",
            "    vmp_code_ptr resq 1",
            "    vmp_data_ptr resq 1",
            "    vmp_code_size resq 1",
            "    vmp_ip resd 1",
        ]
        
        return stub

# asm_transformer/__init__.py
from .parser import InstructionParser
from .transformer import ASMToVMPTransformer

__all__ = ['InstructionParser', 'ASMToVMPTransformer']
