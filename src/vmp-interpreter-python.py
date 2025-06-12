# vmp_interpreter/opcodes.py
from enum import IntEnum

class Opcode(IntEnum):
    ALLOCA_OP = 0x01
    LOAD_OP = 0x02
    STORE_OP = 0x03
    BINARY_OPERATOR_OP = 0x04
    GEP_OP = 0x05
    CMP_OP = 0x06
    CAST_OP = 0x07
    BR_OP = 0x08
    CALL_OP = 0x09
    RET_OP = 0x0A

class BinaryOp(IntEnum):
    ADD = 12
    SUB = 14
    MUL = 16
    UDIV = 18
    SDIV = 19
    UREM = 21
    SREM = 22
    SHL = 24
    LSHR = 25
    ASHR = 26
    AND = 27
    OR = 28
    XOR = 29

class CompareOp(IntEnum):
    ICMP_EQ = 32
    ICMP_NE = 33
    ICMP_UGT = 34
    ICMP_UGE = 35
    ICMP_ULT = 36
    ICMP_ULE = 37
    ICMP_SGT = 38
    ICMP_SGE = 39
    ICMP_SLT = 40
    ICMP_SLE = 41

# vmp_interpreter/utils.py
import struct

class XorShift32:
    def __init__(self, seed):
        self.state = seed if seed != 0 else 1
    
    def next(self):
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state

def pack_value(value, size):
    """将值打包为字节数组"""
    formats = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    if size in formats:
        return struct.pack(f'<{formats[size]}', value & ((1 << (size * 8)) - 1))
    else:
        # 处理非标准大小
        result = []
        for i in range(size):
            result.append((value >> (i * 8)) & 0xFF)
        return bytes(result)

def unpack_value(data, offset, size):
    """从字节数组解包值"""
    formats = {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}
    if size in formats:
        return struct.unpack(f'<{formats[size]}', data[offset:offset+size])[0]
    else:
        # 处理非标准大小
        result = 0
        for i in range(size):
            result |= data[offset + i] << (i * 8)
        return result

# vmp_interpreter/interpreter.py
from .opcodes import Opcode, BinaryOp, CompareOp
from .utils import XorShift32, pack_value, unpack_value

class VMPInterpreter:
    def __init__(self, code_seg_size=5000, data_seg_size=5000):
        self.code_seg = bytearray(code_seg_size)
        self.data_seg = bytearray(data_seg_size)
        self.ip = 0  # Instruction pointer
        self.opcode_xorshift = None
        self.vm_code_xorshift = None
        self.pointer_size = 8
    
    def load_code(self, code_bytes):
        """加载代码段"""
        self.code_seg[:len(code_bytes)] = code_bytes
        self.ip = 0
    
    def load_data(self, data_bytes, offset=0):
        """加载数据段"""
        self.data_seg[offset:offset+len(data_bytes)] = data_bytes
    
    def get_byte_code(self):
        """获取加密的字节码"""
        if self.ip >= len(self.code_seg):
            raise IndexError("Code segment overflow")
        
        byte_val = self.code_seg[self.ip]
        self.ip += 1
        
        if self.vm_code_xorshift:
            byte_val ^= (self.vm_code_xorshift.next() & 0xFF)
        
        return byte_val
    
    def get_xorshift_seed(self):
        """获取xorshift种子（未加密）"""
        seed = 0
        for i in range(4):
            seed |= self.code_seg[self.ip] << (8 * i)
            self.ip += 1
        return seed
    
    def unpack_code(self, size):
        """从代码段解包数据"""
        result = 0
        for i in range(size):
            result |= self.get_byte_code() << (8 * i)
        return result
    
    def get_opcode(self):
        """获取操作码（使用xorshift加密）"""
        if not self.opcode_xorshift:
            return self.get_byte_code()
        
        curr_byte = self.get_byte_code()
        history = set()
        
        for i in range(len(Opcode)):
            tmp = self.opcode_xorshift.next() & 0xFF
            if tmp == curr_byte:
                return i + 1
            
            # 防止冲突
            while tmp in history:
                tmp = self.opcode_xorshift.next() & 0xFF
            history.add(tmp)
        
        return 0xFF
    
    def get_value(self):
        """获取值（变量或常量）"""
        value_size = self.get_byte_code()
        value_type = self.get_byte_code()
        
        if value_type == 0:  # 变量
            var_offset = self.unpack_code(self.pointer_size)
            return unpack_value(self.data_seg, var_offset, value_size)
        else:  # 常量
            return self.unpack_code(value_size)
    
    def handle_alloca(self):
        """处理ALLOCA指令"""
        var_size = self.get_byte_code()
        var_type = self.get_byte_code()
        var_offset = self.unpack_code(self.pointer_size)
        area_offset = self.unpack_code(self.pointer_size)
        
        # 存储区域地址到变量
        addr_bytes = pack_value(area_offset, var_size)
        self.data_seg[var_offset:var_offset+var_size] = addr_bytes
    
    def handle_load(self):
        """处理LOAD指令"""
        var_size = self.get_byte_code()
        var_type = self.get_byte_code()
        var_offset = self.unpack_code(self.pointer_size)
        
        ptr_size = self.get_byte_code()
        ptr_type = self.get_byte_code()
        ptr_offset = self.unpack_code(self.pointer_size)
        
        # 加载指针
        ptr = unpack_value(self.data_seg, ptr_offset, self.pointer_size)
        
        # 从指针加载值
        if ptr < len(self.data_seg):
            load_value = unpack_value(self.data_seg, ptr, var_size)
        else:
            load_value = 0
        
        # 存储到变量
        value_bytes = pack_value(load_value, var_size)
        self.data_seg[var_offset:var_offset+var_size] = value_bytes
    
    def handle_store(self):
        """处理STORE指令"""
        var_size = self.get_byte_code()
        var_type = self.get_byte_code()
        
        if var_type == 0:
            var_offset = self.unpack_code(self.pointer_size)
            store_value = unpack_value(self.data_seg, var_offset, var_size)
        else:
            store_value = self.unpack_code(var_size)
        
        ptr_size = self.get_byte_code()
        ptr_type = self.get_byte_code()
        ptr_offset = self.unpack_code(self.pointer_size)
        
        ptr = unpack_value(self.data_seg, ptr_offset, self.pointer_size)
        
        # 存储值到指针
        if ptr < len(self.data_seg):
            value_bytes = pack_value(store_value, var_size)
            self.data_seg[ptr:ptr+var_size] = value_bytes
    
    def handle_binary_op(self):
        """处理二元运算"""
        op_code = self.get_byte_code()
        
        res_size = self.get_byte_code()
        res_type = self.get_byte_code()
        res_offset = self.unpack_code(self.pointer_size)
        
        op1_value = self.get_value()
        op2_value = self.get_value()
        
        # 执行运算
        if op_code == BinaryOp.ADD:
            res_value = op1_value + op2_value
        elif op_code == BinaryOp.SUB:
            res_value = op1_value - op2_value
        elif op_code == BinaryOp.MUL:
            res_value = op1_value * op2_value
        elif op_code == BinaryOp.UDIV:
            res_value = op1_value // op2_value if op2_value != 0 else 0
        elif op_code == BinaryOp.UREM:
            res_value = op1_value % op2_value if op2_value != 0 else 0
        elif op_code == BinaryOp.SHL:
            res_value = op1_value << op2_value
        elif op_code == BinaryOp.LSHR:
            res_value = op1_value >> op2_value
        elif op_code == BinaryOp.AND:
            res_value = op1_value & op2_value
        elif op_code == BinaryOp.OR:
            res_value = op1_value | op2_value
        elif op_code == BinaryOp.XOR:
            res_value = op1_value ^ op2_value
        else:
            res_value = 0
        
        # 存储结果
        value_bytes = pack_value(res_value, res_size)
        self.data_seg[res_offset:res_offset+res_size] = value_bytes
    
    def handle_cmp(self):
        """处理比较指令"""
        predicate = self.get_byte_code()
        
        res_size = self.get_byte_code()
        res_type = self.get_byte_code()
        res_offset = self.unpack_code(self.pointer_size)
        
        op1_value = self.get_value()
        op2_value = self.get_value()
        
        # 执行比较
        if predicate == CompareOp.ICMP_EQ:
            res_value = 1 if op1_value == op2_value else 0
        elif predicate == CompareOp.ICMP_NE:
            res_value = 1 if op1_value != op2_value else 0
        elif predicate == CompareOp.ICMP_UGT:
            res_value = 1 if op1_value > op2_value else 0
        elif predicate == CompareOp.ICMP_UGE:
            res_value = 1 if op1_value >= op2_value else 0
        elif predicate == CompareOp.ICMP_ULT:
            res_value = 1 if op1_value < op2_value else 0
        elif predicate == CompareOp.ICMP_ULE:
            res_value = 1 if op1_value <= op2_value else 0
        else:
            res_value = 0
        
        # 存储结果
        value_bytes = pack_value(res_value, res_size)
        self.data_seg[res_offset:res_offset+res_size] = value_bytes
    
    def handle_br(self):
        """处理分支指令"""
        br_type = self.get_byte_code()
        
        if br_type == 0:  # 无条件分支
            target_addr = self.unpack_code(self.pointer_size)
        else:  # 条件分支
            condition_value = self.get_value()
            true_br = self.unpack_code(self.pointer_size)
            false_br = self.unpack_code(self.pointer_size)
            
            target_addr = true_br if condition_value else false_br
        
        self.ip = target_addr
    
    def handle_ret(self):
        """处理返回指令"""
        var_size = self.get_byte_code()
        var_type = self.get_byte_code()
        
        if var_size != 0 or var_type != 0:
            if var_type == 0:
                var_offset = self.unpack_code(self.pointer_size)
                ret_value = unpack_value(self.data_seg, var_offset, var_size)
            else:
                ret_value = self.unpack_code(var_size)
            
            # 存储返回值到数据段开始
            value_bytes = pack_value(ret_value, var_size)
            self.data_seg[0:var_size] = value_bytes
    
    def execute(self):
        """执行虚拟机"""
        self.ip = 0
        is_new_bb = True
        
        while True:
            if is_new_bb:
                # 新基本块，获取加密种子
                self.opcode_xorshift = XorShift32(self.get_xorshift_seed())
                self.vm_code_xorshift = XorShift32(self.get_xorshift_seed())
                is_new_bb = False
            
            opcode = self.get_opcode()
            
            if opcode == Opcode.ALLOCA_OP:
                self.handle_alloca()
            elif opcode == Opcode.LOAD_OP:
                self.handle_load()
            elif opcode == Opcode.STORE_OP:
                self.handle_store()
            elif opcode == Opcode.BINARY_OPERATOR_OP:
                self.handle_binary_op()
            elif opcode == Opcode.CMP_OP:
                self.handle_cmp()
            elif opcode == Opcode.BR_OP:
                self.handle_br()
                is_new_bb = True
            elif opcode == Opcode.RET_OP:
                self.handle_ret()
                break
            else:
                # 未知操作码
                break
        
        return self.data_seg
