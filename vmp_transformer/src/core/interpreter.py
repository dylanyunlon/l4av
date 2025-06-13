# src/core/interpreter.py

from .opcodes import *
from .crypto import XorShift32, VMCodeEncryptor
from .memory import VMMemory

class VMPInterpreter:
    """Virtual Machine Protection Interpreter"""
    
    def __init__(self, seg_size=5000, debug=False):
        self.memory = VMMemory(seg_size)
        self.encryptor = VMCodeEncryptor()
        self.ip = 0  # Instruction pointer
        self.debug = debug
        
        # Encryption states
        self.opcode_xorshift_state = XorShift32()
        self.vm_code_state = XorShift32()
        
        # Execution trace for analysis
        self.execution_trace = []
    
    def load_program(self, code_bytes, data_bytes=None):
        """Load program into VM"""
        self.memory.load_code_segment(code_bytes)
        if data_bytes:
            self.memory.load_data_segment(data_bytes)
        self.ip = 0
    
    def get_byte_code(self):
        """Get next encrypted byte from code segment"""
        if self.ip >= self.memory.seg_size:
            raise IndexError("Code pointer out of bounds")
        
        byte = self.memory.code_seg[self.ip]
        self.ip += 1
        
        # Decrypt byte
        decrypted = byte ^ (self.vm_code_state.next() & 0xFF)
        return decrypted
    
    def get_xorshift_seed(self):
        """Get xorshift seed (unencrypted)"""
        result = 0
        for i in range(4):
            if self.ip >= self.memory.seg_size:
                raise IndexError("Code pointer out of bounds")
            result |= self.memory.code_seg[self.ip] << (8 * i)
            self.ip += 1
        return result
    
    def unpack_code(self, size):
        """Unpack value from code segment"""
        result = 0
        for i in range(size):
            result |= self.get_byte_code() << (8 * i)
        return result
    
    def get_opcode(self):
        """Get decrypted opcode"""
        seen = set()
        curr_byte = self.get_byte_code()
        
        for i in range(OP_TOTAL + 1):
            while True:
                tmp = self.opcode_xorshift_state.next() & 0xFF
                if tmp == curr_byte:
                    return i + 1
                
                if tmp not in seen:
                    seen.add(tmp)
                    break
        
        return 0xFF
    
    def get_value(self):
        """Get value (variable or constant)"""
        value_size = self.get_byte_code()
        value_type = self.get_byte_code()
        
        if value_type == 0:
            # Variable
            var_offset = self.unpack_code(POINTER_SIZE)
            return self.memory.unpack_value(var_offset, value_size)
        else:
            # Constant
            return self.unpack_code(value_size)
    
    def get_value_with_size(self, value_size, value_type):
        """Get value with known size and type"""
        if value_type == 0:
            var_offset = self.unpack_code(POINTER_SIZE)
            return self.memory.unpack_value(var_offset, value_size)
        else:
            return self.unpack_code(value_size)
    
    # Handler methods
    def alloca_handler(self):
        """Handle ALLOCA operation"""
        var_size = self.get_byte_code()
        var_type = self.get_byte_code()
        var_offset = self.unpack_code(POINTER_SIZE)
        area_offset = self.unpack_code(POINTER_SIZE)
        
        # Store area address to variable
        addr = self.memory.data_seg_addr + area_offset
        self.memory.pack_value(var_offset, addr, var_size)
        
        if self.debug:
            self.execution_trace.append(f"ALLOCA: var_offset={var_offset}, area_offset={area_offset}")
    
    def load_handler(self):
        """Handle LOAD operation"""
        var_size = self.get_byte_code()
        var_type = self.get_byte_code()
        var_offset = self.unpack_code(POINTER_SIZE)
        
        ptr_size = self.get_byte_code()
        ptr_type = self.get_byte_code()
        ptr_offset = self.unpack_code(POINTER_SIZE)
        
        # Load pointer value
        ptr = self.memory.unpack_value(ptr_offset, POINTER_SIZE)
        
        # Load value from pointer address
        # Note: In text transformation, we track this symbolically
        load_value = self.memory.unpack_value(ptr - self.memory.data_seg_addr, var_size)
        
        # Store to variable
        self.memory.pack_value(var_offset, load_value, var_size)
        
        if self.debug:
            self.execution_trace.append(f"LOAD: ptr={ptr}, value={load_value}, var_offset={var_offset}")
    
    def store_handler(self):
        """Handle STORE operation"""
        var_size = self.get_byte_code()
        var_type = self.get_byte_code()
        store_value = self.get_value_with_size(var_size, var_type)
        
        ptr_size = self.get_byte_code()
        ptr_type = self.get_byte_code()
        ptr_offset = self.unpack_code(POINTER_SIZE)
        
        ptr = self.memory.unpack_value(ptr_offset, POINTER_SIZE)
        
        # Store value to pointer address
        self.memory.pack_value(ptr - self.memory.data_seg_addr, store_value, var_size)
        
        if self.debug:
            self.execution_trace.append(f"STORE: ptr={ptr}, value={store_value}")
    
    def binary_operator_handler(self):
        """Handle binary operations"""
        op_code = self.get_byte_code()
        
        res_size = self.get_byte_code()
        res_type = self.get_byte_code()
        res_offset = self.unpack_code(POINTER_SIZE)
        
        op1_value = self.get_value()
        op2_value = self.get_value()
        
        # Perform operation
        if op_code == BINOP_ADD:
            res_value = op1_value + op2_value
        elif op_code == BINOP_SUB:
            res_value = op1_value - op2_value
        elif op_code == BINOP_MUL:
            res_value = op1_value * op2_value
        elif op_code == BINOP_UDIV or op_code == BINOP_SDIV:
            res_value = op1_value // op2_value if op2_value != 0 else 0
        elif op_code == BINOP_UREM or op_code == BINOP_SREM:
            res_value = op1_value % op2_value if op2_value != 0 else 0
        elif op_code == BINOP_SHL:
            res_value = op1_value << op2_value
        elif op_code == BINOP_LSHR or op_code == BINOP_ASHR:
            res_value = op1_value >> op2_value
        elif op_code == BINOP_AND:
            res_value = op1_value & op2_value
        elif op_code == BINOP_OR:
            res_value = op1_value | op2_value
        elif op_code == BINOP_XOR:
            res_value = op1_value ^ op2_value
        else:
            res_value = 0
        
        # Store result
        self.memory.pack_value(res_offset, res_value, res_size)
        
        if self.debug:
            op_name = BINOP_NAMES.get(op_code, f"OP_{op_code}")
            self.execution_trace.append(f"BINOP {op_name}: {op1_value} {op_name} {op2_value} = {res_value}")
    
    def gep_handler(self):
        """Handle GEP (GetElementPtr) operation"""
        gep_size = self.get_byte_code()
        gep_type = self.get_byte_code()
        
        res_size = self.get_byte_code()
        res_type = self.get_byte_code()
        res_offset = self.unpack_code(POINTER_SIZE)
        
        ptr_value = self.get_value()
        idx_value = self.get_value()
        
        if gep_size != 0 and gep_type != 0:
            # Array type
            res_value = ptr_value + gep_size * idx_value
        else:
            # Struct type
            res_value = ptr_value + gep_size
        
        self.memory.pack_value(res_offset, res_value, res_size)
        
        if self.debug:
            self.execution_trace.append(f"GEP: ptr={ptr_value}, idx={idx_value}, result={res_value}")
    
    def cmp_handler(self):
        """Handle comparison operations"""
        predicate = self.get_byte_code()
        
        res_size = self.get_byte_code()
        res_type = self.get_byte_code()
        res_offset = self.unpack_code(POINTER_SIZE)
        
        op1_value = self.get_value()
        op2_value = self.get_value()
        
        # Perform comparison
        if predicate == ICMP_EQ:
            res_value = 1 if op1_value == op2_value else 0
        elif predicate == ICMP_NE:
            res_value = 1 if op1_value != op2_value else 0
        elif predicate == ICMP_UGT or predicate == ICMP_SGT:
            res_value = 1 if op1_value > op2_value else 0
        elif predicate == ICMP_UGE or predicate == ICMP_SGE:
            res_value = 1 if op1_value >= op2_value else 0
        elif predicate == ICMP_ULT or predicate == ICMP_SLT:
            res_value = 1 if op1_value < op2_value else 0
        elif predicate == ICMP_ULE or predicate == ICMP_SLE:
            res_value = 1 if op1_value <= op2_value else 0
        else:
            res_value = 0
        
        self.memory.pack_value(res_offset, res_value, res_size)
        
        if self.debug:
            cmp_name = CMP_NAMES.get(predicate, f"CMP_{predicate}")
            self.execution_trace.append(f"CMP {cmp_name}: {op1_value} {cmp_name} {op2_value} = {res_value}")
    
    def cast_handler(self):
        """Handle CAST operation"""
        res_size = self.get_byte_code()
        res_type = self.get_byte_code()
        res_offset = self.unpack_code(POINTER_SIZE)
        
        op_value = self.get_value()
        
        # Simple cast - just store value with new size
        self.memory.pack_value(res_offset, op_value, res_size)
        
        if self.debug:
            self.execution_trace.append(f"CAST: value={op_value}, new_size={res_size}")
    
    def br_handler(self):
        """Handle branch operations"""
        br_type = self.get_byte_code()
        
        if br_type == 0:
            # Unconditional branch
            target_addr = self.unpack_code(POINTER_SIZE)
        else:
            # Conditional branch
            condition_value = self.get_value()
            true_br = self.unpack_code(POINTER_SIZE)
            false_br = self.unpack_code(POINTER_SIZE)
            
            target_addr = true_br if condition_value else false_br
        
        self.ip = target_addr
        
        if self.debug:
            self.execution_trace.append(f"BR: target={target_addr}")
        
        return True  # Indicates new basic block
    
    def return_handler(self):
        """Handle return operation"""
        var_size = self.get_byte_code()
        var_type = self.get_byte_code()
        
        if var_size != 0 or var_type != 0:
            ret_value = self.get_value_with_size(var_size, var_type)
            self.memory.pack_value(0, ret_value, var_size)
            
            if self.debug:
                self.execution_trace.append(f"RET: value={ret_value}")
        else:
            if self.debug:
                self.execution_trace.append("RET: void")
    
    def call_handler(self):
        """Handle function call"""
        target_func_id = self.unpack_code(POINTER_SIZE)
        
        if self.debug:
            self.execution_trace.append(f"CALL: func_id={target_func_id}")
    
    def execute(self):
        """Execute loaded program"""
        self.ip = 0
        is_new_bb = True
        
        while True:
            if is_new_bb:
                # New basic block - get encryption seeds
                self.opcode_xorshift_state.reset(self.get_xorshift_seed())
                self.vm_code_state.reset(self.get_xorshift_seed())
                is_new_bb = False
                
                if self.debug:
                    print(f"New BasicBlock at IP={self.ip}")
            
            # Get and execute opcode
            opcode = self.get_opcode()
            
            if self.debug:
                opcode_name = OPCODE_NAMES.get(opcode, f"UNKNOWN_{opcode}")
                print(f"IP={self.ip}, Opcode={opcode_name}")
            
            if opcode == ALLOCA_OP:
                self.alloca_handler()
            elif opcode == LOAD_OP:
                self.load_handler()
            elif opcode == STORE_OP:
                self.store_handler()
            elif opcode == BinaryOperator_OP:
                self.binary_operator_handler()
            elif opcode == GEP_OP:
                self.gep_handler()
            elif opcode == CMP_OP:
                self.cmp_handler()
            elif opcode == CAST_OP:
                self.cast_handler()
            elif opcode == BR_OP:
                is_new_bb = self.br_handler()
            elif opcode == Ret_OP:
                self.return_handler()
                break
            elif opcode == Call_OP:
                self.call_handler()
            else:
                if self.debug:
                    print(f"Unknown opcode: {opcode}")
                break
        
        return self.execution_trace