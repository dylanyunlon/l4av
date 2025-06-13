# src/core/memory.py

class VMMemory:
    """Virtual machine memory management for code and data segments"""
    
    def __init__(self, seg_size=5000):
        self.seg_size = seg_size
        self.code_seg = bytearray(seg_size)
        self.data_seg = bytearray(seg_size)
        self.code_seg_addr = 0
        self.data_seg_addr = 0
    
    def load_code_segment(self, code_bytes):
        """Load bytecode into code segment"""
        if len(code_bytes) > self.seg_size:
            raise ValueError(f"Code size {len(code_bytes)} exceeds segment size {self.seg_size}")
        
        self.code_seg[:len(code_bytes)] = code_bytes
    
    def load_data_segment(self, data_bytes):
        """Load data into data segment"""
        if len(data_bytes) > self.seg_size:
            raise ValueError(f"Data size {len(data_bytes)} exceeds segment size {self.seg_size}")
        
        self.data_seg[:len(data_bytes)] = data_bytes
    
    def read_code(self, offset, size):
        """Read bytes from code segment"""
        if offset + size > self.seg_size:
            raise IndexError("Code segment read out of bounds")
        return self.code_seg[offset:offset + size]
    
    def read_data(self, offset, size):
        """Read bytes from data segment"""
        if offset + size > self.seg_size:
            raise IndexError("Data segment read out of bounds")
        return self.data_seg[offset:offset + size]
    
    def write_data(self, offset, data_bytes):
        """Write bytes to data segment"""
        size = len(data_bytes)
        if offset + size > self.seg_size:
            raise IndexError("Data segment write out of bounds")
        self.data_seg[offset:offset + size] = data_bytes
    
    def unpack_value(self, offset, size, from_code=False):
        """Unpack integer value from memory"""
        segment = self.code_seg if from_code else self.data_seg
        result = 0
        
        for i in range(size):
            if offset + i >= self.seg_size:
                raise IndexError("Memory read out of bounds")
            result |= segment[offset + i] << (8 * i)
        
        return result
    
    def pack_value(self, offset, value, size):
        """Pack integer value into data segment"""
        for i in range(size):
            if offset + i >= self.seg_size:
                raise IndexError("Memory write out of bounds")
            self.data_seg[offset + i] = (value >> (8 * i)) & 0xFF
    
    def clear_data_segment(self, start_offset=0):
        """Clear data segment from given offset"""
        for i in range(start_offset, self.seg_size):
            self.data_seg[i] = 0