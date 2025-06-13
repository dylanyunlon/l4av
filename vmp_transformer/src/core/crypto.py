# src/core/crypto.py

class XorShift32:
    """XorShift32 pseudo-random number generator for opcode encryption"""
    
    def __init__(self, seed=0):
        self.state = seed if seed != 0 else 1  # State must be non-zero
    
    def next(self):
        """Generate next pseudo-random number"""
        x = self.state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self.state = x & 0xFFFFFFFF
        return self.state
    
    def reset(self, seed):
        """Reset generator with new seed"""
        self.state = seed if seed != 0 else 1


class VMCodeEncryptor:
    """Handles encryption/decryption of VM bytecode"""
    
    def __init__(self):
        self.opcode_generator = XorShift32()
        self.code_generator = XorShift32()
    
    def encrypt_bytecode(self, bytecode, opcode_seed, code_seed):
        """Encrypt bytecode with given seeds"""
        self.opcode_generator.reset(opcode_seed)
        self.code_generator.reset(code_seed)
        
        encrypted = []
        for byte in bytecode:
            encrypted_byte = byte ^ (self.code_generator.next() & 0xFF)
            encrypted.append(encrypted_byte)
        
        return bytes(encrypted)
    
    def decrypt_byte(self, byte, generator):
        """Decrypt a single byte"""
        return byte ^ (generator.next() & 0xFF)
    
    def generate_opcode_mapping(self, seed):
        """Generate opcode encryption mapping"""
        self.opcode_generator.reset(seed)
        seen = set()
        mapping = {}
        
        for opcode in range(1, 11):  # OP_TOTAL + 1
            while True:
                encrypted = self.opcode_generator.next() & 0xFF
                if encrypted not in seen:
                    seen.add(encrypted)
                    mapping[opcode] = encrypted
                    break
        
        return mapping