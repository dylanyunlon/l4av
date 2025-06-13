# src/core/opcodes.py

# Opcode definitions
ALLOCA_OP = 0x01
LOAD_OP = 0x02
STORE_OP = 0x03
BinaryOperator_OP = 0x04
GEP_OP = 0x05
CMP_OP = 0x06
CAST_OP = 0x07
BR_OP = 0x08
Call_OP = 0x09
Ret_OP = 0x0A
NOP_OP = 0x0B      # No Operation
PUSH_OP = 0x0C     # Push to stack
POP_OP = 0x0D      # Pop from stack
LEA_OP = 0x0E      # Load Effective Address
Jmp_OP = 0x0F      # Unconditional jump
Jcc_OP = 0x10      # Conditional jump
COMPARE_OP = 0x11  # Compare operation
TEST_OP = 0x12     # Test operation
LOADADDR_OP = 0x13 # Load address (same as LEA but clearer)
XCHG_OP = 0x14     # Exchange values
INC_OP = 0x15      # Increment
DEC_OP = 0x16      # Decrement
NEG_OP = 0x17      # Negate
NOT_OP = 0x18      # Bitwise NOT
IMUL_OP = 0x19     # Signed multiply
IDIV_OP = 0x1A     # Signed divide
ROL_OP = 0x1B      # Rotate left
ROR_OP = 0x1C      # Rotate right
SYSCALL_OP = 0x1D  # System call
INT_OP = 0x1E      # Software interrupt
CPUID_OP = 0x1F    # CPU identification
LoadAddr_OP = 0x2F
OP_TOTAL = 0x1F
Compare_OP  = 0x2B
# Pointer size
POINTER_SIZE = 8

# Binary operator codes
BINOP_ADD = 12
BINOP_FADD = 13
BINOP_SUB = 14
BINOP_FSUB = 15
BINOP_MUL = 16
BINOP_FMUL = 17
BINOP_UDIV = 18
BINOP_SDIV = 19
BINOP_FDIV = 20
BINOP_UREM = 21
BINOP_SREM = 22
BINOP_FREM = 23
BINOP_SHL = 24
BINOP_LSHR = 25
BINOP_ASHR = 26
BINOP_AND = 27
BINOP_OR = 28
BINOP_XOR = 29

# Comparison predicates
ICMP_EQ = 32   # equal
ICMP_NE = 33   # not equal
ICMP_UGT = 34  # unsigned greater than
ICMP_UGE = 35  # unsigned greater or equal
ICMP_ULT = 36  # unsigned less than
ICMP_ULE = 37  # unsigned less or equal
ICMP_SGT = 38  # signed greater than
ICMP_SGE = 39  # signed greater or equal
ICMP_SLT = 40  # signed less than
ICMP_SLE = 41  # signed less or equal

# Jump condition codes (for Jcc_OP)
JCC_EQ = 0   # Jump if equal (ZF=1)
JCC_NE = 1   # Jump if not equal (ZF=0)
JCC_GT = 2   # Jump if greater than (ZF=0 and SF=OF)
JCC_LT = 3   # Jump if less than (SF≠OF)
JCC_GE = 4   # Jump if greater or equal (SF=OF)
JCC_LE = 5   # Jump if less or equal (ZF=1 or SF≠OF)
JCC_A = 6    # Jump if above (CF=0 and ZF=0)
JCC_B = 7    # Jump if below (CF=1)
JCC_AE = 8   # Jump if above or equal (CF=0)
JCC_BE = 9   # Jump if below or equal (CF=1 or ZF=1)
JCC_O = 10   # Jump if overflow (OF=1)
JCC_NO = 11  # Jump if no overflow (OF=0)
JCC_S = 12   # Jump if sign (SF=1)
JCC_NS = 13  # Jump if no sign (SF=0)
JCC_P = 14   # Jump if parity (PF=1)
JCC_NP = 15  # Jump if no parity (PF=0)

# Cast operation types
CAST_TRUNC = 0    # Truncate to smaller type
CAST_ZEXT = 1     # Zero extend
CAST_SEXT = 2     # Sign extend
CAST_FPTOUI = 3   # Float to unsigned int
CAST_FPTOSI = 4   # Float to signed int
CAST_UITOFP = 5   # Unsigned int to float
CAST_SITOFP = 6   # Signed int to float
CAST_FPTRUNC = 7  # Float truncate
CAST_FPEXT = 8    # Float extend
CAST_PTRTOINT = 9 # Pointer to int
CAST_INTTOPTR = 10 # Int to pointer
CAST_BITCAST = 11  # Bitwise cast

# Value types for operands
VALUE_TYPE_VARIABLE = 0
VALUE_TYPE_CONSTANT = 1
VALUE_TYPE_REGISTER = 2
VALUE_TYPE_STRING = 3
VALUE_TYPE_LABEL = 4
VALUE_TYPE_MEMORY = 5

# Special registers
REG_FLAGS = 40  # CPU flags register
REG_IP = 41     # Instruction pointer

# Opcode names for debugging/display
OPCODE_NAMES = {
    ALLOCA_OP: "ALLOCA",
    LOAD_OP: "LOAD",
    STORE_OP: "STORE",
    BinaryOperator_OP: "BINOP",
    GEP_OP: "GEP",
    CMP_OP: "CMP",
    CAST_OP: "CAST",
    BR_OP: "BR",
    Call_OP: "CALL",
    Ret_OP: "RET",
    NOP_OP: "NOP",
    PUSH_OP: "PUSH",
    POP_OP: "POP",
    LEA_OP: "LEA",
    Jmp_OP: "JMP",
    Jcc_OP: "JCC",
    COMPARE_OP: "COMPARE",
    TEST_OP: "TEST",
    LOADADDR_OP: "LOADADDR",
    XCHG_OP: "XCHG",
    INC_OP: "INC",
    DEC_OP: "DEC",
    NEG_OP: "NEG",
    NOT_OP: "NOT",
    IMUL_OP: "IMUL",
    IDIV_OP: "IDIV",
    ROL_OP: "ROL",
    ROR_OP: "ROR",
    SYSCALL_OP: "SYSCALL",
    INT_OP: "INT",
    CPUID_OP: "CPUID"
}

BINOP_NAMES = {
    BINOP_ADD: "ADD",
    BINOP_FADD: "FADD",
    BINOP_SUB: "SUB",
    BINOP_FSUB: "FSUB",
    BINOP_MUL: "MUL",
    BINOP_FMUL: "FMUL",
    BINOP_UDIV: "UDIV",
    BINOP_SDIV: "SDIV",
    BINOP_FDIV: "FDIV",
    BINOP_UREM: "UREM",
    BINOP_SREM: "SREM",
    BINOP_FREM: "FREM",
    BINOP_SHL: "SHL",
    BINOP_LSHR: "LSHR",
    BINOP_ASHR: "ASHR",
    BINOP_AND: "AND",
    BINOP_OR: "OR",
    BINOP_XOR: "XOR"
}

CMP_NAMES = {
    ICMP_EQ: "EQ",
    ICMP_NE: "NE",
    ICMP_UGT: "UGT",
    ICMP_UGE: "UGE",
    ICMP_ULT: "ULT",
    ICMP_ULE: "ULE",
    ICMP_SGT: "SGT",
    ICMP_SGE: "SGE",
    ICMP_SLT: "SLT",
    ICMP_SLE: "SLE"
}

JCC_NAMES = {
    JCC_EQ: "JE",
    JCC_NE: "JNE",
    JCC_GT: "JG",
    JCC_LT: "JL",
    JCC_GE: "JGE",
    JCC_LE: "JLE",
    JCC_A: "JA",
    JCC_B: "JB",
    JCC_AE: "JAE",
    JCC_BE: "JBE",
    JCC_O: "JO",
    JCC_NO: "JNO",
    JCC_S: "JS",
    JCC_NS: "JNS",
    JCC_P: "JP",
    JCC_NP: "JNP"
}

CAST_NAMES = {
    CAST_TRUNC: "TRUNC",
    CAST_ZEXT: "ZEXT",
    CAST_SEXT: "SEXT",
    CAST_FPTOUI: "FPTOUI",
    CAST_FPTOSI: "FPTOSI",
    CAST_UITOFP: "UITOFP",
    CAST_SITOFP: "SITOFP",
    CAST_FPTRUNC: "FPTRUNC",
    CAST_FPEXT: "FPEXT",
    CAST_PTRTOINT: "PTRTOINT",
    CAST_INTTOPTR: "INTTOPTR",
    CAST_BITCAST: "BITCAST"
}

# Helper functions
def get_opcode_name(opcode):
    """Get human-readable name for an opcode"""
    return OPCODE_NAMES.get(opcode, f"UNKNOWN_{opcode:02X}")

def get_binop_name(binop):
    """Get human-readable name for a binary operation"""
    return BINOP_NAMES.get(binop, f"UNKNOWN_BINOP_{binop}")

def get_cmp_name(cmp):
    """Get human-readable name for a comparison"""
    return CMP_NAMES.get(cmp, f"UNKNOWN_CMP_{cmp}")

def get_jcc_name(jcc):
    """Get human-readable name for a conditional jump"""
    return JCC_NAMES.get(jcc, f"UNKNOWN_JCC_{jcc}")

def get_cast_name(cast):
    """Get human-readable name for a cast operation"""
    return CAST_NAMES.get(cast, f"UNKNOWN_CAST_{cast}")