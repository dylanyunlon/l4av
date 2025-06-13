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

OP_TOTAL = 0x0A

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
    Ret_OP: "RET"
}

BINOP_NAMES = {
    BINOP_ADD: "ADD",
    BINOP_SUB: "SUB",
    BINOP_MUL: "MUL",
    BINOP_UDIV: "UDIV",
    BINOP_SDIV: "SDIV",
    BINOP_UREM: "UREM",
    BINOP_SREM: "SREM",
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