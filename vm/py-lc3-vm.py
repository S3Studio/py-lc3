#encoding: utf-8

import os
import sys
import platform
from io import BytesIO

# 65536 locations
UINT16_MAX = 0xFFFF
g_memory = [0] * UINT16_MAX

INT16_MAX = 0x7FFF

R_R0 = 0
R_R1 = 1
R_R2 = 2
R_R3 = 3
R_R4 = 4
R_R5 = 5
R_R6 = 6
R_R7 = 7
R_PC = 8 # program counter
R_COND = 9
R_COUNT = 10

g_reg = [0] * R_COUNT

OP_BR = 0 # branch
OP_ADD = 1 # add
OP_LD = 2 # load
OP_ST = 3 # store
OP_JSR = 4 # jump register
OP_AND = 5 # bitwise and
OP_LDR = 6 # load register
OP_STR = 7 # store register
OP_RTI = 8 # unused
OP_NOT = 9 # bitwise not
OP_LDI = 10 # load indirect
OP_STI = 11 # store indirect
OP_JMP = 12 # jump
OP_RES = 13 # reserved (unused)
OP_LEA = 14 # load effective address
OP_TRAP = 15 # execute trap

FL_POS = 1 << 0 # P
FL_ZRO = 1 << 1 # Z
FL_NEG = 1 << 2 # N

g_running = True

class InvalidArgError(ValueError):
    def __init__(self, arg_name: str, fn_name: str, value) -> None:
        super().__init__(f'Invalid argument {arg_name} in {fn_name}: {str(value)}')

def sign_extend(x: int, bit_count: int) -> int:
    if x < 0:
        raise InvalidArgError('x', 'sign_extend', x)
    if bit_count < 0 or bit_count > 15:
        raise InvalidArgError('bit_count', 'sign_extend', bit_count)

    if (x >> (bit_count - 1)) != 0:
        return ((x | (UINT16_MAX << bit_count)) & UINT16_MAX)
    else:
        return x

def update_flags(r: int) -> None:
    global g_reg

    if r < 0 or r >= R_COUNT:
        raise InvalidArgError('r', 'update_flags', r)

    if g_reg[r] == 0:
        g_reg[R_COND] = FL_ZRO
    else:
        g_reg[R_COND] = FL_NEG if g_reg[r] > INT16_MAX else FL_POS

def swap16(x: int) -> int:
    if x < 0 or x > UINT16_MAX:
        raise InvalidArgError('x', 'swap16', x)

    return ((x << 8) | (x >> 8)) & UINT16_MAX

if platform.system().lower() == 'windows':
    from msvcrt import kbhit, getch

    def check_key_win() -> bool:
        return kbhit()

    check_key = check_key_win

    def get_key_win() -> int:
        return ord(getch()) & UINT16_MAX

    get_key = get_key_win

    class TattrMgrNull(object):
        def __enter__(self):
            return self

        def __exit__(self, type, value, trace):
            pass

    TattrMgr = TattrMgrNull

else:
    def check_key_nix() -> bool:
        return len(sys.stdin.buffer.peek(1)) > 0

    check_key = check_key_nix

    def get_key_nix() -> int:
        return ord(sys.stdin.read(1)) & UINT16_MAX

    get_key = get_key_nix

    import termios
    import tty

    class TattrMgrNix(object):
        def __enter__(self):
            self.tattr = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno(), termios.TCSANOW)
            return self.tattr

        def __exit__(self, type, value, trace):
            termios.tcsetattr(sys.stdin, termios.TCSANOW, self.tattr)

    TattrMgr = TattrMgrNix

MR_KBSR = 0xFE00 # keyboard status
MR_KBDR = 0xFE02 # keyboard data

def mem_write(address: int, val: int) -> None:
    global g_memory

    g_memory[address & UINT16_MAX] = val & UINT16_MAX

def mem_read(address: int) -> int:
    global g_memory

    address &= UINT16_MAX
    if address == MR_KBSR:
        if check_key():
            g_memory[MR_KBSR] = (INT16_MAX + 1)
            g_memory[MR_KBDR] = get_key()
        else:
            g_memory[MR_KBSR] = 0

    return g_memory[address] & UINT16_MAX

class OpBase(object):
    def __init__(self, instr: int) -> None:
        if instr < 0 or instr > UINT16_MAX:
            raise InvalidArgError('instr', type(self), instr)

        self.instr = instr

    def do_op(self) -> None:
        raise NotImplementedError('function do_op not implemented')

class OpAdd(OpBase):
    def do_op(self) -> None:
        global g_reg

        r0 = (self.instr >> 9) & 0x7 # destination register (DR)
        r1 = (self.instr >> 6) & 0x7 # first operand (SR1)
        imm_flag = (self.instr >> 5) & 0x1 # whether we are in immediate mode

        if imm_flag != 0:
            imm5 = sign_extend(self.instr & 0x1F, 5)
            g_reg[r0] = g_reg[r1] + imm5
        else:
            r2 = self.instr & 0x7
            g_reg[r0] = g_reg[r1] + g_reg[r2]
        g_reg[r0] &= UINT16_MAX

        update_flags(r0)

class OpAnd(OpBase):
    def do_op(self) -> None:
        global g_reg

        r0 = (self.instr >> 9) & 0x7
        r1 = (self.instr >> 6) & 0x7
        imm_flag = (self.instr >> 5) & 0x1

        if imm_flag != 0:
            imm5 = sign_extend(self.instr & 0x1F, 5)
            g_reg[r0] = g_reg[r1] & imm5
        else:
            r2 = self.instr & 0x7
            g_reg[r0] = g_reg[r1] & g_reg[r2]

        update_flags(r0)

class OpNot(OpBase):
    def do_op(self) -> None:
        global g_reg

        r0 = (self.instr >> 9) & 0x7
        r1 = (self.instr >> 6) & 0x7

        g_reg[r0] = ~g_reg[r1]
        g_reg[r0] &= UINT16_MAX
        update_flags(r0)

class OpBr(OpBase):
    def do_op(self) -> None:
        global g_reg

        pc_offset = sign_extend((self.instr) & 0x1ff, 9)
        cond_flag = (self.instr >> 9) & 0x7
        if (cond_flag & g_reg[R_COND]) != 0:
            g_reg[R_PC] += pc_offset
            g_reg[R_PC] &= UINT16_MAX

class OpJmp(OpBase):
    def do_op(self) -> None:
        global g_reg

        # Also handles RET
        r1 = (self.instr >> 6) & 0x7
        g_reg[R_PC] = g_reg[r1]

class OpJsr(OpBase):
    def do_op(self) -> None:
        global g_reg

        r1 = (self.instr >> 6) & 0x7
        long_pc_offset = sign_extend(self.instr & 0x7ff, 11)
        long_flag = (self.instr >> 11) & 1

        g_reg[R_R7] = g_reg[R_PC]
        if long_flag != 0:
            g_reg[R_PC] += long_pc_offset # JSR
            g_reg[R_PC] &= UINT16_MAX
        else:
            g_reg[R_PC] = g_reg[r1] # JSRR

class OpLd(OpBase):
    def do_op(self) -> None:
        global g_reg

        r0 = (self.instr >> 9) & 0x7
        pc_offset = sign_extend(self.instr & 0x1ff, 9)
        g_reg[r0] = mem_read(g_reg[R_PC] + pc_offset)
        update_flags(r0)

class OpLdi(OpBase):
    def do_op(self) -> None:
        global g_reg

        r0 = (self.instr >> 9) & 0x7 # destination register (DR)
        pc_offset = sign_extend(self.instr & 0x1ff, 9) # PCoffset 9
        # add pc_offset to the current PC, look at that memory location to get the final address
        g_reg[r0] = mem_read(mem_read(g_reg[R_PC] + pc_offset))
        update_flags(r0)

class OpLdr(OpBase):
    def do_op(self) -> None:
        global g_reg

        r0 = (instr >> 9) & 0x7
        r1 = (instr >> 6) & 0x7
        offset = sign_extend(instr & 0x3F, 6)
        g_reg[r0] = mem_read(g_reg[r1] + offset)
        update_flags(r0)

class OpLea(OpBase):
    def do_op(self) -> None:
        global g_reg

        r0 = (self.instr >> 9) & 0x7
        pc_offset = sign_extend(self.instr & 0x1ff, 9)
        g_reg[r0] = g_reg[R_PC] + pc_offset
        g_reg[r0] &= UINT16_MAX
        update_flags(r0)

class OpSt(OpBase):
    def do_op(self) -> None:
        global g_reg

        r0 = (self.instr >> 9) & 0x7
        pc_offset = sign_extend(self.instr & 0x1ff, 9)
        mem_write(g_reg[R_PC] + pc_offset, g_reg[r0])

class OpSti(OpBase):
    def do_op(self) -> None:
        global g_reg

        r0 = (self.instr >> 9) & 0x7
        pc_offset = sign_extend(self.instr & 0x1ff, 9)
        mem_write(mem_read(g_reg[R_PC] + pc_offset), g_reg[r0])

class OpStr(OpBase):
    def do_op(self) -> None:
        global g_reg

        r0 = (self.instr >> 9) & 0x7
        r1 = (self.instr >> 6) & 0x7
        offset = sign_extend(self.instr & 0x3F, 6)
        mem_write(g_reg[r1] + offset, g_reg[r0])

TRAP_GETC = 0x20 # get character from keyboard, not echoed onto the terminal
TRAP_OUT = 0x21 # output a character
TRAP_PUTS = 0x22 # output a word string
TRAP_IN = 0x23 # get character from keyboard, echoed onto the terminal
TRAP_PUTSP = 0x24 # output a byte string
TRAP_HALT = 0x25 # halt the program

def trap_getc() -> None:
    global g_reg

    g_reg[R_R0] = get_key()

def trap_out() -> None:
    global g_reg

    print(chr(g_reg[R_R0]), end = '', flush = True)

def trap_puts() -> None:
    # one char per word 
    global g_reg
    global g_memory

    p = g_reg[R_R0]
    while g_memory[p] != 0:
        print(chr(g_memory[p]), end = '')
        p += 1
    sys.stdout.flush()

def trap_in() -> None:
    global g_reg

    print('Enter a character: ')
    trap_getc()
    trap_out()

def trap_putsp() -> None:
    # one char per byte (two bytes per word)
    # here we need to swap back to big endian format
    p = g_reg[R_R0]
    while g_memory[p] != 0:
        c = g_memory[p]
        c1 = c & 0xFF
        print(chr(c1), end = '')
        c2 = c >> 8
        if c2 != 0:
            print(chr(c2), end = '')
        p += 1
    sys.stdout.flush()

def trap_halt() -> None:
    global g_running

    print('HALT')
    g_running = False

g_op_trap_map = {
    TRAP_GETC: trap_getc,
    TRAP_OUT: trap_out,
    TRAP_PUTS: trap_puts,
    TRAP_IN: trap_in,
    TRAP_PUTSP: trap_putsp,
    TRAP_HALT: trap_halt,
}

class OpTrap(OpBase):
    def do_op(self) -> None:
        global g_reg

        trap_code = self.instr & 0xFF
        trap_fn = g_op_trap_map[trap_code]
        trap_fn()

def op_bad(instr: int) -> None:
    raise Exception(f'Invalid op in instruction: {instr}')

g_op_map = {
    OP_BR: OpBr,
    OP_ADD: OpAdd,
    OP_LD: OpLd,
    OP_ST: OpSt,
    OP_JSR: OpJsr,
    OP_AND: OpAnd,
    OP_LDR: OpLdr,
    OP_STR: OpStr,
    OP_NOT: OpNot,
    OP_LDI: OpLdi,
    OP_STI: OpSti,
    OP_JMP: OpJmp,
    OP_LEA: OpLea,
    OP_TRAP: OpTrap,
}

def read_image_file(fp: BytesIO) -> None:
    global g_memory

    origin_bytes = fp.read(2) # the origin tells us where in memory to place the image
    origin = (origin_bytes[0] << 8) + origin_bytes[1]

    # we know the maximum file size so we only need one fread */
    max_read = (UINT16_MAX - origin) * 2
    read_data = fp.read(max_read)
    # swap to little endian
    for i in range(0, len(read_data), 2):
        g_memory[origin + i // 2] = (read_data[i] << 8) + read_data[i + 1]

def read_image(image_path: str) -> bool:
    if not os.path.exists(image_path):
        return False

    with open(image_path, 'rb') as fp:
        read_image_file(fp)
    return True

def parse_args() -> None:
    if len(sys.argv) < 2:
        this_file = os.path.basename(__file__)
        print(f'python {this_file} [image-file1] ...')
        sys.exit(2)

    for each_file in sys.argv[1:]:
        if not read_image(each_file):
            print(f'failed to load image: {each_file}')
            sys.exit(1)

if __name__ == '__main__':
    parse_args()

    with TattrMgr():
        # set the PC to starting position
        pc_start = 0x3000 # 0x3000 is the default
        g_reg[R_PC] = pc_start

        while(g_running):
            instr = mem_read(g_reg[R_PC]) # FETCH
            g_reg[R_PC] += 1
            op = instr >> 12

            op_obj = g_op_map.get(op, op_bad)(instr)
            op_obj.do_op()
