#encoding: utf-8

import os
import sys
import platform
from io import BytesIO
from typing import Union

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

g_debugging = False
g_bp_entry = True
g_bp_list = {}
g_bp_step = False

def check_num_base(s: str) -> Union[int, None]:
    if s.isdigit():
        return int(s)

    if len(s) > 1:
        if s[0].lower() in 'd#' and s[1:].isdigit():
            return int(s[1:])
        elif s[0].lower() == 'x' and all([x in '0123456789abcdef' for x in s[1:].lower()]):
            return int(s[1:], 16)
        elif s[0].lower() == 'b' and all([x in '01' for x in s[1:]]):
            return int(s[1:], 2)

    return None

def debug_help(debug_cmd: str) -> bool:
    all_cmds = list(g_debug_ctrl_map.keys())
    all_cmds.append('CtrlC')
    print(f'Avaliable commands: {",".join(all_cmds)}')

    return False

def debug_continue(debug_cmd: str) -> bool:
    if debug_cmd.lower() != 'c':
        return False

    return True

def debug_next(debug_cmd: str) -> bool:
    global g_bp_step

    if debug_cmd.lower() != 'n':
        return False

    g_bp_step = True
    return True

def debug_dump_regs(debug_cmd: str) -> bool:
    cmd_list = [x.upper() for x in debug_cmd.split(' ')]

    int_base = 'd'
    literal_base = 'DEC'
    adjust = 0
    if len(cmd_list) > 1 and cmd_list[1][0] == '-':
        if cmd_list[1] == '-X':
            int_base = 'x'
            literal_base = 'HEX'
            adjust = 4
        elif cmd_list[1] == '-B':
            int_base = 'b'
            literal_base = 'BIN'
            adjust = 16

    reg_list = [
        (R_R0, 'R0'),
        (R_R1, 'R1'),
        (R_R2, 'R2'),
        (R_R3, 'R3'),
        (R_R4, 'R4'),
        (R_R5, 'R5'),
        (R_R6, 'R6'),
        (R_R7, 'R7'),
        (R_PC, 'PC'),
        (R_COND, 'COND'),
    ]
    print(f'REG: {literal_base}')
    for each_reg in reg_list:
        print(f'{each_reg[1]}: {format(g_reg[each_reg[0]], int_base).rjust(adjust, "0")}', end = '')
        print(' *' if each_reg[1] in cmd_list else '')

    return False

def debug_dump_mems(debug_cmd: str) -> bool:
    cmd_list = [x.upper() for x in debug_cmd.split(' ')]

    int_base = 'd'
    literal_base = 'DEC'
    adjust = 0
    if len(cmd_list) > 1 and cmd_list[1][0] == '-':
        if cmd_list[1] == '-X':
            int_base = 'x'
            literal_base = 'HEX'
            adjust = 4
        elif cmd_list[1] == '-B':
            int_base = 'b'
            literal_base = 'BIN'
            adjust = 16
        elif cmd_list[1] == '-C':
            int_base = 'x'
            literal_base = 'CHAR'
            adjust = 4

    target = cmd_list[1:3]
    if len(target) > 0 and target[0][0] == '-':
        target = cmd_list[2:4]

    target = [check_num_base(x) for x in target]
    if None not in target:
        print(f'MEM: {literal_base}')

        start = max(target[0] if len(target) != 0 else g_reg[R_PC] - 1, 0)
        length = target[1] if len(target) == 2 else 8
        for pt in range(start, start + length):
            if pt >= UINT16_MAX:
                break

            print(f'x{format(pt, "x").rjust(4, "0")}: {format(g_memory[pt], int_base).rjust(adjust, "0")}', end = '')
            if literal_base == 'CHAR':
                print('\t', end = '')
                chs = [chr(g_memory[pt] & 0xFF), chr((g_memory[pt] >> 8) & 0xFF)]
                for ch in chs:
                    print(f' {ch if ch.isprintable() else "."}', end = '')
            print()

    return False

def debug_breakpoint(debug_cmd: str) -> bool:
    global g_bp_list

    cmd_list = debug_cmd.split(' ')
    for s in cmd_list[1:]:
        pt = check_num_base(s)
        if pt is None or pt < 0 or pt >= UINT16_MAX:
            continue

        if pt not in g_bp_list:
            g_bp_list[pt] = {'enabled': True}

    return False

def debug_bp_enable(debug_cmd: str) -> bool:
    global g_bp_list

    cmd_list = debug_cmd.split(' ')
    for s in cmd_list[1:]:
        pt = check_num_base(s)
        if pt is None or pt < 0 or pt >= UINT16_MAX:
            continue

        if pt in g_bp_list:
            g_bp_list[pt]['enabled'] = True

    return False

def debug_bp_disable(debug_cmd: str) -> bool:
    global g_bp_list

    cmd_list = debug_cmd.split(' ')
    for s in cmd_list[1:]:
        pt = check_num_base(s)
        if pt is None or pt < 0 or pt >= UINT16_MAX:
            continue

        if pt in g_bp_list:
            g_bp_list[pt]['enabled'] = False

    return False

def debug_bp_remove(debug_cmd: str) -> bool:
    global g_bp_list

    cmd_list = debug_cmd.split(' ')
    for s in cmd_list[1:]:
        pt = check_num_base(s)
        if pt is None or pt < 0 or pt >= UINT16_MAX:
            continue

        if pt in g_bp_list:
            del g_bp_list[pt]

    return False

def debug_bp_clear(debug_cmd: str) -> bool:
    global g_bp_list

    if debug_cmd.lower() != 'bc':
        return False

    g_bp_list.clear()
    return False

def debug_bp_list(debug_cmd: str) -> bool:
    if debug_cmd.lower() != 'bl':
        return False

    print('BP: ENABLED')
    for k, v in g_bp_list.items():
        print(f'x{format(k, "x").rjust(4, "0")}: {v["enabled"]}')

g_debug_ctrl_map = {
    'h' : debug_help,
    'c' : debug_continue,
    'n' : debug_next,
    'r' : debug_dump_regs,
    'm' : debug_dump_mems,
    'bp' : debug_breakpoint,
    'be' : debug_bp_enable,
    'bd' : debug_bp_disable,
    'br' : debug_bp_remove,
    'bc' : debug_bp_clear,
    'bl' : debug_bp_list,
}

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

if platform.system().lower() == 'windows':
    from msvcrt import kbhit, getch

    def check_key_win() -> bool:
        return kbhit()

    check_key_real = check_key_win

    def get_key_win() -> int:
        return ord(getch()) & UINT16_MAX

    get_key_real = get_key_win

    class TattrMgrNull(object):
        def __enter__(self):
            return self

        def __exit__(self, type, value, trace):
            pass

    TattrMgr = TattrMgrNull

else:
    import select

    def check_key_nix() -> bool:
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return len(dr) > 0

    check_key_real = check_key_nix

    def get_key_nix() -> int:
        return ord(sys.stdin.read(1)) & UINT16_MAX

    get_key_real = get_key_nix

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

g_debug_kbin = [] # keyboard input cache for debugging

def check_key() -> bool:
    if g_debugging and len(g_debug_kbin) > 0:
        return True
    else:
        return check_key_real()

def get_key() -> int:
    global g_debug_kbin

    if g_debugging and len(g_debug_kbin) > 0:
        ch = g_debug_kbin.pop(0)
        return ch
    else:
        return get_key_real()

def get_key_imm() -> Union[int, None]:
    global g_debug_kbin

    if check_key_real():
        ch = get_key_real()
        g_debug_kbin.append(ch)
        return ch
    else:
        return None

def clear_kbin() -> None:
    global g_debug_kbin

    g_debug_kbin.clear()

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

    while True:
        ch = get_key()

        if g_debugging and ch == 0x10: # CtrlP
            wait_debugging()
        else:
            g_reg[R_R0] = ch
            break

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

g_op_cache = {}
def get_op(instr):
    global g_op_cache

    if instr not in g_op_cache:
        op = instr >> 12
        g_op_cache[instr] = g_op_map.get(op, op_bad)(instr)
    return g_op_cache[instr].do_op

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
    global g_debugging

    if len(sys.argv) < 2 or (len(sys.argv) == 2 and sys.argv[1].lower() == '-d'):
        this_file = os.path.basename(__file__)
        print(f'python {this_file} [-d] image-file1 ...')
        sys.exit(2)

    file_list = sys.argv[1:]
    if file_list[0].lower() == '-d':
        file_list = file_list[1:]
        g_debugging = True

    for each_file in file_list:
        if not read_image(each_file):
            print(f'failed to load image: {each_file}')
            sys.exit(1)

def check_bp() -> bool:
    global g_bp_entry
    global g_bp_step

    if g_debugging == False:
        return False

    if g_bp_entry:
        g_bp_entry = False
        return True

    if g_bp_list.get(g_reg[R_PC] - 1, {}).get('enabled', False):
        return True

    if g_bp_step:
        g_bp_step = False
        return True

    ch = get_key_imm()
    if ch == 0x10: # CtrlP
        clear_kbin()
        return True

    return False

g_last_cmd = None

def wait_debugging() -> None:
    global g_last_cmd

    print(f'\nProgram paused at x{format(g_reg[R_PC] - 1, "x").rjust(4, "0")}')

    debug_cmd = None
    while True:
        if debug_cmd is None:
            print('*>', end = '', flush = True)
            debug_cmd = ''
        ch = chr(get_key())
        if ch not in '\x1b\x5b' and debug_cmd[-2:] != '\x1b\x5b':
            print(ch, end = '', flush = True)

        if ch in '\r\n':
            debug_ctrl = g_debug_ctrl_map.get(debug_cmd.split(' ')[0].lower(), None)
            if debug_ctrl is not None:
                if debug_ctrl(debug_cmd):
                    return
                g_last_cmd = debug_cmd # last_cmd used when pc is different
            debug_cmd = None
        elif ch == '\x7f':
            print('[', end = '', flush = True)
            debug_cmd = debug_cmd[:-1]
        elif debug_cmd[-2:] == '\x1b\x5b':
            if debug_cmd is None:
                debug_cmd = []
            if ch == '\x41':
                # arrow up, use last input
                debug_cmd = g_last_cmd
                if debug_cmd is not None:
                    print(f'\n*>{debug_cmd}', end = '', flush = True)
                else:
                    print()
            elif ch == '\x42':
                debug_cmd = None # arrow down, clean input
                print()
        else:
            debug_cmd += ch

if __name__ == '__main__':
    parse_args()

    with TattrMgr():
        # set the PC to starting position
        pc_start = 0x3000 # 0x3000 is the default
        g_reg[R_PC] = pc_start

        while(g_running):
            instr = mem_read(g_reg[R_PC]) # FETCH
            g_reg[R_PC] += 1

            if check_bp():
                wait_debugging()

            op_obj = get_op(instr)
            op_obj()
