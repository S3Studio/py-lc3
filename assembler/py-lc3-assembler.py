#encoding: utf-8
import os
import sys
from typing import Union
import ply.lex as lex
import ply.yacc as yacc

UINT16_MAX = 0xFFFF

### global section ###

g_symbol_table = {}
g_symbol_needed = set()
g_mli = [] # machine language instructions
g_orig = -1
g_error = []

def init_global() -> None:
    global g_symbol_table, g_symbol_needed, g_mli, g_orig, g_error
    g_symbol_table = {}
    g_symbol_needed = set()
    g_mli = [] # machine language instructions
    g_orig = -1
    g_error = []

def print_errors() -> None:
    print(f'{len(g_error)} errors found')
    for each_err in g_error:
        print(each_err)

def get_orig_with_default() -> int:
    return g_orig if g_orig != -1 else 0x3000

### lex section ###

reserved_list = [
    'ADD',
    'AND',
    'BR',
    'BRN',
    'BRZ',
    'BRP',
    'BRZP',
    'BRNP',
    'BRNZ',
    'BRNZP',
    'JMP',
    'RET',
    'JSR',
    'JSRR',
    'LD',
    'LDI',
    'LDR',
    'LEA',
    'NOT',
    'RTI',
    'ST',
    'STI',
    'STR',
    'TRAP',
    'GETC',
    'OUT',
    'PUTS',
    'IN',
    'PUTSP',
    'HALT',
]

tokens = (
    'NEWLINE',
    'STRING',
    'NUMBER',
    'OPCODE',
    'COMMENT',
    'LABEL', # includes registers (r1, r2, ...)
    'PSEUDO',
)

def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += len(t.value)
    return t

def t_STRING(t):
    r'"([^"\r\n]|(\"))*"'
    t.value = t.value[1:-1]
    t.value = t.value.replace('\\e', '\033').encode('utf-8').decode('unicode_escape')
    return t

def t_NUMBER(t):
    r'((\#)?(-)?[0-9]+|x(-)?[0-9A-Z]+|b(-)?[01]+)'
    if t.value[0] == '#':
        res = int(t.value[1:])
    elif t.value[0] == 'x':
        res = int(t.value[1:], 16)
    elif t.value[0] == 'b':
        res = int(t.value[1:], 2)
    else:
        res = int(t.value)
    t.value = res
    return t

def t_COMMENT(t):
    r';.*'
    pass

def t_LABEL(t):
    r'[A-Za-z_][A-Za-z0-9_]*'
    if t.value.upper() in reserved_list:
        t.value = t.value.upper()
        t.type = 'OPCODE'
    return t

t_PSEUDO = r'\.[A-Za-z][A-Za-z0-9]*'

literals = ':,'

t_ignore = ' \t\r'

def t_error(t):
    global g_error
    g_error.append(f"{t.lexer.lineno}: Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

### yacc section ###

def p_start(p):
    '''start : expression newlines
             | expression
             | newlines
             | empty'''
    p[0] = p[1]

def p_start_2(p):
    '''start : newlines expression newlines
             | newlines expression'''
    p[0] = p[2]

def p_empty(p):
    '''empty : '''
    p[0] = []

def p_expression_line(p):
    '''expression : expression newlines line
                  | line'''
    if len(p) > 2:
        p[0] = p[1]
        if len(p) > 3:
            p[0].append(p[3])
    else:
        p[0] = [p[1]]

def p_line_term(p):
    '''line : LABEL newlines term
            | LABEL term
            | term'''
    p[0] = p[len(p) - 1]
    if len(p) > 2:
        p[0]['label'] = p[1]

def p_multi_newline(p):
    '''newlines : newlines NEWLINE
          | NEWLINE'''
    p[0] = []

def p_term(p):
    '''term : instruction
            | pseudo'''
    p[0] = p[1]

def p_instruction(p):
    '''instruction : OPCODE operand_list
                   | OPCODE'''
    p[0] = {'opcode': p[1], 'line': p.lineno(1)}
    if len(p) == 3:
        p[0]['operands'] = p[2]

def p_operand_list(p):
    '''operand_list : operand ',' operand ',' operand
                    | operand ',' operand
                    | operand'''
    p[0] = [p[1]]
    if len(p) > 3:
        p[0].append(p[3])
    if len(p) > 5:
        p[0].append(p[5])

def p_operand(p):
    '''operand : LABEL
               | NUMBER'''
    p[0] = p[1]

def p_pseudo(p):
    '''pseudo : PSEUDO NUMBER
              | PSEUDO STRING
              | PSEUDO LABEL
              | PSEUDO'''
    p[0] = {'pseudo': p[1], 'line': p.lineno(1)}
    if len(p) > 2:
        p[0]['param'] = p[2]

def p_error(p):
    global g_error
    g_error.append(f'{str(p.lineno) + ": " if p is not None else ""}syntax error at "{p if p is not None else "EOF" }"')

### opcode util section ###

def operand_is_register(s) -> bool:
    return type(s) == str and len(s) == 2 and s[0] in 'Rr' and s[1] in '01234567'

def operand_is_label(s) -> bool:
    return type(s) == str and not operand_is_register(s)

def operand_within(i, l: int) -> bool:
    return isinstance(i, int) and i >= (-1<<(l-1)) and i < (1<<l)

def operand_is_imm5(i) -> bool:
    return operand_within(i, 5)

def operand_is_off6(i) -> bool:
    return operand_within(i, 6) 

def operand_is_off9(i) -> bool:
    return operand_within(i, 9) 

def get_line(asm_code: dict) -> str:
    return f'{asm_code.get("line", -1)}: '

### psudo opcode analyze section ###

def po_orig(asm_code: dict) -> list:
    global g_orig, g_error

    param = asm_code.get('param', '')
    if not isinstance(param, int):
        g_error.append(get_line(asm_code) + 'invalid .ORIG statement')
    elif g_orig != -1:
        g_error.append(get_line(asm_code) + 'multi-define of .ORIG')
    else:
        g_orig = param

    return []

def po_end(asm_code: dict) -> Union[list, None]:
    global g_error

    if 'param' in asm_code:
        g_error.append(get_line(asm_code) + 'invalid .END statement')
        return []
    else:
        return None

def po_fill(asm_code: dict) -> list:
    global g_error

    param = asm_code.get('param', '')
    if not isinstance(param, int) and not operand_is_label(param):
        g_error.append(get_line(asm_code) + 'invalid .FILL statement')
        return []
    else:
        return [asm_code]

def po_blkw(asm_code: dict) -> list:
    global g_error

    param = asm_code.get('param', '')
    if not isinstance(param, int):
        g_error.append(get_line(asm_code) + 'invalid .BLKW statement')
        return []
    else:
        return [{'pseudo': '.FILL', 'param': 0}] * param

def po_stringz(asm_code: dict) -> list:
    global g_error

    param = asm_code.get('param', 0)
    if type(param) != str:
        g_error.append(get_line(asm_code) + 'invalid .STRINGZ statement')
        return []
    else:
        ret = [{'pseudo': '.FILL', 'param': ord(x) & UINT16_MAX}
            for x in param]
        ret.append({'pseudo': '.FILL', 'param': 0})
        return ret

g_pseudo_map = {
    '.ORIG': po_orig,
    '.END': po_end,
    '.FILL': po_fill,
    '.BLKW': po_blkw,
    '.STRINGZ': po_stringz,
}

def generate_pseudo_param(lc: int, asm_code: dict, param) -> list:
    global g_error

    if param is None:
        return []

    elif isinstance(param, int):
        if not operand_within(param, 16):
            g_error.append(f'operand too large for pseudo code {asm_code.get("pseudo", "")}: {param}')
            return []
        return [param]

    elif operand_is_label(param):
        pt = g_symbol_table[param] + get_orig_with_default()
        if not operand_within(pt, 16):
            g_error.append(f'label "{param}" too far for pseudo code {asm_code.get("pseudo", "")}')
            return []
        return [pt]

### opcode analyze section ###

g_type_map = {
    'r': operand_is_register,
    'l': operand_is_label,
    '5': operand_is_imm5,
    '6': operand_is_off6,
    '9': operand_is_off9,
    # check for offset11(x) will be done with label offset afterwards
    'x': operand_is_label,
}

def check_operands(asm_code: dict, each_o_type: list) -> bool:
    global g_error

    operands = asm_code.get('operands', [])
    l = len(operands)
    if l != len(each_o_type):
        g_error.append(get_line(asm_code) + f'invalid operand num for {asm_code["opcode"]}')
        return False

    for i in range(l):
        if not any([g_type_map[x](operands[i]) for x in each_o_type[i]]):
            g_error.append(get_line(asm_code) + f'invalid operand for {asm_code["opcode"]}: {operands[i]}')
            return False
    return True

def generate_operands(lc: int, asm_code: dict, each_o_type: list) -> list:
    global g_error

    ret = []
    operands = asm_code.get('operands', [])
    l = len(operands)
    for i in range(l):
        if 'r' in each_o_type[i] and operand_is_register(operands[i]):
            ret.append(int(operands[i][1]))
        elif len(set('56') & set(each_o_type[i])) > 0 and isinstance(operands[i], int):
            ret.append(operands[i])
        elif len(set('9xl') & set(each_o_type[i])) > 0:
            if operand_is_label(operands[i]):
                off = g_symbol_table[operands[i]] - lc - 1
            elif isinstance(operands[i], int):
                off = operands[i]
            ret.append(off & UINT16_MAX)
            if ('9' in each_o_type[i] and not operand_within(off, 9)) or \
                    ('x' in each_o_type[i] and not operand_within(off, 11)):
                g_error.append(f'label "{operands[i]}" too far for instruction {asm_code.get("opcode", "")}')

    return ret

g_op_map = {
    'ADD': ['r', 'r', 'r5'],
    'AND': ['r', 'r', 'r5'],
    'BR': ['9l'],
    'BRN': ['9l'],
    'BRZ': ['9l'],
    'BRP': ['9l'],
    'BRZP': ['9l'],
    'BRNP': ['9l'],
    'BRNZ': ['9l'],
    'BRNZP': ['9l'],
    'JMP': ['r'],
    'RET': [],
    'JSR': ['x'],
    'JSRR': ['r'],
    'LD': ['r', '9l'],
    'LDI': ['r', '9l'],
    'LDR': ['r', 'r', '6'],
    'LEA': ['r', '9l'],
    'NOT': ['r', 'r'],
    'RTI': [],
    'ST': ['r', '9l'],
    'STI': ['r', '9l'],
    'STR': ['r', 'r', '6'],
    'TRAP': ['5'],
    'GETC': [],
    'OUT': [],
    'PUTS': [],
    'IN': [],
    'PUTSP': [],
    'HALT': []
}

### opcode generation section ###

g_mask = [(1 << x) - 1 for x in range(16)]

def gen_and(lc: int, asm_code: dict, int_operands: list) -> int:
    if operand_is_imm5(asm_code['operands'][2]):
        return (0b101 << 12) + (int_operands[0] << 9) + (int_operands[1] << 6) + (1 << 5) + (int_operands[2] & g_mask[5])
    else:
        return (0b101 << 12) + (int_operands[0] << 9) + (int_operands[1] << 6) + (int_operands[2])

def gen_add(lc: int, asm_code: dict, int_operands: list) -> int:
    if operand_is_imm5(asm_code['operands'][2]):
        return (0b1 << 12) + (int_operands[0] << 9) + (int_operands[1] << 6) + (1 << 5) + (int_operands[2] & g_mask[5])
    else:
        return (0b1 << 12) + (int_operands[0] << 9) + (int_operands[1] << 6) + (int_operands[2])

def gen_br(lc: int, asm_code: dict, int_operands: list) -> int:
    sub_op = asm_code['opcode'][2:]
    if len(sub_op) == 0:
        sub_op = 'NZP'
    n = 1 if 'N' in sub_op else 0
    z = 1 if 'Z' in sub_op else 0
    p = 1 if 'P' in sub_op else 0
    return (n << 11) + (z << 10) + (p << 9) + (int_operands[0] & g_mask[9])

def gen_jmp(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b1100 << 12) + (int_operands[0] << 6)

def gen_jsr(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b100 << 12) + (1 << 11) + (int_operands[0] & g_mask[11])

def gen_jsrr(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b100 << 12) + (int_operands[0] << 6)

def gen_ld(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b10 << 12) + (int_operands[0] << 9) + (int_operands[1] & g_mask[9])

def gen_ldi(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b1010 << 12) + (int_operands[0] << 9) + (int_operands[1] & g_mask[9])

def gen_ldr(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b110 << 12) + (int_operands[0] << 9) + (int_operands[1] << 6) + (int_operands[2] & g_mask[6])

def gen_lea(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b1110 << 12) + (int_operands[0] << 9) + (int_operands[1] & g_mask[9])

def gen_not(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b1001 << 12) + (int_operands[0] << 9) + (int_operands[1] << 6) + g_mask[6]

def gen_rti(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b1000 << 12)

def gen_st(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b11 << 12) + (int_operands[0] << 9) + (int_operands[1] & g_mask[9])

def gen_sti(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b1011 << 12) + (int_operands[0] << 9) + (int_operands[1] & g_mask[9])

def gen_str(lc: int, asm_code: dict, int_operands: list) -> int:
    return (0b111 << 12) + (int_operands[0] << 9) + (int_operands[1] << 6) + (int_operands[2] & g_mask[6])

def gen_fill(lc: int, asm_code: dict, int_operands: list) -> int:
    return (int_operands[0] & UINT16_MAX)

def gen_trap_subcode(subcode: int) -> int:
    return (0b1111 << 12) + (subcode)

g_gen_ml_map = {
    'ADD': gen_add,
    'AND': gen_and,
    'BR': gen_br,
    'BRN': gen_br,
    'BRZ': gen_br,
    'BRP': gen_br,
    'BRZP': gen_br,
    'BRNP': gen_br,
    'BRNZ': gen_br,
    'BRNZP': gen_br,
    'JMP': gen_jmp,
    'RET': lambda x, y, z: gen_jmp(x, y, [0b111]),
    'JSR': gen_jsr,
    'JSRR': gen_jsrr,
    'LD': gen_ld,
    'LDI': gen_ldi,
    'LDR': gen_ldr,
    'LEA': gen_lea,
    'NOT': gen_not,
    'RTI': gen_rti,
    'ST': gen_st,
    'STI': gen_sti,
    'STR': gen_str,
    'TRAP': (lambda x, y, z: gen_trap_subcode(z[0])),
    'GETC': (lambda x, y, z: gen_trap_subcode(0x20)),
    'OUT': (lambda x, y, z: gen_trap_subcode(0x21)),
    'PUTS': (lambda x, y, z: gen_trap_subcode(0x22)),
    'IN': (lambda x, y, z: gen_trap_subcode(0x23)),
    'PUTSP': (lambda x, y, z: gen_trap_subcode(0x24)),
    'HALT': (lambda x, y, z: gen_trap_subcode(0x25)),
    '.FILL': gen_fill,
}

### main route section ###

g_overwrite_bin = False
g_debug = False

def generate_codes() -> list:
    bin_bytes = []
    for i, each_instr in enumerate(g_mli):
        err_cnt_prev = len(g_error)

        if 'opcode' in each_instr:
            op = each_instr['opcode']
            int_operands = generate_operands(i, each_instr, g_op_map[op])
        else:
            op = each_instr['pseudo']
            param = each_instr.get('param', None)
            int_operands = generate_pseudo_param(i, each_instr, param)

        if err_cnt_prev == len(g_error):
            bin_bytes.append(g_gen_ml_map[op](i, each_instr, int_operands))

    if g_debug:
        print(bin_bytes)

    return bin_bytes

def generate_bin(filename: str, bin_bytes: list) -> None:
    bin_filename = os.path.splitext(filename)[0] + '.bin'
    if not g_overwrite_bin and os.path.exists(bin_filename):
        print(f'error: target bin file "{bin_filename}" already exists, bypassing...')
        return

    with open(bin_filename, 'wb') as f:
        orig = get_orig_with_default()
        f.write(orig.to_bytes(2, byteorder = 'big'))
        for each_byte in bin_bytes:
            f.write(each_byte.to_bytes(2, byteorder = 'big'))
    print(f'success to write to file "{bin_filename}"')

def parse_args() -> None:
    if len(sys.argv) < 2:
        this_file = os.path.basename(__file__)
        print(f'python {this_file} [assembly-file1] ...')
        sys.exit(2)

    return sys.argv[1:]

if __name__ == '__main__':
    files = parse_args()

    for each_file in files:
        print()
        print(f'-=start to compile {each_file}=-')

        if not os.path.exists(each_file):
            print(f'failed to load assembly file: {each_file}')
            continue

        init_global()

        with open(each_file, 'r') as f:
            code = f.read()

        lexer = lex.lex()
        parser = yacc.yacc()
        parsed_res = parser.parse(code)
        if len(g_error) > 0:
            print_errors()
            continue
        elif not isinstance(parsed_res, list):
            print('-1: failed to check syntax')
            continue

        for asm_code in parsed_res:
            if len(g_mli) > (UINT16_MAX - 3000):
                g_error.append('too many instructions')
                break

            p = asm_code.get('pseudo', '') # pseudo code
            o = asm_code.get('opcode', '') # opcode
            new_instr = []
            err_cnt_prev = len(g_error)

            if p != '':
                if p not in g_pseudo_map:
                    g_error.append(get_line(asm_code) + f'invalid pseudo opcode "{p}"')
                else:
                    new_instr = g_pseudo_map[p](asm_code)
                    if new_instr is None:
                        # return of .END
                        break

            elif o != '':
                if o == 'TRAP':
                    operands = asm_code.get('operands', [])
                    if len(operands) != 1:
                        g_error.append(get_line(asm_code) + f'invalid operand num for TRAP')
                    elif operands[0] not in [0x20, 0x21, 0x22, 0x23, 0x24, 0x25]:
                        g_error.append(get_line(asm_code) + f'invalid operand for TRAP: {operands[0]}')
                    else:
                        new_instr = [asm_code]
                elif o not in g_op_map:
                    g_error.append(get_line(asm_code) + f'invalid opcode {o}')
                else:
                    new_instr = [asm_code] if check_operands(asm_code, g_op_map[o]) else []

            else:
                g_error.append('invalid syntax')

            if len(new_instr) > 0 and len(g_error) == err_cnt_prev:
                declare_label = asm_code.get('label', '')
                if declare_label != '':
                    if declare_label in g_symbol_table:
                        g_error.append(get_line(asm_code) + f'multi-define of symbol "{declare_label}"')
                    else:
                        g_symbol_table[declare_label] = len(g_mli)

                operands = asm_code.get('operands', [])
                for each_o in operands:
                    if operand_is_label(each_o):
                        g_symbol_needed.add(each_o)

                param = new_instr[0].get('param', None) # asm_code may be .STRINGZ str whose param should not be a label
                if operand_is_label(param):
                    g_symbol_needed.add(param)

                g_mli.extend(new_instr)

        for each_symbol in g_symbol_needed:
            if each_symbol not in g_symbol_table:
                g_error.append(f'-1: symbol "{each_symbol}" not declared')

        if g_debug:
            print(g_mli)
        if len(g_error) > 0:
            print_errors()
            continue

        bin_bytes = generate_codes()
        print_errors() # will not generate compile errors afterwards
        if len(g_error) > 0:
            continue

        generate_bin(each_file, bin_bytes)