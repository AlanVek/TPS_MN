import numpy as np
from math import log2
from time import time


# Binary to decimal (int).
def bin2dec(num : np.array) -> int:
    return np.dot(num[::-1], 2 ** np.arange(len(num) + 0.0))

# Binary IEEE 754 to decimal (float)
def binf2dec(num : np.array, ne : int) -> float:
    if ne >= len(num) - 1 or ne < 1: return 0
    if np.count_nonzero((num!=0) & (num!=1)): return 0

    signpart, expopart, mantpart = num[0], num[1 : ne + 1], num[ne + 1 :]
    sign, bias = (-1) ** signpart,  2 ** (ne - 1) - 1
    normal = 1

    one_count_exp = np.count_nonzero(expopart)
    all_zeros_exp, all_ones_exp = one_count_exp == 0, one_count_exp == len(expopart)
    all_zeros_mant = not np.count_nonzero(mantpart)

    if all_zeros_exp and all_zeros_mant: return 0
    elif all_zeros_exp and not all_zeros_mant: normal = 0
    elif all_ones_exp and all_zeros_mant: return np.inf * sign
    elif all_ones_exp and not all_zeros_mant: return np.nan

    expo = bin2dec(expopart) - bias + (1 - normal)
    mantisa = normal + bin2dec(mantpart) * 2.0 ** (-len(num) + ne + 1)
    return sign * 2.0 ** expo * mantisa

def bin2string(num : np.array) -> str: return ''.join(map(str, num))
def string2bin(num : str) -> np.array: return np.array(list(num), dtype = int)

as_int = np.vectorize(int)

# Decimal (int) to binary.
def dec2bin(num: int, width: int = -1) -> np.array:
    if num <= 0: res = np.zeros(1, dtype = int)
    else:
        length = int(log2(num) + 1)
        expos = np.arange(length - 1, -1, -1)
        res = (num & as_int(2.0**expos)).astype(bool).astype(int)

    if res.size < width:
        res = np.concatenate((np.zeros(width - res.size, dtype = int), res))
    return res

def dec2binf_gen(num: float, ne: int, nm: int) -> np.array:

    if ne < 1 or nm < 1:
        return np.zeros(1 + nm + ne, dtype = int)

    BIAS = 2 ** (ne - 1) - 1

    # Nan
    if np.isnan(num):
        return np.concatenate(([0], np.ones(ne + nm, dtype = int)))

    abs_num, sign_bit = abs(num),  num < 0

    # Infinito o Cero
    if abs_num >= 2 ** (1 + BIAS):
        return np.concatenate(([sign_bit], np.ones(ne), np.zeros(nm))).astype(int)
    elif abs_num < 2 ** (1 - BIAS - nm):
        return np.concatenate(([sign_bit], np.zeros(ne + nm, dtype = int)))

    # Normal o Sub-Normal
    if abs_num >= 2**(1 - BIAS):
        exp = int(np.floor(log2(abs_num)))
        dec = abs_num / 2**exp - 1
    else:
        exp = -BIAS
        dec = abs_num / 2**(exp + 1)

    mantissa = dec2bin(int(2**nm * dec), nm)

    return np.concatenate(([sign_bit], dec2bin(exp + BIAS, ne), mantissa))

dec2binf = lambda num: dec2binf_gen(num, ne = 5, nm = 10)

def tests(ne : int, nm : int, maxrepe : int = 1):
    errors = 0

    # Loops through maxrepe random IEEE 754 numbers with 1 sign bit, ne expo bits and nm mantissa bits
    for i in range(maxrepe):

        # Generates IEEE 754 number
        init = np.random.randint(0, 2, 1 + ne + nm, dtype = int)

        # Decimal representation
        to_dec = binf2dec(init, ne=ne)

        # Back again to IEEE 754
        to_flo = dec2binf(to_dec)

        if np.any(init != to_flo):
            if not (np.isnan(to_dec) and np.all(init[1 : 1 + ne] == to_flo[1 : 1 + ne])):
                if not (to_dec == 0 and np.all(init[1:] == to_flo[1:])):
                    errors += 1
                    print(f'Initial: {bin2string(init)}')
                    print(f'Decimal: {to_dec}')
                    print(f'Floating: {bin2string(to_flo)}')

    return errors

if __name__ == '__main__':

    ne = 5
    nm = 10

    errors = tests(ne=ne, nm=nm, maxrepe = 50000)

    print(f'\nFinished with {errors} errors.')
