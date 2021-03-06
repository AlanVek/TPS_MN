import numpy as np
from math import log2

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

def test():
    
    bin2string = lambda n: ''.join(map(str, n))
    
    cant_ok, cant_tot = 0, 13
    
    # Mínimo representable
    cant_ok += bin2string(dec2binf(2**-24)) ==  '0000000000000001'
    
    # Infinito
    cant_ok += bin2string(dec2binf(np.inf)) == '0111110000000000'
                          
    cant_ok += bin2string(dec2binf(-np.inf)) == '1111110000000000'
    
    # Mayor al máximo -> infinito
    cant_ok += bin2string(dec2binf(2**16)) == '0111110000000000'
    
    # Menor al mínimo -> cero
    cant_ok += bin2string(dec2binf(2**-25)) == '0000000000000000'
    
    cant_ok += bin2string(dec2binf(-2**-25)) == '1000000000000000'
    
    # NaN
    cant_ok +=  bin2string(dec2binf(np.nan)) == '0111111111111111'
    
    # Normales
    cant_ok += bin2string(dec2binf(23.18914)) == '0100110111001100'
    
    cant_ok += bin2string(dec2binf(-56219.098)) == '1111101011011100'
    
    cant_ok += bin2string(dec2binf(-0.5938)) == '1011100011000000'
                          
    # Subnormales                          
    cant_ok += bin2string(dec2binf(1.53e-5)) == '0000000100000000'
                          
    cant_ok += bin2string(dec2binf(-4.274e-5)) == '1000001011001101'
    
    cant_ok += bin2string(dec2binf(-3.26e-5)) == '1000001000100010'
    
    print('Ok!' if cant_ok == cant_tot else 'Failed')

if __name__ == '__main__':
    test()