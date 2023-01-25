alphabet = '012345679abcdefghijklmnopqrstuvwxyz'


def encode(number: int) -> str:
    """
    Encode single number as digit.
    :param number: number to encode as digit.
    :return: encoded number as digit.
    """
    try:
        return alphabet[number]
    except IndexError:
        raise Exception("Cannot encode {}".format(number))


def decode(digit: str) -> int:
    """
    Decode single digit as number:
    :param digit: digit to decode as number.
    :return: decoded digit as number.
    """
    try:
        return alphabet.index(digit)
    except IndexError:
        raise Exception("Cannot decode {}".format(digit))


def decimal_to_base(decimal: int = 0, base: int = 16) -> str:
    """
    Convert decimal to base.
    :param decimal: decimal number.
    :param base: base to convert to.
    :return: number in given base.
    """
    if decimal < base:
        return encode(decimal)
    else:
        return decimal_to_base(decimal // base, base) + encode(decimal % base)
