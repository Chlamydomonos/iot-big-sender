import json
import numpy as np
import wave
import reedsolo


# 采样率，必须为 44100
SAMPLE_RATE = 44100


# 调制符号的采样点数量，需要是128的倍数
SYMBOL_SIZE = 1280


# 每个调制符号最后的空白采样点数量，需要小于SYMBOL_SIZE
EMPTY_SIZE = 384


# 数据包的Reed-Solomon编码后块的数量，由于包长度用7bit表示，所以最多为8
PACKET_SIZE = 3


def generate_wave(frequencies: list[int], sample_points: int, max_frequency_count: int=4) -> np.ndarray:
    '''
    生成音频，使用一系列频率叠加
    '''
    if not frequencies:
        return np.zeros(sample_points)

    t = np.linspace(0, sample_points / SAMPLE_RATE, sample_points, False)
    composite_wave = sum(1 / max_frequency_count * np.sin(2 * np.pi * frequency * t) for frequency in frequencies)
    return composite_wave


def save_wave(wave_data: np.ndarray, filename: str):
    '''
    保存音频
    '''
    wave_data = (wave_data * 32767).astype(np.int16)
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        f.writeframes(wave_data.tobytes())


codec = reedsolo.RSCodec(4)


def reedsolo_encode(data: bytes) -> bytes:
    '''
    使用 reedsolo 编码
    输入数据的长度必须是 8 的倍数
    '''

    data_blocks = []
    for i in range(0, len(data), 8):
        data_blocks.append(data[i:i+8])

    encoded_data_blocks = []
    for data_block in data_blocks:
        encoded_data_blocks.append(codec.encode(data_block))

    encoded_data = b''
    for encoded_data_block in encoded_data_blocks:
        encoded_data += encoded_data_block
    return encoded_data


def reedsolo_decode(data: bytes) -> bytes:
    '''
    使用 reedsolo 解码
    输入数据的长度必须是 12 的倍数
    '''

    data_blocks = []
    for i in range(0, len(data), 12):
        data_blocks.append(data[i:i+12])

    decoded_data_blocks = []
    for data_block in data_blocks:
        try:
            decoded_data_blocks.append(codec.decode(data_block)[0])
        except reedsolo.ReedSolomonError:
            decoded_data_blocks.append(data_block[:8])

    decoded_data = b''
    for decoded_data_block in decoded_data_blocks:
        decoded_data += decoded_data_block

    return decoded_data


def binary_to_base_n(binary_num: list[int], base: int):
    '''
    将二进制转换为n进制
    输入的二进制数是一个列表，如[1, 0, 1, 1]
    输出的n进制数也是一个列表，如[1, 2, 3]
    '''
    decimal_num = 0
    for i, bit in enumerate(reversed(binary_num)):
        decimal_num += int(bit) * (2 ** i)

    base_n_num = []
    while decimal_num > 0:
        remainder = decimal_num % base
        base_n_num.insert(0, remainder)
        decimal_num //= base

    return base_n_num if base_n_num else [0]


def base_n_to_binary(base_n_num: list[int], base: int):
    '''
    将n进制转换为二进制
    输入的n进制数是一个列表，如[1, 2, 3]
    输出的二进制数也是一个列表，如[1, 0, 1, 1]
    '''
    decimal_num = 0
    for i, digit in enumerate(reversed(base_n_num)):
        decimal_num += int(digit) * (base ** i)

    binary_num = []
    while decimal_num > 0:
        remainder = decimal_num % 2
        binary_num.insert(0, remainder)
        decimal_num //= 2

    return binary_num if binary_num else [0]


def bytes_to_binary(bytes: bytes) -> list[int]:
    '''
    把字节转换为二进制数组
    输入的字节是一个字节列表，如b'abc'
    输出的二进制数组也是一个列表，如[1, 0, 1, 1]
    '''
    binary_list = []
    for byte in bytes:
        binary = format(byte, '08b')
        binary_list += [int(bit) for bit in binary]
    return binary_list


def binary_to_bytes(binary: list[int]):
    '''
    把二进制数组转换为字节
    输入的二进制数组是一个列表，如[1, 0, 1, 1]
    输出的字节也是一个列表，如b'abc'
    '''
    if len(binary) % 8 != 0:
        binary += [0] * (8 - len(binary) % 8)
    out = []
    for i in range(0, len(binary), 8):
        out.append(int(''.join(map(str, binary[i:i + 8])), 2))
    return bytes(out)


num_codec = reedsolo.RSCodec(3)


def reedsolo_encode_num(data: int) -> bytes:
    '''
    使用 reedsolo 编码数字
    只能编码1-256之间的数字
    '''
    data -= 1
    data = data.to_bytes(1, 'big')
    data = num_codec.encode(data)
    return data


def reedsolo_decode_num(data: bytes) -> int:
    '''
    使用 reedsolo 解码数字
    只能解码1-256之间的数字
    '''
    try:
        data = num_codec.decode(data)[0]
        data = int.from_bytes(data, 'big')
        data += 1
        return data
    except reedsolo.ReedSolomonError:
        data = int.from_bytes(data[:1], 'big')
        data += 1
        return data


frequency_combinations: list[list[int]] = json.load(open('frequency_combinations.json', 'r'))
head_frequencies: list[int] = json.load(open('head_frequencies.json', 'r'))


def generate_symbol(frequencies: list[int]) -> np.ndarray:
    '''
    生成一个调制符号
    '''
    return np.concatenate([
        generate_wave(frequencies, SYMBOL_SIZE - EMPTY_SIZE),
        generate_wave([], EMPTY_SIZE)
    ])


head = np.concatenate([
    generate_symbol([head_frequencies[0]]),
    generate_symbol([head_frequencies[1]]),
    generate_symbol([head_frequencies[2]]),
    generate_symbol([head_frequencies[3]])
])


def encode_packet(packet: bytes, is_last: bool=False) -> np.ndarray:
    '''
    编码一个数据包
    '''
    packet_length = len(packet)
    if is_last:
        packet_length += 128

    if len(packet) % 8 != 0:
        packet += b'\x00' * (8 - len(packet) % 8)

    packet = reedsolo_encode(packet)
    packet = reedsolo_encode_num(packet_length) + packet

    base_1820_packet = []
    for i in range(0, len(packet), 4):
        binary_unit = (packet[i:i+4])
        binary_unit = bytes_to_binary(binary_unit)
        base_1820_unit = binary_to_base_n(binary_unit, 1820)
        base_1820_packet += base_1820_unit

    waves = [head]
    for i in range(len(base_1820_packet)):
        waves.append(generate_symbol(frequency_combinations[len(waves) % 2 == 0][base_1820_packet[i]]))
    waves.append(generate_wave([], SYMBOL_SIZE * 2))
    return np.concatenate(waves)


def letters_to_binary(letters: str) -> list[int]:
    '''
    将字母转换为二进制
    输入的字母序列长度必须为4
    '''
    binary = []
    for letter in letters:
        letterOrd = ord(letter)
        if letterOrd > ord('A') and letterOrd < ord('Z'):
            letterOrd -= ord('A')
        else:
            letterOrd -= ord('a')
            letterOrd += 26
        letter_binary = format(letterOrd, '06b')
        binary += [int(bit) for bit in letter_binary]
    return binary


def binary_to_letters(binary: list[int]) -> str:
    '''
    将二进制转换为字母
    输入的二进制序列长度必须为24
    '''
    letters = ''
    for i in range(0, len(binary), 6):
        letterOrd = 0
        for j in range(6):
            letterOrd += binary[i + j] * (2 ** (5 - j))
        if letterOrd < 26:
            letterOrd += ord('A')
        else:
            letterOrd -= 26
            letterOrd += ord('a')
        letters += chr(letterOrd)
    return letters


def num_to_binary(num: int) -> list[int]:
    '''
    将数字转换为二进制
    输入的数字必须在0-65535之间
    '''
    binary = []
    for _ in range(16):
        binary.append(num % 2)
        num //= 2
    return binary


def binary_to_num(binary: list[int]) -> int:
    '''
    将二进制转换为数字
    输入的二进制序列长度必须为16
    '''
    num = 0
    for i in range(16):
        num += binary[i] * (2 ** i)
    return num


def encode(text: str) -> np.ndarray:
    '''
    编码一段文本
    '''
    text_length = len(text)

    if text_length % 4 != 0:
        text += 'a' * (4 - text_length % 4)

    binary = num_to_binary(text_length)
    for i in range(0, len(text), 4):
        binary += letters_to_binary(text[i:i+4])

    byte_list = binary_to_bytes(binary)

    waves = [generate_wave([], SYMBOL_SIZE * 2)]
    for i in range(0, len(byte_list), PACKET_SIZE * 8):
        is_last = False
        if i + PACKET_SIZE * 8 >= len(byte_list):
            is_last = True
        waves.append(encode_packet(byte_list[i:i+PACKET_SIZE * 8], is_last))
    waves.append(generate_wave([], SYMBOL_SIZE * 2))

    return np.concatenate(waves)
