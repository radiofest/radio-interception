import komm
import matplotlib.pyplot as plt
import numpy as np
import graycode

modulation_dict = {'BPSK': 2,
                   'QPSK': 4,
                   '8-PSK': 8,
                   'QAM': 4,
                   '8-QAM': 8}

def modulate(message: str, mod: str = 'BPSK', vis: bool = False):
    ordr = modulation_dict[mod]

    message_bits = ''.join('{0:08b}'.format(ord(x), 'b') for x in message)

    ups = 3 #задаем количество дополнительных сэмплов определяющих
            #параметр sps
    if ordr == 4:
        ups = 6
    elif ordr == 8:
        ups = 8
    message_bits = np.array(list(message_bits), dtype=int)
    if ordr == 8:#для модуляций 8го порядка необходимо расширить символ с 8 бит до 9, так как
                 # у модуляций этого порядка отводится три бита на один символ
        n = 6
        k = []
        while n <= len(message_bits):
            k.append(n)
            n = n+8
        message_bits = np.insert(message_bits, k, 0)

    print(f'msg:{message_bits}')
    print(f'len:{len(message_bits)}')

    if (mod == 'BPSK') | (mod == 'QPSK') | (mod == '8-PSK'):
        points = PSKmod(message_bits, ordr, vis)
    else:
        points = QAMmod(message_bits, ordr, vis)
    plotIQ(points)
    print(f'modualated points:{points}')
    points_upsampled = upsample(points, ups)

    pnt = RRCfilter(0.35, 4, 4, points_upsampled)

    plotIQ(pnt)
    plotTime(pnt)
    plotFreq(pnt)

    print(f'post filter points komm:{pnt}')
    print(f'len:{len(pnt)}')

    return pnt

def PSKmod(message, ordr,visualize: bool = False):
    r"""
    komm PSK modulator wrapper

    Parameters:

        message: binary list to modulate

        ordr: PSK modulation order

        visualize(bool): flag to plot a modulation constellation

    Returns:
        compl_points: output complex symbols
    """
    modulator = komm.PSKModulation(ordr)
    compl_points = modulator.modulate(message)
    const = modulator.constellation

    if visualize:
        match ordr:
            case 2:
                key = 'BPSK'
            case 4:
                key = 'QPSK'
            case 8:
                key = '8-PSK'
            case _:
                print('wrong order')
                return 0
        labels = []
        for i in range(ordr):
            labels.append(list(modulator.demodulate(const[i])))
        real = np.real(const)
        imag = np.imag(const)
        print(f'labels:{labels}')
        print(f'const:{const}')
        plt.plot(real, imag, '.')
        plt.title(key)
        plt.grid(True)
        for i,(x,y) in enumerate(zip(real, imag)):
            plt.text(x=x, y=y, s = f'{labels[i]}', ha="center", va= "bottom")
        plt.show()
    return compl_points



def QAMmod(message, ordr, visualize: bool = False):
    r"""
    komm QAM modulator wrapper

    Parameters:

        message: binary list to modulate

        ordr: PSK modulation order

        visualize(bool): flag to plot a modulation constellation

    Returns:
        compl_points: output complex symbols
    """
    if ordr != 8:
        modulator = komm.QAModulation(int(ordr))
    else:
        modulator = komm.QAModulation((4, 2), 1)
    compl_points = modulator.modulate(message)
    const = modulator.constellation

    if visualize:
        match ordr:
            case 4:
                key = 'QAM'
            case 8:
                key = '8-QAM'
            case _:
                print('wrong order')
                return 0
        labels = []
        for i in range(ordr):
            labels.append(list(modulator.demodulate(const[i])))
        real = np.real(const)
        imag = np.imag(const)
        print(f'labels:{labels}')
        print(f'const:{const}')
        plt.plot(real, imag, '.')
        plt.title(str(key))
        plt.grid(True)
        for i,(x,y) in enumerate(zip(real, imag)):
            plt.text(x=x, y=y, s = f'{labels[i]}', ha="center", va= "bottom")
        plt.show()
    return compl_points


def upsample(bits, upsamp_points: int):
    r"""
    upsample func

    Parameters:

        bits: input samples(some weird naming why bits????)

        upsamp_points(int): int value equal to number of padded zeros
    Returns:
        points_upsampled: output sequence with padded zeros
    """
    points_upsampled = np.zeros((len(bits) - 1) * upsamp_points + upsamp_points, dtype=np.complex64)
    points_upsampled[::upsamp_points] = bits

    return points_upsampled

def RRCfilter(beta,len, sps, symbols):
    r"""
    Root raised cosine filter func

    Parameters:

        beta: rolloff factor

        len: filter length

        sps: samples per symbol value

        symbols: input complex sequence
    Returns:
         symbols: output filtered sequence
    """
    t0 = -len/2
    t1 = len/2
    t = np.arange(t0, t1, step=1 / sps)
    taps = (np.vectorize(imp_resp))(beta, t)
    symbols = np.convolve(taps, symbols)
    return symbols

def imp_resp(a, t):
    r"""

    Parameters:
        a: roloff factor

        t: "time" point
    Returns:
          impulse response
    """
    t += 1e-8
    return (np.sin(np.pi * (1 - a) * t) + 4 * a * t * np.cos(np.pi * (1 + a) * t)) / (
            np.pi * t * (1 - (4 * a * t) ** 2)
    )

def gray_code(n):
    if n < 1:
        g = []
    else:
        g = ['0', '0']
        n -= 1
        while n > 0:
            k = len(g)
            for i in range(k-1, -1, -1):
                char = '1' + g[i]
                g.append(char)
            for i in range(k-1, -1, -1):
                g[i] = '0' + g[i]
            n -= 1
    return g
def plotIQ(symbols):
    plt.plot(symbols.real[0:1000], symbols.imag[0:1000], '.')
    plt.title("IQ")
    plt.grid(True)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.show()

def plotTime(symbols):
    x = np.arange(0, len(symbols))
    _, ax = plt.subplots(figsize=(20, 20))
    plt.title("Time")
    ax.plot(x, symbols, 'g', linewidth=3)
    ax.grid(True)
    plt.show()


def plotFreq(symbols):
    x = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(symbols)))**2)
    f = np.linspace(-0.5, 0.5, len(symbols))
    plt.plot(f, x,'.')
    plt.title("SPECTRUM")
    plt.grid(True)
    plt.xlabel("Frequency[Normalized]")
    plt.ylabel("PSD [dB]")
    plt.show()

if __name__ == '__main__':
    modulate('ar', '8-PSK', True)