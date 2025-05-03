import os
import cv2
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt


# Standard quantization matrices
Q_Y = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

Q_C = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
])

def bgr_to_ycbcr(image):
    img = image.astype(np.float32)
    Y  = 0.299 * img[:,:,2] + 0.587 * img[:,:,1] + 0.114 * img[:,:,0]
    Cb = -0.1687 * img[:,:,2] - 0.3313 * img[:,:,1] + 0.5 * img[:,:,0] + 128
    Cr = 0.5 * img[:,:,2] - 0.4187 * img[:,:,1] - 0.0813 * img[:,:,0] + 128
    return Y, Cb, Cr

def downsample(channel):
    return channel[::2, ::2]

def block_process(channel, quant_matrix):
    h, w = channel.shape
    compressed = np.zeros_like(channel)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8] - 128
            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            quant_block = np.round(dct_block / quant_matrix)
            compressed[i:i+8, j:j+8] = quant_block
    return compressed

def zigzag_order(block):
    index_order = sorted(((x, y) for x in range(8) for y in range(8)),
                         key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))
    return [block[i, j] for i, j in index_order]

def run_length_encode(zigzag_list):
    encoded = []
    zero_count = 0

    # First element is DC, store it separately
    encoded.append(("DC", zigzag_list[0]))

    # Process AC coefficients
    for coeff in zigzag_list[1:]:
        if coeff == 0:
            zero_count += 1
        else:
            encoded.append((zero_count, coeff))
            zero_count = 0

    # End-of-Block marker
    if zero_count > 0:
        encoded.append(("EOB", 0))

    return encoded


def jpeg_compress(image_path):

    img = cv2.imread(image_path)
    h, w, _ = img.shape
    print("Height = ", h , "\nWidth = " ,w)

    # STEP 1 : Converting BGR to YCbCr   ->  Y = luminance , Cb = Chrominance Blue , Cr = Chrominance Red
    # The normal conversion is from RGB but since openCV opens the image in BGR format
    Y, Cb, Cr = bgr_to_ycbcr(img)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(Y, cmap='gray')
    plt.title('Y (Luminance)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(Cb, cmap='gray')
    plt.title('Cb (Chrominance Blue)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(Cr, cmap='gray')
    plt.title('Cr (Chrominance Red)')
    plt.axis('off')

    plt.tight_layout()
    # plt.show()

    # Downsample chroma channels 
    Cb_ds = downsample(Cb)
    Cr_ds = downsample(Cr)

    # Pad image to make dimensions multiple of 8
    def pad(channel):
        h, w = channel.shape
        pad_h = 8 - (h % 8) if h % 8 else 0
        pad_w = 8 - (w % 8) if w % 8 else 0
        return np.pad(channel, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    print("Before Padding = ",Y)
    Y = pad(Y)
    print("After Padding = ",Y)

    Cb_ds = pad(Cb_ds)
    Cr_ds = pad(Cr_ds)

    # Process blocks
    Yq = block_process(Y, Q_Y)
    Cbq = block_process(Cb_ds, Q_C)
    Crq = block_process(Cr_ds, Q_C)

    return Yq, Cbq, Crq


Yq, Cbq, Crq = jpeg_compress("download.jpeg")
print("Compressed matrices generated.")

original_size = os.path.getsize("download.jpeg")
compressed_size = Yq.size * 2 + Cbq.size * 2 + Crq.size * 2  # 2 bytes per int16 value

print(f"Original image size: {original_size} bytes")
print(f"Approximate compressed size (after quantization): {compressed_size} bytes")

# Test zig-zag scan on one 8x8 block (for verification)
sample_block = Yq[:8, :8]
zigzagged = zigzag_order(sample_block)
# print("Zig-zag output of top-left block in Y channel:\n", zigzagged)

rle_encoded = run_length_encode(zigzagged)
print("\nRun-Length Encoded:\n", rle_encoded)