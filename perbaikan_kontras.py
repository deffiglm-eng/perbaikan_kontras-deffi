"""
Peningkatan Kualitas Citra: Kontras Terang
Library : imageio, numpy, scipy.ndimage (+ matplotlib untuk menampilkan histogram)

Rumus transformasi linier (contrast & brightness):
    s = alpha * r + beta
    r     = nilai piksel citra awal (0..255)
    s     = nilai piksel citra hasil (dipotong ke 0..255)
    alpha = faktor kontras   (alpha > 1 -> kontras naik)
    beta  = faktor kecerahan (beta  > 0 -> citra lebih terang)

Cara pakai:
    1. Taruh foto: plat_motor.jpg dan bakso.jpg di folder yang sama dengan file ini
    2. pip install imageio numpy scipy matplotlib
    3. python peningkatan_citra.py
"""

import imageio.v3 as iio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Parameter default (bisa diatur per foto di bagian bawah file)
ALPHA = 1.4   # kontras
BETA = 40     # kecerahan (terang)


def ke_grayscale(img):
    """Ubah RGB ke grayscale: Y = 0.299 R + 0.587 G + 0.114 B"""
    if img.ndim == 2:
        return img.astype(np.float64)
    return img[..., :3] @ np.array([0.299, 0.587, 0.114])


def tingkatkan_kontras(img, alpha=ALPHA, beta=BETA):
    """Haluskan noise ringan (scipy.ndimage), lalu s = alpha * r + beta."""
    f = img.astype(np.float64)
    sigma = (0.8, 0.8, 0) if f.ndim == 3 else 0.8
    f = ndimage.gaussian_filter(f, sigma=sigma)
    s = alpha * f + beta
    return np.clip(s, 0, 255).astype(np.uint8)


def histogram(img_gray):
    """Hitung histogram 256 bin dengan scipy.ndimage.histogram."""
    return ndimage.histogram(img_gray, 0, 255, 256)


def proses(nama_file, label, alpha=ALPHA, beta=BETA):
    awal = iio.imread(nama_file)
    hasil = tingkatkan_kontras(awal, alpha, beta)
    iio.imwrite(f"hasil_{label}.png", hasil)

    g_awal = ke_grayscale(awal)
    g_hasil = ke_grayscale(hasil)

    print(f"[{label}] mean awal = {g_awal.mean():.1f}, std awal = {g_awal.std():.1f}")
    print(f"[{label}] mean hasil = {g_hasil.mean():.1f}, std hasil = {g_hasil.std():.1f}")

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].imshow(awal)
    ax[0, 0].set_title("Citra Awal")
    ax[0, 1].imshow(hasil)
    ax[0, 1].set_title(f"Citra Ditingkatkan (alpha={alpha}, beta={beta})")
    ax[1, 0].bar(range(256), histogram(g_awal), width=1.0, color="gray")
    ax[1, 0].set_title("Histogram Citra Awal")
    ax[1, 1].bar(range(256), histogram(g_hasil), width=1.0, color="gray")
    ax[1, 1].set_title("Histogram Citra Ditingkatkan")
    for a in ax[0]:
        a.axis("off")
    for a in ax[1]:
        a.set_xlabel("Intensitas piksel (0-255)")
        a.set_ylabel("Jumlah piksel")
        a.set_xlim(0, 255)
    plt.tight_layout()
    plt.savefig(f"screenshot_{label}.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    # Foto plat motor sudah cukup terang -> parameter dibuat lebih kecil
    # agar piksel tidak jenuh (putih semua).
    proses("plat_motor.jpg", "plat_motor", alpha=1.2, beta=10)
    proses("plat_motor.jpg", "plat_motor", alpha=1.2, beta=10)
    proses("telor.jpg", "telor", alpha=1.1, beta=10)