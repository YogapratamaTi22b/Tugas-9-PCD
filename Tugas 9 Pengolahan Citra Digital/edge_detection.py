import imageio
import numpy as np
import matplotlib.pyplot as plt

def roberts_operator(image):
    """
    Implementasi Deteksi Tepi dengan Operator Robert
    """
    # Kernel Robert
    kernel_robert_x = np.array([[1, 0], [0, -1]])
    kernel_robert_y = np.array([[0, 1], [-1, 0]])
    
    # Konvolusi dengan kernel Robert
    gx = convolve2d(image, kernel_robert_x)
    gy = convolve2d(image, kernel_robert_y)
    
    # Magnitudo gradien
    gradient_magnitude = np.hypot(gx, gy)
    return gradient_magnitude

def sobel_operator(image):
    """
    Implementasi Deteksi Tepi dengan Operator Sobel
    """
    # Kernel Sobel
    kernel_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Konvolusi dengan kernel Sobel
    gx = convolve2d(image, kernel_sobel_x)
    gy = convolve2d(image, kernel_sobel_y)
    
    # Magnitudo gradien
    gradient_magnitude = np.hypot(gx, gy)
    return gradient_magnitude

def convolve2d(image, kernel):
    """
    Fungsi untuk melakukan konvolusi 2D pada gambar grayscale
    """
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Padding untuk menjaga ukuran output sama dengan input
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    
    # Output hasil konvolusi
    output = np.zeros_like(image)
    for i in range(image_h):
        for j in range(image_w):
            region = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(region * kernel)
    return output

def main():
    # Membaca gambar dalam format grayscale
    image_path = r'C:\Users\ASUS\Documents\Smester 5\Pengolahan citra digital\Tugas 9 Pengolahan Citra Digital\pohon.jpg'
    image = imageio.imread(image_path, mode='F')  # Use mode='F' for floating-point grayscale
    
    # Robert Operator
    edge_roberts = roberts_operator(image)
    
    # Sobel Operator
    edge_sobel = sobel_operator(image)
    
    # Menampilkan hasil perbandingan
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Robert Operator")
    plt.imshow(edge_roberts, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Sobel Operator")
    plt.imshow(edge_sobel, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()