import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    

# THE SLOW VERSION

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        output = np.zeros(shape=img.shape)

        w = self.wndw_size
        r = self.pad_w
        sigma_r = self.sigma_r
        sigma_s = self.sigma_s

        # Spatial Kernel

        Gs = np.zeros((w,w,3))
        for i in range(w):
            for j in range(w):
                Gs[i,j,:] = ((i-r)**2 + (j-r)**2)
        Gs = np.divide(Gs, 2*(sigma_s**2))
        Gs = np.negative(Gs)
        Gs = np.exp(Gs)

        # Lookup Table for Range Kernel
        # LUT = np.exp(np.negative(np.divide(np.arange(256) * np.arange(256) , 2*(sigma_r**2))))

        # Applying the Bilateral Filter / Joint Bilateral Filter
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                Iq = padded_img[x:x+w, y:y+w]

                # Range Kernel
                Tq = padded_guidance[x:x+w, y:y+w] / 255 # Normalize
                Tp = padded_guidance[x+r, y+r] / 255     # Normalize

                if len(Tq.shape) == 3: # For RGB images
                    Gr = np.sum(np.square(Tq-Tp), axis=2)
                else: # For Gray images
                    Gr = np.square(Tq-Tp)
                Gr = np.divide(Gr, 2*(sigma_r**2))
                Gr = np.negative(Gr)
                Gr = np.exp(Gr)
                Gr = np.stack((Gr, Gr, Gr), axis=-1)

                output[x,y] = np.sum(np.sum((Gs*Gr*Iq), axis=0), axis=0) / np.sum(np.sum((Gs*Gr), axis=0), axis=0)
        
        return np.clip(output, 0, 255).astype(np.uint8)
