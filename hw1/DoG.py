from matplotlib.transforms import Bbox
import numpy as np
import cv2
import matplotlib.pyplot as plt


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        gaussian_images.append(image)
        height, width= image.shape

        for octave in range(self.num_octaves):
            for num in range(1,self.num_guassian_images_per_octave):
                    img_blur = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=self.sigma**num, sigmaY=self.sigma**num)
                    gaussian_images.append(img_blur)
            image = cv2.resize(gaussian_images[-1], (width//2, height//2), interpolation=cv2.INTER_NEAREST)
            gaussian_images.append(image)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(0, self.num_DoG_images_per_octave):
            img_dog = cv2.subtract(gaussian_images[i+1], gaussian_images[i])
            dog_images.append(img_dog)

            # cv2.imshow("dog", img_dog)
            # cv2.waitKey(0)
            # img_dog_norm = img_dog
            # cv2.normalize(img_dog, img_dog_norm, 0, 255, cv2.NORM_MINMAX)
            # plt.imshow(img_dog_norm, cmap='gray')
            # plt.savefig("1_norm_dog_"+str(i)+".png", bbox_inches='tight')
            # plt.close()
            # cv2.imwrite("1_norm_dog_"+str(i)+".png", img_dog_norm)  
        for i in range(self.num_DoG_images_per_octave+1, self.num_DoG_images_per_octave+5):
            img_dog = cv2.subtract(gaussian_images[i+1], gaussian_images[i])
            dog_images.append(img_dog)

            # cv2.imshow("dog", img_dog)
            # cv2.waitKey(0)
            # img_dog_norm = img_dog
            # cv2.normalize(img_dog, img_dog_norm, 0, 255, cv2.NORM_MINMAX)
            # plt.imshow(img_dog_norm, cmap='gray')
            # plt.savefig("1_norm_dog_"+str(i)+".png", bbox_inches='tight')
            # plt.close()
            # cv2.imwrite("1_norm_dog_"+str(i)+".png", img_dog_norm)  

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint

        keypoints = []
        for octave in range(self.num_octaves):
            for num in range(self.num_DoG_images_per_octave-2):
                d = octave*self.num_DoG_images_per_octave+num # (0,1,4,5)
                h,w = dog_images[d].shape

                img1 = dog_images[d]
                img2 = dog_images[d+1]
                img3 = dog_images[d+2]

                for x in range(1,h-1):
                    for y in range(1,w-1):
                        if(abs(img2[x][y]) > self.threshold):
                            kernel1 = img1[x-1:x+2,y-1:y+2]
                            kernel2 = img2[x-1:x+2,y-1:y+2]
                            kernel3 = img3[x-1:x+2,y-1:y+2]
                            if img2[x][y] >= np.amax(kernel1) and img2[x][y] >= np.amax(kernel2) and img2[x][y] >= np.amax(kernel3):
                                keypoints.append([x*(octave+1),y*(octave+1)])
                            elif img2[x][y] <= np.amin(kernel1) and img2[x][y] <= np.amin(kernel2) and img2[x][y] <= np.amin(kernel3):
                                keypoints.append([x*(octave+1),y*(octave+1)])
                                
        keypoints = np.array(keypoints)



        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return keypoints
