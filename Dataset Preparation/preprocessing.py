import numpy as np
import cv2 as cv
import os
import tensorflow as tf 


class Preprocess:

    """-----------------------DOcstring for Preprocess--------------------------------

    This class is used to preprocess the images in the dataset. The class has the following methods:

    1. image_resizing: This method is used to resize the images in the dataset. The method takes the following parameters:
        - width: The width of the resized image.
        - height: The height of the resized image.
        - imagefile: The image file to be resized.
        - resizedsubpath: The subpath where the resized images will be saved.
        - savefile: A boolean value to save the resized images.

    2. normalization: This method is used to normalize the images in the dataset. The method takes the following parameters:
        - imagefile: The image file to be normalized.
        - normalizedsubpath: The subpath where the normalized images will be saved.
        - savefile: A boolean value to save the normalized images.

    3. random_crop_with_flip: This method is used to randomly crop the images in the dataset. The method takes the following parameters:
        - num_crops: The number of crops to be made.
        - imagefile: The image file to be cropped.
        - normalizedsubpath: The subpath where the normalized images will be saved.
        - randomcropsubpath: The subpath where the randomly cropped images will be saved.
        - savefile: A boolean value to save the randomly cropped images.

    4. center_corner_crop_with_flip: This method is used to crop the images in the dataset from the center and corners. The method takes the following parameters:
        - imagefile: The image file to be cropped.
        - normalizedsubpath: The subpath where the normalized images will be saved.
        - centercornersubpath: The subpath where the center and corner cropped images will be saved.
        - savefile: A boolean value to save the center and corner cropped images.

    5. fancy_pca: This method is used to apply fancy PCA to the images in the dataset. The method takes the following parameters:
        - imagefile: The image file to be augmented.
        - cropsubpath: The subpath where the cropped images will be saved.
        - augmentedsubpath: The subpath where the augmented images will be saved.

    Example:

    process = Preprocess("/media/HDD/WorkI/AlexNet/Dataset/","Animals90")
    process.image_resizing(256,256, "Animals90ResizedImages")
    process.normalization("Animals90ResizedImages", "Animals90NormalizedImages")
    process.random_crop_with_flip(2, "Animals90ResizedImages", "Animals90RandomCroppedImages")
    process.center_corner_crop_with_flip("Animals90ResizedImages", "Animals90CenterCornerCroppedImages")
    process.fancy_pca("Animals90ResizedImages", "Animals90AugmentedImages")


    Example 2:

    process = Preprocess("/media/HDD/WorkI/AlexNet/Dataset/","Animals90")
    process.image_resizing(256,256, "Animals90ResizedImages", savefile = True)
    process.normalization("Animals90ResizedImages", "Animals90NormalizedImages", savefile = True)
    process.random_crop_with_flip(2, "Animals90ResizedImages", "Animals90RandomCroppedImages", savefile = True)
    process.center_corner_crop_with_flip("Animals90ResizedImages", "Animals90CenterCornerCroppedImages", savefile = True)
    process.fancy_pca("Animals90ResizedImages", "Animals90AugmentedImages", savefile = True)

    Example 3:

    process = Preprocess("/media/HDD/WorkI/AlexNet/Dataset/","Animals90")
    process.image_resizing(256,256, imagefile = image, resizedsubpath = "Animals90ResizedImages")
    process.normalization(imagefile = image, normalizedsubpath = "Animals90NormalizedImages")
    process.random_crop_with_flip(2, imagefile = image, normalizedsubpath = "Animals90ResizedImages", randomcropsubpath = "Animals90RandomCroppedImages")
    process.center_corner_crop_with_flip(imagefile = image, normalizedsubpath = "Animals90ResizedImages", centercornersubpath = "Animals90CenterCornerCroppedImages")
    process.fancy_pca(imagefile = image, cropsubpath = "Animals90ResizedImages", augmentedsubpath = "Animals90AugmentedImages")


    ---------------------------------------------------------------------------------
    """



    def __init__(self, mainpath, originalsubpath):
        self.mainpath = mainpath
        self.originalsubpath = originalsubpath


    def image_resizing(self, width, height, imagefile = None, resizedsubpath = None, savefile = False):

        """This method is used to resize the images in the dataset. The method takes the following parameters:
        - width: The width of the resized image.
        - height: The height of the resized image.
        - imagefile: The image file to be resized.
        - resizedsubpath: The subpath where the resized images will be saved.
        - savefile: A boolean value to save the resized images.
        
        Returns:
        - resizedimage: The resized image.

        Example:

        process = Preprocess("/media/HDD/WorkI/AlexNet/Dataset/","Animals90")
        process.image_resizing(256,256, "Animals90ResizedImages")
        """

        resizedimage = cv.resize(imagefile, (width,height))
            
        if savefile:

            originaldatapath = os.path.join(self.mainpath, self.originalsubpath)


            dirlist = os.listdir(originaldatapath)


            resizeddirectory = os.path.join(self.mainpath, resizedsubpath)

            if not os.path.isdir(resizeddirectory):
                    os.mkdir(resizeddirectory)


            for subpath in dirlist:

                resizedsubdirectory = os.path.join(resizeddirectory , subpath)


                if not os.path.isdir(resizedsubdirectory):
                    os.mkdir(resizedsubdirectory)


                for imagename in os.listdir(os.path.join(originaldatapath,subpath)):

                    
                    print(os.path.join(resizedsubdirectory, imagename))

                    resizedimagepath = os.path.join(resizedsubdirectory, imagename)
                    
                    image = cv.imread(os.path.join(originaldatapath, subpath, imagename))
                    

                    resizedimage = cv.resize(image, (width,height))

                    cv.imwrite(resizedimagepath, resizedimage)

            return None
        
        return resizedimage
        


    def normalization(self, imagefile = None, resizedsubpath = None, normalizedsubpath = None, savefile = False):
        
        """This method is used to normalize the images in the dataset. The method takes the following parameters:
        - imagefile: The image file to be normalized.
        - normalizedsubpath: The subpath where the normalized images will be saved.
        - savefile: A boolean value to save the normalized images.
        
        Returns:
        - image: The normalized image.
        
        Example:
        process = Preprocess("/media/HDD/WorkI/AlexNet/Dataset/","Animals90")
        process.normalization("Animals90ResizedImages", "Animals90NormalizedImages")
        """


        image = imagefile/255.0
            

        if savefile:

            resizeddatapath = os.path.join(self.mainpath, resizedsubpath)

            resizeddirlist = os.listdir(resizeddatapath)

            normalizeddirectory = os.path.join(self.mainpath, normalizedsubpath)

            if not os.path.isdir(normalizeddirectory):
                    os.mkdir(normalizeddirectory)

            for resizedsubdir in resizeddirlist:

                normalizedsubdirectory = os.path.join(normalizeddirectory, resizedsubdir)

                if not os.path.isdir(normalizedsubdirectory):
                    os.mkdir(normalizedsubdirectory)

                for imagename in os.listdir(os.path.join(resizeddatapath, resizedsubdir)):
                    image = cv.imread(os.path.join(resizeddatapath, resizedsubdir, imagename))
            
                    print(os.path.join(normalizedsubdirectory, imagename))
                    
                    image = image/255.0

                    normalizedimagepath = os.path.join(normalizedsubdirectory, imagename)

                    cv.imwrite(normalizedimagepath, image)

            return None
        
        return image
    

    
    def random_crop_with_flip(self, num_crops, imagefile = None, normalizedsubpath = None, randomcropsubpath = None, savefile = False):

        """This method is used to randomly crop the images in the dataset. The method takes the following parameters:
        - num_crops: The number of crops to be made.
        - imagefile: The image file to be cropped.
        - normalizedsubpath: The subpath where the normalized images will be saved.
        - randomcropsubpath: The subpath where the randomly cropped images will be saved.
        - savefile: A boolean value to save the randomly cropped images.

        Returns:
        - listofimages: A list of the randomly cropped images.

        Example:
        process = Preprocess("/media/HDD/WorkI/AlexNet/Dataset/","Animals90")
        process.random_crop_with_flip(2, "Animals90ResizedImages", "Animals90Random
        """
        
        listofimages = []

        for i in range(num_crops):

            croped_image = tf.image.random_crop(imagefile, [224, 224,3])

            croped_flip_image = tf.image.random_flip_left_right(croped_image)

            listofimages.append(croped_flip_image)
            

        if savefile:

            normalizeddatapath = os.path.join(self.mainpath, normalizedsubpath)

            normalizeddirlist = os.listdir(normalizeddatapath)

            randomcropdirectory = os.path.join(self.mainpath, randomcropsubpath)

            if not os.path.isdir(randomcropdirectory):
                    os.mkdir(randomcropdirectory)

            for normalizedsubdir in normalizeddirlist:

                randomcropsubdirectory = os.path.join(randomcropdirectory, normalizedsubdir)

                if not os.path.isdir(randomcropsubdirectory):
                        os.mkdir(randomcropsubdirectory)

                for imagename in os.listdir(os.path.join(normalizeddatapath, normalizedsubdir)):
                    image = cv.imread(os.path.join(normalizeddatapath, normalizedsubdir, imagename))

                    print(os.path.join(randomcropsubdirectory, imagename))

                    for i in range(num_crops):

                        croped_image = tf.image.random_crop(image , [224, 224,3])

                        croped_flip_image = tf.image.random_flip_left_right(croped_image)

                        croped_flip_image = croped_flip_image.numpy()

                        randomcropimagepath = os.path.join(randomcropsubdirectory, "crop_" + str(i) + "_" + imagename)

                        cv.imwrite(randomcropimagepath, croped_flip_image)

            return None
        
        return listofimages
    

    def center_corner_crop_with_flip(self, imagefile = None, normalizedsubpath = None, centercornersubpath = None, savefile = False):

        """This method is used to crop the images in the dataset from the center and corners. The method takes the following parameters:
        - imagefile: The image file to be cropped.
        - normalizedsubpath: The subpath where the normalized images will be saved.
        - centercornersubpath: The subpath where the center and corner cropped images will be saved.
        - savefile: A boolean value to save the center and corner cropped images.

        Returns:
        - croped_left_top: The cropped image from the left top.
        - croped_right_top: The cropped image from the right top.
        - croped_left_bottom: The cropped image from the left bottom.
        - croped_right_bottom: The cropped image from the right bottom.
        - croped_center: The cropped image from the center.
        - croped_left_top_flip: The flipped cropped image from the left top.
        - croped_right_top_flip: The flipped cropped image from the right top.
        - croped_left_bottom_flip: The flipped cropped image from the left bottom.
        - croped_right_bottom_flip: The flipped cropped image from the right bottom.
        - croped_center_flip: The flipped cropped image from the center.

        Example:
        process = Preprocess("/media/HDD/WorkI/AlexNet/Dataset/","Animals90")
        process.center_corner_crop_with_flip("Animals90ResizedImages", "Animals90CenterCornerCroppedImages")
        """

        croped_left_top = tf.image.crop_to_bounding_box(image, 0, 0, 224, 224)

        croped_right_top = tf.image.crop_to_bounding_box(image, 0, 32, 224, 224)

        croped_left_bottom = tf.image.crop_to_bounding_box(image, 32, 0, 224, 224)

        croped_right_bottom = tf.image.crop_to_bounding_box(image, 32, 32, 224, 224)

        croped_center = tf.image.central_crop(image, 0.875)

        croped_left_top_flip = tf.image.random_flip_left_right(croped_left_top)

        croped_right_top_flip = tf.image.random_flip_left_right(croped_right_top)

        croped_left_bottom_flip = tf.image.random_flip_left_right(croped_left_bottom)

        croped_right_bottom_flip = tf.image.random_flip_left_right(croped_right_bottom)

        croped_center_flip = tf.image.random_flip_left_right(croped_center)

            
        if savefile:

            normalizeddatapath = os.path.join(self.mainpath, normalizedsubpath)

            normalizeddirlist = os.listdir(normalizeddatapath)

            centercornerdirectory = os.path.join(self.mainpath, centercornersubpath)

            if not os.path.isdir(centercornerdirectory):
                    os.mkdir(centercornerdirectory)

            for normalizedsubdir in normalizeddirlist:
                
                centercornersubdirectory = os.path.join(centercornerdirectory, normalizedsubdir)

                if not os.path.isdir(centercornersubdirectory):
                        os.mkdir(centercornersubdirectory)

                for imagename in os.listdir(os.path.join(normalizeddatapath, normalizedsubdir)):

                    image = cv.imread(os.path.join(normalizeddatapath, normalizedsubdir, imagename))

                    print(image.shape)

                    print(os.path.join(normalizeddatapath, normalizedsubdir, imagename))
                    
                    print(os.path.join(centercornersubdirectory, imagename))
            

                    croped_left_top = tf.image.crop_to_bounding_box(image, 0, 0, 224, 224).numpy()

                    croped_right_top = tf.image.crop_to_bounding_box(image, 0, 32, 224, 224).numpy()

                    croped_left_bottom = tf.image.crop_to_bounding_box(image, 32, 0, 224, 224).numpy()

                    croped_right_bottom = tf.image.crop_to_bounding_box(image, 32, 32, 224, 224).numpy()

                    croped_center = tf.image.central_crop(image, 0.875).numpy()

                    croped_left_top_flip = tf.image.random_flip_left_right(croped_left_top).numpy()

                    croped_right_top_flip = tf.image.random_flip_left_right(croped_right_top).numpy()

                    croped_left_bottom_flip = tf.image.random_flip_left_right(croped_left_bottom).numpy()

                    croped_right_bottom_flip = tf.image.random_flip_left_right(croped_right_bottom).numpy()

                    croped_center_flip = tf.image.random_flip_left_right(croped_center).numpy()

                    croped_left_top_imagepath = os.path.join(centercornersubdirectory, "left_top_" + imagename)

                    croped_right_top_imagepath = os.path.join(centercornersubdirectory, "right_top_" + imagename)

                    croped_left_bottom_imagepath = os.path.join(centercornersubdirectory, "left_bottom_" + imagename)

                    croped_right_bottom_imagepath = os.path.join(centercornersubdirectory, "right_bottom_" + imagename)

                    croped_center_imagepath = os.path.join(centercornersubdirectory, "center_" + imagename)

                    croped_center_flip_imagepath = os.path.join(centercornersubdirectory, "center_flip_" + imagename)

                    croped_left_top_flip_imagepath = os.path.join(centercornersubdirectory, "left_top_flip_" + imagename)

                    croped_right_top_flip_imagepath = os.path.join(centercornersubdirectory, "right_top_flip_" + imagename)

                    croped_left_bottom_flip_imagepath = os.path.join(centercornersubdirectory, "left_bottom_flip_" + imagename)

                    croped_right_bottom_flip_imagepath = os.path.join(centercornersubdirectory, "right_bottom_flip_" + imagename)

                    cv.imwrite(croped_left_top_imagepath, croped_left_top)

                    cv.imwrite(croped_right_top_imagepath, croped_right_top)

                    cv.imwrite(croped_left_bottom_imagepath, croped_left_bottom)

                    cv.imwrite(croped_right_bottom_imagepath, croped_right_bottom)

                    cv.imwrite(croped_center_imagepath, croped_center)

                    cv.imwrite(croped_center_flip_imagepath, croped_center_flip)

                    cv.imwrite(croped_left_top_flip_imagepath, croped_left_top_flip)

                    cv.imwrite(croped_right_top_flip_imagepath, croped_right_top_flip)

                    cv.imwrite(croped_left_bottom_flip_imagepath, croped_left_bottom_flip)

                    cv.imwrite(croped_right_bottom_flip_imagepath, croped_right_bottom_flip)


            return None
        
        return croped_left_top, croped_right_top, croped_left_bottom, croped_right_bottom, croped_center, croped_left_top_flip, croped_right_top_flip, croped_left_bottom_flip, croped_right_bottom_flip, croped_center_flip
        



    def fancy_pca(self, imagefile = None, cropsubpath = None, augmentedsubpath = None, savefile = False):

        """This method is used to apply fancy PCA to the images in the dataset. The method takes the following parameters:
        - imagefile: The image file to be augmented.
        - cropsubpath: The subpath where the cropped images will be saved.
        - augmentedsubpath: The subpath where the augmented images will be saved.
        - savefile: A boolean value to save the augmented images.

        Returns:
        - augmented: The augmented image.

        Example:
        process = Preprocess("/media/HDD/WorkI/AlexNet/Dataset/","Animals90")
        process.fancy_pca("Animals90ResizedImages", "Animals90AugmentedImages")
        """


        original_shape = imagefile.shape
        pixels = imagefile.reshape(-1, 3).astype(np.float32)

        mean = np.mean(pixels, axis=0)
        pixels_centered = pixels - mean


        covariance_matrix = np.cov(pixels_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)


        alpah = np.random.normal(0, 0.1, 3)
        perturbation = np.dot(eigenvectors, alpah * eigenvalues)


        perturbed = pixels + perturbation


        augmented = np.clip(perturbed, 0, 255).reshape(original_shape).astype(np.uint8)


        if savefile:

            cropdatapath = os.path.join(self.mainpath, cropsubpath)

            cropdirlist = os.listdir(cropdatapath)

            augmenteddirectory = os.path.join(self.mainpath, augmentedsubpath)

            if not os.path.isdir(augmenteddirectory):
                    os.mkdir(augmenteddirectory)

            for cropsubdir in cropdirlist:
                
                augmentedsubdirectory = os.path.join(augmenteddirectory, cropsubdir)

                if not os.path.isdir(augmentedsubdirectory):
                        os.mkdir(augmentedsubdirectory)

                for imagename in os.listdir(os.path.join(cropdatapath, cropsubdir)):
                    image = cv.imread(os.path.join(cropdatapath, cropsubdir, imagename))
                    
                    print(os.path.join(augmentedsubdirectory, imagename))


                    original_shape = image.shape
                    pixels = image.reshape(-1, 3).astype(np.float32)


                    mean = np.mean(pixels, axis=0)
                    pixels_centered = pixels - mean

                
                    covariance_matrix = np.cov(pixels_centered, rowvar=False)

                    
                    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    

                    alpah = np.random.normal(0, 0.1, 3)
                    perturbation = np.dot(eigenvectors, alpah * eigenvalues)

                    

                    perturbed = pixels + perturbation

                   
                    augmented = np.clip(perturbed, 0, 255).reshape(original_shape).astype(np.uint8)

                    augmentedimagepath = os.path.join(augmentedsubdirectory, "fancy_pca_" + imagename)


                    cv.imwrite(augmentedimagepath, augmented)
                    cv.imwrite(augmentedimagepath, image)

            return None

        return augmented
    
    

process = Preprocess("/media/HDD/WorkI/AlexNet/Dataset/","Animals90")

#process.image_resizing(256,256, "Animals90ResizedImages")
#process.normalization("Animals90ResizedImages", "Animals90NormalizedImages")
#process.random_crop_with_flip(2, "Animals90ResizedImages", "Animals90RandomCroppedImages")
#process.center_corner_crop_with_flip("Animals90ResizedImages", "Animals90CenterCornerCroppedImages")
#process.fancy_pca("Animals90ResizedImages", "Animals90AugmentedImages")




