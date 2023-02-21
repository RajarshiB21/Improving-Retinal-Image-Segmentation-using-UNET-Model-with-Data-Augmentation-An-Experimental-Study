# Improving-Retinal-Image-Segmentation-using-UNET-Model-with-Data-Augmentation-An-Experimental-Study
Improving Retinal Image Segmentation using UNET Model with Data Augmentation: An Experimental Study


  Retinal Vessel Segmentation with U-NET
Debkumar Chowdhury1*, Rajarshi Banerjee1,  Kaustuv Ghosh 1, Arnab Kumar Dey1,  
Shreyas Saha1, Sayak Sil 1and Sayan Rakshit 1  
1University of Engineering and  
Management, Kolkata, India 
1debkumar.cse@gmail.com,1kaustavghosh35@gmail.com 
,1shreyassaha4@gmail.com ,1arnabkdey2000@gmail.com 
1banerjeerajarshi21@gmail.com 
,1sayaksundar@gmail.com ,1sayan10rakshit@gmail.com 
Abstract. Accurate segmentation of the Retinal vasculature is vital for diagnosing of many major diseases. Currently, doctors determine the cause of Retinopathy primarily by diagnosing fundus images. Large-scale manual screening is difficult to achieve for retinal health screen. Retinal vessel segmentation is a fundamental step for various ocular imaging applications .In this paper, we formulate the retinal vessel segmentation problem as a boundary detection task and solve it using a novel deep learning architecture .Firstly, due to the lack of retinal data, pre-processing of the raw data is required. The data processed by grayscale transformation, normalization. Data augmentation can prevent over fitting in the training process. Secondly, the basic network structure model U-net is built .Datasets from a public challenge are used to evaluate the performance of the proposed method, which is able to detect vessel F1 of 0.7967, Recall of 0.7848, Precision of 0.8138 , Accuracy of 0.9652.
Keywords:  Retinal vessel segmentation , U-net , Deep learning , Segmentation
1 Introduction 
Retinal image segmentation is a critical task in the medical field, as it allows for the identification of various retinal pathologies and aids in the diagnosis and treatment of diseases such as diabetic retinopathy, glaucoma, and macular degeneration.
The goal of this research is to investigate whether the performance of the UNET model for retinal image segmentation can be improved through the use of data augmentation. Specifically, we aim to explore whether augmenting the training data can enhance the model's ability to accurately segment retinal structures and improve the quality of the segmentation results.To accomplish this, we used the DRIVE dataset, which consists of 40 color fundus images of the retina, each with a corresponding manually segmented image. We divided the dataset such that there are 20 training images and 20 validation images.. We employed the UNET architecture, a convolutional neural network designed for image segmentation, to train the model on the retinal images.
We applied two different data augmentation techniques, namely rotation, horizontal flipping and vertical flipping to the training images to increase the diversity of the data and improve the model's robustness resulting in 80 training images. During training, we used two loss functions, DiceLoss and DiceBCELoss, to optimize the model's performance.
The evaluation of the segmentation results was based on multiple metrics, including f1 score, jaccard score, precision, recall, and accuracy. These metrics were used to assess the performance of the model and to compare the results obtained using different techniques, such as with and without data augmentation.

2 Literature Survey 
Retinal image segmentation is a critical task in medical image analysis, and numerous studies have investigated the use of various deep learning models for this purpose. Among these models, the UNET architecture has gained considerable attention due to its effectiveness in segmenting medical images. For instance, Zhang et al. (2020) used a modified UNET model for segmenting the optic disc and cup in retinal images, achieving high segmentation accuracy. Similarly, Sinha et al. (2018) employed a UNET-based model for the detection of diabetic retinopathy, obtaining high accuracy and sensitivity.
In recent years, data augmentation techniques have also gained popularity in medical image segmentation. Augmenting the training data can increase the diversity of the dataset, improve the robustness of the model, and prevent overfitting. Various augmentation techniques have been proposed, including rotation, horizontal and vertical flipping. For instance, Lu et al. (2018) used data augmentation techniques, such as rotation and flipping, to improve the segmentation accuracy of the UNET model in retinal images.
DiceLoss and DiceBCELoss are commonly used loss functions in image segmentation tasks, and both have been shown to perform well in the segmentation of retinal images. For example, Li et al. (2020) used the DiceLoss function to train a UNET-based model for segmenting the optic disc and macula in retinal images. They obtained a high dice coefficient and sensitivity, demonstrating the effectiveness of the loss function in this application.
Finally, several evaluation metrics have been proposed for assessing the performance of segmentation models in retinal images. These metrics include f1 score, jaccard score, precision, recall, and accuracy. For instance, Zhang et al. (2020) used the f1 score, precision, and recall to evaluate the performance of their UNET-based model for optic disc and cup segmentation, while Li et al. (2020) employed the jaccard index and dice coefficient to assess the accuracy of their model for macula and optic disc segmentation.
In summary, the literature survey highlights the effectiveness of the UNET model and data augmentation techniques in retinal image segmentation, as well as the relevance of the loss functions and evaluation metrics used in this study. The proposed research builds on this existing knowledge and aims to investigate the impact of data augmentation on the performance of the UNET model for retinal image segmentation.

3 Methodology

In this section, our proposed segmentation method will be described in detail. The preprocessing includes normalization. To increase the number of training samples, we use data augmentation techniques such as Rotation, Horizontal Flipping and Vertical Flipping to increase the number of training samples from 20 to 80. The validation dataset contains 20 images.

3.1 Dataset : 

The Digital Retinal Images for Vessel Extraction (DRIVE) dataset (https://paperswithcode.com/dataset/drive) is a dataset for retinal vessel segmentation. It consists of a total of JPEG 40 color fundus images; including 7 abnormal pathology cases. The images were obtained from a diabetic retinopathy screening program in the Netherlands. The images were acquired using Canon CR5 non-mydriatic 3CCD camera with FOV equals to 45 degrees. Each image resolution is 584*565 pixels with eight bits per color channel (3 channels).
The set of 40 images was equally divided into 20 images for the training set and 20 images for the validation set. Inside both sets, for each image, there is circular field of view (FOV) mask of diameter that is approximately 540 pixels. Inside training set, for each image, one manual segmentation by an ophthalmological expert has been applied. Inside testing set, for each image, two manual segmentations have been applied by two different observers, where the first observer segmentation is accepted as the ground-truth for performance evaluation.
 


3.2 Pre-processing :

Fundus images have disadvantages such as uneven brightness, poor contrast, and strong noise, requiring per-processing before input the network for training. Figures 1(A) and 1(B) show the original fundus image and manually labeled vascular map of the retinal vessels.















Figure 1. Original fundus image (A) and Manually labeled vascular map of the retinal vessels (B).


                  3.3 Image Enhancement:

The DRIVE dataset was used to train and test the UNET model for retinal image segmentation. The dataset consists of 40 color fundus images of size 565 x 584 pixels, with a resolution of 8 bits per color channel. The images were preprocessed by normalizing the pixel values to [0,1]. The images were then divided into 80 training images and 20 validation images.

To increase the size of the training dataset and to prevent overfitting, various data augmentation techniques were applied to the training images. These techniques included rotation, horizontal and vertical flipping. The augmented images were then used for training the model.


3.4 U-NET :

The U-net consists of an input layer, a hidden layer, and an output layer. Its structure is shown in Figure 3.The hidden layer can be divided into an up-sampling and a down-sampling part, distinguishing decoders and encoders. Down-sampling consists of convolution layer and pooling layer, which play the role of path shrinking in the network and capture global information. The up-sampling consists of convolutional and deconvolution  layers, which play the role of path expansion in the network and locate pixel points. The output layer is an end-to-end network that classifies each pixel point of the feature map with the same size as the original image after up-sampling by the soft max activation function. That is, the input image is of the same size as the output image.













                                                            Figure 3 : U-NET Structure

The UNET model architecture was implemented using PyTorch, with 3 encoding layers and 3 decoding layers. Each encoding layer consists of two 3x3 convolutional layers with batch normalization and ReLU activation, followed by a max-pooling layer. Each decoding layer consists of an upsampling layer, followed by two 3x3 convolutional layers with batch normalization and ReLU activation. Skip connections were added between corresponding encoding and decoding layers to help preserve spatial information.



4 Experiment

4.1 Experimental Process:

In this study, we used two loss functions for training our UNET model: DiceLoss and DiceBCELoss.
DiceLoss is a loss function commonly used in segmentation tasks that measures the similarity between two sets of pixels. Specifically, it calculates the overlap between the predicted segmentation mask and the ground truth mask. The formula for DiceLoss is as follows:
DiceLoss = 1 - (2 * intersection) / (prediction + ground truth)               (1)
Here, intersection represents the number of pixels that are common in both the prediction and ground truth masks, and prediction and ground truth represent the total number of pixels in their respective masks. The DiceLoss function produces a value between 0 and 1, where 1 indicates perfect overlap between the masks.
DiceBCELoss is a combination of DiceLoss and binary cross-entropy (BCE) loss. BCE loss is a standard loss function used in binary classification tasks that measures the difference between the predicted probabilities and the actual binary labels. The formula for BCE loss is as follows:
BCELoss = -(y * log(p) + (1 - y) * log(1 - p))                                 (2)
Here, y represents the binary label (0 or 1), p represents the predicted probability, and log represents the natural logarithm. BCE loss produces a value between 0 and infinity, where 0 indicates perfect classification accuracy.
The DiceBCELoss function combines these two loss functions by adding them together, and it is used to train the UNET model in this study. The formula for DiceBCELoss is as follows:
DiceBCELoss = DiceLoss + BCELoss                                                (3)
By using these two loss functions, we aim to optimize the model to produce accurate segmentation masks that closely match the ground truth masks.


4.2 Evaluation Methodology :

Experiments were conducted to analyze and compare the performance of the segmentation algorithm proposed in this paper with other algorithms using evaluation indexes such as accuracy, recall and precision, which are defined as follows:

Accuracy=TP + TNTP + TN + FP + FN                                                                   (4)
Precision=TPTP + FP                                                                                 (5)
Recall=TPTP + FN                                                                                       (6)

TP (True Positive), FP (False Positive), FN (False Negative), TN (True Negative).

As shown in Eq. (4), (5), and (6), we can see the relationship between evaluation index and TP、FP、FN、TN.







5 Conclusion

In this study, we trained a UNET model with and without data augmentation to perform retinal image segmentation on the DRIVE dataset. The following quantitative results were obtained:
With data augmentation, the UNET model achieved an f1 score of 0.7947 jaccard score of 0.6597 precision of 0.8207, recall of 0.7749, and accuracy of 0.9653 on the validation set.

Without data augmentation, the UNET model achieved an f1 score of 0.7775, jaccard score of 0.6364, precision of 0.8206, recall of 0.7441, and accuracy of 0.9631 on the validation set.
These results suggest that data augmentation can significantly improve the performance of the UNET model for retinal image segmentation.


To visually evaluate the performance of the UNET model, we provide examples of segmentation results on the validation set in Figure 1. The figure shows that the model with data augmentation is able to segment the retinal blood vessels and optic disc with high accuracy.


   


















Figure 1. Original label (A), With data augmentation(B) and (C) Without data augmentation


Figure 1: Segmentation results on the validation set. From left to right, the first picture shows the original segmentation or the ground truth, the middle picture shows the segmentation masks predicted by the UNET with data augmentation, and the bottom row shows the segmentation masks predicted by the UNET model without data augmentation.

During the experiments, we generated several graphs to analyze the performance of the UNET model. Figure 2 shows the training loss graph, which indicates that the model converges after around 75 epochs for the model with data augmentation and around 175 for the model without data augmentation.


             

                        (A)                                                                         (B)

Figure 2. Model with data augmentation (A) and (B) Model without data augmenta

Figure 3. Jaccard vs F1 score

Figure 3 shows the jaccard vs f1 graph, which indicates a positive correlation between these two metrics. 

In conclusion, this research explored the effect of data augmentation on the performance of the UNET model for retinal image segmentation. The results demonstrated that the model achieved higher values for f1 score, jaccard score, precision, recall, and accuracy on the validation set when data augmentation was applied. The f1 score improved from 0.7775 to 0.7947, demonstrating the potential of data augmentation to improve the model's performance. The visual examples of the segmentation results on the validation set further support the effectiveness of data augmentation.

The significance of this work lies in the potential to improve the accuracy and reliability of retinal image segmentation, which can aid in the early detection of various eye diseases. This can help doctors and medical professionals make more informed decisions about patient care and treatment.

For future research, it would be interesting to explore other data augmentation techniques and their impact on the UNET model's performance. It would also be valuable to compare the UNET model's performance with other segmentation models to determine which is the most effective for retinal image segmentation. Additionally, more extensive datasets can be used to train and evaluate the model, potentially leading to even better segmentation results.

             6 References 

T.Y. Wong, R. Klein, A.R. Sharrett, et al.
Retinal arteriolar diameter and risk for hypertension[J]
Ann. Intern. Med., 140 (4) (2004), pp. 248-255

S. Chaudhuri, S. Chatterjee, N. Katz, et al.
Detection of blood vessels in retinal images using two-dimensional matcher filters[J]
IEEE Trans. Med. Imag., 8 (3) (1989), pp. 263-269

G. Azzopardi, N. Strisciuglio, M. Vento, et al.
Traniable COSFIRE filters for vessel delineation with application in retinal images[J]
Med. Image Anal., 19 (1) (2015), pp. 46-57

J. Zhang, B. Dashtbozorg, E. Bekkers
Robust Retinal Vessel Segmentation via Locally Adaptive Derivative Frames in Orientation Scores[J]
IEEE Transactions on Medical Imaging, 35 (12) (2016), pp. 2631-2644

A. Salazar-Gonzalez, D. Kaba, Y. Li
Segmentation of blood vessels and optic disc in retinal images[J]
IEEE Journal of Bio-medical and Health Informatics, 18 (6) (2014), pp. 1874-1886

M.M. Fraz, S.A. Barman, P. Remagnino, et al.
An approach to localize the retinal blood vessels using bit planes and centerline detection[J]
Comput. Methods Progr. Biomed., 108 (2) (2012), pp. 600-616

D. Marin, M.E. Gegundez-Arias, B. Ponte, et al.
An exudate detection method for diagnosis risk of diabetic macular edema in retinal images using feature-based and supervised classification[J]
Med. Biol. Eng. Comput., 56 (8) (2018), pp. 1379-1390

E. Ricci, R. Perfetti
Retinal blood vessel segmentation using line operators and support vector classification[J]
IEEE Trans. Med. Imag., 26 (10) (2007), pp. 1357-1365

R. Sohini, D.D. Koozekanani, K.K. Parhi
Blood vessel segmentation of fundus images by major vessel extraction and subimage classification[J]
IEEE Journal of Biomedical and Health Informatics, 19 (3) (2015), pp. 1118-1128

A. Oliveira, S. Pereira, C.A. Silva
Retinal vessel segmentation based on fully convolutional neural networks [J]
Expert Syst. Appl., 112 (2018), pp. 229-242

J. Staal, M.D. Abramoff, NIEMEIJER, et al.
Ridge-based vessel segmentation in color images of the retinal[J]
IEEE Trans. Med. Imag., 23 (4) (2004), pp. 501-509

A.A. Nahid, M.A. Mehrabi, Y.N. Kong
Histopathoiogical Breast Cancer Image Classification by Deep Neural Network Techniques Guided by Local clustering[J], 2018, BidMed Research International (2018),

F. Zana, J.C. Klein
Segmentation of vessel-like petterns using mathematical morphology and curvature evaluation[J]
IEEE Trans. Image Process., 10 (7) (2001), pp. 1010-1019

B. Al-Diri, A. Hunter, D. Steel
An active contour model for segmenting and measuring retinal vessels[J]
IEEE Trans. Med. Imag., 28 (9) (2009), pp. 1488-1497

M.S. Miri, A. Mahloojifar
Retinal image analysis using curvelet transform and multistructure elements morphology by reconstruction[J]
IEEE (Inst. Electr. Electron. Eng.) Trans. Biomed. Eng., 58 (5) (2011), pp. 1183-1192

M.M. Fraz, S.A. Barman, P. Remagnino, et al.
An approach to localize the retinal blood vessels using bit planes and centerline detection[J]
Comput. Methods Progr. Biomed., 108 (2) (2012), pp. 600-616

X.G. You, Q.M. Peng, Y. Yuan, et al.
Segmentation of retinal blood vessels using the radial projection and semi-supervised approach[J]
Pattern Recogn., 44 (10-11) (2011), pp. 2314-2324

M.M. Fraz, P. Remagnino, A. Hoppe, et al.
Blood vessel segmentation methodologies in retinal images: a survey[J]
Comput. Methods Progr. Biomed., 108 (1) (2012), pp. 407-433

D. Marin, M.E. Gegundez-Arias, B. Ponte, et al.
An exudate detection method for diagnosis risk of diabetic macular edema in retinal images using feature-based and supervised classification[J]
Med. Biol. Eng. Comput., 56 (8) (2018), pp. 1379-1390

E. Ricci, R. Perfetti
Retinal blood vessel segmentation using line operators and support vector classification[J]
IEEE Trans. Med. Imag., 26 (10) (2007), pp. 1357-1365

R. Sohini, D.D. Koozekanani, K.K. Parhi
Blood vessel segmentation of fundus images by major vessel extraction and subimage classification[J]
IEEE Journal of Biomedical and Health Informatics, 19 (3) (2015), pp. 1118-1128

A. Oliveira, S. Pereira, C.A. Silva
Retinal vessel segmentation based on fully convolutional neural networks[J]
Expert Syst. Appl., 112 (2018), pp. 229-242

Alom M Z, Hasan M, Yakopcic C, et al. Recurrent Residual Convolutional Neural Network Based on U-Net (R2U-Net) for medical image segmentation [DB/OL].

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions. CVPR
(2017), pp. 1610-2357

Sifre Laurent
Rigid-motion scattering for image classification
Ph.D. thesis section, 6 (2) (2014)
