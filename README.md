 
![](https://img.shields.io/static/v1?label=&message=Domain/Writer-Adaptation&color=green&size=30)

# Description

This is an implementation of the work presented in paper "Domain and writer adaptation of offline Arabic handwriting recognition using deep neural networks".
Arabic Handwritten Text Recognition (AHTR) based on deep learning approaches remains a challenging problem due to the inevitable domain shift like the variability among writers’ styles and the scarcity of labelled data. To alleviate such problems, we investigate in this paper different domain adaptation strategies of AHTR system. The main idea is to exploit the knowledge of a handwriting source domain and to transfer this knowledge to another domain where only few labelled data are available. Different writer-dependent and writer-independent domain adaptation strategies are explored using a convolutional neural networks (CNN) and Bidirectional Long Short Term Memory (BSTM) - connectionist temporal classification (CTC) architecture. To discuss the interest of the proposed techniques on the target domain, we have conducted extensive experiments using three Arabic handwritten text datasets, mainly, the MADCAT, the AHTID/MW and the IFN/ENIT. Concurrently, the Arabic handwritten text dataset KHATT was used as the source domain.  


![image](https://user-images.githubusercontent.com/15616524/148753690-fbbc4fb7-1349-4095-a57c-3e2cf9aea17d.png)



## License
This work is only allowed for academic research use. For commercial use, please contact the author.


### To test the global adaptation using the MADCAT dataset, run the following code

python main.py

#### Experimental results of the writer adaptation strategies carried on the MADCAT dataset for 15 writers (Test samples per writer : 150, total test samples : 2250)

![image](https://user-images.githubusercontent.com/15616524/148755225-f648a36b-b128-4728-ab2d-a11a3988321d.png)

#### An overview of the proposed two-stage fine-tuning process for the writer adaptation task
![](https://img.shields.io/static/v1?label=&message=The-Two-Stage-HTR-Architecture&color=orange&size=30)

![image](https://user-images.githubusercontent.com/15616524/148754875-9ddd7455-b0be-46c8-9be4-b60daf1390ae.png)
