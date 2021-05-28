# Low-Light-Image-Enhancement-using-GAN
In this project, image taken in low lighting conditions, night time, or without much ambient light are converted into and enhanced image as if the image was taken with good lighting condition. Generative Adversarial Networks (GANs) is used to generate the enhanced image from scratch.

<h4>Dataset Used:</h4>
LOw Light Paired (LOL) Dataset - https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view <br/>
Synthetic Image Pairs from Raw Images - https://drive.google.com/file/d/1G6fi9Kiu7CDnW2Sh7UQ5ikvScRv8Q14F/view <br/>
Sony and Fuji low-light Images - https://storage.googleapis.com/isl-datasets/SID/Sony.zip / https://storage.googleapis.com/isl-datasets/SID/Fuji.zip <br/>
Single Image Contrast Enhancement (SICE) - https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view / https://drive.google.com/file/d/16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN/view <br/>
Custom Dataset created by adding noise to Google Scraped Images Dataset - 

<h4>Kaggle Link for Dataset:</h4>https://www.kaggle.com/crawford/emnist

<h4>Link to Published Paper:</h4>https://www.irjet.net/archives/V7/i5/IRJET-V7I5964.pdf

<h4>System Configuration</h4>
<ul>
  <li>Processor: Intel i7 9th gen</li>
  <li>GPU: Nvidia RTX 2060 Mobile</li>
  <li>RAM: 16GB</li>
</ul>

<h4>Softwares used for this project</h4>
<ol>
  <li>Python 3.7</li>
  <li>Tensorflow 2.2.1</li>
</ol>

<h3>Steps to run the project</h3>
  <b>**Change the File paths in the codes to their respective paths before execution.**</b>
<ol>
  <li>Run csv_to_image(emnist).py to convert from csv to image format.</li>
  <li>Run letters(emnist).py to train the CNN model on EMNIST dataset.</li>
  <li>Run bounding box.py to predict to extract text from image.</li>
</ol>

<h3>CNN Architecture used</h3>
<img src="https://github.com/DragonSinMeliodas99/Optical-Character-Recognition/blob/main/Layers.png" alt="CNN layers" width="900"></img>
