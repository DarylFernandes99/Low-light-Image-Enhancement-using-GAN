# Low-Light-Image-Enhancement-using-GAN
In this project, image taken in low lighting conditions, night time, or without much ambient light are converted into an enhanced image as if the image was taken under better lighting condition. Generative Adversarial Networks (GANs) is used to generate the enhanced image from scratch.

<h4>Datasets Used:</h4>
LOw Light Paired (LOL) Dataset (<a href="https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB/view">Link</a>) <br/>
Synthetic Image Pairs from Raw Images (<a href="https://drive.google.com/file/d/1G6fi9Kiu7CDnW2Sh7UQ5ikvScRv8Q14F/view">Link</a>) <br/>
Sony (<a href="https://storage.googleapis.com/isl-datasets/SID/Sony.zip">Link</a>) and Fuji (<a href="https://storage.googleapis.com/isl-datasets/SID/Fuji.zip ">Link</a>) low-light Images<br/>
Single Image Contrast Enhancement (SICE) - Part1: <a href="https://drive.google.com/file/d/1HiLtYiyT9R7dR9DRTLRlUUrAicC4zzWN/view">Link</a> | Part2: <a href="https://drive.google.com/file/d/16VoHNPAZ5Js19zspjFOsKiGRrfkDgHoN/view">Link</a> <br/>
Custom Dataset created by adding noise to Google Scraped Images Dataset (<a href="https://www.kaggle.com/basu369victor/low-light-image-enhancement-with-cnn">Link</a>)

<h4>Link to Published Paper:</h4> https://www.irjet.net/archives/V8/i6/IRJET-V8I6136.pdf

<h4>System Configuration</h4>
<ul>
  <li>GPU: Nvidia Tesla T4 16GB / Nvidia Tesla P100 16GB</li>
  <li>RAM: 12GB</li>
</ul>

<h4>Softwares used for this project</h4>
<ol>
  <li>Python 3.7</li>
  <li>Tensorflow 2.4.1</li>
</ol>

<h3>Steps to run the project</h3>
<ol>
  <li><h4>Convert images from png/jpeg/jpg to npz format (png2npz.py)</h4></li>
    <ul>
      <li>Change the path variable to desired path</li>
      <li>Execute the code</li>
      <li>Upload the npz file on drive for training model on colab</li>
    </ul>
  <li><h4>Train model on Colab (Low-Light Image Enhancemnet using GAN.ipynb)</h4></li>
    <ul>
      <li>Save the converted dataset on drive</li>
      <li>Change the drive path to the path of dataset in drive</li>
      <li>Change the path to load dataset from colab in "Loading Dataset" cell</li>
      <li>Change the path for saving model under "Creating Model Driectory" cell</li>
      <li>Change the "batch" variable passed to train function (default batch size is 12)</li>
      <li>Execute all the cells in order, the model is summarized at regular intervals</li>
    </ul>
  <li><h4>Testing our model (main.py)</h4></li>
    <ul>
      <li>Add path of model and input image in variables model_path and image_path respectively</li>
      <li>Execute the main.py file</li>
      <li>The ouput will be stored as "output.png" in the current working directory</li>
    </ul>
</ol>

<h3>GAN Architecture used</h3>
<ul>
  <li><h4>Generator</h4></li>
  <ul>
    <li><h5>Encoder</h5></li>
      <img src="../main/Images/Encoder.JPG" alt="CNN layers" width="700"></img>
    <li><h5>Decoder</h5></li>
      <img src="../main/Images/Decoder.JPG" alt="CNN layers" width="700"></img>
  </ul>
  <li><h4>Discriminator</h4></li>
    <img src="../main/Images/Discriminator.JPG" alt="CNN layers" width="700"></img>
</ul>
