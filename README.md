<h1>Use CNN to recognize gender</h1>
<p>
    <h2>Requirements</h2>   
    <ul class="mylist">
        <li>Datasets of image</li>
        <li>Feature extractor : haarcascade_frontalface_default.xml</li>
    </ul>
    <h2>Lab1 : CNN_gender_train.py</h2>   
    <ul class="mylist">
        <li>Use opencv feature extractor to detect face (Input)</li>
        <li>Model structure (reference pic)</li>
            <img src="https://ujwlkarn.files.wordpress.com/2016/08/screen-shot-2016-08-07-at-9-15-21-pm.png?w=748">
        <li>Keep training until confidence level reach 90%</li>
    </ul>
    <h2>Lab2 : CNN_gender_test.py</h2>   
    <ul class="mylist">
        <li>Loading model from lab1</li>
        <li>Take a selfie to demo</li>
            <img src="https://i.imgur.com/WpHLnpz.png">
        <li>If recongnize incorrect, force model to learn the image</li>
    </ul>
</p>