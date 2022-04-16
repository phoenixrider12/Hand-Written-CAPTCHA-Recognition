<p align="center">
  <img src="https://user-images.githubusercontent.com/78701055/163567065-86a208ac-d2e5-4dac-96ad-0ecad66649b7.jpg" alt="">
</p>



<h2 align="center"> Introduction : </h2>

CAPTCHAs (Completely Automated Public Turing
Tests to Tell Computers and Humans Apart) are
something that almost every internet user has
encountered. Many users are presented with
strange-looking, stretched, fuzzy, coloured, and shape
distorted visuals that look more like a Dali painting
than English text while signing in or creating an
account, making an online purchase, or even
publishing a comment.

With the growth of Artificial Intelligence, Machine
Learning and Computer Vision, the need for strong
captchas has increased, so that Computer Vision
models are unable to automatically detect these
captchas.

So for the same we bring to you the hand written captcha detector which has a pretrained model
consisting of 19 letters:
#### 'A','D','E','G','H','J','K','M','N','P','R','S','W','X', 'Z', 'a', 'b', 'd', 'g'
and 7 emojis mapped as:

|<img src="https://user-images.githubusercontent.com/78701055/163564439-fd8aa49a-7604-41c0-9c83-55a2b4c2dac8.jpg" alt="" width="100"/> | <img src="https://user-images.githubusercontent.com/78701055/163565375-cb875b9b-9698-45e1-9c3a-48b3cae8fbd3.jpg" alt="" width="100"/> |  <img src="https://user-images.githubusercontent.com/78701055/163565378-e8e768c1-b9db-410f-a30a-e9f0171d4ec4.jpg" alt="" width="100"/> | <img src="https://user-images.githubusercontent.com/78701055/163565383-34e5dd5a-e40c-4346-8f74-4dfefa261027.jpg" alt="" width="100"/> | <img src="https://user-images.githubusercontent.com/78701055/163565391-5f52a4c5-0ae8-463d-8dfc-6df3958f3eee.jpg" alt="" width="100"/> |  <img src="https://user-images.githubusercontent.com/78701055/163565395-d51900b7-d46e-44d9-b188-0058e1e2c4ce.jpg" alt="" width="100"/> | <img src="https://user-images.githubusercontent.com/78701055/163565402-b88c2d4f-9171-44ab-b5ee-b0ba254a2861.jpg" alt="" width="100"/> |
|--|--|--|--|--|--|--|
|Checkmark : 1 | Cloud: 2 | Croissant: 3 | Heart: 4 | Laugh: 5 | Smile: 6 | Sun: 7 |

<br>
<h2 align="left"> Setting up the project : </h2>

1. Creating virtual environment:
```bash
python -m venv venv
```
2. Activate virtual environment

Linux:
```bash
source venv/bin/activate
```
Windows:
```cmd
./venv/Scripts/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Install the package: 
```bash
pip install -e .
```
5. Move to examples folder: 
```bash
cd examples
```
6. Run main.py : 
```bash
python main.py
```
<br>
<h2 align="left"> How to use : </h2>

<p align="center">
<img src="https://user-images.githubusercontent.com/78701055/163561542-863fc2c4-2f58-467d-ab4a-f1887db78f0e.png" alt="" width="70%">
<img src="https://user-images.githubusercontent.com/78701055/163562737-e6e29827-aa61-4dd8-804a-906d5abf8ab0.png" alt="" width="25%">
</p>

After setting up project just place your images in `line` folder under the `data` directory and run the `main.py` file again. The program will show you images and its output in terminal one by one. If you think that your images is not recognised clearly by program just change the `threshold value` according to your needs for best output.

NOTE: The model is trained for the above mentioned letters so give them only in input images for best output.

<br>
<h3 align="left"> Made and maintained by : </h3>

|<img src="https://avatars.githubusercontent.com/u/78701055?v=4" alt="drawing" width="150"/> | <img src="https://avatars.githubusercontent.com/u/76533398?v=4" alt="drawing" width="150"/> | <img src="https://avatars.githubusercontent.com/u/75940729?v=4" alt="drawing" width="150"/> | 
|--|--|--|
|[Ankur Agarwal](https://github.com/Ankur-Agrawal-ece20) |[Aryaman Gupta](https://github.com/phoenixrider12) |[Vivek Agrawal](https://github.com/vivekagarwal2349) |
