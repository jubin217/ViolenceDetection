
- Clone this repository and change directory to this repository

```bash
git clone https://github.com/jubin217/ViolenceDetection.git
cd ViolenceDetection
```

- Download Inbuilt Keras VGG16 model by running the following script:

```bash
python download_imagenet.py
```

- Now download the pretrained model H5 file from [here](https://drive.google.com/drive/folders/1SYD0dbfOLRBcidaACw5aiQsbtN4ySXLh?usp=sharing) and put it inside the model folder so it'll be "model/vlstm_92.h5".

- Install Dependencies

 ```bash
 pip install -r requirements.txt
 ```

- to run inference on a video existing in 'data' folder run following command:

```bash
python infer.py <Video name existing in data folder>
# Example
python infer.py fi3_xvid.avi
```

- to run inference live:

```bash
python infer_cam.py
```
