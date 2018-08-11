1) Command line application python scripts.
- train.py.......command line app to train a neural net model.
- predict.py.....command line app to predict class of an image
- model_api.py...neural net model API
- data_api.py....data loaders and checkpoints API
- image_api.py...image API

2) Jupyter notebook files.
Image Classifier Project.ipynb
Image Classifier Project.html

3) Helper shell scripts
- train.sh
- predict.sh

4) Train neural net model example usage.

python train.py adjust-learn-rate \
 --data-path=flowers \
 --checkpoint-path=checkpoints \
 --gpu \
 --arch-name=vgg19 \
 --hidden-layers 4096 \
 --output-size=102 \
 --dropout=0.3 \
 --batch-size=96 \
 --max-learning-rate=0.02 \
 --epochs-per-cycle=4 \
 --num-cycles=6 \
 --momentum=0.8 \
 --dump-koeff=0.9

5) Predict class of images example usage.

 python predict.py --image flowers/test/2/image_05100.jpg \
                          --checkpoint checkpoints/best_model.pth \
                          --top-k 5 \
                          --class-names cat_to_name.json \
                          --gpu
