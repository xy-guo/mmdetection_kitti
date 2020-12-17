# python convert_cityscapes_to_coco.py  --dataset cityscapes_instance_only --datadir ~/data/cityscapes/ --outdir ../data/cityscapes/annotations/
python tools/convert_datasets/cityscapes.py --img-dir leftImg8bit/ --gt-dir gtFine_trainvaltest/gtFine -o data/cityscapes/annotations/ data/cityscapes/ 
