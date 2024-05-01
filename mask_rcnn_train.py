import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp

@DATASETS.register_module(force=True)
class BalloonDataset(CocoDataset):
    #단일 class일때 class명 마지막에 ,쓰는 것 유의
    CLASSES = ('balloon', )


if __name__ == "__main__":
    config_file = 'configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'

    cfg = Config.fromfile(config_file)

    # dataset에 대한 환경 파라미터 수정. 
    cfg.dataset_type = 'BalloonDataset'
    cfg.data_root = 'balloon'

    # train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정. 
    cfg.data.train.type = 'BalloonDataset'
    cfg.data.train.data_root = 'balloon'
    cfg.data.train.ann_file = 'train_coco.json'
    cfg.data.train.img_prefix = 'train'

    cfg.data.val.type = 'BalloonDataset'
    cfg.data.val.data_root = 'balloon'
    cfg.data.val.ann_file = 'val_coco.json'
    cfg.data.val.img_prefix = 'val'


    # class의 갯수 수정. balloon 단일 class
    cfg.model.roi_head.bbox_head.num_classes = 1
    cfg.model.roi_head.mask_head.num_classes = 1

    # pretrained 모델
    cfg.load_from = 'checkpoints/mask_rcnn_r101_fpn_1x_coco_20200204-1efe0ed5.pth'

    # 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정. 
    cfg.work_dir = 'sg_weights'

    # 학습율 변경 환경 파라미터 설정. 
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # CocoDataset의 경우 metric을 bbox로 설정해야 함.
    #bbox로 설정시 IOU 값을 0.5~0.95까지 변경하면서 측정
    cfg.evaluation.metric = ['bbox', 'segm']
    cfg.evaluation.interval = 12
    cfg.checkpoint_config.interval = 12

    # epochs 횟수는 36
    cfg.runner.max_epochs = 36

    cfg.lr_config.policy='step'
    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device='cuda'
    #dataset 생성
    datasets = [build_dataset(cfg.data.train)]
    #model 설정
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.CLASSES = datasets[0].CLASSES
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    #학습 진행
    train_detector(model, datasets, cfg, distributed=False, validate=True)