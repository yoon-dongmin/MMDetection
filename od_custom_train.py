import copy
import os.path as osp
import cv2
import mmcv
from mmcv import Config
from mmdet.apis import set_random_seed, train_detector
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

# 반드시 아래 Decorator 설정 할것.@DATASETS.register_module() 설정 시 force=True를 입력하지 않으면 Dataset 재등록 불가. 
@DATASETS.register_module(force=True)
class KittyTinyDataset(CustomDataset):
  CLASSES = ('Car', 'Truck', 'Pedestrian', 'Cyclist')
  
  ##### self.data_root: /content/kitti_tiny/ self.ann_file: /content/kitti_tiny/train.txt self.img_prefix: /content/kitti_tiny/training/image_2
  #### ann_file: /content/kitti_tiny/train.txt
  # annotation에 대한 모든 파일명을 가지고 있는 텍스트 파일을 __init__(self, ann_file)로 입력 받고, 이 self.ann_file이 load_annotations()의 인자로 입력
  def load_annotations(self, ann_file):
    print('##### self.data_root:', self.data_root, 'self.ann_file:', self.ann_file, 'self.img_prefix:', self.img_prefix)
    print('#### ann_file:', ann_file)
    cat2label = {k:i for i, k in enumerate(self.CLASSES)}
    image_list = mmcv.list_from_file(self.ann_file)
    # 포맷 중립 데이터를 담을 list 객체
    data_infos = []
    
    for image_id in image_list:
      filename = '{0:}/{1:}.jpeg'.format(self.img_prefix, image_id)
      # 원본 이미지의 너비, 높이를 image를 직접 로드하여 구함. 
      image = cv2.imread(filename)
      height, width = image.shape[:2]
      # 개별 image의 annotation 정보 저장용 Dict 생성. key값 filename 에는 image의 파일명만 들어감(디렉토리는 제외)
      data_info = {'filename': str(image_id) + '.jpeg',
                   'width': width, 'height': height}
      # 개별 annotation이 있는 서브 디렉토리의 prefix 변환. 
      label_prefix = self.img_prefix.replace('image_2', 'label_2')
      # 개별 annotation 파일을 1개 line 씩 읽어서 list 로드 
      lines = mmcv.list_from_file(osp.join(label_prefix, str(image_id)+'.txt'))

      # 전체 lines를 개별 line별 공백 레벨로 parsing 하여 다시 list로 저장. content는 list의 list형태임.
      # ann 정보는 numpy array로 저장되나 텍스트 처리나 데이터 가공이 list 가 편하므로 일차적으로 list로 변환 수행.   
      content = [line.strip().split(' ') for line in lines]
      # 오브젝트의 클래스명은 bbox_names로 저장. 
      bbox_names = [x[0] for x in content]
      # bbox 좌표를 저장
      bboxes = [ [float(info) for info in x[4:8]] for x in content]

      # 클래스명이 해당 사항이 없는 대상 Filtering out, 'DontCare'sms ignore로 별도 저장.
      gt_bboxes = []
      gt_labels = []
      gt_bboxes_ignore = []
      gt_labels_ignore = []

      for bbox_name, bbox in zip(bbox_names, bboxes):
        # 만약 bbox_name이 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가, 그렇지 않으면 gt_bboxes_ignore, gt_labels_ignore에 추가
        if bbox_name in cat2label:
          gt_bboxes.append(bbox)
          # gt_labels에는 class id를 입력
          gt_labels.append(cat2label[bbox_name])
        else:
          gt_bboxes_ignore.append(bbox)
          gt_labels_ignore.append(-1)
      # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값은 모두 np.array임. 
      data_anno = {
          'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
          'labels': np.array(gt_labels, dtype=np.long),
          'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
          'labels_ignore': np.array(gt_labels_ignore, dtype=np.long)
      }
      # image에 대한 메타 정보를 가지는 data_info Dict에 'ann' key값으로 data_anno를 value로 저장. 
      data_info.update(ann=data_anno)
      # 전체 annotation 파일들에 대한 정보를 가지는 data_infos에 data_info Dict를 추가
      data_infos.append(data_info)

    return data_infos
  
if __name__ == "__main__":
    ### Config 설정하고 Pretrained 모델 다운로드
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    cfg = Config.fromfile(config_file)
    # dataset에 대한 환경 파라미터 수정.
    cfg.dataset_type = 'KittyTinyDataset'
    cfg.data_root = 'kitti_tiny'

    # train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정.
    cfg.data.train.type = 'KittyTinyDataset'
    cfg.data.train.data_root = 'kitti_tiny'
    cfg.data.train.ann_file = 'train.txt'
    cfg.data.train.img_prefix = 'training/image_2'

    cfg.data.val.type = 'KittyTinyDataset'
    cfg.data.val.data_root = 'kitti_tiny/'
    cfg.data.val.ann_file = 'val.txt'
    cfg.data.val.img_prefix = 'training/image_2'

    cfg.data.test.type = 'KittyTinyDataset'
    cfg.data.test.data_root = 'kitti_tiny'
    cfg.data.test.ann_file = 'val.txt'
    cfg.data.test.img_prefix = 'training/image_2'

    # class의 갯수 수정.
    cfg.model.roi_head.bbox_head.num_classes = 4
    
    # pretrained 모델
    cfg.load_from = checkpoint_file

    # 학습 weight 파일로 로그를 저장하기 위한 디렉토리 설정.
    cfg.work_dir = './weights'

    # 학습율 변경 환경 파라미터 설정.
    cfg.optimizer.lr = 0.02 / 8

    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # config 수행 시마다 policy값이 없어지는 bug로 인하여 설정.
    cfg.lr_config.policy = 'step'

    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'
    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 12
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 12

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # ConfigDict' object has no attribute 'device 오류 발생시 반드시 설정 필요. https://github.com/open-mmlab/mmdetection/issues/7901
    cfg.device='cuda'


    # We can initialize the logger for training and have a look
    # at the final config used for training
    #print(f'Config:\n{cfg.pretty_text}')

    # train용 Dataset 생성.
    datasets = [build_dataset(cfg.data.train)]

    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # epochs는 config의 runner 파라미터로 지정됨. 기본 12회
    train_detector(model, datasets, cfg, distributed=False, validate=True)