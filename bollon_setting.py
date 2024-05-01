import os.path as osp
import json
import cv2

#---------------json을 coco json 형태로 변환------------
def convert_balloon_to_coco(ann_file, out_file, image_prefix):
  
  #ann_file에 balloon annotation이 들어간다.
  with open(ann_file) as json_file:
    data_infos = json.load(json_file)

  #-----coco dataset의 주요 key값인 annotations정보와 images를 담을 list를 생성한다. 
  annotations = []
  images = []
  obj_count = 0
  
  # 해당 json은 image의 filename+size로 고유 image id를 가짐. 
  # 개별 고유 image id 별로 regions key값으로 object별 segmentation 정보를 polygon으로 가짐
  for idx, v in enumerate(data_infos.values()):
    filename = v['filename']
    # images에 담을 개별 image의 정보를 dict로 생성. 
    img_path = osp.join(image_prefix, filename)
    height, width = cv2.imread(img_path).shape[:2]

    images.append(dict(
        id = idx,
        file_name = filename,
        height = height,
        width = width
    ))
    # annotations 리스트에 담을 bboxes정보와 polygon 정보를 생성한다.
    for _, obj in v['regions'].items():
      assert not obj['region_attributes']
      obj = obj['shape_attributes']
      # polygon 좌표 추출.  
      px = obj['all_points_x']
      py = obj['all_points_y']

      # polygon (x, y) 좌표로 변환. # segmentation 좌표 추출
      poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)] #0.5를 더하는 것은 객체의 경계를 픽셀의 중심에 맞추어 좀 더 정확하고, 실제 시각적으로도 자연스러운 표현을 돕기 위한 일반적인 처리 방법
      # polygon x,y 연속 좌표 list로 변환
      poly = [p for x in poly for p in x] #세그먼테이션 정보를 x, y 좌표의 연속된 리스트로 저장
      
      # boundig box의 x, y, width, height를 segmentation 좌표 기반으로 구하기 위해, 최소/최대 x,y 좌표값을 구함. 
      x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py)) #왼쪽 상단 모서리, 오른쪽 하단 모서리
      # 개별 object의 segmentation 정보와 bbox, image id, 자신의 id 정보를 Dict로 형태. 
      data_anno = dict(
          image_id = idx,
          id=obj_count,
          category_id = 0,
          bbox = [x_min, y_min, x_max - x_min, y_max - y_min],
          area = (x_max - x_min) * (y_max - y_min),
          segmentation = [poly],
          iscrowd = 0
      )
      # 개별 object의 정보를 annotations list에 추가. 
      annotations.append(data_anno)
      obj_count += 1
  # images와 annotations, categories를 Dict형태로 저장. 
  coco_format_json = dict(
      images = images,
      annotations = annotations,
      categories = [{'id':0, 'name':'balloon'}]
  )
  
  # json 파일로 출력. 
  with open(out_file, 'w') as json_out_file:
    json.dump(coco_format_json, json_out_file)

if __name__ == "__main__":
    convert_balloon_to_coco('balloon/train/via_region_data.json', 'balloon/train_coco.json', 'balloon/train')
    convert_balloon_to_coco('balloon/val/via_region_data.json', 'balloon/val_coco.json', 'balloon/val')