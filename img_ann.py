import os

def generate_voc_annotation_file(root: str, ann_file: str = "test.txt", output_file: str = "test_pairs.txt"):
    with open(os.path.join(root, ann_file), 'r') as f:
        lines = [x.strip() for x in f.readlines()]  # ["000005", "000007", ...]

    with open(os.path.join(root, output_file), 'w') as f:
        for line in lines:
            image_path = os.path.join("JPEGImages", f"{line}.jpg")
            target_path = os.path.join("Annotations", f"{line}.xml")
            print(f"{image_path} {target_path}")
            f.write(f"{image_path} {target_path}\n")

# 사용 예시
root = "/media/data/VOCdevkit/VOC2007/ImageSets/Main"
generate_voc_annotation_file(root)