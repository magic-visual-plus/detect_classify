import sys
import os
import cv2
from tqdm import tqdm
from hq_det.models.dino import hq_dino


if __name__ == '__main__':
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    os.makedirs(output_path, exist_ok=True)
    
    model = hq_dino.HQDINO(model=sys.argv[1])
    model.eval()
    
    model.to("cuda:0")

    filenames = os.listdir(input_path)

    filenames = [os.path.join(input_path, f) for f in filenames if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(filenames)

    for filename in tqdm(filenames):
        img = cv2.imread(filename)

        results = model.predict([img], bgr=True, confidence=0.3, max_size=1536)

        result = results[0]

        print(len(result.bboxes))
        for bbox in result.bboxes:
            img = cv2.rectangle(
                img.copy(),
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 255, 0),
                2
            )
            pass
        save_path = os.path.abspath(os.path.join(output_path, os.path.basename(filename)))
        cv2.imwrite(save_path, img)
        pass
    pass