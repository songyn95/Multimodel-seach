# coding=utf-8

import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import numpy as np
import pandas as pd
from pathlib import Path
import os
from PIL import Image
import torch
from torchvision.utils import make_grid
from torchvision.io import read_image
from matplotlib import pyplot as plt

features_path = Path("features").absolute()
# Load the features and the corresponding IDs
photo_features = np.load(features_path / "features.npy")
photo_ids = pd.read_csv(features_path / "photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])


# search_query = "小学生在排队"
def search_(search_query, top_k, models, top_p, search_text=False, search_img=False):
    assert search_text or search_img, "only support text or img!"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = load_from_name(models, device=device, download_root='./models')

    with torch.no_grad():
        encode_content = ''
        # Encode and normalize the description using CLIP
        if search_text:
            encode_content = model.encode_text(clip.tokenize(search_query))
        if search_img:
            image = preprocess(Image.open(search_query)).unsqueeze(0).to(device)
            encode_content = model.encode_image(image)

        encode_content /= encode_content.norm(dim=-1, keepdim=True)
    # Retrieve the description vector and the photo vectors
    text_features = encode_content.cpu().numpy()
    # Compute the similarity between the descrption and each photo using the Cosine similarity
    similarities = list((text_features @ photo_features.T).squeeze(0))

    # Sort the photos by their similarity score
    best_photos = sorted(zip(range(photo_features.shape[0]), similarities), key=lambda x: x[1], reverse=True)
    file_list = []
    display_pic(best_photos, file_list, top_k, top_p)

    return file_list


def display_pic(photos, file_list, top_k, top_p):
    for i in range(top_k):
        # Retrieve the photo ID
        idx = photos[i][0]
        sim = photos[i][1]
        if sim >= top_p:
            photo_id = photo_ids[idx]
            file_list.append(os.path.join('images', 'gallery', photo_id + '.jpg'))
        else:
            return


def convert_tensor(file_list):
    # 读取图片
    images = [read_image(i) for i in file_list]  # 图片
    # 把图片转换为Tensor
    tensor_images = [img.to(torch.float32).div(255) for img in images]
    # 把图片合成一张大图
    grid = make_grid(tensor_images, nrow=1)  # nrow表示每行的图片数量
    # 把Tensor转换为numpy数组
    grid_image = grid.permute(1, 2, 0).numpy()
    # plt.imsave(grid_image, 'result.jpg')
    print(type(grid_image))
    return grid_image
    # 使用matplotlib显示图片
    # plt.imshow(grid_image)
    # plt.title()
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()


if __name__ == '__main__':
    topK = 10
    pred = open('eval/vit-h-14/pred.jsonl', 'w')
    with open('images/text.txt', 'r') as f:
        number = 0
        for each in f:
            search_query = each.strip()
            file_list = search_(search_query, topK)
            file_list = [i.split(r"/")[-1] for i in file_list]  # 图片
            res = {"text_id": number, "text": search_query, "image_ids": file_list}
            pred.write(str(res) + '\n')
            number += 1

    pred.close()
