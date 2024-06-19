# 跨模态检索(文搜图/图搜图)
2024.06.19 本项目使用[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)搭建文搜图/图搜图页面,旨在帮助用户快速使用跨模态检索任务。本项目代码针对MUGE数据集约19w(189585张)数据作为底库数据。本项目提供了提取特征, 检索, 以及uI代码, 下文中将详细介绍细节。


# 开始用起来！
## 安装要求
本项目代码目前完全使用CPU版本, GPU版本需要代码稍作修改。

运行下列命令即可安装本项目所需的三方库,主要用到cn_clip库。
```bash
pip install -r requirements.txt
```


# 跨模态检索
## 代码组织
下载本项目后, 工作区目录结构如下：

```
search_text_img/ 
└── extract_features.py    #特征提取
└── search.py              #特征比对
└── web.py                 #gradio UI界面
└── README.md
└── requirements.txt
└── models        #目录主要方便保存模型,若没有模型,会自动下载存放在该路径
└── features/
    ├── freatures.npy      #底库特征数据落盘
    ├── photo_ids.csv      #底库特征数据id落盘
```

## 运行代码
```
python web.py
```
## 界面展示
* 运行代码后,访问

## 注意事项
* clip不同模型对应的特征纬度不同,因此web界面上提供了多种模型的选择,但是底库数据(MUGE)目前是使用vit-B-16提取特征的, 因此支持vit-B-16快速检索. 若要使用其他模型, 需要重新对底库数据提取特征, 此处读者可自行修改。



# 引用
如果觉得本项目好用，希望能给我们提个star并分享给身边的用户，欢迎给相关工作citation，感谢支持！

