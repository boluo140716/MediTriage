# ICD-10 中文码表数据来源

供 `disease-code` skill 做确定性查表使用（meditriage/skills/disease-code）。

- `disease.csv`：1586 条 ICD-10 三位类目码，制表符分隔 `code<TAB>中文病名`。
- `disease_catalog.csv`：238 条章节 / 区间目录（含层级与中文名）。

仅取三位类目码；四位及以上细分编码以临床版 / 医保版为准。ICD-10 本身为 WHO
公开发布的国际疾病分类标准，此处采用其中文命名。

## 来源与许可

数据取自开源仓库 chaseliu/ICD-10-CN（https://github.com/chaseliu/ICD-10-CN），
master 分支的 `disease.csv` / `disease_catalog.csv`，遵循 MIT License：

```
MIT License
Copyright (c) 2017 Chase Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction ... THE SOFTWARE IS PROVIDED "AS IS".
```

完整许可见上述仓库 LICENSE。
