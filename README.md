# Реализация Network-free метода с поддержкой Restyle-pSp энкодера
Выполнено в рамках курсового проекта "Приложение генеративных моделей для решения дискриминативных задач", основной задачей которого являлось исследование применения StyleGAN2 для решения задачи сегментации портретных изображений без учителя. 

### Детали реализации
Данный репозиторий основан на открытом репозитории https://github.com/yuval-alaluf/restyle-encoder. Внесены собственные изменения, отвечающие за поддержку Network-free метода. 

### Использование
- Склонировать данный репозиторий:
``` 
git clone https://github.com/uivvyd/Network-free-method-with-restyle-encoder.git
cd Network-free-method-with-restyle-encoder
```
- Запустить основной скрипт:
```
python scripts/network_free_segmentation.py \
--exp_dir=[Directory for saving results] \
--restyle_psp_path=[Path for a Restyle-pSp encoder checkpoint] \
--generator_path=[Path for a StyleGAN2 checkpoint] \
--data_path=[Directory for testing data] \
```
Можно изменять:
- количество версий style-mixed изображений для кластеризаций: `--n_stylemixed`
- количество кластеров для KMeans: `--n_clusters`
- количество запусков для получения итоговой маски: `--n_repeat_kmeans`

### Примеры работы
<p align="center">
<img src="docs/Network_free_method_FFHQ.png" width="800px"/>
<img src="docs/Network_free_method_CelebAMask.png" width="800px"/>
<br>
Результат работы улучшенного Network-free метода, датасеты: FFHQ, CelebAMask-HQ
</p>
