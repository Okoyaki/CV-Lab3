# CV Лабораторная работа №3

## Цель работы:
Научиться создавать простые системы классификации изображений на основе сверточных нейронных сетей.

## Задание
1. Выбрать цель для задачи классификации и датасет (собрать либо найти, например, на Kaggle)
2. Зафиксировать архитектуру сети, Loss, метрику качества
3. Натренировать (либо дотренировать сеть) на выбранном датасете
4. Оценить качество работы по выбранной метрике на валидационной выборке, определить, переобучилась ли модель
5. Сделать отчёт в виде readme на GitHub, там же должен быть выложен исходный код.

## Теоретическая база
Сверточные нейронные сети (Convolutional Neural Networks, CNN) — это разновидность искусственных нейронных сетей, специально разработанных для обработки данных, имеющих пространственную структуру, таких как изображения. Главной особенностью этих сетей является использование сверточных операций, которые позволяют выявлять важные признаки в данных, например края, текстуры или более сложные объекты, сохраняя пространственные взаимосвязи между пикселями.

Основой работы CNN являются несколько ключевых компонентов:

<b>Сверточные слои:</b> Эти слои выполняют свертку входного изображения с набором фильтров. Каждый фильтр представляет собой небольшую матрицу, которая движется по изображению и вычисляет свертку, выделяя определенные признаки, такие как линии, углы или текстуры. Благодаря этому сети не требуется обрабатывать каждый пиксель отдельно — анализируются только локальные области изображения.

<b>Функции активации:</b> После применения фильтров к данным вводится функция активации (например, ReLU), которая добавляет нелинейность и помогает сети лучше справляться с разнообразными задачами классификации.

<b>Слоев подвыборки (Pooling):</b> Эти слои уменьшают размерность данных, сохраняя наиболее важную информацию. Например, метод max pooling выбирает максимальное значение в каждой локальной области, что помогает уменьшить объем вычислений и снизить риск переобучения.

<b>Полносвязные слои:</b> В конце сети данные уплощаются и передаются в один или несколько полносвязных слоев, где происходит финальная обработка информации для выполнения задачи, например классификации объектов на изображении.

## Описание разработанной системы
В качестве датасета изображений был взят датасет 5 разных видов цветков: маргаритка (633 изображения), одуванчик (898 изображений), роза (641 изображние), подсолнух (699 изображений), тюльпан (799 изображений):

![alt text](https://github.com/Okoyaki/CV-Lab3/blob/3aedc1313a0523807993d5309e2cb222dab3c4f5/data/images/daisy/5673551_01d1ea993e_n.jpg)
![alt text](https://github.com/Okoyaki/CV-Lab3/blob/3aedc1313a0523807993d5309e2cb222dab3c4f5/data/images/dandelion/7355522_b66e5d3078_m.jpg)
![alt text](https://github.com/Okoyaki/CV-Lab3/blob/3aedc1313a0523807993d5309e2cb222dab3c4f5/data/images/roses/99383371_37a5ac12a3_n.jpg)
![alt text](https://github.com/Okoyaki/CV-Lab3/blob/3aedc1313a0523807993d5309e2cb222dab3c4f5/data/images/sunflowers/6953297_8576bf4ea3.jpg)
![alt text](https://github.com/Okoyaki/CV-Lab3/blob/3aedc1313a0523807993d5309e2cb222dab3c4f5/data/images/tulips/5547758_eea9edfd54_n.jpg)

Ниже приведена архитектура сверточной нейронной сети:

![alt text](https://github.com/Okoyaki/CV-Lab3/blob/b4847db2839e2be60b9862b4db2d3a368bed84df/data/result/summary.png)

## Результаты работы и тестирования системы
В начале датасет был разделен на обучающую и валидационную выборки (80% обучающая и 20% валидационная). Все изображения были приведены к общему размеру 128х128 и обучены на сверточной нейронной сети, приведенной выше, количество эпох - 20. Ниже приведен график обучения, на котором приведены метрика точности и метрика потерь:

![alt text](https://github.com/Okoyaki/CV-Lab3/blob/3dae1659789bd9e03d5db63f761c40e35dee01f8/data/result/train.png)

Результирующие метрики, полученные в процессе обучения: accuracy: 0.9661 - loss: 0.0908

Метрики, полученные в ходе проведения валидации: val_acc: 0.6555 - val_loss: 1.4713 

Отдельно были проведены тесты на взятых из открытых источников изображениях трех цветков: маргаритки, одуванчика и подсолнуха:

![alt text](https://github.com/Okoyaki/CV-Lab3/blob/3dae1659789bd9e03d5db63f761c40e35dee01f8/data/result/test0.jpg)
![alt text](https://github.com/Okoyaki/CV-Lab3/blob/3dae1659789bd9e03d5db63f761c40e35dee01f8/data/result/test1.jpg)
![alt text](https://github.com/Okoyaki/CV-Lab3/blob/3dae1659789bd9e03d5db63f761c40e35dee01f8/data/result/test2.jpg)

## Выводы по работе

В результате выполнения работы можно сделать следующие выводы:

1. Исходя из графика метрик, полученных в процессе обучения можно сделать вывод, что модель переобучилась
2. Несмотря на высокую точность при обучении, показатели валидации вышли гораздо ниже (показатель точности уменьшился ~на 0.3)
