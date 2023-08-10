
CV (Computer Vision) - это область искусственного интеллекта, которая занимается разработкой алгоритмов и программ 
для анализа и обработки изображений. Она используется в различных сферах, таких как медицина,
автомобильная промышленность, розничная торговля и многие другие. Задачи CV могут включать в себя распознавание 
объектов на изображении, классификацию изображений, сегментацию изображений и детектирование объектов. Например, 
CV может использоваться для распознавания лиц на фотографиях, определения состояния дорог на основе изображений с камер 
наблюдения или для автоматического анализа рентгеновских снимков.

### Computer Vision
В этой части проекта ты познакомишься с компьютерным зрением, со сверточными нейронными сетями 
и их обучением для классификации изображений с использованием библиотеки [PyTorch](https://pytorch.org/). 
Работать мы будем с набором данных [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html). \
CIFAR-10 состоит из 60000 цветных изображений 32x32 в 10 классах, по 6000 изображений в каждом классе. \
Имеется 50000 обучающих изображений и 10000 тестовых изображений.

![cifar10](misc/images/cifar10.png)


#### 1
Дописываем функцию [load_dataloaders] с помощью:
- [torchvision.datasets.CIFAR10]
- [torch.utils.data.DataLoader]  
Функция принимает ТЕНЗОРЫ и количество БАТЧЕЙ 
Функция возвращает DataLoaderы для train и test частей датасета. 
Загружаем датасет
C помощью функции `len` выводим количество батчей в train_loader и test_loader.
Оставляем параметры DataLoader `transform` и `batch_size`  по умолчанию.

#### 2
Передаем первые 4 изображения и метки из первого батча тестовой выборки в функцию [imshow] и визуализируем датасет

#### 3
Попробуем написать небольшую сверточную нейронную сеть, которую мы будем обучать классифицировать изображения.

Напишем сеть, основанную на одном блоке архитектуры [ResNet]. Схема этого блока приведена ниже:

<img src="../misc/images/rediual_block.png" width="500"/>

Допишим класс ResidualNet:
- Все сверточные слои должны иметь 32 выходных канала, а также не должны изменять ширину и высоту изображения.
- Также в сверточных слоях `padding = 1`

Функции, которые вам понадобяться: 
- [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), 
- [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html), 
- [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html).

Для базовой проверки, что сеть написана верно, этот код не должен выдавать ошибку\
`assert net(torch.zeros((10, 3, 32, 32))).shape == (10, 10)`

#### 4
Для обучения кроме самой модели требуется определить оптимизатор и функцию ошибок:
* В качестве оптимизатора выберите [стохастический градиентный спуск](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
* В качестве функции ошибок
[кросс-энтропия](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

Обучим сеть и с помощью функции [plot_train_log](./code-samples/cv_utils.py) визуализируем процесс обучения модели.

#### 5
Обучи сеть с аугментацией данных и с помощью функции [plot_train_log](./code-samples/cv_utils.py), визуализируй процесс обучения модели.

_________________________________________________________________________________________________________________________

CV (Computer Vision) is a field of artificial intelligence that develops algorithms and programs
for image analysis and processing. It is used in various fields, such as medicine,
the automotive industry, retail, and many others. CV tasks may include recognition
image objects, image classification, image segmentation and object detection. For example,
CV can be used to recognize faces in photos, determine the state of roads based on images from cameras
observations or for automated X-ray analysis.

### Computer Vision
In this part of the project, you will get acquainted with computer vision, convolutional neural networks
and learning them to classify images using the [PyTorch] library (https://pytorch.org/).
We will work with the data set [CIFAR10] (https://www.cs.toronto.edu/~kriz/cifar.html).
CIFAR-10 consists of 60,000 32x32 color images in 10 classes, 6,000 images in each class.
There are 50,000 training images and 10,000 test images.

#### 1
Add the [load_dataloaders] function using:
- [torchvision.datasets.CIFAR10]
- [torch.utils.data.DataLoader]
The function accepts TENSORS and the number of BATCHES
The function returns the DataLoaders for the train and test parts of the datacet.
Loading Datacet
Using the function'len ', we output the number of batches in train_loader and test_loader.
Leave the DataLoader parameters' transform'and' batch _ size'by default.

#### 2
We pass the first 4 images and marks from the first batch of the test sample to the [imshow] function and render the datacet

#### 3
Let's try to write a small convolutional neural network, which we will train to classify images.

Let's write a network based on one block of the [ResNet] architecture. The diagram of this unit is given below:

<img src="../misc/images/rediual_block.png" width="500"/>

Let's add the ResidualNet class:
- All convolutional layers must have 32 output channels, and must not change the width and height of the image.
- Also in convolutional layers' padding = 1'

The features you need are:
- [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html),
- [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html),
- [ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html).

For basic verification that the network is written correctly, this code should not issue an error\
`assert net(torch.zeros((10, 3, 32, 32))).shape == (10, 10)`

#### 4
To learn, in addition to the model itself, you need to define an optimizer and error function:
* Choose [stochastic gradient descent] as the optimizer (https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
* As error function
[cross-entropy] (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

We train the network and use the [plot_train_log] (./code-samples/cv_utils.py) function to visualize the model learning process.

#### 5
Train the network with data augmentation and use the [plot_train_log] function (./code-samples/cv_utils.py), visualize the model training process.
