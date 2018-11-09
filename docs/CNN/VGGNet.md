author: SongChiYoung
# VGG Net 
Very Deep Convolutional Networks for Large-Scale Image Recognition SIMONYAN, Karen; ZISSERMAN, Andrew.  [^1]

## 1. 서론 

VGG Net은 2014년 ImageNet 대회에서 Classification 에서  2위, Localization에서 1위를 한 모델이다. Classification 1위는 Google의 GoogLeNet[^3] 으로, 이 모델에서 나온 Inception 모델은 아직도 많이 논의되고있다. VGGNet은 GoogLeNet 보다 비록 성능은 뒤졌지만, 훨신 간단하고, 따라서 각 Epoch가 빠르며, 이해와 변형에 용이하여 현재도 많은 연구에서 차용하고 있다.

이 모델의 이 논문 이전의 논문들과 다른, 가장 큰 특징은 두가지 이다. 

* **첫번째**는  3x3의 작은 filter만을 사용한 CNN모델 이라는 점 이다.
* **두번째**는 16/19 레이어의 깊은 레이어를 사용한다는 점 이다. 
(현재 일반적으로 두 모델을 각 VGG16/VGG19로 부른다.)

## 2. 특징
 VGGNet을 설명한 논문인 `Very Deep Convolutional Networks for Large-Scale Image Recognition`[^1] 은 모델의 깊이의 영향을 분석하고자 했다. 이를 위해 3x3의 매우 작은 Filter를 가지는 Conv레이어를 여러 층 쌓아 깊게 만들어서, 깊이의 영향력을 보였다.

![레이어의 깊이와 오류율 변화 추세](https://lh3.googleusercontent.com/z7dB7oa9-B_R8gDrd2dHJhnuW-MHX1aXEte3tdUQzGD_NCKxUX6x9Vk0acdVlWP0_TuZsxJnaHM)

레이어의 깊이와 오류율 변화 추세[^2]

위의 그림에서 보이듯, VGG Net 전까지는 최대 8레이어의 비교적 얕은 모델이 주류를 이뤘으나, VGG Net 이후 깊은 레이어가 잦게 사용되면서, 다음년도 1위를 차지한 ResNet[^2] 의 경우에는 152 레이어를 사용하였다.

이렇듯 깊은 레이어를 사용하는것은 결정 함수의 결정력을 키워주는 효과가 있다. *(본문 : makes the decision function more discriminative.)*  하지만, Layer의 수가 늘어나면 (망이 더 깊어(Deep)지면) 문제가 생긴다. 우선, 망이 깊어질수록 Paremeter의 수가 증가하며, 이는 연산속도의 저하와, Overfitting 의 위험을 키우는 효과가 있다.

이를 위해, VGG Net 에서는 각 필터의 크기를 3x3으로 작은 필터를 사용하였다. 이 논문 이전, 기존의 State Of the Art 였던 7x7 필터 하나 대신, 3x3 필터 3개를 사용하였을 경우, 레이어의 수를 C 로 하였을 때, 파라미터의 수는 아래와 같다.

* 3x3 필터 3 레이어의 경우 : \( 3(3^2^C^2^) = 27C^2^ \)
* 7x7 필터 1 레이어의 경우 : \( 7^2^C^2^ = 49C^2^ \)

본 논문에서는 깊이에 따른 성능 변화를 비교하기 위해 다양한 모델을 구성하였다.
A부터 E 까지 각각, A를 기본으로, LRN(Local Response Normalization: 정규화의 일종) 을 추가한 모델 A에 레이어를 추가시킨 B, B에 레이어를 추가시키되, Conv로 인한 영향이 없이 Non-Linearity만을 늘이기 위한 방법으로 1x1 Conv를 3개 추가한 C, C의 대조군 으로, 3x3 Conv레이어를 3개 추가한 D(VGG 16) 그리고 D에 3x3 Conv 레이어를 3개 추가한 E(VGG 19)로 구성했다.

![네트워크 구성 A-E](https://lh3.googleusercontent.com/K4nMuGsEA7pgesgCQwn7T5Zu9oHCYm_8_rHkgovPRS2JwRy4QNAfXgdn0Z04PLkDOYtT0DB9n5M)

네트워크 구성 A-E[^1]

1x1 Conv 레이어는 특이한 발상인데, 이는 Convolutional의 영향은 줄이면서 Non-linear 의 영향은 발휘하는 레이어 이다. 이 논문에서는 같은 사이즈의 Input/Output 을 적용하여, 단지 위의 이유만으로 사용했지만 Network in Network[^4](이하 NIN) 라는 논문에서는 Mlpconv (NIN에서 제안하는 모델로, Conv 레이어의 결과를 그대로 Conv에 넘기는 기존 방식에서, FC레이어를 뒤에 붙여 Non-linearity를 높힌 모델, Global Average Pooling도 NIN에서 처음 제안되었다.)에도 이를 적용하여 사용하였다.

## 3. 결과

![결과](https://lh3.googleusercontent.com/bKwLJSwUxkLjUYhiSFNqZvKL7cSnqaCV1NIiUVV3TYcC-QCMFlndWwxnWxgRj4uTURueg_nl19E)

결과[^1]

각 레이어별로 실험한 결과 위와 같은 결과가 나왔다.

A와 A-LRN 을 비교한 경우 큰 차이가 없으므로, CNN모델에서 중간에 들어가는 정규화는 큰 의미가 없음을 확인 가능하다. B와 C를 비교했을 때, 큰 차이가 없으므로, 깊은 학습은 Convolutional을 통한 분할이 중요함을 확인 가능하다. 여기서 주목할 만한 사항은, D와 E를 비교했을 때, 깊을수록 더 좋은 결과를 보이던 이전까지와 달리 같거나 나쁜 성능을 보이고 있는 점 이다. 이는, Vanishing-gradient로 해석 가능하다. 하지만, 이 논문의 경우 Activation-function 으로 ReLU를 사용하였는데도 위와 같은 현상이 발생하였다. 따라서 저자들은 E 이후로 더이상 레이어를 늘리기를 멈추었다. 이를 해결하여 레이어를 152개까지 쌓은 논문이 ResNet으로, ResNet은 일정 갯수의 레이어 마다, 몇 단계 이전의 레이어의 결과를 앞으로 가져오는 정책을 통해, 해당 이슈를 해결하였다. (이외에도, DenseNet 등이 해당 이슈를 각자의 방식으로 해결하여 논문을 출판)

## 4. 학습 방식
이 논문에서, VGG Net은 독특한 학습 방식을 가진다. 이전보다 깊어진 네트워크를 효율적으로 학습시키기 위해, A부터 E 까지 A를 학습 시키고, 여기에 레이어를 추가 시켜서 B를 추가 학습 시키는 방식으로 학습을 진행하였다. *-Bootstrapping?-*

![enter image description here](https://lh3.googleusercontent.com/3haVD5EjvgwKtz52Jo8rOGJN-bsEHKpML4U_k3CmuOUNKs96T4U4G4EEoEUXo3s3_4vBkTUqu1I)

또한, 이미지의 갯수를 늘이기 위해, 244x244 보다 큰 크기의 이미지의 경우 다양한 방식으로 이미지를 잘라내어 사용하였다. 이 부분의 경우 자세히 다루지 않고 넘어가도록 하겠다.


[^1]: SIMONYAN, Karen; ZISSERMAN, Andrew. Very deep convolutional networks for large-scale image recognition. _arXiv preprint arXiv:1409.1556_, 2014.

[^2]:HE, Kaiming, et al. Deep residual learning for image recognition. In: _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2016. p. 770-778.

[^3]:SZEGEDY, Christian, et al. Going deeper with convolutions. In: _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2015. p. 1-9.

[^4]:LIN, Min; CHEN, Qiang; YAN, Shuicheng. Network in network. _arXiv preprint arXiv:1312.4400_, 2013.
