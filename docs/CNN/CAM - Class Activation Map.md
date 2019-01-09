author: SongChiYoung

# CAM - Class Activation Map
Learning deep features for discriminative localization, Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A (MIT, csail)[^1]

## 서론
서두에, 이 논문을 읽으며 YbigTa 강병규 님의 글[^참고] 을 다소 참고하였음을 밝힌다.

### CAM 이란?
CAM(Class Activation Map)은 CNN(Convolutional Neural Network)을 해석 하고자 하는 생각에서 시작했다. 일반적으로, CNN은 특정 지역의 정보를 알고 해석한다고 믿어져 왔지만 이 부분이 명확히 해석되지 못하고 있다. 이를 위해, 기존 연구들 에서는 각 Filter들에 대한 해석을 노력해왔다. 

![enter image description here](https://lh3.googleusercontent.com/Tw-s-2fv1o9lB06NW-192XiKzILO7XlG2KaISynKGxHlLYir3ps6aW47AJvkY4ULhlbWmxNRcAk)
<center>
출처 : http://cs231n.github.io/convolutional-networks/
</center>

위 그림은 스탠포드 대학에서 온라인으로 제공하는 수업, cs231n[^2] 의 강의 자료중 일부 이다. 위 그림에서는 CNN이 동작하는 과정을 설명하기 위해 각 레이어를 지날때 마다의 값을 이미지로 표현하였다. 위 그림에서, 낮은 부분의 Layer에서는 엣지를 Detect하고, 깊은 레이어 일수록 특정 Feature를 찾아낸다는 점은 알 수 있지만, 왜 특정 Class로 분류 되는지는 보이지 않는다.


이 논문은 그러한 부분은 해결하기 위한 노력을 기하였다. 즉, 이미지가 분류될 때, 어느 부분이 해당 이미지 분류에 영향을 미쳤는지를 분석하기 위해 노력하였다.


### 논문의 특징
이 논문은 앞서 적은, 이미지의 어느 영역이 Class 분류에 큰 영향을 끼쳤는지를 분석하기 위하여 특징적인 구현을 하였다. 우선, 첫번째로는 기존의 분류 방법 대신 GAP라는 것을 사용하였다. GAP는 Global Average Pooling의 약자로, 기존 CNN분류기에서 사용하던 3층의 FC레이어(Fully connected Layer) 대신, 1층의 Average Pooling과 FC레이어를 사용하려는 아이디어 이다.

이는, NIN(Network In Network[^NIN])논문에서 처음 제안되었으며, 이후 이미지넷 첼린지에서 우승한 구글의 Inception Net(a.k.a GoogLeNet)[^GoogLeNet] 에도 적용되는 등 많은 참조가 되고 있는 구조 이다.

![GAP vs Ori](https://lh3.googleusercontent.com/Gc5xt-cXepMHzGrr1ThDhkGzcKzav6AmKI-OIdivbIJfhmilc2tuKqfyfnO-oBPw67iDMu6-Kx0)

 저자들은 CAM을 제안하면서 기존에 잘 알려진 모델들을 GAP가 적용되도록 재구현 하여 사용하였으며, GAP를 사용하였을 때에도 사용하지 않은 모델과 유사한 성능이 나옴을 실험적으로 증명하였다.

## 특징
### GAP
논문은 CAM을 구현하기 위해 GAP를 사용하였다. 기존 FC방식은, 최종 Convolution layer 의 결과를 Flatten(최종 Conv레이어의 결과는, 최종 채널 수 x 최종 W x 최종 H 로 3차원이다. 이를 1차원 행렬로 늘어놓아 Fully connected layer(이하 FC)의 인풋으로 바꿔주는 것을 Flatten 이라고 한다) 한 이후, 해당 데이터를 3번 정도의 FC레이어를 거쳐 Class별로 분류하는 방식이다. 이 과정은 많은 레이어를 소모 하므로, Network 의 속도가 늘어나고, 소모 메모리가 늘어나고, Over-fitting에 불리한 등의 문제를 가지고 있다. 따라서, NIN[^NIN] 논문 에서는 GAP라는것을 제안하였다.

![NIN 구조](https://lh3.googleusercontent.com/WXfY72DC3cxec_Qzc24-sGYql30xwPe0rhVogqYnlMtJgFOsxr4F82ExXegTer06SNIfcscPjkg)

GAP는 이를 위해, 1x1 사이즈의 filter를 가진 Convolutional network를 기존 CNN모델 마지막에 추가하였다. 이 때 해당 Conv레이어의 Out channel 사이즈는 분류하기 원하는 Class의 갯수와 같게 하였다. 이후 각 채널을 평균낸 값을 기존에 FC로 얻은 값 대신에 SoftMax 계층에 넣도록 하였다. 이렇게 하면, FC레이어 없이 학습 하므로 계수가 없어서 Over-fitting에 보다 안전하다.

하지만, CAM논문의 구현은 위와 같은 NIN의 구현과 다소 다르게 보였다. 공개된 저자들의 CAM구현[^3]을 보면 마지막 레이어가, 3x3 filter를 쓰는 out channerl이 1024인 Conv레이어 이며, 마지막이 1024x1000 인 FC로 끝나는 것을 확인 가능하다. 

따라서, 본 논문을 구현해 보는데 있어서는 저자들의 구현을 우선시 하여, 마지막 레이어에서average pooling --> channel x class FC 로 이어지는 구현을 GAP의 목적으로 사용하였다.

### Class Activation Map
CAM(Class Activation Map)은 CNN을 하는데 있어서, 이미지의 어느 부분이 결정에 큰 영향을 주었는지를 분석하고자 하는 목적에서 시작되었다. 이를 위해 저자들은 모델에 GAP을 적용시켰는데, 이는 GAP의 경우 마지막 판별 전 까지 데이터의 위치 정보가 훼손되지 않기 때문이다.(이전 까진, Flatten또는 FC등이 없기 때문)

![ "cam 구조"](https://lh3.googleusercontent.com/yesbxG9mqnP8uJ10y7zKJ4MRpKjtWuJbBwHzudTxuFW1rrXykjNQFF8GPHsyy74jLiZHrkq69WA)

즉, 마지막 판별 레이어에서 가지는 Weight값을 convolutional layers 와 pooling layers 를 거친 n x n 행렬 에 곱하면, 판별식에서 어느 부분이 큰 값을 가졌는지를 알 수 있다.

이는, 마지막 판별 레이어가 각 레이어의 Average 값을 입력으로 받는 노드라 이론적으로 옳은 표현이 되는데, 뉴런 네트워크에서 각 노드의 결과는 `입력 x Weight` 들을 모두 더한 값 이기 때문이다.  또한 위 구조에서, 마지막 FC레이어의 노드 갯수는 Class의 갯수와 같으므로, 각 노드의 `입력 x Weight` 값 들의 합을 비교해서, 가장 큰 값을 가지는 노드의 클래스로 분류되는 것이다.

![Neural의 구조](https://lh3.googleusercontent.com/6T9tzO3zShM6o5-ZT2CmLG_xztHe5hQCtUxwSYIf8GHR02P_FggmqeqhF1e-n0b88oLQT8DHZObV)

즉, 해당 값이 클수록 해당 클래스로 분류될 확율이 늘어난다. 따라서, `Weight` 값의 의미는 *분류를할 때 해당 채널의 중요도* 라고 볼 수 있을 것 이다. 이 때, 각 채널에서 높은 값을 가지는 부분은 해당 채널의 Average 값을 높히는데 가장 큰 역할을 하는 부분이다. 즉, 각 채널에서 가장 중요한 부분이라고 해석 가능하다.

따라서, CAM의 구현은, 해당 채널의 각 값에 Weight값을 곱한 수치를 중요도로 해석한다. 이후, 위와 같이 계산된 모든 채널을 더하여서 나온 n x n 행렬을 Heat-map 으로 그린다. 그려진 Heat-map은 이미지에서의 위치 정보를 여전히 포함하고 있으므로, 가장 높은 값을 가지는 위치가, 이미지에서 CNN의 Classification에 있어서 가장 영향력이 높은 부분이라 할 수 있다.

이를 수식으로 표현하면 아래와 같다.(M 은 CAM, x,y는 좌표 c 는 판별 클래스, k는 각 채널)
![enter image description here](https://lh3.googleusercontent.com/Q_ly_K8eX5IHXyYmyRKe0J9WmMXLTxTk6eIcRcNU9R36DmFaxbF7ZCGDcxbLKtawGmRmB4H6hTLg)

[VGG19 + CAM을 직접 구현한 글](http://blog.ees.guru/49)의 마지막에는 논문을 따라 구현하면서 얻은 결과를 포함하고 있지만, 여기에는 논문에서 나온 결과를 첨부하며 마무리 짓도록 하겠다.

![CAM result](https://lh3.googleusercontent.com/fsaJHcLNZrcDimwE9BwaDid6IWUicJgWBDD4P7yy8eA4xgDkfM4zBdsqpKf0BiPgSyFEts2VnDeB)

위 결과에서 보이듯,  이미지의 특징적인 부분에서 높은 값을 가지는 Class Activation Map(CAM)이 만들어진 것을 확인 가능하다.

특히 아래 dome으로 판별된 이미지의 top 5 결과를 보면, 각 판별 결과에 따라 중요하게 판별된 부분이 다른것 또한 확인 가능하다.

----

[^참고]: https://kangbk0120.github.io/articles/2018-02/cam
[^1]: ZHOU, Bolei, et al. Learning deep features for discriminative localization. In: _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_. 2016. p. 2921-2929. 
[^2]: [Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)
[^3]: https://github.com/metalbubble/CAM
[^NIN]:LIN, Min; CHEN, Qiang; YAN, Shuicheng. Network in network. _arXiv preprint arXiv:1312.4400_, 2013.
[^GoogLeNet]:SZEGEDY, Christian, et al. Going deeper with convolutions. In: _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2015. p. 1-9.
