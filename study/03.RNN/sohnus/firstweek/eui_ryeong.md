> Written with Coursera's [Deeplearing.io nlp sequence model](https://www.coursera.org/learn/nlp-sequence-models)

# RNN FIRST WEEK

RNN 이란 Recurrent Neural Network 의 약자로서 회귀적 특성을 가진 인공 신경망이다.
RNN은  sequence data, 각 요소간 순서가 중요한 연속적인 데이터를 다뤄 딥러닝을 해야할때 일반적인 인공신경망에 비해 효율성을 얻을 수 있다.

sequence  데이터를 rnn 네트워크 구조로 처리하면 데이터의 길이에 관계없이 인풋과 아웃풋을 받아들일 수 있으며, 다른 위치에 있는
 요소끼리도 학습을 공유할 수 있다는 장점이 있다.

## rnn 구조
rnn 의 기본적인 구조는 다음과 같다.
![](https://lh3.googleusercontent.com/7pRypQuZFJbvKNmWwrRDZH2zlZlSpC7nPEfoUgCy_JHjoIwTnvNTfv3kjvbhfMrJ_XbRal5-nV4 "rnn1")

위의 그림에서 x< t > 는 입력값의 한 요소 , y(hate)< t > 는 결과 값, 그리고 그 사이에 있는 단계는 hidden state 이다.  

rnn 의 기본 구조에서 입력값은 한꺼번에 전달되지 않으며, 입력값의 요소들이 타입 스탭에 따라 순서대로 들어간다.   

x 는 one hot vector 행렬 형태로 전달이 되는데, 예를 들어 rnn 으로 어떤 문장을 입력값으로 넣었다고 하면 x<3> 는 그 문장의 세번째 단어가 가지고 있는 사전(들어있는 단어의 갯수가 10000 개라고 가정하자) 에서 검색하여 k 번째에 있는 단어임을 확인 한 후, (10000,1) 크기에서 k 열만 1이고 나머지는 다 0인 행렬이 x<3> 가 된다. 
  
위의 그림과 같이 한번 x 입력을 받고 히든 state 와 그에 따른 y 를 계산한 다음 히든 state 를 다시 다음 히든 state 의 계산을 위해 전달해 주는 한번의 그래프가 한번의 타임스탭이다. 

타임스탭들을 쭉 펼쳐서 표현해 보면 아래 그림과 같다. 그림에서 화살표는 그 방향대로 계산이 됨을 의미한다. 
![](https://lh3.googleusercontent.com/CRH_SLWKYklTlmsZ9eexwcB-X_pCyx69F2Y1hV8znqpSY2FXdzJ7oR4a2SpnNQZ2d8lBUkPPF2Q "rnn2")
히든 스테이트는 a< t > 로 표현한다.

 a< t > 는 x< t > 와 그 이전 타임 스탭의 a< t-1 > 를 변수로 계산되는데, 이 때문에 특정 타임스탭 k 에서 a < k > 는 그 이전 타임스탭들의 입력값들의 정보에 다 영향을 받은 셈이 된다. 이 부분이 rnn 의 가장 큰 특징이다.
 
맨 처음의 a<0> 는 초기화된 행렬이다. 

위의 그림만 보면 마치 다음 타임스탭으로 넘어갈 때 다른 공간으로 가는 것 같이 보일 수 있지만 모든 타임스탭들은 전부 하나의 신경망에서 진행됨을 인지해야 한다.

위의 그림에서는 각 타임스탭마다 계속 입력값이 들어가고 결과값도 계속 나오는데, 이는 many to many 를 기준으로 예시를 들었기 때문에 그런 것이고,  실제로는 다음과 같이 여러 경우가 있다.
- one to one :
- one to many :
- many to one :
- many to many :

## rnn 순전파
![](https://lh3.googleusercontent.com/RrcQ4k2W9RfWh4J_zbI95jSqDqwOv-RyYoC5Gpd0N7AaUlmMmOcmrHB01g1nrGWmMAuifhTVw18 "rnn3")

각 요소를 계산하기 위한 실질적인 계산식은 아래와 같다.
![](https://lh3.googleusercontent.com/bmFabDYnmtqVwpNfCMOOA0ltc8se1QHdXHHPT3rPqtWUWjADaMdwbfsCNG19imqQ6Gcp3X4vwkA "rnn")

g(활성함수, activation function) 부분을 구체화시키면 일반적으로 다음과 같다.
![](https://lh3.googleusercontent.com/zIVU9BjKLzcmPd9hOTzYVpsi6ZKdsYufZmFk3hu1F_RjqM3wpXsFGFMNh90hFTu0uKg2FKyTtOQ "rnn6")

위 식을 통해 결과값을 계산할 수 있다. 이렇게 식을 통해 결과값을 계산하는 과정이 순전파(foward propagation) 이다. 

순전파 과정을 표현한 것이 아래 그림이다.
![](https://lh3.googleusercontent.com/niyimsuc6VBqNxBGdo5ZZmccRfg7VJWIDl0pol-hiqcnBsUUFsf4J6zB2xUgU61M4XEj5E0nZgo "rnn4")
(그림상에서는 일부만 그렸지만 실제로는 타임스탭 갯수만큼 앞뒤로 계속 펼쳐져 있어야 한다.)

## rnn 역전파
물론 아무런 학습없이 식을 그대로 쓴다면 의미있는 결과값이 나오지 않을 것이다. 역전파(back propagation)과정으로 실제 결과값과 계산한 결과값의 차이를 이용하여 가중치인 Waa, Wax, Wya 를 수정해나가며 식들이 좀더 정확한 예측을 할 수 있도록 해야 한다.

아래의 그림과 같이 순전파 과정에서 파란색 화살표 방향으로 로우값이 계산이 되면 거기서 화살표방향으로 되돌아오며 역전파를 한다.
 
![](https://lh3.googleusercontent.com/_MnESIWmyd5DORHfjoCMkxMUWBNvXLZhfT9QaaO1V6K1th2nwh6Qg2E0ch0jblAKOEyquSTwJ6g "rnn7")

역전파 과정에서는 편미분을 이용하여, 각 요소들이 실제 결과값과  계산값 사이의 차이에 얼만큼 영향을 끼쳤는데 역으로 돌아가면서 계산한다. 
그 값을 그래디언트라고 하는데,  이 그래디언트에 따라 가중치들을 업데이트한다.
역전파 과정을 통하여 가중치 값들을 업데이트함으로써 인공신경망이 의미있는 예측을 하게 할 수 있다.

지금까지 설명한 것처럼 rnn y 가 앞의 모든 타임스탭들의 입력값에 영향을 받아 나오게 되있다. 그러나 연속적인 데이터를 다루는데 있어 한 지점의 결과값을 구하는데 앞의 입력값만 알고 뒤의 입력값들을 고려하지 못한다면 정확한 예측을 하는데 문제가 생길 수 있다.
번역을 할 때 한 단어의 정확한 의미를 파악하기 위해 앞의 내용만 보는 것이 아니라 뒤의 단어들도 볼 필요가 있는 것처럼 말이다.

## 양방향 rnn
이 문제를 해결하기 위해 양방향으로 순전파를 진행하는 rnn 이  bi-directional recurrent neural networks 이다. 
![](https://lh3.googleusercontent.com/0knR_VRgtR6ovw47YlToANn5rRBtkuL2AxaJPwW8cxtsAbPUOzJZqNlTKR4FnboIeLig4aMs6M8 "rnn5")
bi-directional recurrent neural networks 는 앞 입력값들의 정보를 가지고 계산되는 히든 레이어와, 뒷 입력값들의 정보를 이용하여 계산되는 히든 레이어가 나뉘어 있다. 

이 둘은 연결되어 있지 않으며, 히든레이어 행렬을 계산하는 식또한 일반 rnn 과 동일하다.

결과값 y(hate) 은 다음과 같이 두가지의 히든레이어를 모두 이용하여 계산된다. 
![](https://lh3.googleusercontent.com/aS17aAFdvpkxS3P1DrjGbVR-Txxwmf0RBW2jPJsf4f7AHUCDGfCaO8iuO1CPvTDmGDbLWjJdE7A "rnn8")

역전파(back propagation) 와 bi-directional recurrent neural networks 에 대해서는 1주차 강의에서는 자세하게 설명하지 않으므로 구체적인 계산법 등을 생략한다.

## LSTM

일반적인 RNN 은 로우값(실제 결과값과 계산값 사이의 차이)과 거기에 관련된 요소가 있는 지점 사이 거리가 멀면 역전파시 그래디언트가 점차 줄어든다. 가중치 업데이트는 그래디언트 값에 따라 이루어지기 때문에, 그 요소가 결과에 크게 영향을 끼치는 요소였더라도 그래디언트 값이 작으면 수정이 빠르게 되지 않는다는 문제가 생긴다.  
이것을 vanishing gradient problem 이라고 한다. 

이 문제를 해결하기 위한 특별한 종류의 RNN 이 LSTM (Long Short Term Memory networks) 이다. 
