# CAM (Clss Activation Map)

논문 : https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf 을 읽고 각자 구현해본다.

CAM 은, 정답을 고른 Weight값과 Conv레이어의 결과를 곱하면, 해당 정답을 고르는데 많은 영향을 끼친 부분일수록 높은 값을 가질것 이라고 예측한 데서 시작한다.

이를 위해, GAP(Global Average Pooling)을 사용하는데, 이는 GAP에서는 기존 3층 정도의 FC를 쌓을때 대비, Conv레이어를 그대로 반영하기 때문이다.

구현
Alexnet : 황태윤
VGG19 : 송치영
GoogLeNet : 김일환
