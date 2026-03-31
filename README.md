# rust-autograd-engine

`rust-autograd-engine`는 Rust로 구현한 경량 텐서/자동미분 학습 엔진(CoreGrad)입니다.

## 프로젝트 소개
- `Tensor` 기반 다차원 데이터 연산 (`add/sub/mul/matmul`, 브로드캐스팅)
- 동적 연산 그래프 + 역전파(`backward`)를 통한 기울기 계산
- `nn.Linear`, `ReLU`, `Sigmoid`, `MSELoss`, `SGD`로 구성한 최소 신경망 API
- `examples/xor_training.rs`로 XOR 학습 E2E 동작 확인

## 구조
- `src/tensor`: 텐서 데이터 구조와 연산
- `src/autograd`: 위상 정렬 및 역전파 엔진
- `src/nn`: 레이어/활성화/손실
- `src/optim`: 옵티마이저
- `tests`: 연산/역전파 회귀 테스트
