# 프로젝트 기획서: CoreGrad - Rust 기반 경량 자동 미분 및 텐서 연산 엔진

## 1. 프로젝트 개요 (Overview)
본 프로젝트는 PyTorch나 TensorFlow와 같은 거대 딥러닝 프레임워크에 의존하지 않고, Rust를 활용해 밑바닥부터 텐서(Tensor) 연산과 자동 미분(Autograd) 시스템을 구축하는 것을 목표로 합니다. 

단순히 AI 모델을 사용하는 것을 넘어, 동적 연산 그래프(Dynamic Computation Graph)의 생성부터 역전파(Backpropagation)를 통한 기울기(Gradient) 계산, 그리고 메모리 안전성을 보장하는 저수준 시스템 설계 역량을 증명합니다.

### 1.1. 주요 목표
- 다차원 배열을 안전하고 효율적으로 다루는 커스텀 `Tensor` 데이터 구조 정의
- 메모리 누수 없는 동적 연산 그래프(DAG) 구축 및 Autograd 엔진 설계
- 행렬 곱셈(Matmul) 등 핵심 연산의 SIMD 가속화
- 다층 퍼셉트론(MLP)을 구성하여 XOR 문제 학습 및 검증

---

## 2. 시스템 아키텍처 (Architecture)
엔진은 크게 세 가지 계층으로 분리되어 동작합니다.

1. **Storage & Math Layer:** 실제 데이터(`f32` 배열)를 메모리에 연속적으로 할당하고, 형상(Shape)과 보폭(Stride)을 관리하며 기본 수학 연산을 수행합니다.
2. **Autograd Engine:** 텐서 간의 연산 기록을 방향성 비순환 그래프(DAG) 형태로 추적하고, 위상 정렬을 통해 역전파를 스케줄링합니다.
3. **Neural Network API:** 사용자가 직관적으로 모델을 구성할 수 있도록 `nn.Linear`, `ReLU`, `MSELoss`, `SGD` 등의 딥러닝 모듈을 제공합니다.

---

## 3. 핵심 기술 스택 (Tech Stack)
- **Language:** Rust (Edition 2021)
- **Math & Acceleration:** `std::simd` (Nightly) 또는 OpenBLAS 바인딩 (`cblas-sys`)
- **Data Structures:** `std::rc::Rc`, `std::cell::RefCell`, `std::rc::Weak` (내부 가변성 및 참조 관리)
- **Testing & Benchmarking:** `criterion` (성능 측정)

---

## 4. 메모리 및 소유권 설계 전략 (Memory Management)
동적 연산 그래프 구축 시 발생하는 순환 참조 및 메모리 누수를 방지하기 위해 Rust의 스마트 포인터를 전략적으로 활용합니다.

- **그래프 노드 관리:** 각 텐서 노드는 다중 소유 및 내부 상태 변경을 위해 `Rc<RefCell<Node>>` 구조를 채택합니다.
- **순환 참조 방지:** 순전파 시 부모 노드가 자식 노드를 가리키는 트리가 형성되나, 역전파나 캐싱 로직에서 자식이 부모를 참조해야 할 경우 메모리 누수 방지를 위해 반드시 `Weak` 포인터를 사용합니다.

---

## 5. 단계별 개발 마일스톤 (Milestones)

### Phase 1: 다차원 텐서(Tensor) 기초 구현
- [ ] 1차원 `Vec<f32>`를 활용한 다차원 데이터 메모리 레이아웃 설계
- [ ] Shape 및 Stride 기반의 인덱싱 로직 구현
- [ ] 텐서 간 사칙연산 및 행렬 곱셈(Matmul) 구현
- [ ] 메모리 복사 없는 브로드캐스팅(Broadcasting) 메커니즘 구축

### Phase 2: Autograd 엔진 및 연산 그래프
- [ ] `Tensor` 구조체에 연산 이력(Creator/Op) 추적 기능 추가
- [ ] 각 연산에 대한 국소 미분(Local Gradient) 수식 하드코딩
- [ ] 위상 정렬(Topological Sort)을 이용한 연산 그래프 순회 로직 구현
- [ ] 연쇄 법칙(Chain Rule)을 적용한 자동 `.backward()` 메서드 완성

### Phase 3: Neural Network 모듈 및 최적화기
- [ ] `Module` 트레이트 정의 및 가중치(Weight/Bias) 관리
- [ ] `nn.Linear` (완전 연결 계층) 모듈 구현
- [ ] 활성화 함수 (`ReLU`, `Sigmoid`) 모듈 구현
- [ ] 손실 함수 (`MSELoss`) 및 `SGD` 옵티마이저 구현

### Phase 4: 성능 최적화 및 튜토리얼
- [ ] MLP 모델을 이용한 XOR 논리 게이트 학습 스크립트 작성
- [ ] `criterion`을 활용한 연산 프로파일링 및 병목 분석
- [ ] SIMD 명령어 또는 BLAS 연동을 통한 행렬 연산 가속화 적용

---

## 6. 핵심 알고리즘 및 수학적 배경 (Algorithms & Math)

### 6.1. 역전파와 연쇄 법칙 (Backpropagation & Chain Rule)
특정 노드에 대한 손실 함수 $L$의 기울기는 다음과 같이 연쇄 법칙을 통해 누적 계산됩니다. Autograd 엔진은 이 수식을 그래프 순회 시 자동 적용합니다.

$$
\frac{\partial L}{\partial x} = \sum \left( \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} \right)
$$

### 6.2. 위상 정렬 (Topological Sorting)
역전파 시 자식 노드의 기울기가 부모 노드보다 먼저 계산되어야 하므로, DFS(깊이 우선 탐색) 기반의 위상 정렬 알고리즘을 구현하여 노드 실행 순서를 역순으로 스케줄링합니다.

---

## 7. 디렉토리 구조 (Directory Structure)
```text
coregrad/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── tensor/
│   │   ├── mod.rs         # Tensor 구조체 및 기본 메서드
│   │   ├── ops.rs         # 사칙연산 및 브로드캐스팅
│   │   └── linalg.rs      # 행렬 곱셈 등 선형대수
│   ├── autograd/
│   │   ├── mod.rs         # 역전파 코어 엔진
│   │   └── graph.rs       # 연산 그래프 및 위상 정렬
│   ├── nn/
│   │   ├── mod.rs         # 신경망 모듈 트레이트
│   │   ├── linear.rs      # Linear Layer
│   │   └── activations.rs # ReLU, Sigmoid
│   └── optim/
│       ├── mod.rs
│       └── sgd.rs         # 확률적 경사 하강법
└── examples/
    └── xor_training.rs    # 최종 엔진 검증 예제