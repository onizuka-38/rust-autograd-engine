# 구현 계획서 (plan.md)

## 0. 전제 및 범위 (Assumptions)
- 목표 기능은 CoreGrad 최소 동작 구현(텐서 연산, autograd, nn/optim, XOR 예제)으로 확정한다.
- 비즈니스 결과는 학습 가능한 최소 엔진 확보와 회귀 가능한 테스트 기반 개발로 확정한다.
- 변경 범위는 요청된 파일 집합으로 제한하고, 불필요한 리팩토링은 제외한다.

---

## 1. 변경 파일 경로 목록
- `plan.md` (신규): 구현 계획/리스크/스니펫 정의
- `src/tensor/mod.rs`: Tensor 데이터 구조/shape/stride 핵심 로직
- `src/tensor/ops.rs`: element-wise 연산/브로드캐스팅
- `src/tensor/linalg.rs`: matmul 및 선형대수 연산
- `src/autograd/mod.rs`: backward 진입점/gradient 누적
- `src/autograd/graph.rs`: DAG 추적/위상 정렬
- `src/nn/mod.rs`: Module trait 및 파라미터 접근 패턴
- `src/nn/linear.rs`: Linear 계층 연산/가중치 관리
- `src/nn/activations.rs`: ReLU/Sigmoid
- `src/optim/sgd.rs`: SGD step/zero_grad
- `examples/xor_training.rs`: E2E 학습 검증 시나리오
- `tests/` 하위(필요 시 신규): 회귀 테스트/정확성 검증

주의: 실제 작업 시 위 목록에서 **요청 범위와 직접 관련된 파일만** 최소 수정한다.

---

## 2. 목표/제약 정의 템플릿
### 2.1 목표
- 텐서 연산/자동미분/신경망 API를 Rust로 구현해 최소 학습 파이프라인이 동작하도록 만든다.

### 2.2 고정 제약
- 바꾸면 안 되는 항목: 모듈 경로(`src/tensor`, `src/autograd`, `src/nn`, `src/optim`)와 공개 API의 핵심 시그니처
- 비기능 제약: 메모리 안전성(Rc/RefCell/Weak), 테스트 가능성, 범위 외 변경 금지

### 2.3 성공 기준 (검증 가능 형태)
- 기능 기준: XOR 예제에서 forward/backward/optimizer step이 끝까지 수행된다.
- 회귀 기준: 텐서 연산/역전파 테스트가 통과한다.
- 품질 기준: API 호환성을 유지하고 요청 범위 외 파일을 수정하지 않는다.

---

## 3. 단계별 구현 전략 (왜 이 방식인지)
1. 요구사항 고정 및 영향도 맵 작성
- 이유: 플레이스홀더 상태에서 바로 코딩하면 범위 누락/과잉 구현 가능성이 높다.
- 산출물: 변경 대상 함수/모듈 목록 + 수정 금지 영역 명시.

2. 실패 재현/기준 테스트 먼저 정의
- 이유: 버그/요구사항을 재현 가능한 테스트로 고정하면 구현 후 검증이 명확해진다.
- 방식: 최소 단위 테스트 + 필요 시 E2E(`examples/xor_training.rs`) 검증.

3. 데이터 계층(Tensor) 최소 수정
- 이유: 상위 레이어(autograd/nn)는 tensor 동작에 강하게 의존하므로, 기반 동작을 먼저 안정화해야 한다.
- 범위: shape/stride, broadcasting, dtype/메모리 규칙 중 요청 관련 부분만.

4. autograd 경로 보정
- 이유: 계산 결과가 맞아도 gradient 전파가 깨지면 비즈니스 결과(학습 가능성) 달성 불가.
- 범위: creator/op 추적, topo 순회, gradient 누적/초기화 규칙.

5. API 레이어(nn/optim) 호환성 유지
- 이유: 사용자 관점의 변경 비용을 줄이기 위해 외부 API 시그니처를 유지하는 것이 우선.
- 범위: 내부 구현만 조정하고 공개 인터페이스는 제약에 맞게 보존.

6. 성능/안정성 점검 후 마무리
- 이유: 기능 충족 이후 비기능 제약(성능/안정성) 위반 여부를 마지막 게이트로 확인.
- 검증: 테스트 통과 + 기준 벤치/러닝 시나리오 확인.

---

## 4. 필요한 코드 스니펫 (핵심 부분만)
아래는 구조 예시이며, 실제 구현 시 기존 코드 스타일/시그니처에 맞춘다.

```rust
// src/autograd/graph.rs (개념 스니펫)
fn topo_sort(root: NodeRef) -> Vec<NodeRef> {
    let mut visited = HashSet::new();
    let mut order = Vec::new();

    fn dfs(n: &NodeRef, visited: &mut HashSet<NodeId>, order: &mut Vec<NodeRef>) {
        if !visited.insert(n.borrow().id) {
            return;
        }
        for p in n.borrow().parents.iter() {
            if let Some(parent) = p.upgrade() {
                dfs(&parent, visited, order);
            }
        }
        order.push(n.clone());
    }

    dfs(&root, &mut visited, &mut order);
    order
}
```

```rust
// src/autograd/mod.rs (개념 스니펫)
pub fn backward(loss: &Tensor) {
    loss.set_grad_scalar(1.0);
    for node in topo_sort(loss.node()).into_iter().rev() {
        let grad_out = node.borrow().grad.clone();
        let locals = node.borrow().op.local_grads(&grad_out);
        for (parent, local_grad) in locals {
            parent.accumulate_grad(local_grad);
        }
    }
}
```

```rust
// tests/feature_regression.rs (개념 스니펫)
#[test]
fn feature_regression_case() {
    // given
    // when
    // then: 기대 동작/수치 오차 허용 범위 검증
    assert!(actual.approx_eq(expected, 1e-5));
}
```

---

## 5. 리스크 / 트레이드오프
- 리스크: Tensor 수정이 광범위 회귀를 유발할 수 있음
- 대응: 요청 관련 분기만 최소 변경 + 회귀 테스트 우선
- 대응 검증: `tests/tensor_ops.rs`와 `tests/autograd.rs`의 연산 회귀 테스트 통과

- 리스크: autograd의 `Rc<RefCell<_>>` 사용 시 런타임 borrow 충돌 가능
- 대응: borrow 범위를 짧게 유지하고 중첩 mutable borrow 패턴 회피
- 대응 검증: `backward_scalar_expression_matches_manual_gradient` 테스트 통과

- 리스크: `Weak`/`Rc` 설계 변경 시 메모리 누수 또는 dangling 참조 위험
- 대응: 부모 참조는 `Weak` 유지, drop 시나리오 테스트 추가
- 대응 검증: 연산 그래프 순회 시 `Weak` 업그레이드 실패 노드 스킵 동작 확인

- 트레이드오프: API 안정성 유지 vs 내부 단순성 개선
- 판단: 이번 범위에서는 API 안정성 우선, 내부 리팩터링은 별도 작업으로 분리

- 트레이드오프: 빠른 기능 구현 vs 성능 최적화(SIMD/BLAS)
- 판단: 먼저 정합성 확보 후 병목이 확인된 경로만 최적화

---

## 6. 실행 체크리스트
- [x] 플레이스홀더 값 확정: 목표/제약/성공기준
- [x] 영향 파일 최소 집합 확정
- [x] 실패 재현 테스트 작성
- [x] 구현 후 테스트/예제 검증
- [x] 성능/안정성 제약 충족 여부 확인

---

## 7. 상세 todo list
1. [x] TODO: 플레이스홀더를 실제 요구사항으로 확정
- 작업: 기능/버그, 비즈니스 결과/요구사항, 제약 항목을 문서에 실제 값으로 치환
- 검증 기준: `plan.md` 내 꺾쇠 괄호 플레이스홀더가 0개
- 확인 명령: `rg "<기능/버그>|<비즈니스 결과/요구사항>|<제약>|<바꾸면 안 되는" plan.md`

2. [x] TODO: 변경 파일 최소 집합 확정
- 작업: 1번 결과 기준으로 변경 파일 목록에서 불필요 항목 제거
- 검증 기준: 각 파일 옆에 "왜 변경하는지" 1줄 설명 존재
- 확인 명령: `rg "src/|tests/|examples/" plan.md`

3. [x] TODO: 실패 재현 테스트 명세 작성
- 작업: 실패 케이스를 `given/when/then` 형식으로 1개 이상 정의
- 검증 기준: 재현 조건, 기대 결과, 허용 오차가 모두 명시됨
- 확인 명령: `rg "given|when|then|허용 오차" plan.md`

4. [x] TODO: 테스트 파일/함수 이름 선확정
- 작업: 추가/수정할 테스트 파일 경로와 테스트 함수명을 명시
- 확정 항목: `tests/tensor_ops.rs`, `tests/autograd.rs`
- 확정 항목: `add_supports_row_broadcast`, `matmul_computes_expected_values`, `backward_scalar_expression_matches_manual_gradient`, `sgd_step_reduces_loss_on_simple_linear_fit`
- 검증 기준: 경로와 함수명이 각각 최소 1개 이상 존재
- 확인 명령: `rg "tests/.*\\.rs|fn .*\\(\\)" plan.md`

5. [x] TODO: 구현 단계별 완료 조건을 명령 단위로 고정
- 작업: 각 단계에 대응하는 실행 명령을 문서화
- 검증 기준: 최소 `cargo test`, 필요 시 `cargo test <target>`, `cargo run --example xor_training` 포함
- 확인 명령: `rg "cargo test|cargo run --example xor_training" plan.md`

6. [x] TODO: 리스크별 대응 테스트 매핑
- 작업: 각 리스크마다 대응 검증 방법(테스트 또는 실행 시나리오) 연결
- 검증 기준: 리스크 항목마다 "대응 검증" 문장이 1개 이상
- 확인 명령: `rg "리스크|대응|검증" plan.md`

7. [x] TODO: 최종 게이트(머지 가능 조건) 정의
- 작업: 구현 시작 전 통과해야 할 체크 항목을 명시
- 검증 기준: "모든 테스트 통과", "API 호환성 유지", "요청 범위 외 변경 없음" 3개 항목 포함
- 확인 명령: `rg "모든 테스트 통과|API 호환성 유지|요청 범위 외 변경 없음" plan.md`
