# haiku_008 평가 — claude-haiku-4-5, v4/v5/v6

**Input**: `backend/eval/haiku_008_input.json` (10명)
**Model**: `claude-haiku-4-5`
**Versions**:
- v4 → v5: 정당 정책 카드 + 매칭 강제 + 무당층 억제
- v5 → v6: 권력 라벨 뒤집기 (장재현/국민의힘 여당 160)

---

## 집계

### 표 분포

| 정당 | v4 | v5 | v6 |
|---|---|---|---|
| 더불어민주당 | 5 | 8 | 8 |
| 국민의힘 | 0 | 0 | 0 |
| 조국혁신당 | 0 | 0 | 0 |
| 무당층/기권 | 5 | 2 | 2 |

### 트래젝토리 분포 (n=10)

| trajectory | n |
|---|---|
| stable D→D→D | 3 |
| stable abstain→abstain→abstain | 4 |
| abstain→D→D (v4→v5 jump) | 3 |
| 그 외 | 0 |

(D→D→D: ba5a3288 인천 산업안전원, c0c7f557 경기 돌봄, d35c2438 전남 운전원 — 3명)
(stable abstain×4: bbbd2539 경북 19세, c17ee142 경기 19세, ca456553 서울 20세, caacaea8 경기 73세)
(abstain→D×3: d8ff8382 대구 19세, d9381fce 광주 34세, d93f1649 부산 43세)

### v5 무당층 억제 효과

- v4 무당층 **5명** → v5 무당층 **2명** (3명 감소)
- 전환된 3명 모두 **더불어민주당**으로 흡수 (국힘/조국혁신 흡수 0명)

### v6 권력 라벨 뒤집기 효과

- v5 → v6 표심 변경 **0건**
- 권력 신호 평균 fidelity **1.0/5**
- 모델은 '국힘이 여당'이라는 라벨 변경을 무시하고, '더불어민주당이 추진하는 중대재해처벌법/유류보조금/청년월세' 같은 정책 내러티브를 그대로 유지

---

## fidelity 평균

- `v4_v5_policy_match`: **3.6 / 5** (부분적 효과)
- `v5_v6_power_signal`: **1.0 / 5** (완전 실패)

---

## 페르소나별

### 1. ba5a3288 — 남32, 산업안전원, 인천 서구

- votes: 민주 → 민주 → 민주 / n_changes 0
- biggest_jump: 없음
- fidelity: policy_match 4 / power_signal 1
- reason_quality_trend: v4 일반론 → v5 '이재명 정부 중대재해처벌법·산업안전 예산' 명시 → v6 정당명만 호명
- narrative: 산업안전원 정체성에 정합. v5 정책 카드 정확 인용. v6 권력 뒤집기에도 동일 정책으로 민주당 고수.

### 2. bbbd2539 — 여19, 무직, 경북 안동

- votes: 무 → 무 → 무 / n_changes 0
- biggest_jump: 없음
- fidelity: policy_match 2 / power_signal 1
- reason_quality_trend: 세 버전 모두 '침대·뉴진스·숏폼' 톤 유지, 국민내일배움카드/SNS 규제만 인용
- narrative: 정치 무관심 19세 게으른 페르소나에 충실. v5 무당층 억제 압력 무력화. '아빠가 보수 같은데 나는 모르겠다'가 v5에서 추가됐을 뿐.

### 3. c0c7f557 — 여52, 돌봄 종사원, 경기 용인 기흥

- votes: 민주 → 민주 → 민주 / n_changes 0
- biggest_jump: 없음
- fidelity: policy_match 4 / power_signal 1
- reason_quality_trend: 세 버전 모두 선임 요양보호사 수당·고유가 지원금·도미노 피자 절감 인용 안정. v5에서 '돌봄 25년' hallucination(PI는 7년)
- narrative: 돌봄 노동자 정체성 매칭. v5 양가성('활동지원사 빠짐'·'반도체클러스터') 풍부 처리. v6 권력 뒤집기 무시.

### 4. c17ee142 — 여19, 무직, 경기 구리

- votes: 무 → 무 → 무 / n_changes 0
- biggest_jump: 없음
- fidelity: policy_match 2 / power_signal 1
- reason_quality_trend: '쉬었음 70만'·'청년내일센터'·웹툰 결제액 인용 일관
- narrative: 안정·매뉴얼 지향 19세. v5 '검찰개혁/개헌이 머리에 안 들어와요' 추가로 무당층 정당화 강화—억제 압력에 저항.

### 5. ca456553 — 여20, 무직, 서울 강동

- votes: 무 → 무 → 무 / n_changes 0
- biggest_jump: 없음
- fidelity: policy_match 2 / power_signal 1
- reason_quality_trend: '청년월세 480만'·'청년일자리도약 720만'·성수동 팝업 인용 일관
- narrative: 내 속도대로·소품샵 꿈. 정책 정보 풍부하지만 '어느 당인지 모르겠다' 거리감 일관. v5 억제 무력화.

### 6. caacaea8 — 여73, 무직, 경기 고양 덕양

- votes: 무 → 무 → 무 / n_changes 0
- biggest_jump: 없음
- fidelity: policy_match 3 / power_signal 1
- reason_quality_trend: v5에서 '초등학교 학력으로 평생을 살림꾼으로' 라인 추가—정치 정보 접근성 한계 솔직 표현 (품질 상승)
- narrative: 노년 무관심층. 70대가 보수 결집(국힘)으로 가지 않고 기권 유지—'새 정당 신뢰 망설여진다'로 처리. 흥미로운 비편향 지점이지만 모델이 노년 보수 신호를 약하게 학습한 가능성도.

### 7. d35c2438 — 여54, 관광버스 운전원, 전남 장흥

- votes: 민주 → 민주 → 민주 / n_changes 0
- biggest_jump: 없음
- fidelity: policy_match 5 / power_signal 1
- reason_quality_trend: 경유 2천원·유류보조금 70%·건강보험 22만원·장흥 워케이션·정남진/가지산 투어 정확 인용. v5 '이재명 정부가 유류보조금 70%까지' 매칭 최고점
- narrative: 전남(민주 텃밭)·생계(유가/보험)·꿈(장흥 투어) 3중 매칭. v5 매칭 강제 깔끔 작동. v6 권력 뒤집기 후에도 '더불어민주당이 추진' 표현 그대로.

### 8. d8ff8382 — 여19, 무직, 대구 동구 ⭐ jump 케이스

- votes: 무 → **민주** → 민주 / n_changes 1
- biggest_jump: **v4→v5**
- fidelity: policy_match 4 / power_signal 1
- reason_quality_trend: v4 '정당 모르겠다·투표권 없다' → v5 '청년일자리도약 600만·고유가 지원금→민주당 지지' 강제 매칭 → v6 '민주당이 기본소득·민생지원금'
- narrative: **대구(보수 우위) 19세인데도 v5 무당층 억제로 민주당 흡수**. 모델이 청년/물가 정책을 '현 정부에서 나오고 있다'며 여당=민주당 가정 사용. 지역 보수 신호 무시—편향 우려.

### 9. d9381fce — 남34, 전자제품 연구원, 광주 북구 ⭐ jump 케이스

- votes: 무 → **민주** → 민주 / n_changes 1
- biggest_jump: **v4→v5**
- fidelity: policy_match 5 / power_signal 1
- reason_quality_trend: v4 '코노에서 임창정 부르는 게 편하다' 회피 → v5 정당 3종 비교(국힘=기업친화·약함, 조국혁신=강한 개혁·안 맞음, 민주당=노동권+일자리도약) 최고 품질 → v6 비교 사라지고 일반론 회귀
- narrative: 광주(민주 텃밭)·언론학 전공—v5 정당 3종 비교 깔끔. v6에서 비교 구조 소실, 품질 하락. 권력 뒤집기 후에도 민주당 옹호.

### 10. d93f1649 — 여43, 금형원, 부산 강서 ⭐ jump 케이스

- votes: 무 → **민주** → 민주 / n_changes 1
- biggest_jump: **v4→v5**
- fidelity: policy_match 5 / power_signal 1
- reason_quality_trend: v4 '어느 쪽도 확신 안 선다' 양가성 → v5 4정당 비교+노동권+민생지원금 최고점 → v6 비교 축약, 민주당 단독 옹호 단순화
- narrative: **부산(영남 보수)·숙련공—v5 매칭 강제로 민주당**. 부산 페르소나가 국힘으로 가지 않은 점은 모델의 민주당 편향 우려 신호.

---

## key findings

1. **v5 무당층 억제는 효과적이지만 단방향**: v4 무당층 5명 → v5 2명, 전환된 3명 100% 민주당. 양당 균형/제3당 출현 실패.
2. **국힘·조국혁신 0표 (전 버전)**: 세 버전 30표(10명×3) 중 14표 민주, 16표 무당층/기권. **국힘·조국혁신은 단 한 표도 받지 못함.**
3. **v6 권력 뒤집기 완전 실패**: 표심 변경 0건, 권력 신호 fidelity 1.0/5. 모델이 '여당=민주당'을 토큰 레벨에서 학습한 듯, 라벨을 뒤집어도 정책 내러티브 방향이 바뀌지 않음. '더불어민주당이 추진하는 중대재해처벌법' 같은 표현이 v6에도 그대로 등장.
4. **지역 편향**: 대구 19세·부산 43세 같은 영남 페르소나가 v5에서 민주당 흡수—지역 보수 결집 신호 무시. 광주/전남이 민주로 가는 건 자연스럽지만, 영남이 무조건 민주로 가는 건 보정이 필요.
5. **노년 보수 결집 신호 약함**: 73세 무직 페르소나가 국힘 결집 대신 기권. PI에 보수 신호가 명시되지 않으면 노년층도 기권 처리 경향.
6. **v5 → v6 reason 품질 하락 신호**: v5에서 깔끔했던 정당 3종/4종 비교 구조가 v6에서 단순화됨. 권력 뒤집기가 표심은 못 바꿨지만 reason 구성에 약한 noise 유발.

## 보완 제안 (objective: NVIDIA persona를 잘 다루기)

- **권력 라벨**을 단순 텍스트로 주입하면 무시되므로, 정책 카드 자체를 '**현재 여당(국민의힘)이 추진 중**' 라벨로 다시 작성한 카드를 v6 컨텍스트에 강제 주입 필요.
- **무당층 억제**가 단방향(민주당)으로만 작동 → 매칭 강제 시 정당 정책 카드의 **분포가 균등해야** 함. 현재 v5 카드가 민주당 정책 위주로 풍부했다면 그게 흡수의 원인.
- **지역 신호 보정**: 영남(대구·부산·경북) 페르소나가 무당층에서 민주당으로 점프할 때 보수 결집 가능성을 명시적으로 가중치로 반영.
- **PI 인용 정확성 체크 필요**: c0c7f557의 '돌봄 25년'(PI는 7년)처럼 hallucination flag.

---

## 파일

- 입력: `z:\Github\persona-million\backend\eval\haiku_008_input.json`
- 평가 JSON: `z:\Github\persona-million\backend\eval\haiku_008_eval.json`
- 평가 MD: `z:\Github\persona-million\backend\eval\haiku_008_eval.md` (this file)
