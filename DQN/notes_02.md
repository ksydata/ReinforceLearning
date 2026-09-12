### 7. 딥러닝과 DQN

### 7.2.2. Deep Q-Learning Network

- replay memory method 
- 강화학습의 데이터는 상관관계가 높은 경우 과적합(overfitting) 발생, 성능 저하, 그 데이터의 확률분포 자체가 변동성이 높으면 target(reward + gamma * Q_t+1)이 변함. 이를 비정상성(non-stationary)라고 함
  
> (1) $\sim y = q_w(s)$
> (2) $a = argmax(\sim y)$
> (3) $s', r = env.step(a)$
> (4) $(s, a, r, s')$를 리플레이 메모리 R에 삽입함
> deque 자료형을 활용하여 구현 가능함(deque는 꽉 차면 오래된 것부터 제거하는 자료구조)
> 미니배치 단위에서 서로 이질적인 데이터들이 뽑힐 확률이 높아짐


### 7.2.3. DQN 학습 알고리즘

- 행위 신경망과 타겟 신경망으로 나누어 타겟 신경망에 집중함
- q_behavior : input - current_state
- **q_target : input - next_state**


### 7.4. 딥러닝 모델

- 깊은 다층 퍼셉트론(Deep MLP)
- 컨볼루션 뉴럴 네트워크 신경망(CNN) [V]
- 순환 신경망(Recurrent Neural Network, RNN) time sequence, NLP(text, language)
- 트랜스포머(Recent) long sequence text, self attention


### Final exam
프로그램 7-5와 7-6의 구현
퐁 아타리 게임과의 차이 - Atari Break out(벽돌 게임) / action 0(no operation),2(up), 3(down)
차이가 무엇인지 - 1 (fire), start action = 1
wrapper / info item(info["lives"], maximum 5)
reward값이 20~30초 변함이 없다면 다시 fire(start action = 1)

20 hours 소요(n_match = 500, n_match >= 1500) - reasonable score and result
open_source(n_match = 50,000)
