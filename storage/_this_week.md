# GRDL 2020 프로그램 리뷰



## 1. GRDL 이란?

Geometric and Relational Deep Learning

This workshop is part of the ELLIS program on Geometric Deep Learning. For more information on the **European Lab for Learning and Intelligent Systems** (ELLIS) visit https://ellis.eu/.

그렇다면 ellis란? 유럽의 ai를 연구하는 ?

- https://en.wikipedia.org/wiki/European_Laboratory_for_Learning_and_Intelligent_Systems





## 2. Organizers 소개

- 





## 3. Speaker와 Program 소개

- 





## 4. 내용 리뷰



| Time        | Presenter                                                    |
| :---------- | :----------------------------------------------------------- |
| 9.20-9.30   | *Opening remarks* (Thomas Kipf)                              |
| 9.30-10.00  | **Peter Battaglia**: Learning Physics with Graph Neural Networks [[video](https://youtu.be/Us8072uy9UY)] |
| 10.00-10.30 | **Natalia Neverova**: Entity-level Video Understanding [[video](https://youtu.be/5sgDhc-piDw)] |
| 10.30-11.00 | *Break & networking* (topic-specific breakout rooms)         |
| 11.00-11.30 | **Stephan Günnemann**: Adversarial Robustness of Machine Learning Models for Graphs [[video](https://youtu.be/Ze0kefjZwfs)] |
| 11.30-12.00 | **Yaron Lipman**: Deep Learning of Irregular and Geometric Data [[video](https://youtu.be/fveyx5zKReo)] |
| 12.00-12.15 | **Qi Liu**: Hyperbolic Graph Neural Networks [[video](https://youtu.be/h1wH5pr6cPE)] |
| 12.15-13.30 | *Lunch & networking* (topic-specific breakout rooms)         |
| 13.30-14.00 | **Miltos Allamanis**: Typilus: Neural Type Hints [[video](https://youtu.be/vvtPISQ8sH4)] |
| 14.00-14.15 | **Noemi Montobbio**: KerCNNs: Biologically Inspired Lateral Connections for Classification of Corrupted Images [[video](https://youtu.be/cZT2vNWtsSk)] |
| 14.15-14.30 | **Pim de Haan**: Natural Graph Convolutions [[video](https://youtu.be/DvVGNqV1mCc)] |
| 14.30-15.00 | *Break & networking* (topic-specific breakout rooms)         |
| 15.00-15.30 | **Poster spotlights** (1min pre-recorded video per poster)   |
| 15.30-17.00 | **Poster session / networking** in breakout rooms            |



### 4.1 Peter Battaglia : Learning Physics with GNNs [34:18]

- deepmind

- 제목 그대로 GNN을 사용해, phsical dynamics와 physical knowledge를 배운다.

Intro

- What is deep learning good at?
  - What do many of dl's successes havei n common?
    - Vectors, Grids, Sequences
    - 그런데 자연 상에 존재하는 중요한 문제들의 포맷들은 이런 형태가 보통 아님
    - richly structed임
      - molecules graph, chemical graph
      - biological species (trees)
      - code.... natural language... 블라블라
  - The classic deep learning toolkit
    - 데이터 형태에 따라, MLP, CNN, RNN를 구분
    - 그렇지만, 아까 말한 다양한 포맷의 데이터에 적합하지 안흥ㄹ 수 있다.
    - 그러나 그래프로 표현한다면 가능
    - 최근 그래프 신경망에 대한 관심이 높아진 이유도 이런 것일 것
  - Background 서베이 논문들 링크 제공
- General Idea
  - conv net과 비슷함
  - 하지만, 그리드 형태라기 보다 임의의 그래프에 적합해지는 것
  - 가장 중요한건 엔티티와 그들간의 관계를 학습할 수 있다는 것 (key power)

Graph Network (Blocks)

- why do wee need another graph neural network variant? (왜 그래프에 또다른 variant가 필요할까?)
  - GN을 expressive, and easy to implement하게 디자인함
  - GN 블록은 graph-to-graph function approximator
    - 함수로 생각한다고 했으니, output (graph)의 구조는 input(graph)와 match됨 (nodes수, edge 연결성)
      - 대신, attribute가 바뀌겠지.
    - output graph-level, edge-, node- 레벨 속성들은 input graphs의 것들의 함수가 될 수 있음
- 이 사람이 정의한 Graph Network를 사용하면, 다양한 variant를 손쉽게 하나의 GN 블록으로 해석할 수 있다.
  - transformer network 또한 맥락에서...
- 블록처럼 GN을 쌓아서 기존 딥러닝 디자인 패러다임을 적용할 수 있다.

Physics part

- interaction network
  - physics : like gravitational system with a little n body system represents planets
    - => graph representatiton, nodes=planet the bodies, edges= forces, possibility of them interacting
    - v (node) planet mass, position
    - e(edge) forces charged particles
  - edge function - message aggregation - node function
  - trained to predict body coordinates (supervised learning, future positions)
- 3가지 경우 모두 그래프로 해석 가능 (n-body, balls, smalls)
- larger systems에 대해 zeroshot도 가능
  - GN이 physics의 rules를 잘 학습했기때문

--

- full graph network genearlizes "Interaction Network"
  - Interaction Network -> GN

--

- Systems: "DeepMind Control Suite" (Mujoco) & real JACO
  - mujoco : phsycis engine 3d simulation
  - 역시, Kinematic tree of actuated system as a graph
    - bodies = nodes, joints = edges, gloabl properties
  - 모델
    - G_D(dynmaic graph, velocities in the systems at some initial conditions)와 G_S(Static properties, masses , length of the body)
    - concat 후에, skip connection
  - 이 모델을 단일 모델로서 학습함
    - 왜? 모든건 phsyics를 학습하는 것
    - bodies와 joints가 어떻게 움직이는지 서로 interaction하는지를 배우는 거니까
  - zero-shot generalizaiton for swimmer model
  - real jaco
    - real robot arm data 를 일종의 time series로 생각
    - 15:00







### 4.2 Entity-level video understanding [28:34]

- facebook ai research
- 



### 4.3 Stephan Guuenmann : Adversarial Robustness of ML models for Graphs

- 그럴듯한 제목



### 4.4 Yaron Lipman : Deep Learning of Irregular and Geometric Data





### 4.5





### 4.6













## 5. 기타

- 항상 재밌다고 생각하며 관심을 두고 있는 분야가 바로 이 분야라고 생각함

- 아직 아는게 적지만... 페이스북의 GNN 커뮤니티에 올라온 글을 보고 리뷰해보고 싶다는 생각이 들어

- Thomas Kipf 라는 분만 익히 들어본 것 같음
  - GCN의 창시자...
- 암스테르담...이런 학회에 가서 연구했던 내용들을 멋있게 발표하고 싶다는 생각이 듭니당...
- 이해 못한 내용들이 워낙 많아, 아쉬웠지만...ㅎㅎ
- 

















