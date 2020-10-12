---
title: 우리가 PyTorch Lightning을 써야 하는 이유
date: 2020-03-09 12:06:00
categories: [Deep Learning, Tutorial]
tags: [Deep Learning, PyTorch]
use_math: true
seo:
  date_modified: 2020-04-14 14:19:20 +0900
---



다음 글은 [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) 라이브러리에 있는 여러 내용들을 참고하여 작성했습니다.

<br/>

<img src="https://avatars2.githubusercontent.com/u/58386951?s=200&v=4" width="100" height="100">

<div align="center">
<h1><a id="user-content-pytorch-lightning" class="anchor" aria-hidden="true" href="#pytorch-lightning"><svg class="octicon octicon-link" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9h1v1H4c-1.5 0-3-1.69-3-3.5S2.55 3 4 3h4c1.45 0 3 1.69 3 3.5 0 1.41-.91 2.72-2 3.25V8.59c.58-.45 1-1.27 1-2.09C10 5.22 8.98 4 8 4H4c-.98 0-2 1.22-2 2.5S3 9 4 9zm9-3h-1v1h1c1 0 2 1.22 2 2.5S13.98 12 13 12H9c-.98 0-2-1.22-2-2.5 0-.83.42-1.64 1-2.09V6.25c-1.09.53-2 1.84-2 3.25C6 11.31 7.55 13 9 13h4c1.45 0 3-1.69 3-3.5S14.5 6 13 6z"></path></svg></a>PyTorch Lightning</h1>
<p><strong>The lightweight PyTorch wrapper for ML researchers. Scale your models. Write less boilerplate.</strong></p>
</div>
---

<br/>



[`PyTorch Geometric`](https://baeseongsu.github.io/posts/pytorch-geometric-introduction/)에 이어 <kbd>PyTorch Lightning</kbd>라는 라이브러리를 소개하려고 합니다. **"도대체 왜 PyTorch를 사용하면서, 거기에 또 다른 라이브러리들까지 사용해야 할까?"**라는 의문이 들 수 있는데요. 이는 <u>정말 다루고 싶은 문제들에 더 집중</u>하고 싶기 때문이라고 생각합니다. 마찬가지로, PyTorch Lightning 또한 이런 목적에서 탄생했습니다. 몇 가지 예제를 통해 **PyTorch Lightning 라이브러리**에 대해 살펴보도록 하겠습니다.

<br/>

- TOC 
{:toc}
<br/>

---

<br/>

## **PyTorch Lightning이란 무엇인가?**

`PyTorch Lightning`은 PyTorch에 대한 High-level 인터페이스를 제공하는 오픈소스 Python 라이브러리입니다. PyTorch만으로도 충분히 다양한 AI 모델들을 쉽게 생성할 수 있지만 GPU나 TPU, 그리고 16-bit precision, 분산학습 등 더욱 복잡한 조건에서 실험하게 될 경우, 코드가 복잡해집니다. 따라서 코드의 추상화를 통해, 프레임워크를 넘어 **하나의 코드 스타일로 자리 잡기 위해 탄생한 프로젝트**가 바로 **PyTorch Lightning**입니다.



<img src="https://pytorch-lightning.readthedocs.io/en/latest/_images/pt_to_pl.jpg" style="max-width:100%;">

위 그림은 MNIST 예제에 대해 PyTorch와 PyTorch Lightning을 각각 사용해 작성한 코드입니다. 왼쪽보다 오른쪽이 훨씬 정돈된 느낌을 주는 데요. 특히, 기존 `PyTorch`를 사용하시는 분들은 파란 영역에 해당하는 **모델 학습/평가 부분이 단 2줄로 줄어들었다는 사실**을 확연히 보실 수 있습니다. 이에 대한 기능은 아래에서 조금 더 자세히 설명해 드리도록 하겠습니다.

<br/>

<br/>

<br/>

## **우리가 Lightning을 써야하는 이유**

라이브러리를 구현해주신 분들이 추천한 **Lightning 사용 대상**은 다음과 같습니다.

- PyTorch를 사용하는 모든 사람
- 딥러닝 연구자들
- 추상화하여 딥러닝 모델을 훈련하고 싶은 사람들
- PyTorch로 딥러닝 모델을 서비스화하는 엔지니어들

*(저자들은 특히, dataset, dataloader, train/valid/test loop 등을 사용한 기존 PyTorch 예제를 몇 가지 살펴본 뒤, PyTorch Lightning을 적용해보는 것이 가장 효율적이다! 라고 말하고 있네요. 전적으로 공감합니다!)*

<br/>

먼저, `PyTorch Lightning`이 매력적인 이유는, (저처럼 추상화를 하지 않고) 코드를 작성하던 기존 PyTorch 사용자들이 더욱 **정돈된 코드 스타일을 갖추게 된다는 점**입니다. 제가 아직도 써먹는... 모델 학습/평가 과정 템플릿을 통해 구체적으로 설명해 드리도록 하겠습니다.

<br/>

아래 예제는 크게 \<TRAINING LOOP\>와 \<VALIDATION LOOP\>로 이루어져 있습니다. `mnist_train`과 `mnist_val`이라는 `dataloader`를 통해 배치학습을 진행 중입니다. (제가 처음 이 코드를 볼 때는 정말 힘들었습니다 :cold_sweat: 만약 여기에 early stooping 등의 기법을 추가한다면 더욱 복잡한 코드가 되겠죠...?)

```python
# ----------------
# TRAINING LOOP
# ----------------
num_epochs = 1
for epoch in range(num_epochs):

  # TRAINING LOOP
  for train_batch in mnist_train:
    x, y = train_batch

    logits = pytorch_model(x)
    loss = cross_entropy_loss(logits, y)
    print('train loss: ', loss.item())

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

  # VALIDATION LOOP
  with torch.no_grad():
    val_loss = []
    for val_batch in mnist_val:
      x, y = val_batch
      logits = pytorch_model(x)
      val_loss.append(cross_entropy_loss(logits, y).item())

    val_loss = torch.mean(torch.tensor(val_loss))
    print('val_loss: ', val_loss.item())
```

<br/>

이를 PyTorch Lightning을 바꿔본다면 어떻게 될까요?

```python
# train
model = LightningMNISTClassifier()
trainer = pl.Trainer()

trainer.fit(model)
```

참 쉽죠...? 뭔가 `sklearn`의 `.fit` 메소드를 보는 듯합니다.

<br/>

이 앞에서 살펴본 건 단순히 "복잡한 양의 코드 작업이 간단하게 줄었다"라는 부분입니다. 그렇다면, **그 복잡한 양의 코드 작업은 어떻게 변형(추상화)되었을까요?** 이 부분에 바로 PyTorch Lightning의 핵심이 담겨 있습니다.



<img src="https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_images/general/pl_overview.gif" style="max-width:100%;">



크게 파란색 영역인 `Trainer`와 빨간색 영역인 `Lightning Module`로 나누어 살펴볼 수 있습니다.

먼저, <span style="color:blue;">파란색 영역</span>인 `Trainer`는 모델의 학습에 관여되는 **engineering을 담당하는 클래스**라고 볼 수 있습니다. 모델의 학습 epoch이나 batch 등의 상태뿐만 아니라, 모델을 저장해 로그를 생성하는 부분까지 전반적으로 담당합니다. 실제로 코드에서는 `pl.Trainer()`라고 정의하면 끝입니다.

두 번째로, <span style="color:red;">빨간색 영역</span>인 `Lightning Module`은 모델 내부의 구조를 설계하는 **research & science 클래스**라고 생각할 수 있습니다. 모델의 구조나 데이터 전처리, 손실함수 등의 설정 등을 통해 모델을 초기화하게 됩니다. 실제로 코드에서는 `pl.LightningModule`클래스를 상속받아 새로운 `lightningmodule`클래스를 생성합니다. 기존 PyTorch의 `nn.Module`과 같은 방식이라고 보시면 됩니다.

결국 두 가지의 큰 클래스를 통해, 복잡한 양의 작업들을 2가지 영역으로 추상화할 수 있게 됩니다.

<br/>

---

이밖에 PyTorch Lightning 장점은 다음과 같습니다.

- 굉장히 유연함
  - PyTorch Code에 아주 적합한 구조
  - Trainer를 다양한 방식으로 Override 가능
  - Callback System을 통해 추가적인 작업을 할 수 있음
- 딥러닝 학습 시 다뤄야할 부분을 잘 구조화 하였음
  - Lightning module
  - Trainer
  - 기타(Callbacks, Loggers...)
- 다양한 학습 방법에 적용가능함
  - GPU, TPU learning
  - 16-bit precision
- PyTorch Ecosystem에 속해 있음
  - 엄격한 Testing 과정
  - PyTorch 친화적
- 다양한 예제와 풍부한 Documentation
- 많은 contributor들이 존재함
- integration with logging/visualization frameworks
  - Tensorboard, MLFLow, Neptune.ai, Comet.ml...
- 기타 등등...

<br/>

<br/>

<br/>

## **Lightning의 핵심 요소 2가지**

PyTorch Lightning은 크게 2가지 영역으로 추상화하여, 코드 스타일의 혁신을 추구하고 있는데요.  
이 2가지 영역의 핵심 요소, `LightningModule`과 `Trainer`에 대해 더 자세히 살펴보도록 하겠습니다.

<br/>

### LightningModule 클래스

먼저 `LightningModule`에 대해 살펴보도록 하겠습니다. 

- **1) 모델의 기본적인 구조정의** (기존 코드와 동일)

  - 기존 모델을 초기화하듯이 그대로 사용 가능

    ```python
      def __init__(self):
        super(LightningMNISTClassifier, self).__init__()
    
        # mnist images are (1, 28, 28) (channels, width, height) 
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
    ```

  - 기존 모델의 `forward`를 정의하는 부분을 그대로 사용 가능

    ```python
    def forward(self, x):
          batch_size, channels, width, height = x.size()
    
          # (b, 1, 28, 28) -> (b, 1*28*28)
          x = x.view(batch_size, -1)
    
          # layer 1 (b, 1*28*28) -> (b, 128)
          x = self.layer_1(x)
          x = torch.relu(x)
    
          # layer 2 (b, 128) -> (b, 256)
          x = self.layer_2(x)
          x = torch.relu(x)
    
          # layer 3 (b, 256) -> (b, 10)
          x = self.layer_3(x)
    
          # probability distribution over labels
          x = torch.log_softmax(x, dim=1)
    
          return x
    ```

  - 손실함수 또한 클래스 내부에 정의하여 사용하는 것이 구조화되어 좋음

    ```python
    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)
    ```

<br/>

- **2) 모델 학습 루프 (Training Loop Sturcutre)**
  
  - 복잡하게 작성하던 내용을 추상화한 부분
  - 일반적인 루프 패턴은 루프마다 3가지 메소드를 가지고 있음
  - (Training, validation, test loop) * (`___step`(스텝마다), `___step_end`(스텝 종료), `___epoch_end`(1 epoch 종료))
  - <u>해당되는 이름에 루프 패턴을 붙여서 정의</u>
  - 일반적으로 쓰는 구조는 다음과 같음
  - `training_step` - `validation_step` - `validation_epoch_end`  
  
    
    
    
    ```python
    def training_step(self, train_batch, batch_idx):
          x, y = train_batch
          logits = self.forward(x)
          loss = self.cross_entropy_loss(logits, y)
    
          logs = {'train_loss': loss}
          return {'loss': loss, 'log': logs}
    
    def validation_step(self, val_batch, batch_idx):
          x, y = val_batch
          logits = self.forward(x)
          loss = self.cross_entropy_loss(logits, y)
          return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
          # called at the end of the validation epoch
          # outputs is an array with what you returned in validation_step for each batch
          # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
          avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
          tensorboard_logs = {'val_loss': avg_loss}
          return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
    ```
    
    

<br/>

- **3) 데이터 준비**
  - PyTorch의 데이터 준비하는 과정을 크게 5가지 형태로 구조화
    - 1) 다운로드
    - 2) 데이터 정리 혹은 메모리 저장
    - 3) 데이터셋 로드
    - 4) 데이터 전처리 (특히, `transforms`를 의미)
    - 5) `dataloader` 형태로 wrapping
    
  - 이에 맞게 코드를 추상화
    - `prepare_data()`
    - `train_dataloader`, `val_dataloader`, `test_dataloader`
  - 코드 참조

    
    
    
    ```python
    def prepare_data(self):
        # transforms for images
        transform=transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize((0.1307,), (0.3081,))])
          
        # prepare transforms standard to MNIST
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
    
      def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64)
    
      def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=64)
    
      def test_dataloader(self):
        return DataLoader(self,mnist_test, batch_size=64)
    ```
    
    

<br/>

- **4) optimizer 설정**

  ```python
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.02)
  ```

<br/>

기존 PyTorch 학습 과정에 관여하는 여러 코드가 위와 같은 추상화된 함수형태로 `LightningModule`안에 포함되어 있다는 것을 확인할 수 있습니다. 특히, 상속받은 `LightningModule` 클래스는 위와 같은 함수들을 순서에 따라 실행하는데, 이를 바로 **Lifecycle**이라고 부릅니다. (즉, 해당하는 순서에 따라 함수를 작성하는 것이 중요합니다.)

1. `__init__`
2. `prepare_data`
3. `configure_optimizers`
4. `train_dataloader`
5. `val_dataloader`
6. `test_dataloader` (`.test()`가 호출될 때 호출)

또한, 각 배치와 에폭마다 루프 메소드는 함수 이름에 맞게 정해진 순서대로 호출됩니다.

1. `validation_step` : 배치마다 실행
2. `validation_epoch_end` : 에폭마다 실행

<br/>

---

<br/>

### Trainer 클래스

다음은 `Trainer` 입니다.

- 기본 사용

  ```python
  from pytorch_lightning import Trainer
  
  model = MyLightningModule()
  
  trainer = Trainer()
  trainer.fit(model)
  ```



- `main.py`로 작성 시

  ```python
  from argparse import ArgumentParser
  
  def main(hparams):
      model = LightningModule()
      trainer = Trainer(gpus=hparams.gpus)
      trainer.fit(model)
  
  if __name__ == '__main__':
      parser = ArgumentParser()
      parser.add_argument('--gpus', default=None)
      args = parser.parse_args()
  
      main(args)
  ```

  ```python
  # 실행
  $ python main.py --gpus 2
  ```



- Testing

  ```python
  trainer.test()
  ```



- Deployment / prediction

  ```python
  # load model
  pretrained_model = LightningModule.load_from_checkpoint(PATH)
  pretrained_model.freeze()
  
  # use it for finetuning
  def forward(self, x):
      features = pretrained_model(x)
      classes = classifier(features)
  
  # or for prediction
  out = pretrained_model(x)
  api_write({'response': out}
  ```

<br/>

이밖에도 다양한 [Trainer flags](!https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags) 옵션이 존재합니다.

- `callbacks`
- `logger`
- `max_epochs`
- `min_epochs`
- `distributed_backend`
- `overfit_pct`

<br/>

<br/>

<br/>

## **PyTorch Lightning에 contribute하기**

현재 PyTorch Lightning은 꾸준한 **Issues**와 **Pull requests**가 올라오고 있습니다. 해당 라이브러리를 사용하면서 불편하거나, 개선되어야 할 점을 제시했을 떄, contributor들이 적극적으로 디스커션하는 것을 볼 수 있습니다. 저도 아주 단순한 내용이긴 하지만... 태어나자마자 처음으로 Issues부터 시작해 PR까지 도전했던 라이브러리가 바로 PyTorch Lightning입니다. (해당 라이브러리 예제에 argparser 옵션에 관련된 내용이었습니다.)

![Desktop View]({{ "/assets/img/post/pytorch_lighting_pr.png" | relative_url }})

알고보니 이전에 저의 내용보다 더 포괄적으로, `default argparser`를 만드는 것이 좋겠다는 이슈를 제시하신 분이 있었습니다. 이를 여러 개발자 분이 해결해, 해당 클래스에 `get_init_arguments_and_types()`라는 classmethod로 구현된 것을 확인할 수 있었습니다. (`argparser`에 대한 기초 개념을 공부해야 하겠다는 생각이 들었습니다ㅠㅠ)

![Desktop View]({{ "/assets/img/post/pytorch_lighting_pr2.png" | relative_url }})



<br/>

<br/>

<br/>

---

## **PyTorch Lightning 소개를 마치며**

**PyTorch Lightning**은 2018년도부터 시작해 현재까지 활성화된 PyTorch Project입니다. 현재 `lightning`과 비슷한 역할을 하는 High-level api로는 `keras`, `Ignite`, `fast.ai` 등이 있습니다. 각각의 장단점을 살펴보는 것도 좋을 것 같네요!

이러한 라이브러리들을 사용하면서, 장단점을 비교할 때 어떤 점을 보면 좋을까 곰곰이 생각해봤는데,  
다음과 같은 질문들이 있을 수 있을 것 같아요.

- 단순히 MNIST를 넘어 빠르게 다양한 예제들을 구성할 수 있는가?
- 실제로 프로젝트, 논문 등에 구현된 사례들이 많은가?
- 독자적인 생태계를 구축하는 것이 아니라 flexible한 연동이 가능한가?
- 추가적인 작업을 하고 싶을 때, 구조적 변경이 자주 일어나는가?

그런 점에서는 만드신 모든 분이 대단하신 것 같아요 :smiley:

















