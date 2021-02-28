---
title: Jekyll git blog에서 이미지 경로 잘 찾아주기
author: JaeKwan Koo
date: 2021-02-28 12:50:00 +0900
categories: [R Blog, R]
tags: [getting started, Jekyll]
pin: false
---

-   [Git Blog](#git-blog)

## Git Blog

깃 블로그를 입문한지 얼마되지 않았다.  
지킬이라는 것을 이용해서 블로그를 생성했는데, md파일과 연동이 되서
포스팅이 편리하다고 느꼈다.

깃 블로그를 생성하는 것은 물론 엄청난 진입장벽이 있다.  
깃허브를 사용해왔어야 조금 익숙하다.(물론 난 아직도 익숙하지 않다.)  
깃 블로그를 입문하는 것은 처음 깃허브를 만났을 때만큼 어려운 작업이었다.

본인은 그냥 마음에 드는 테마를 찾아서 저장소를 복사해온 뒤, 내 입맛에
맞게 정보만 수정해준 것에 불과하다.(아마 이게 제일 쉬울 것이다.)  
다른 블로그들을 많이 참조하긴 했는데 `git bash`를 많이 써서
`sourcetree`를 쓰는 나에게는 와닿지가 않았다.

내가 추천하는 커리큘럼은  
1. 깃허브에 익숙해지자  
2. 동시에 md형식의 파일을 다뤄보자  
3. 마음에 드는 테마를 찾아서 그냥 그대로 가져오자  
4. 내 저장소에서 하나하나 스스로 폴더들에 대한 설명을 읽어가며 정보를
수정하자(가장 많은 노력이 듦)

<br>

한 가지 포스팅 테스트를 진행하면서 화가났던 점은 분명 깃허브 상에서
rmd파일로 작성하고 md파일을 생성하고 올리게 되면 자동으로 경로가 잘
잡혔던 것이 git blog에서는 그림이 표현되지가 않았다.

이 때문에, 고생을 좀 했는데 결국 답을 찾았다.

[**githack**](https://raw.githack.com/)을 사용하면 내 생각에는 사진
파일에 대해서 더 편리한 포스팅을 할 수 있을 것으로 예상한다.

<br>

한 예를 가지고 진행을 해보도록 하겠다.  
간단한 데이터로 이미지만 생성하고 확인해보자.

``` r
plot(1:10)
```

<img src="https://raw.githack.com/koojaekwan/koojaekwan.github.io/master/_posts/picture_root_files/figure-gfm/unnamed-chunk-1-1.png" style="display: block; margin: auto;" />

rmd파일로 간단한 이미지를 생성하기 위한 산점도를 만들고, md파일로
변환하게 되면 이미지 파일 폴더도 같이 생겨난다.  
파일명에 대한 md와 파일명에 대한 이미지폴더 파일이 생겨났음을 알 수
있다.

여기서 본래 깃허브에 업로드를 하게 되면, 알아서 인식하고 이미지가
보여지지만 깃블로그에서는 그렇지 않았는데 githack으로 경로를 지정하여
해결했다.

<br>

먼저 이미지폴더만 업로드하여 이미지 파일주소를 githack으로 주소변환할
준비를 한다.

현재 내 파일 이름은 `picture_root`이며 이를 md파일로 변환하면
picture\_root.md파일과 picture\_root\_files라는 이름으로 이미지폴더
파일이 생겨나게 된다.

<img src="https://raw.githack.com/koojaekwan/koojaekwan.github.io/master/_posts/picture_root_files/figure-gfm/image1.png" style="display: block; margin: auto;"/>

<img src="https://raw.githack.com/koojaekwan/koojaekwan.github.io/master/_posts/picture_root_files/figure-gfm/image2.png" style="display: block; margin: auto;"/>

이미지 폴더를 먼저 업로드하고 이 업로드 된 이미지 주소를 githack으로
변환한다.  
이후, 생성된 md파일을 열어 넣고 싶은 이미지 위치에 이미지 삽입 문법으로
githack으로 변환된 이미지 주소를 넣어주면 되는 것이다.

<img src="https://raw.githack.com/koojaekwan/koojaekwan.github.io/master/_posts/picture_root_files/figure-gfm/githack_sample_pic.png" style="display: block; margin: auto;"/>

나는 오른쪽 주소를 이용한다.

이대로 다시 md를 변환하고 결과를 확인하면 될 것같다.  
키포인트는 이미지 주소를 바꾸는 것이다. 보통 rmd에서 그래프를 생성하고
md로 변환하면 이미지 파일이 나오고 이를 같이 깃허브에 올리면
깃허브상에서도 잘 로드된다. 하지만 디폴트로 되어있는 것을 그대로
쓰지말고 필요한 이미지만 먼저 깃블로그 저장소로 업로드한 후, githack으로
이미지 주소를 변환시켜 이를 다시 rmd파일에 적용하고 변환 후 올리길
바란다.
