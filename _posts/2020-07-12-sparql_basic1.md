---
title: Wikidata를 통한 SPARQL Tutorial Basic 01
date: 2020-07-03 17:00:00
categories: [Study Note, Query Language]
tags: [SPARQL, rdf]
use_math: true
seo:
  date_modified: 2020-07-29 10:34:14 +0900
---



해당 글은 [https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial)를 참고하여 정리한 노트입니다.

<br/>

![sparql]({{ "/assets/img/post/sparql.png" | relative_url }}) 

<br/>

## 1. 시작하기 전에

### :bulb: Wikidata란?

우리가 흔히 말하는 위키피디아 등과 같은 정보를 담은 DB로, 위키미디어 재단에서 운영하는 **협력적으로 편집 가능한 지식 데이터 베이스**를 의미합니다. 위키데이터는 항목에 초점을 맞춘 도큐먼트 지향(document-oriented) 데이터베이스입니다.

위키데이터 저장소는 items로 구성되어 있습니다. 하나의 item은 하나의 label, description 등으로 이루어져있습니다. item은 고유한 Q 식별자를 가집니다. 아래의 사진처럼, Douglas Adams를 설명하기 위해 하나의 문서가 item이 됩니다. 즉, <u>item의 label은 "Douglas Adams"라는 값이고, 이 item의 고유식별자(identifier)는 "Q42"입니다.</u>

이밖에도, item을 설명하기 위해 property라는 것이 존재합니다. <u><Douglas Adams는 St John's College를 다녔다>라는 사실이 있다면, "Douglas Adams"는 "St John's College"와 "educated at"이라는 property로 연결</u>되어 있다는 것을 알 수 있습니다. 그렇다면, "St John's College"라는 item은 새로운 property에 의해 다른 item과 연결되어 있을 수 있겠죠? (실제로는 레이블이 아니라, Q42와 같은 고유식별자로 연결되어 있습니다.)

**이런 방식으로, 모든 Wikidata를 거미줄처럼 연결할 수 있습니다!**

<br/>

![Wikidata](https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Datamodel_in_Wikidata.svg/880px-Datamodel_in_Wikidata.svg.png)

<br/>

### :bulb: RDF란?

RDF(Resource Description Framework)는 **웹 상의 자원을 기술하기 위한** W3C(World Wide Web Consortium)의 **표준규격**입니다. 쉽게 말하면, 웹상에 존재하는 데이터를 거미줄처럼 표현하기 위한 표준 모델이라고 보면 됩니다. <u>RDF 데이터 모델은 주어-서술어-목적어 형태의 트리플(triple)</u>이 됩니다.

예를 들어, 위 사진에서는 "Douglas Adams"(주어)-"educated at"(서술어)-"St John's College"(목적어)라는 트리플을 얻을 수 있습니다. 여기서 주의할 점은 목적어가 항상 대상이 될 필요는 없다는 점입니다. "Seongsu"-"isAge"-"27"이라는 문장처럼 목적어는 특징의 값이 될 수 있습니다.

<br/>

### :bulb: SPARQL이란?

SPARQL(SPARQL Protocol and RDF Query Language)은 W3C(World Wide Web Consortium)의 표준으로서 **RDF 질의어**입니다. 흔히 우리가 관계형 데이터 베이스(RDBMS)에 접근하기 위해 SQL을 사용하듯이, <u>RDF로 표현된 데이터를 다루기 위해 SPARQL이라는 쿼리 언어</u>를 사용하게 됩니다.

따라서, 오늘은 RDF로 표현된 데이터인 Wikidata를 기반으로 SPARQL 튜토리얼을 진행해보려고 합니다. 튜토리얼을 위해서 다른 프로그램 설치 없이 [WDQS](https://query.wikidata.org/)을 통해 SPARQL 연습을 할 수 있습니다. WDQS는 Wikidata Query Service로, Wikidata 데이터셋에 대해 웹 상에서 SPARQL 쿼리를 작성해 실행하면 결과값을 바로 보여주는 서비스입니다. 

<br/>

<br/>

<br/>

---

## 2. SPARQL 기본 형태

### 1) SELECT, WHERE, triple

```sparql
SELECT ?a ?b ?c
WHERE
{
  x y ?a.
  m n ?b.
  ?b f ?c.
}
```

- `SELECT` 절을 통해 반환하고 싶은 변수들을 나열 (변수는 `?`로 시작)
- `WHERE` 절은 해당 변수들에 제한 조건을 부여
  - 보통 triple 형식으로 문장을 구성함
  - Wikidata와 비슷한 지식 기반 데이터 베이스는, 모든 정보가 triple형태로 저장되어 있음
- 따라서, triple의 조합에서 변수가 위치한 곳의 결과를 반환하는 방식으로 작동함

<br/>

<br/>

<br/>

### 2) SELECT, WHERE, triple (예시)

```sparql
SELECT ?fruit
WHERE
{
  ?fruit hasColor yellow.
  ?fruit tastes sour.
}
```

- triple은 하나의 문장이고, subject + predicate + object로 구성되어 있음
- 실제로 이렇게 쿼리가 돌아가는 것은 아니고, triple 요소를 해당하는 고유식별자로 변환해야 함

<br/>

<br/>

<br/>

## 3. SPARQL 작성

### 1) pseudo-query

Wikidata에서 <u>바로크 시대 작곡가인 Johann Sebastian Bach의 모든 자식들을 검색</u>해보고 싶을 때, SPARQL 쿼리를 어떻게 작성하면 될까요? 먼저 pseudo-query를 살펴보겠습니다.

<br/>

위 SPARQL 기본 형태에서 알 수 있었던 사실은 다음과 같습니다.

- `SELECT` 절 : 검색하고 싶은 변수를 `?`를 붙여 작성
- `WHERE` 절 : 해당 변수가 포함된 triple을 작성

<br/>

이 작성 규칙에 따라, 검색 대상인 "Bach의 자식들"은 3가지 형태의 triple로 작성할 수 있습니다.

- `?child`의 아버지는 바흐다.
- `?child`의 부모는 바흐다.
- 바흐의 아이들은 `?child`이다.

<br/>

따라서 *pseudo-query*로 다음과 같이 작성할 수 있습니다.

```sparql
SELECT ?child
WHERE
{
  # child "has parent" Bach
  ?child parent Bach.
}
```

혹은

```sparql
SELECT ?child
WHERE
{
  # child "has father" Bach
	?child father Bach.
}
```

혹은

```sparql
SELECT ?child
WHERE
{
  # Bach "has child" child
  Bach child ?child.
}
```

<br/>

<br/>

<br/>

### 2) executable-query

하지만 위에서 적은 pseudo-query를 실제 WDQS에서 실행하면 에러가 나게 됩니다. [[실행]](https://query.wikidata.org/#SELECT%20%3Fchild%0AWHERE%0A%7B%0A%20%20%23%20Bach%20%22has%20child%22%20child%0A%20%20Bach%20child%20%3Fchild.%0A%7D)

사실 SPARQL 쿼리가 Bach나 child와 같은 언어를 이해하는 방식은 사람이 읽는 방식과 다릅니다. 따라서 <u>Wikidata의 item과 property에 부여해준 고유식별자(identifier)를 사용하여 쿼리를 작성</u>해야 합니다. 이 고유식별자는 [Search](https://www.wikidata.org/wiki/Special:Search) 사이트에서 검색할 수 있습니다.

<br/>

**item 식별자 찾기**

-  Search 사이트에서 해당 item 검색
- 의미가 동일한 단어에 해당하는 `Q숫자`를 가져옴
- `Bach`라는 item을 찾아보기 [[실행]](https://www.wikidata.org/w/index.php?sort=relevance&search=Bach&title=Special:Search&profile=advanced&fulltext=1&advancedSearch-current={}&ns120=1&ns0=1)

**property 식별자 찾기**

- Search 사이트에서 해당 `P:`를 앞에 붙여 검색
- 의미가 동일한 단어에 해당하는 `P숫자`를 가져옴
- `father`라는 property를 찾아보기 [[실행]](https://www.wikidata.org/w/index.php?sort=relevance&search=P%3Afather&title=Special:Search&profile=advanced&fulltext=1&advancedSearch-current={}&ns0=1&ns120=1)

<br/>

따라서, 위에 해당하는 유명한 작곡가인 Johann Sebastian Bach는 `Q1339`에 해당하고, 어떤 아이템의 아버지라는 관계를 나타내는 property는 `P:P22`에 해당됩니다.

추가로, <u>Wikidata triple를 작성할 때는 prefix를 포함</u>해야 합니다. item은 `wd:`, property는 `wdt:`를 붙여줍니다. <u>고정 값에 대해서만 prefix를 붙여주고, 변수는 따로 붙여주지 않습니다.</u>

<br/>

이제 pseudo-query를 executable-query로 바꿔 재작성해볼까요?

```sparql
SELECT ?child
WHERE
{
  ?child wdt:P22 wd:Q1339.
}
```

해당 쿼리를 실행하면 다음과 같은 결과를 얻을 수 있습니다.

![sparql_result1]({{ "/assets/img/post/sparql_basic1_1.png" | relative_url }}){: width="50%" height="50%"}  

드디어 궁금했던 Bach의 자식들을 Wikidata로부터 SPARQL 쿼리로 찾아냈습니다. 자식들이 굉장히 많네요...

<br/>

<br/>

<br/>

### 3) 식별자에 label 붙여주기 

사실 앞에서 얻은 결과를 통해 우리가 알 수 있는 것은 이상한 코드말고는 없습니다. 따라서 자식들의 이름을 알고 싶다면, 하나하나 클릭해봐야 합니다. 더 나은 방법이 없을까요? 결과값마다 사람이 읽을 수 있는 name label을 붙여줄 수 있다면 편하지 않을까요?

<br/>

기존 작성했던 쿼리에 `SERVICE`문과 `?childLabel`이라는 변수를 추가해봤습니다.

```sparql
SELECT ?child ?childLabel
WHERE
{
# ?child  father   Bach
  ?child wdt:P22 wd:Q1339.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
}
```

- `WHERE` 절 안에 새로운 문장 하나가 삽입됨
- `SERVICE` 문장으로 인해 쿼리 내 `?child`의 label에 해당하는 값을 `?childLabel`이라는 변수로 표현하게 됨
- `SELECT` 절로 아이템뿐만 아니라 그에 대한 label을 반환하게 함

<br/>

해당 쿼리를 통해 다음과 같은 결과를 얻을 수 있습니다. [[실행]](https://query.wikidata.org/#SELECT%20%3Fchild%20%3FchildLabel%0AWHERE%0A%7B%0A%23%20%3Fchild%20%20father%20%20%20Bach%0A%20%20%3Fchild%20wdt%3AP22%20wd%3AQ1339.%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22%5BAUTO_LANGUAGE%5D%22.%20%7D%0A%7D)

![sparql_result1]({{ "/assets/img/post/sparql_basic1_2.png" | relative_url }}){: width="50%" height="50%"}

요한 크리스티안 바흐처럼 `?child` 변수뿐만 아니라, 그에 대한 label인 `?childLabel`도 표시해주는 것을 볼 수 있습니다. 하지만, 2명을 제외한 대부분의 자식들 이름이 나오지 않고 있는 것을 볼 수 있습니다.

<br/>

이는 (auto_language) 옵션으로 인해, 한글명으로 바꿀 수 없는 label을 기존 식별자인 `Q-번호`로 표시하고 있는 것 같습니다. 따라서 우측 상단 옵션을 영어로 바꾼다면 아래와 같이 모든 자식들의 label(이름)을 확인할 수 있습니다. [[실행]](https://query.wikidata.org/#SELECT%20%3Fchild%20%3FchildLabel%0AWHERE%0A%7B%0A%23%20%3Fchild%20%20father%20%20%20Bach%0A%20%20%3Fchild%20wdt%3AP22%20wd%3AQ1339.%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22%5BAUTO_LANGUAGE%5D%22.%20%7D%0A%7D)

![sparql_result1]({{ "/assets/img/post/sparql_basic1_3.png" | relative_url }}){: width="50%" height="50%"}

<br/>

<br/>

<br/>

다음 시간에는 조금 더 복잡한 형태의 SPARQL 쿼리에 대해 알아보도록 하겠습니다.


