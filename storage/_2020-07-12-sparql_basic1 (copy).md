---
title: Wikidata를 통한 SPARQL Tutorial Basic 02
date: 2020-07-29 10:50:00
categories: [Study Note, Query Language]
tags: [SPARQL, rdf]
use_math: true
seo:
  date_modified: 2020-07-29 12:00:00 +0900
---



해당 글은 [https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial)를 참고하여 정리한 노트입니다.

<br/>

![sparql]({{ "/assets/img/post/sparql.png" | relative_url }}) 

<br/>

### Advanced triple patterns

저번 Tutorial에서는 Johann Sebastian Bach의 자식들을 SPARQL의 triple pattern을 통해 검색해봤습니다. 오늘은 조금 더 복잡한 형태의 triple을 다뤄보겠습니다.

사실 Bach에게는 2명의 아내가 있다고 합니다. 이를 확인하기 위해 먼저 SPARQL로 검색해볼까요?

```sparql
SELECT ?wife ?wifeLabel
WHERE
{
# ?child  spouse   Bach
  ?wife wdt:P26 wd:Q1339.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
}
```

![image-20200712185450601](/Users/seongsu/Library/Application Support/typora-user-images/image-20200712185450601.png)

네, 실제로 2명의 아내가 있네요. 그렇다면, 첫번째 



그렇다면, 





우리가 배운 내용에 따르면, 직업(occupation), 작곡가(composer), 피아니스트(pianist)를 각각 search에서 찾아서 filtering 조건을 공통 ?child에 걸어주는 것이니까, ;를 활용해 저렇게 작성



근데 여기마저도 반복하는 것들이 있음...

마치 영어로 본다면, 

Child has occupation composer and occupation pianist. 이런느낌

결국, Child has occupation composer and pianist로 occupation의 관계에 해당하는 wdt:P106이 반복되는 것을 제거할 수 있지 않을까?



`;`가 고정된 ?subject에 대해 predicate-object pair를 계속해서 추가할 수 있는 것처럼, 고정된 ?subject-predicate에 대해 `,`를 활용하여 object를 계속해서 추가할 수 있음 (and 조건처럼 보면 될듯)

```sparql
SELECT ?child ?childLabel
WHERE
{
  ?child wdt:P22 wd:Q1339;
         wdt:P25 wd:Q57487;
         wdt:P106 wd:Q36834,
                  wd:Q486748.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
}
```

조금 더 읽기 좋은 형태로...

```sparql
SELECT ?child ?childLabel
WHERE
{
  ?child wdt:P22 wd:Q1339;
         wdt:P25 wd:Q57487;
         wdt:P106 wd:Q36834, wd:Q486748.
  # both occupations in one line
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
}
```

이렇게 작성해도 상관은 없지만... 별로 안좋은 것 같습니다

```sparql
SELECT ?child ?childLabel
WHERE
{
  ?child wdt:P22 wd:Q1339;
  wdt:P25 wd:Q57487;
  wdt:P106 wd:Q36834,
  wd:Q486748.
  # no indentation; makes it hard to distinguish between ; and ,
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
}
```

WDQS가 자동으로 들여쓰기를 해주기 때문에 위처럼 적어도 상관은 없으나 가독성이 떨어집니다.



---

요약해봅시다.

- 쿼리가 마치 텍스트처럼 구조화되어 있는 것을 볼 수 있다.

- subject에 대한 각 triple은 period에 의해 끝이 남

- Multiple predicates about the same subject are separated by semicolons, 

- and multiple objects for the same subject and predicate can be listed separated by commas.

```sparql
SELECT ?s1 ?s2 ?s3
WHERE
{
  ?s1 p1 o1;
      p2 o2;
      p3 o31, o32, o33.
  ?s2 p4 o41, o42.
  ?s3 p5 o5;
      p6 o6.
}
```



---

마지막으로 SPARQL이 제공하는 더 abbreviation을 소개하려고 합니다.

만약 우리가 Bach의 자식에 관심이 없고, 그의 grandchildren(손자)들에 관심이 있다고 가정해봅시다.



grandchild의 엄마/아빠 (gender)가 있는 조건으로 위를 찾아가면 굉장히 찾기 어려움

따라서, 아래로 내려가는 방향이 더 편함

Bach의 자식들을 찾고, 그 자식들의 자식을 찾는ㄴ 것이 편함 이를 쿼리로 나타내면 다음과 같다.

```sparql
SELECT ?grandChild ?grandChildLabel
WHERE
{
  wd:Q1339 wdt:P40 ?child.
  ?child wdt:P40 ?grandChild.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
}
```



이를 영어로 나타내면,

Bach has a child `?child`.

`?child` has a child `?grandchild`.



Once more, I propose that we abbreviate this English sentence, and then I want to show you how SPARQL supports a similar abbreviation. Observe how we actually don’t care about the child: we don’t use the variable except to talk about the grandchild. We could therefore abbreviate the sentence to:

Bach has as child someone who has a child `?grandchild`.



Bach의 child라고 지칭하는 것보다, someone이라고 보통 말함 (왜냐면, 구체적으로 누군지 관심없기 때문에) 

But we can refer back to them because we’ve said “someone *who*”: this starts a relative clause, and within that relative clause we can say things about “someone” (e.g., that he or she “has a child `?grandChild`”). In a way, “someone” is a variable, but a special one that’s only valid within this relative clause, and one that we don’t explicitly refer to (we say “someone who is this and does that”, not “someone who is this and someone who does that” – that’s two different “someone”s).



즉, 이를 SPARQL로 나타내면,

```sparql
SELECT ?grandChild ?grandChildLabel
WHERE
{
  wd:Q1339 wdt:P40 [ wdt:P40 ?grandChild ].
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
}
```



`[]` brackets 쌍을 variable에 있는 위치에 사용하면, 마치 익명 변수처럼 작용한다. barckets 안에서 

You can use a pair of brackets (`[]`) in place of a variable, which acts as an anonymous variable. Inside the brackets, you can specify predicate-object pairs, just like after a `;` after a normal triple; the implicit subject is in this case the anonymous variable that the brackets represent. (Note: also just like after a `;`, you can add more predicate-object pairs with more semicolons, or more objects for the same predicate with commas.



---

요약하자면, 다음과 같음



|   natural language   |                  example                  |  SPARQL   |                 example                 |
| :------------------: | :---------------------------------------: | :-------: | :-------------------------------------: |
|       sentence       |            Juliet loves Romeo.            |  period   |        `juliet loves romeo**.**`        |
| conjunction (clause) | Romeo loves Juliet **and** killshimself.  | semicolon |  `romeo loves juliet**;** killsromeo.`  |
|  conjunction (noun)  |    Romeo kills Tybalt **and** himself.    |   comma   |    `romeo kills tybalt**,** romeo.`     |
|   relative clause    | Juliet loves **someone who** killsTybalt. | brackets  | `juliet loves **[** killstybalt **]**.` |







---







### Instances and classes

지금까지 봤던 "relation"은 A, B 사이의 관계를 나타내는 속성으로, 





reference

- rdf에 대한 이해
  - https://bitnine.tistory.com/46
- Wikidata:SPARQL tutorial
  - https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial#Before_we_start
- Wikidata Query Service
  - https://query.wikidata.org
- SPARQL 사용하기
  - https://joyhong.tistory.com/65





SPARQL(SPARQL Protocol and RDF Query Language)은 W3C의 표준으로서 RDF 쿼리 언어입니다. 데이터베이스에서 정보를 찾거나 입력하고자 할 경우 Query를 사용하듯이 RDF로 표현된 데이터를 찾기 위해서 SPARQL 이라는 언어를 사용합니다. (https://joyhong.tistory.com/65)



