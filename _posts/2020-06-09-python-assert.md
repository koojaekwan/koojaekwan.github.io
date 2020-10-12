---
title: Python Assert Statement 정리
date: 2020-06-09 18:48:00
categories: [Study Note, Python]
tags: [Python]
use_math: true
seo:
  date_modified: 2020-06-21 23:25:50 +0900
---



해당 글은 [https://docs.python.org/3/reference/simple_stmts.html#assert](https://docs.python.org/3/reference/simple_stmts.html#assert)를 참고하여 정리한 노트입니다.

<br/>

## 1. Basic

`assert` 함수는 거의 모든 프로그래밍 언어에 존재한다.   
<u>디버깅 모드에서 개발자가 오류가 생기면 치명적일 것이라는 곳에 심어 놓은 에러 검출용 코드</u>  
즉, 프로그램에 debugging assertions를 삽입하기 위한 방법임

- 디버깅 모드, 릴리즈 모드
  - 디버깅 모드 = 컴파일할 때 디버깅 정보를 삽입하여 디버깅을 할 수 있도록 하는 컴파일 모드
  - 릴리즈 모드 = 디버깅 정보 없이, 순수한 코드 자체의 기능만 사용하는 컴파일 모드
  - 당연히, debug mode가 더 큰 메모리를 사용

<br/>

<br/>

<br/>

---

## 2. Implementation

<br/>	

simple form

```python
assert expression
```

is equivalent to the following

```python
if __debug__:
    if not expression: raise AssertionError
```

<br/>

extended from

```python
assert expression1, expression2
```

is equivalent to the following

```python
if __debug__:
    if not expression1: raise AssertionError(expression2)
```



<br/>

<br/>

<br/>

---


## 3 Extra

<br/>

`python`의 `__debug__`란?  

- python의 Built-in Constants
  - Built-in Constants 예시 : False, True
- https://docs.python.org/3/library/constants.html
  - python -O option일 경우, \_debug__ = False (Command line option)
  - 즉, assert statements를 제거하는 효과를 뜻함


