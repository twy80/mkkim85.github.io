---
title: "Hive 테이블의 많은 수의 작은 파일들 합치기"
date: "2018-08-02 11:00:00 +0900"
tags:
  - Hadoop
  - Hive
  - MapReduce
  - HDFS
use_math: true
---

아래의 문제들을 해결하기 위한 Hive 테이블을 구성하는 많은 수의 작은 파일들을 적은 수의 큰 파일들로 병합하는 2가지 방법

{% capture notice-text %}
* [Hive](https://hive.apache.org/)로 여러개의 테이블을 생성해서 사용하다보면 어느 순간 쿼리 실행 시간이 늘어나는 현상 발생    
* HDFS에서 관련 테이블을 찾아보면 수 많은 파일들로 인해서 성능이 느려진것을 알 수 있음    
* 특히 실시간 로그 데이터를 Hive 테이블에 삽입 하다보면 기하급수적으로 파일 숫자가 늘어나서 문제가 될 수 있음
{% endcapture %}

<div class="notice--warning">
  {{ notice-text | markdownify }}
</div>

## 쿼리를 사용한 방법: `INSERT OVERWRITE`

- 간단하게 Hive 쿼리를 통해서 테이블을 구성하는 작은 크기의 많은 수의 파일들을 합쳐주는 방법
- 먼저 MapReduce 작업 수를 설정
```sql
set mapred.reduce.tasks=1
```
- 그리고 아래 쿼리를 실행해서 테이블 내용을 읽어서 다시 작은 수의 파일들로 병합
```sql
insert overwrite table <table_name> select * from <table_name> limit 999999999
```

- `limit` 다음에는 쿼리 실행 결과 출력되는 레코드 수보다 큰 값을 지정

## Hive Merge 설정을 통한 방법

Hive 쿼리 실행 결과 출력 파일 수에 대한 설정

### `hive.merge` 설정

아래 설정을 적용하여 Hive 쿼리 수행 후 많은 수로 생성된 결과 파일들을 병합

```xml
hive.merge.mapredfiles=true (default: false)
hive.merge.mapfiles=true (default: true)
hive.merge.size.per.task=256000000 (default: 256000000)
hive.merge.smallfiles.avgsize=200000000 (default: 16000000)
```

{% capture notice-text %}
* 출력 파일들의 평균 크기가 `hive.merge.smallfiles.avgsize` 보다 작으면 병합    
* 합병된 파일의 최대 크기는 `hive.merge.size.per.task` 의 설정 값을 따름
{% endcapture %}

<div class="notice--info">
  {{ notice-text | markdownify }}
</div>

### 중간 결과 파일 압축 설정

압축을 활용하여 파일 처리 성능 향상

```xml
hive.exec.compress.intermediate=true
hive.intermediate.compression.codec=org.apache.hadoop.io.compress.GzipCodec
hive.intermediate.compression.type=BLOCK
```

## References

- [Hive query - INSERT OVERWRITE LOCAL DIRECTORY creates multiple files for a single table](https://stackoverflow.com/questions/28272591/hive-query-insert-overwrite-local-directory-creates-multiple-files-for-a-singl)
- [Hive Multiple Small Files - Hortonworks](https://community.hortonworks.com/questions/106987/hive-multiple-small-files.html)
- [HIVE QUERY의 OUTPUT이 작은 파일로 쪼개지는 경우](https://kidokim509.wordpress.com/2015/06/10/hive-query%EC%9D%98-output-%ED%8C%8C%EC%9D%BC%EC%9D%B4-%EC%9E%91%EC%9D%80-%EC%9A%A9%EB%9F%89%EC%9C%BC%EB%A1%9C-%EC%AA%BC%EA%B0%9C%EC%A0%B8-%EC%9E%88%EB%8A%94-%EA%B2%BD%EC%9A%B0/)
