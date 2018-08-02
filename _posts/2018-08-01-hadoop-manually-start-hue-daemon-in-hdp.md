---
title: "HDP에서 수동으로 HUE 데몬 시작하기"
date: "2018-08-01 12:00:00 +0900"
tags:
  - Hadoop
  - Apache Ambari
  - CentOS
  - HDP
  - HUE
use_math: true
---

## 방법 1
`/etc/init.d/hue`가 있는 경우는 아래 방법을 사용
{: .notice--info}

### 1.	HUE 데몬 시작 명령
```shell
/etc/init.d/hue start
```

### 2.	HUE 데몬 정지 명령
```shell
/etc/init.d/hue stop
```

### 3.	HUE 데몬 재시작 명령
```shell
/etc/init.d/hue restart
```

---

## 방법 2: Supervisor 활용
만약 `/etc/init.d/hue`가 없는 경우는 아래 방법을 사용
{: .notice--warning}

### 1.	Supervisor를 통한 HUE 데몬 시작하기

- HUE가 설치된 디렉토리로 이동 (예: `/usr/local/hue`)
```shell
/usr/local/hue/build/env/bin/supervisor -d
```

- `-d`: HUE 데몬 시작 옵션

### 2.	HUE 데몬 정지 (프로세스 죽이기)

```shell
ps-ef | grep hue  # PID 확인
kill -9 <PID>
```

### 3.	HUE 데몬 재시작

- 2번 과정 $\rightarrow$ 1번 과정 수행

---

## References

-	[Start, Stop, and Restart Hue - Hortonworks Data Platform](https://docs.hortonworks.com/HDPDocuments/HDP2/HDP-2.6.2/bk_command-line-installation/content/start_stop_restart_hue.html)
-	[Hue 3 on HDP installation tutorial](http://gethue.com/hadoop-hue-3-on-hdp-installation-tutorial/)
-	[리눅스 프로세스명으로 kill - 제타위키](https://zetawiki.com/wiki/%EB%A6%AC%EB%88%85%EC%8A%A4_%ED%94%84%EB%A1%9C%EC%84%B8%EC%8A%A4%EB%AA%85%EC%9C%BC%EB%A1%9C_kill)
