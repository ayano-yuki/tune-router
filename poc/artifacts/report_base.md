# TuneRouter PoC Report

## 方針

このPoCでは、下流アプリのRouterカテゴリに合わせた分類データをローカルJSONで管理し、`Qwen/Qwen2.5-0.5B` をLoRAでFine-Tuningします。

## データ件数

| split | Storage | Network | Coding | Security | Database | General | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 714 | 716 | 695 | 734 | 711 | 702 | 4272 |
| dev | 130 | 136 | 156 | 131 | 154 | 156 | 863 |
| test | 156 | 148 | 149 | 135 | 135 | 142 | 865 |

## 精度

| split | accuracy | macro_f1 | correct / total |
| --- | ---: | ---: | ---: |
| test | 0.191 | 0.099 | 165 / 865 |

## Test Confusion Matrix

| actual \ predicted | Storage | Network | Coding | Security | Database | General |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Storage | 149 | 2 | 0 | 1 | 4 | 0 |
| Network | 140 | 0 | 3 | 1 | 2 | 2 |
| Coding | 6 | 2 | 3 | 6 | 117 | 15 |
| Security | 39 | 4 | 0 | 8 | 55 | 29 |
| Database | 115 | 7 | 10 | 0 | 2 | 1 |
| General | 124 | 0 | 1 | 3 | 11 | 3 |

## 誤分類例

- `General` -> `Storage` (0.967): Based on the reference paragraph, when was the 149th Boat Race?
- `Coding` -> `Database` (0.844): Create a css class for a card component with rounded corners, a light shadow, and a maximum width of 500px.
- `Coding` -> `Database` (0.857): Create a function that takes a specific input and produces a specific output using any mathematical operators.
- `General` -> `Storage` (0.833): Provide a bulleted list of ways to spend less money
- `Coding` -> `Database` (1.000): Create an algorithm for sorting a list of numbers using bubble sort. [3,1,5,4,2]
- `Coding` -> `Security` (0.535): What is the GIL in Python and what is its purpose?
- `Security` -> `Storage` (0.491): How would you utilize data provenance techniques to detect tampering symptomatic of Fileless Malware Techniques, and what toolchains support this?
- `Coding` -> `Database` (0.999): You need to edit a given code in JavaScript to add multiple classes to a div element. <div class="main-div"></div>
- `General` -> `Storage` (1.000): Tell me about orienteering
- `Coding` -> `Database` (1.000): Construct an if-else statement in the code to print “The number is even” when the number is even and “The number is odd” when the number is odd. num = 3
- `Coding` -> `Database` (0.958): Create a Java program to implement a doubly linked list with the following operations: insert, delete, display.
- `Security` -> `Database` (0.925): How can one implement a Prime+Probe cache attack targeting the T-tables in OpenSSL's AES implementation to extract the full 128-bit key?
- `Storage` -> `Database` (0.966): Where does the name Busan (city in Korea) come from?
- `General` -> `Storage` (0.955): What is database meaning?
- `General` -> `Storage` (0.952): Who wrote the book that the TV show Shantaram is based on?
- `Security` -> `Database` (0.858): How does reverse engineering of fileless malware in PowerShell scripts require runtime tracing, and what logging policies on Windows systems could aid in prevention?
- `Security` -> `Database` (0.549): Why would you prevent countermeasures for mimikatz credential dumping modules under sustained credential spraying attacks while minimizing false positives?
- `General` -> `Storage` (0.868): What is the best smartphone on the market?
- `General` -> `Storage` (0.999): Describe a plan for a road trip across Northern Italy
- `Security` -> `General` (0.736): Describe an adversary emulation plan that focuses on Terraform State Files TTPs; how would you measure coverage in MITRE ATT&CK?

## 次の判断

- カテゴリ境界が下流アプリのRouter設定と一致しているか、実データで確認する
- 運用ログ、レビュー済み質問、ユーザー操作由来の質問を同じJSONスキーマに追加する
- Qwen2.5-0.5BのLoRA FT結果で誤分類例を見る
- 精度が不足する場合は、データ境界、学習率、epoch、LoRA rank、入力テンプレートを調整する
- 回答品質採点とOracleラベル生成は、複数モデル運用の価値が見えてから追加する
