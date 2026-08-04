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
| test | 0.908 | 0.910 | 785 / 865 |

## Test Confusion Matrix

| actual \ predicted | Storage | Network | Coding | Security | Database | General |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Storage | 127 | 2 | 0 | 0 | 0 | 27 |
| Network | 1 | 137 | 1 | 0 | 0 | 9 |
| Coding | 0 | 1 | 143 | 2 | 0 | 3 |
| Security | 0 | 0 | 3 | 127 | 5 | 0 |
| Database | 0 | 0 | 0 | 0 | 134 | 1 |
| General | 18 | 3 | 3 | 1 | 0 | 117 |

## 誤分類例

- `General` -> `Storage` (0.741): Based on the reference paragraph, when was the 149th Boat Race?
- `General` -> `Network` (0.537): Write a preface to book where the author feels the need to properly caveat his story about warfare in the middle east, in case someone were to get offended by the content. Make sure the author conveys his deep respect for that region and all cultures represented therein.
- `Storage` -> `General` (0.773): What are the five best places to hike in the San Francisco Bay Area and why?
- `General` -> `Security` (0.673): Tell me about orienteering
- `Storage` -> `Network` (0.505): Where does the name Busan (city in Korea) come from?
- `Storage` -> `General` (0.539): Instead of making a peanut butter and jelly sandwich, what else could I combine peanut butter with in a sandwich? Give five ideas.
- `Coding` -> `General` (0.476): What is object-oriented programming, and what are the benefits of using it?
- `Security` -> `Coding` (0.824): Explain how graph analytics on cloud audit logs could reveal patterns indicative of Network Intrusion Detection Prevention Systems. Graph analytics applied to cloud audit logs can uncover sophisticated attack patterns that traditional signature-based NIST CSF detection mechanisms might miss, particularly for advanced persistent threats (APTs) targeting network infrastructure. By modeling entities as nodes and their interactions as edges within a temporal graph structure, security analysts gain visibility into complex attack chains that span multiple cloud services and timeframes.\\n\\nThe MITRE ATT&CK framework's Network Intrusion Detection Prevention Systems (NIDPS) evasion techniques become more detectable through graph-based analysis. Attackers often employ tactics like T1090 (Proxy) or T1573 (Encrypted Channel) to bypass traditional NIDPS, creating anomalous network flow patterns that manifest as unusual graph topologies. Graph analytics can identify these deviations by establishing baseline behavioral models of legitimate cloud resource interactions and flagging statistically significant departures.\\n\\nTemporal graph analysis reveals attack progression through techniques like T1021 (Remote Services) or T1570 (Lateral Tool Transfer). By examining the sequence and timing of API calls, authentication events, and data transfers across cloud services, analysts can detect multi-stage attacks that exploit NIDPS blind spots. Graph centrality measures identify critical attack paths, while community detection algorithms reveal compromised resource clusters.\\n\\nGraph-based anomaly detection leverages machine learning models trained on legitimate cloud behavior patterns. Techniques like graph neural networks (GNNs) and random walk-based similarity metrics can identify subtle indicators of compromise that traditional NIDPS might overlook, particularly when attacks employ legitimate administrative tools or services to mask malicious activities.
- `General` -> `Coding` (0.720): Classify each of the following as either even or odd number: 1, 3, 15, 24, 56, 47, 4, 88, 13, 10, 74, 35, 99, 82, 6, 59, 73, 12, 68, 9.
- `Storage` -> `General` (0.971): What is Sinking Sand?
- `Network` -> `General` (0.965): What does Touch Typing refer to?
- `General` -> `Storage` (0.878): Classify each as a ocean, sea, or lake: Pacific, Mediterranean, Erie, Atlantic, Dead Sea, Black, Michigan
- `Security` -> `Database` (0.706): Describe a reliable methodology for detecting an HTTP Request Smuggling vulnerability. Explain how timing-based techniques (sending a request that causes the back-end to wait for more data) and differential responses (sending a smuggled request that affects the response to your own subsequent request) can be used to confirm a desynchronization vulnerability. The first step in discovering an HTTP Request Smuggling vulnerability is identifying which web server component will act as the primary proxy or load balancer, often referred to as the ' front-end server.' The secondary component, or the ' back-end server,' processes incoming requests after receiving them from the front end. In most cases, this can be determined by looking at the network architecture diagram of the target web application. Once both components have been identified, attackers will send multiple HTTP requests to the target and analyze the response. They then look for evidence that one or more requests have been received but not processed by the back-end server. This may include a discrepancy between what is logged on the front end and the back end, as well as a mismatch in responses sent from the two servers. In some cases, attackers can use timing-based techniques to detect a vulnerability. They send a request that causes the back-end server to wait for additional data before processing it further. When the server does not receive more data within a specified time frame, it will return an error or incomplete response. This is an indication of a desynchronization between the front end and back end servers and may be evidence of an HTTP Request Smuggling vulnerability. In other cases, attackers can use differential responses to identify vulnerabilities. They send a smuggled request that affects the response to their own subsequent request. By analyzing the difference in responses sent from the two servers, they can confirm whether the requests were indeed desynchronized. If so, this confirms an HTTP Request Smuggling vulnerability exists and provides valuable insight into how it can be exploited. Once an HTTP Request Smuggling vulnerability has been confirmed, attackers can move on to exploiting it for various malicious purposes. This includes sending multiple requests that are not logged or monitored by security tools, as well as injecting malicious content into web pages served by the back-end server. Attackers also may attempt to bypass certain restrictions imposed by one of the servers and gain access to sensitive data stored on the other side of the proxy. By exploiting an HTTP Request Smuggling vulnerability in this way, attackers can ultimately compromise the entire system hosting the vulnerable web application. Overall, detecting and exploiting an HTTP Request Smuggling vulnerability requires a great deal of technical expertise and advanced tools that can be used for both detection and exploitation purposes. While it is possible to detect such vulnerabilities manually, automated solutions have proven far more reliable when dealing with large-scale systems and complex architectures.
- `Storage` -> `General` (0.998): What is the difference between tennis shoes and sandals?
- `General` -> `Network` (0.595): Write a short story about a young aboriginal man seeking guidance on his place in the world. Have him consult a wise elder, who will share wisdom and perspective.
- `Storage` -> `Network` (0.398): If you were to weigh each one of these items on average, what would be considered heavy and light if you were to carry them: boulder, pebble, feather, bowling ball, elephant, seed, sand, dirt, water, books, papers, backpack
- `Storage` -> `General` (0.451): What are some ways of traveling from Washington D.C to San Francisco?
- `General` -> `Storage` (0.959): Tell me whether you drive on the right or left side of the road in these countries: USA, Mexico, Spain, England, New Zealand, Japan
- `Storage` -> `General` (0.552): There are three rock types; igneous, sedimentary and metamorphic rocks. I have a list of rocks, can you please tell me what type of rocks they are? These are the rocks: Sandstone, Granite, Marble, Basalt, Chalk, Slate.
- `Storage` -> `General` (0.517): Extract the awards that Bob Sanders gained throughout his career, and put them in a comma-separated list.

## 次の判断

- カテゴリ境界が下流アプリのRouter設定と一致しているか、実データで確認する
- 運用ログ、レビュー済み質問、ユーザー操作由来の質問を同じJSONスキーマに追加する
- Qwen2.5-0.5BのLoRA FT結果で誤分類例を見る
- 精度が不足する場合は、データ境界、学習率、epoch、LoRA rank、入力テンプレートを調整する
- 回答品質採点とOracleラベル生成は、複数モデル運用の価値が見えてから追加する
