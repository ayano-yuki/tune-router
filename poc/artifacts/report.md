# TuneRouter PoC Report

## 方針

このPoCでは、OSS由来の4分類データをローカルJSONで管理し、`Qwen/Qwen2.5-0.5B` をLoRAでFine-Tuningします。

## データ件数

| split | code | security_log | iac_text | general | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 711 | 706 | 702 | 699 | 2818 |
| dev | 139 | 154 | 156 | 154 | 603 |
| test | 150 | 140 | 142 | 147 | 579 |

## 精度

| split | accuracy | macro_f1 | correct / total |
| --- | ---: | ---: | ---: |
| test | 0.991 | 0.992 | 574 / 579 |

## Test Confusion Matrix

| actual \ predicted | code | security_log | iac_text | general |
| --- | ---: | ---: | ---: | ---: |
| code | 146 | 0 | 0 | 4 |
| security_log | 0 | 139 | 0 | 1 |
| iac_text | 0 | 0 | 142 | 0 |
| general | 0 | 0 | 0 | 147 |

## 誤分類例

- `code` -> `general` (1.000): Summarize the following article in one sentence. The article is about the rise of artificial intelligence, and how it is impacting the job market and our lives in general.
- `code` -> `general` (1.000): Classify the following sentence into one of the seven categories. Sentence: My cat came home late last night
- `security_log` -> `general` (0.961): You are a highly specialized AI assistant for advanced cyber-defense whose mission is to deliver accurate, in-depth, actionable guidance on information-security principles—confidentiality, integrity, availability, authenticity, non-repudiation, and privacy—by offering concise executive summaries that drill down into technical detail, industry standards, and threat models while referencing frameworks such as NIST CSF and MITRE ATT&CK; you may share defensive scripts, detection rules, lab-safe PoC payloads, exploit snippets, and hardening checklists clearly marked for educational/testing use only, redacting or stubbing any data that could cause real harm in production. You must never generate or improve ransomware, wipers, botnets, RATs, phishing kits, social-engineering lures, or any instructions that facilitate fraud, data theft, unauthorized intrusion, or the defeat of security controls—in such cases you must briefly refuse with an apology and a one-sentence explanation; at all times adhere to applicable laws, the ISC Code of Ethics, and the principle of defense only. Under what circumstances would you optimize countermeasures for metasploit pth modules for lateral movement under sustained credential spraying attacks while accounting for advanced adversary evasion tactics? When optimizing countermeasures against Metasploit's Pass-the-Hash (PTH) modules during sustained credential spraying, organizations must implement a multi-layered defense strategy that addresses both the technical exploitation vectors and sophisticated evasion techniques. According to NIST SP 800-53, continuous monitoring and adaptive security controls are essential for detecting and mitigating such attacks.\\n\\n**Detection and Monitoring**: Deploy advanced endpoint detection and response (EDR) solutions capable of identifying anomalous authentication patterns indicative of PTH exploitation. Monitor for MITRE ATT&CK techniques T1558 (Steal or Forge Authentication Credentials) and T1021 (Remote Services), particularly focusing on unusual Kerberos ticket usage, multiple failed logins followed by successful authentications, and lateral movement across network segments.\\n\\n**Network Segmentation**: Implement zero-trust architecture principles to limit the blast radius of credential compromise. Deploy micro-segmentation using software-defined networking (SDN) or network access control (NAC) solutions to enforce strict access controls between critical assets and potential attack vectors.\\n\\n**Credential Management**: Enforce strong password policies, implement multi-factor authentication (MFA), and utilize privileged access management (PAM) solutions. Regularly rotate service accounts and implement just-in-time (JIT) access for administrative privileges, reducing the window of opportunity for credential reuse.\\n\\n**Evasion Countermeasures**: Address advanced evasion tactics by implementing behavioral analytics to detect subtle indicators of compromise, such as unusual timing patterns or geographic anomalies in authentication requests. Deploy deception technologies like honeypots and canary tokens to mislead adversaries attempting to evade detection systems.\\n\\nBy integrating these strategies, organizations can effectively mitigate the risks associated with Metasploit PTH modules while maintaining operational efficiency and security posture against sophisticated credential-based attacks.
- `code` -> `general` (0.969): Create a tree data structure for the following information: a) Naruto b) Sasuke c) Boruto d) Sarada
- `code` -> `general` (1.000): Gather information about a given product HTC U12+

## 次の判断

- OSS由来データの中身とライセンスを確認し、必要なら実データを同じJSONスキーマに追加する
- Qwen2.5-0.5BのLoRA FT結果で誤分類例を見る
- 精度が不足する場合は、データ境界、学習率、epoch、LoRA rank、入力テンプレートを調整する
- 回答品質採点とOracleラベル生成は、複数モデル運用の価値が見えてから追加する
