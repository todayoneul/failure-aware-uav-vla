# 이전 기술 검증 기록

현재 텀프로젝트 안내는 [baseline](../baseline.md), [setup](../setup.md), [experiments](../experiments.md), [failure plan](../failure_plan.md)에 있다. 아래 기록은 성공·실패의 근거로 보존하며 당시 판단을 현재 상태와 구별한다.

| 기록 | 내용 |
|---|---|
| [final_closed_loop_validation.md](final_closed_loop_validation.md) | live single-step, 10/10 loop, latency·GPU·RAM 최종 측정 |
| [communication_stability_test.md](communication_stability_test.md) | direct WSL reconnect, idle, 제한된 자원 부하 시험 |
| [aerovla_int4_validation.md](aerovla_int4_validation.md) | 실제 OpenVLA + LoRA NF4 loader·단독 추론 |
| [projectairsim_smoke_test.md](projectairsim_smoke_test.md) | Windows rendering·camera·script 비행·blur/drift 데모 |
| [projectairsim_aerovla_integration.md](projectairsim_aerovla_integration.md) | adapter 검증과 최초 relay 실패 |
| [compatibility_smoke_test.md](compatibility_smoke_test.md) | CUDA/BF16/NF4 및 최초 TravelUAV gates |
| [feasibility_report.md](feasibility_report.md) | 처음 수행한 실행 가능성 조사 |
| [platform_comparison.md](platform_comparison.md) | 당시 simulator 후보 비교 |

보조 기록: [TravelUAV rendering](traveluav_rendering_decision.md), [native TravelUAV 미실행](native_traveluav_smoke_test.md), [초기 baseline 계획](baseline_reproduction_plan.md), [최초 통합 실패](end_to_end_smoke_test.md), [gateway 미사용 사유](gateway_evaluation.md), [통신 시험 계획](communication_stability_plan.md).

실측 수치는 유지했다. 개인 PC 경로 표기는 `$REPO_ROOT`와 `$UAV_VLA_HOME`으로 치환했다. Git에 없는 raw artifact 링크는 로컬 파일명으로 표시한다. 과거의 checkpoint 보류·통합 실패·논문/benchmark 계획은 당시 기록이며, 현재 과목 프로젝트의 필수 과제가 아니다.
