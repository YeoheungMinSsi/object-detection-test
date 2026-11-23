# Previous single-scale scripts

Moved here to distinguish the previous 30x30 detector from the current multiscale pipeline.
The shared prepare_dataset.py remains in the project root. Current model/ files implement
three-scale outputs and are not compatible with these archived scripts. The old losses
module currently lacks hard_negative_loss, which train.py imports. This was already an
incomplete state before the move; these scripts are preserved, not restored for execution.

Relative imports now refer to singlescale modules. If the old model/loss implementation
is restored later, invoke scripts from the pytorch root with python -m singlescale.train
or python -m singlescale.test_model, not by running files directly inside this folder.
Existing checkpoints and diagnostics folders were not moved. Use train_multiscale.py
and test_multiscale.py for the current model.

### single
singlescale에는 30×30 특징 맵을 만드는 네트워크가 아니라, 그 출력을 처리하던 코드
- 이전 학습 실행 코드
- 30×30 출력용 손실 계산
- 예측 박스 복원·검증·테스트 코드